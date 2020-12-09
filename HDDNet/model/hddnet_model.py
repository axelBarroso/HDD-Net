from HDDNet.utils_hddnet.model_utils import *
import HDDNet.utils_hddnet.geo_utils as geo_utils
import cv2
from skimage.transform import pyramid_gaussian

HDDNET_CONF = {

    # model
    'random_seed': 1234,
    'is_training': False,
    'reuse': False,

    # Detector block architecture
    'factor_scaling': 1.2,
    'pyramid_levels': 3,
    'num_blocks': 3,
    'num_filters': 8,
    'conv_kernel_size': 5,

    # Descriptor block architecture
    'factor_desc': 4,
    'num_canonical_rot_kernels': 1,
    'size_canonical_rot_kernel': 7,
    'num_rotation_in_kernel': 5,
    'num_hc_filters': 8,
    'descrip_size': 128,
    'apply_upsampling': False,
}

class HDDNet(object):
    def __init__(self, args):

        tf.set_random_seed(HDDNET_CONF['random_seed'])
        np.random.seed(HDDNET_CONF['random_seed'])

        self.bn_epsilon = 1e-5
        self.regularizer = tf.contrib.layers.l2_regularizer(1.0)

        # Descriptor block architecture
        self.descrip_size = HDDNET_CONF['descrip_size']
        self.apply_upsampling = HDDNET_CONF['apply_upsampling']
        if not self.apply_upsampling:
            self.net_factor_scale = 1 / 4.
        else:
            self.net_factor_scale = 1
        self.factor_scaling = HDDNET_CONF['factor_scaling']
        self.pyramid_levels = HDDNET_CONF['pyramid_levels']
        self.factor_desc = HDDNET_CONF['factor_desc']

        self.der_desc_kernels = create_derivatives_kernel_desc()
        self.num_canonical_rot_kernels = HDDNET_CONF['num_canonical_rot_kernels']
        self.size_canonical_rot_kernel = HDDNET_CONF['size_canonical_rot_kernel']
        self.num_rotation_in_kernel = HDDNET_CONF['num_rotation_in_kernel']
        self.num_hc_filters = HDDNET_CONF['num_hc_filters']

        self.circular_mask = create_circular_kernel_mask(self.size_canonical_rot_kernel, self.num_canonical_rot_kernels)

        # Detector block architecture
        self.num_blocks = HDDNET_CONF['num_blocks']
        self.num_filters = HDDNET_CONF['num_filters']
        self.conv_kernel_size = HDDNET_CONF['conv_kernel_size']

        name_scope_det = 'model_deep_detector'
        # Smooth Gausian Filter
        gaussian_avg = gaussian_multiple_channels(1, 1.5)
        self.gaussian_avg = tf.constant(gaussian_avg, name=name_scope_det + '_Gaussian_avg')

        # Sobel derivatives
        kernel_x, kernel_y = create_derivatives_kernel_det()
        self.kernel_filter_dx = tf.constant(kernel_x, name=name_scope_det + '_kernel_filter_dx')
        self.kernel_filter_dy = tf.constant(kernel_y, name=name_scope_det + '_kernel_filter_dy')

        self.is_training = False
        self.reuse = False
        self.nms_size = args.nms_size
        self.border_size = args.border_size
        self.order_coord = args.order_coord
        self.scale_factor_levels = args.scale_factor_levels
        self.pyramid_levels_extractor = args.pyramid_levels_extractor
        self.upsampled_levels = args.upsampled_levels
        self.num_points = args.num_points
        self.point_level, self.levels = define_points_per_level(self.scale_factor_levels, self.pyramid_levels_extractor,
                                                                self.upsampled_levels, self.num_points+1)
        # initialize model
        self.network = self.model()

    def define_placeholders(self):

        self.place_holders = {}
        with tf.name_scope('inputs'):

            # Define the input tensor shape
            tensor_input_shape = (None, None, None, 1)
            tensor_input_desc_shape = (None, None, None, self.descrip_size)

            input_network = tf.placeholder(dtype=tf.float32, shape=tensor_input_shape, name='input_network')
            input_desc_network = tf.placeholder(dtype=tf.float32, shape=tensor_input_desc_shape, name='input_desc_network')
            kpts = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='kpts')
            batch_inds = tf.placeholder(dtype=tf.int32, shape=(None,), name='batch_inds')

        self.place_holders['input_network'] = input_network
        self.place_holders['input_desc_network'] = input_desc_network
        self.place_holders['kpts'] = kpts
        self.place_holders['batch_inds'] = batch_inds

    def model(self):

        self.define_placeholders()
        input_data = self.place_holders['input_network']
        dim = [tf.shape(input_data)[1], tf.shape(input_data)[2]]
        dim_float = tf.cast(dim, tf.float32)
        network = {}

        with tf.name_scope('model_deep_detector'):
            network = self.compute_detector(input_data, self.is_training, self.reuse, network, dim_float)

        with tf.name_scope('model_deep_descriptor'):
            network = self.compute_descriptor(input_data, self.is_training, self.reuse, network, dim_float)

        descs_extractor = self.create_descriptor_extractor(self.place_holders)
        network['descs_extractor'] = descs_extractor

        return network

    def compute_detector(self, input_data, is_training, reuse, network, dim_float):

        for idx_level in range(self.pyramid_levels):

            if idx_level == 0:
                input_data_smooth = input_data
            else:
                input_data_smooth = tf.nn.conv2d(input_data, self.gaussian_avg, strides=[1, 1, 1, 1], padding='SAME')

            input_data_resized = tf.image.resize_images(input_data_smooth, size=tf.cast(
                (dim_float[0] / (self.factor_scaling ** idx_level), dim_float[1] / (self.factor_scaling ** idx_level)),
                tf.int32), align_corners=True, method=0)

            input_data_resized = local_norm_image(input_data_resized, k_size=65, clip=False)

            features_t, network = self.compute_handcrafted_features(input_data_resized, network, idx_level)

            for idx_layer in range(self.num_blocks):
                features_t = self.conv_block_det(features_t, str(idx_layer + 1), reuse or idx_level > 0, is_training,
                                             num_filters=self.num_filters, size_kernel=self.conv_kernel_size)

            features_t = tf.image.resize_images(features_t, size=tf.cast((dim_float[0], dim_float[1]), tf.int32),
                                                align_corners=True, method=0)

            if not idx_level:
                features = features_t
            else:
                features = tf.concat([features, features_t], axis=3)

        features = tf.layers.batch_normalization(inputs=features, scale=True, training=is_training,
                                                 name=tf.contrib.framework.get_name_scope() + '_batch_final', reuse=reuse)

        output = self.conv_block_det(features, 'last_layer', reuse, is_training, num_filters=1,
                                     size_kernel=self.conv_kernel_size, batchnorm=False, relu=False)

        network['input_data'] = input_data
        network['features'] = features
        network['score_map'] = tf.nn.relu(output)

        return network

    def compute_descriptor(self, input_data, is_training, reuse, network, dim):

        dim_float = tf.cast(dim, tf.float32)
        for idx_level in range(self.pyramid_levels):

            if idx_level == 0:
                input_data_resized = input_data
            else:
                input_data_resized = tf.nn.conv2d(input_data, self.gaussian_avg, strides=[1, 1, 1, 1], padding='SAME')

                input_data_resized = tf.image.resize_images(input_data_resized, size=tf.cast(
                    (dim_float[0] / (self.factor_scaling ** idx_level),
                     dim_float[1] / (self.factor_scaling ** idx_level)),
                    tf.int32), align_corners=True, method=0)

            reuse_i = idx_level > 0 or reuse

            input_data_resized = local_norm_image(input_data_resized, k_size=31, clip=True)
            hc_features = self.compute_hc_dense_features(input_data_resized)
            hc_features = self.batch_normalization(hc_features, relu=True, name='convhc/bn', reuse=reuse_i, training=is_training)
            data_plus_hc = tf.concat([input_data_resized, hc_features], axis=-1)

            feats_i = self.conv_bn(data_plus_hc, 3, 32, 1, name='conv0', reuse=reuse_i, training=is_training)
            feats_i = self.conv(feats_i, 3, 32, 1, biased=False, relu=False, name='conv1', reuse=reuse_i)
            feats_i = self.batch_normalization(feats_i, relu=True, name='conv1/bn', reuse=reuse_i, training=is_training)
            feats_i = self.conv_bn(feats_i, 3, 64, 2, name='conv2', reuse=reuse_i, training=is_training)
            feats_i = self.conv(feats_i, 3, 64, 1, biased=False, relu=False, name='conv3', reuse=reuse_i)
            feats_i = self.batch_normalization(feats_i, relu=True, name='conv3/bn', reuse=reuse_i, training=is_training)
            feats_i = self.conv_bn(feats_i, 3, 128, 2, name='conv4', reuse=reuse_i, training=is_training)
            feats_i = self.conv_bn(feats_i, 3, 128, 1, name='conv5_0', reuse=reuse_i, training=is_training)

            conv5_i = tf.image.resize_images(feats_i, size=tf.cast(
                (dim_float[0] / 4, dim_float[1] / 4), tf.int32), align_corners=True, method=0)

            if idx_level:
                conv5 = tf.concat([conv5, conv5_i], axis=3)
            else:
                conv5 = conv5_i

        conv6 = self.conv_bn(conv5, 3, 128, 1, name='conv6_0', reuse=reuse, training=is_training)
        conv6 = self.conv_bn(conv6, 3, 128, 1, name='conv6_1', reuse=reuse, training=is_training)
        conv6 = self.conv(conv6, 3, 128, 1, biased=False, relu=False, name='conv6', reuse=reuse)

        if self.apply_upsampling:
            conv6 = tf.image.resize_bicubic(conv6, size=(dim[0], dim[1]), align_corners=True)

        features_norm = tf.nn.l2_normalize(conv6, axis=-1)

        network['features_desc_raw'] = conv6
        network['features_desc'] = features_norm
        network['input_data'] = input_data

        return network


    def conv_block_det(self, features, name, reuse, is_training, num_filters, size_kernel, batchnorm=True, relu=True):

        features = tf.layers.conv2d(inputs=features, filters=num_filters,
                                    kernel_size=size_kernel,
                                    strides=1, padding='SAME', use_bias=True,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                    data_format='channels_last',
                                    name=tf.contrib.framework.get_name_scope() + '_conv_'+name, reuse=reuse)

        if batchnorm:
            features = tf.layers.batch_normalization(inputs=features, scale=True, training=is_training,
                                                 name=tf.contrib.framework.get_name_scope() + '_batch_'+name, reuse=reuse)

        if relu:
            features = tf.nn.relu(features)

        return features

    def conv_bn(self, input_tensor, kernel_size, filters, strides, name, relu=True, center=False, scale=False,
                dilation_rate=1, padding='SAME', biased=False, separable=False, reuse=False,
                training=False):

        conv = self.conv(input_tensor, kernel_size, filters, strides, name, relu=False,
                         dilation_rate=dilation_rate, padding=padding,
                         biased=biased, reuse=reuse, separable=separable)
        conv_bn = self.batch_normalization(conv, name + '/bn',
                                           center=center, scale=scale, relu=relu, reuse=reuse, training=training)
        return conv_bn


    def conv(self, input_tensor, kernel_size, filters, strides, name, relu=True, dilation_rate=1, padding='SAME',
             biased=True, reuse=False, kernel_init=None, bias_init=tf.zeros_initializer, separable=False,
             data_format='channels_last'):
        """2D/3D convolution.
        Implementation code from ASLFeat:
        https://github.com/lzx551402/ASLFeat"""

        kwargs = {'filters': filters,
                  'kernel_size': kernel_size,
                  'strides': strides,
                  'activation': tf.nn.relu if relu else None,
                  'use_bias': biased,
                  'dilation_rate': dilation_rate,
                  'trainable': True,
                  'reuse': reuse,
                  'bias_regularizer': self.regularizer if biased else None,
                  'kernel_initializer': kernel_init,
                  'bias_initializer': bias_init,
                  'name': name,
                  'data_format': data_format}

        if separable:
            kwargs['depthwise_regularizer'] = self.regularizer
            kwargs['pointwise_regularizer'] = self.regularizer
        else:
            kwargs['kernel_regularizer'] = self.regularizer

        if isinstance(padding, str):
            padded_input = input_tensor
            kwargs['padding'] = padding
        else:
            padded_input = caffe_like_padding(input_tensor, padding)
            kwargs['padding'] = 'VALID'

        if not separable:
            return tf.compat.v1.layers.conv2d(padded_input, **kwargs)
        else:
            return tf.layers.separable_conv2d(padded_input, **kwargs)


    def batch_normalization(self, input_tensor, name, training, axis=-1,
                            center=False, scale=False, relu=False, reuse=False):
        """Batch normalization.
        Implementation code from ASLFeat:
        https://github.com/lzx551402/ASLFeat"""

        output = tf.compat.v1.layers.batch_normalization(input_tensor, axis=axis, center=center, scale=scale, fused=True,
                                                         training=training, reuse=reuse, epsilon=self.bn_epsilon,
                                                         gamma_regularizer=None, beta_regularizer=None, name=name)
        if relu:
            output = self.relu(output, name + '/relu')
        return output


    def relu(self, input_tensor, name=None):
        """ReLu activation.
        Implementation code from ASLFeat:
        https://github.com/lzx551402/ASLFeat"""

        return tf.nn.relu(input_tensor, name=name)

    def compute_handcrafted_features(self, image, network, idx):

        # Sobel_conv_derivativeX
        dx = tf.nn.conv2d(image, self.kernel_filter_dx, strides=[1, 1, 1, 1], padding='SAME')
        dxx = tf.nn.conv2d(dx, self.kernel_filter_dx, strides=[1, 1, 1, 1], padding='SAME')
        dx2 = tf.multiply(dx, dx)

        # Sobel_conv_derivativeY
        dy = tf.nn.conv2d(image, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')
        dyy = tf.nn.conv2d(dy, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')
        dy2 = tf.multiply(dy, dy)

        dxy = tf.nn.conv2d(dx, self.kernel_filter_dy, strides=[1, 1, 1, 1], padding='SAME')

        dxdy = tf.multiply(dx, dy)
        dxxdyy = tf.multiply(dxx, dyy)
        dxy2 = tf.multiply(dxy, dxy)

        # Concatenate Handcrafted Features
        features_t = tf.concat([dx, dx2, dxx, dy, dy2, dyy, dxdy, dxxdyy, dxy, dxy2], axis=3)

        network['dx_' + str(idx + 1)] = dx
        network['dx2_' + str(idx + 1)] = dx2
        network['dy_' + str(idx + 1)] = dy
        network['dy2_' + str(idx + 1)] = dy2
        network['dxdy_' + str(idx + 1)] = dxdy
        network['dxxdyy_' + str(idx + 1)] = dxxdyy
        network['dxy_' + str(idx + 1)] = dxy
        network['dxy2_' + str(idx + 1)] = dxy2
        network['dx2dy2_' + str(idx + 1)] = dx2+dy2

        return features_t, network

    def compute_hc_dense_features(self, input):

        rot_angle = np.pi / self.num_hc_filters
        rot_angle_within_block = rot_angle/self.num_rotation_in_kernel

        weights = tf.constant(self.der_desc_kernels)
        weights = tf.transpose(weights, [3, 0, 1, 2])

        for r in range(0, self.num_hc_filters):
            for i in range(self.num_rotation_in_kernel):
                tmp = tf.contrib.image.rotate(weights, rot_angle_within_block * (i - 1) + r * rot_angle - np.pi / 2., interpolation='NEAREST')
                tmp = tf.transpose(tmp, [1, 2, 3, 0])
                tmp = tmp * self.circular_mask
                tmp_block = tf.nn.conv2d(input, tmp, strides=[1, 1, 1, 1], padding='SAME')

                tmp_ort = tf.contrib.image.rotate(weights, rot_angle_within_block * (i - 1) + r * rot_angle, interpolation='NEAREST')
                tmp_ort = tf.transpose(tmp_ort, [1, 2, 3, 0])
                tmp_ort = tmp_ort * self.circular_mask
                tmp_block_ort = tf.nn.conv2d(input, tmp_ort, strides=[1, 1, 1, 1], padding='SAME')

                tmp_block = tf.concat([tmp_block, tmp_block_ort * tmp_block, tf.pow(tmp_block, 2.),
                                       tf.nn.relu(tmp_block), tf.nn.relu(-1 * tmp_block)], axis=-1)

                tmp_block = tf.expand_dims(tmp_block, axis=-1)

                if i == 0:
                    block = tmp_block
                else:
                    block = tf.concat([block, tmp_block], axis=-1)
            rotated_feature = tf.reduce_max(block, axis=-1, keepdims=False)

            if r == 0:
                hc_features = rotated_feature
            else:
                hc_features = tf.concat([hc_features, rotated_feature], axis=-1)

        return hc_features

    def create_descriptor_extractor(self, place_holders):

        input_desc_network = place_holders['input_desc_network']
        batch_inds = place_holders['batch_inds']
        kpts = place_holders['kpts']

        descs_extractor_tmp = geo_utils.descriptor_crop(input_desc_network, batch_inds, kpts)
        descs_extractor = tf.nn.l2_normalize(descs_extractor_tmp, axis=-1)

        return descs_extractor

    def extract_multiscale_features(self, image, sess):

        # Define end nodes
        score_map = self.network['score_map']
        desc_map = self.network['features_desc_raw']
        descs_extractor = self.network['descs_extractor']

        # Define place holders
        input_im = self.place_holders['input_network']
        input_desc_network = self.place_holders['input_desc_network']
        batch_inds = self.place_holders['batch_inds']
        kpts = self.place_holders['kpts']

        pyramid = pyramid_gaussian(image, max_layer=self.pyramid_levels_extractor, downscale=self.scale_factor_levels)

        score_maps = {}
        desc_maps = {}
        for (j, resized) in enumerate(pyramid):
            im = resized.reshape(1, resized.shape[0], resized.shape[1], 1)

            new_im = im[:, :resized.shape[0] - int(resized.shape[0] % (1 / self.net_factor_scale)),
                     :resized.shape[1] - int(resized.shape[1] % (1 / self.net_factor_scale)), :]

            i_scores = sess.run(score_map, feed_dict={input_im: new_im})
            i_desc = sess.run(desc_map, feed_dict={input_im: new_im})

            i_scores = geo_utils.remove_borders(i_scores, borders=self.border_size)
            score_maps['map_' + str(j + 1 + self.upsampled_levels)] = i_scores[0, :, :, 0]
            desc_maps['map_' + str(j + 1 + self.upsampled_levels)] = i_desc[:, :, :, :]

        if self.upsampled_levels:
            for j in range(self.upsampled_levels):
                factor = self.scale_factor_levels ** (self.upsampled_levels - j)
                up_image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

                im = up_image.reshape(1, up_image.shape[0], up_image.shape[1], 1)
                new_im = im[:, :up_image.shape[0] - int(up_image.shape[0] % (1 / self.net_factor_scale)),
                         :up_image.shape[1] - int(up_image.shape[1] % (1 / self.net_factor_scale)), :]

                i_scores = sess.run(score_map, feed_dict={input_im: new_im})
                i_desc = sess.run(desc_map, feed_dict={input_im: new_im})

                i_scores = geo_utils.remove_borders(i_scores, borders=self.border_size)
                score_maps['map_' + str(j + 1)] = i_scores[0, :, :, 0]
                desc_maps['map_' + str(j + 1)] = i_desc[:, :, :, :]

        im_pts = []
        for idx_level in range(self.levels):

            scale_value = (self.scale_factor_levels ** (idx_level - self.upsampled_levels))
            scale_factor = 1. / scale_value

            h_scale = np.asarray([[scale_factor, 0., 0.], [0., scale_factor, 0.], [0., 0., 1.]])
            h_scale_inv = np.linalg.inv(h_scale)
            h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

            num_points_level = self.point_level[idx_level]
            if idx_level > 0:
                res_points = int(np.asarray([self.point_level[a] for a in range(0, idx_level + 1)]).sum() - len(im_pts))
                num_points_level = res_points

            im_scores = geo_utils.apply_nms(score_maps['map_' + str(idx_level + 1)], self.nms_size)
            im_pts_tmp = geo_utils.get_point_coordinates(im_scores, num_points=num_points_level, order_coord='xysr')

            im_pts_input = np.asarray(list(map(lambda x: [x[0] * self.net_factor_scale, x[1] * self.net_factor_scale], im_pts_tmp)))

            feed_dict = {
                input_desc_network: desc_maps['map_' + str(idx_level + 1)],
                kpts: im_pts_input,
                batch_inds: np.zeros((len(im_pts_tmp))),
            }
            im_desc_tmp = sess.run(descs_extractor, feed_dict=feed_dict)
            im_pts_tmp = geo_utils.apply_homography_to_points(im_pts_tmp, h_scale_inv)

            if not idx_level:
                im_pts = im_pts_tmp
                im_desc = im_desc_tmp
            else:
                im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)
                im_desc = np.concatenate((im_desc, im_desc_tmp), axis=0)

        if self.order_coord == 'yxsr':
            im_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], im_pts)))

        sort_idx = (-1 * im_pts[:, 3]).argsort()
        im_pts = im_pts[sort_idx[:self.num_points]]
        im_desc = im_desc[sort_idx[:self.num_points]]

        return im_pts, im_desc

