import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import maximum_filter

def remove_borders(image, borders):

    shape = image.shape
    new_im = np.zeros_like(image)
    if len(shape) == 4:
        shape = [shape[1], shape[2], shape[3]]
        new_im[:, borders:shape[0]-borders, borders:shape[1]-borders, :] = image[:, borders:shape[0]-borders, borders:shape[1]-borders, :]
    elif len(shape) == 3:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders, :] = image[borders:shape[0] - borders,
                                                                               borders:shape[1] - borders, :]
    else:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders] = image[borders:shape[0] - borders,
                                                                            borders:shape[1] - borders]
    return new_im


def apply_nms(score_map, size):

    score_map = score_map * (score_map == maximum_filter(score_map, footprint=np.ones((size, size))))

    return score_map



def apply_homography_to_points(points, h):

    new_points = []

    for point in points:

        new_point = h.dot([point[0], point[1], 1.0])

        tmp = point[2]**2+np.finfo(np.float32).eps

        Mi1 = [[1/tmp, 0], [0, 1/tmp]]
        Mi1_inv = np.linalg.inv(Mi1)
        Aff = getAff(point[0], point[1], h)

        BMB = np.linalg.inv(np.dot(Aff, np.dot(Mi1_inv, np.matrix.transpose(Aff))))

        [e, _] = np.linalg.eig(BMB)
        new_radious = 1/((e[0] * e[1])**0.5)**0.5

        new_point = [new_point[0] / new_point[2], new_point[1] / new_point[2], new_radious, point[3]]
        new_points.append(new_point)

    return np.asarray(new_points)

def getAff(x,y,H):

    h11 = H[0,0]
    h12 = H[0,1]
    h13 = H[0,2]
    h21 = H[1,0]
    h22 = H[1,1]
    h23 = H[1,2]
    h31 = H[2,0]
    h32 = H[2,1]
    h33 = H[2,2]
    fxdx = h11 / (h31 * x + h32 * y + h33) - (h11 * x + h12 * y + h13) * h31 / (h31 * x + h32 * y + h33) ** 2
    fxdy = h12 / (h31 * x + h32 * y + h33) - (h11 * x + h12 * y + h13) * h32 / (h31 * x + h32 * y + h33) ** 2

    fydx = h21 / (h31 * x + h32 * y + h33) - (h21 * x + h22 * y + h23) * h31 / (h31 * x + h32 * y + h33) ** 2
    fydy = h22 / (h31 * x + h32 * y + h33) - (h21 * x + h22 * y + h23) * h32 / (h31 * x + h32 * y + h33) ** 2

    Aff = [[fxdx, fxdy], [fydx, fydy]]

    return np.asarray(Aff)

def find_index_higher_scores(map, num_points = 1000, threshold = -1):

    # Best n points
    if threshold == -1:

        flatten = map.flatten()
        order_array = np.sort(flatten)

        order_array = np.flip(order_array, axis=0)

        threshold = order_array[num_points-1]
        if threshold <= 0.0:
            indexes = np.argwhere(order_array > 0.0)
            if len(indexes) == 0:
                threshold = 0.0
            else:
                threshold = order_array[indexes[len(indexes)-1]]
        # elif threshold == 0.0:
        #     threshold = order_array[np.nonzero(order_array)].min()

    indexes = np.argwhere(map >= threshold)

    return indexes[:num_points]


def get_point_coordinates(map, scale_value=1., num_points=1000, threshold=-1, order_coord='xysr'):

    indexes = find_index_higher_scores(map, num_points=num_points, threshold=threshold)
    new_indexes = []
    for ind in indexes:

        scores = map[ind[0], ind[1]]
        if order_coord == 'xysr':
            tmp = [ind[1], ind[0], scale_value, scores]
        elif order_coord == 'yxsr':
            tmp = [ind[0], ind[1], scale_value, scores]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes)

def _meshgrid(height, width):
    with tf.name_scope('meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid

def descriptor_crop(images, batch_inds, kpts_xy, crop_size=1):
    # images : [B,H,W,C]
    # out_size : (out_width, out_height)
    # batch_inds : [B*K,] tf.int32 [0,B)
    # kpts_xy : [B*K,2] tf.float32 or whatever
    # kpts_scale : [B*K,] tf.float32
    # kpts_ori : [B*K,2] tf.float32 (cos,sin)

    out_width = out_height = crop_size

    with tf.name_scope('extract_descriptors'):

        num_batch = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        C = tf.shape(images)[3]
        num_kp = tf.shape(kpts_xy)[0]  # B*K
        zero = tf.zeros([], dtype=tf.int32)
        max_y = tf.cast(tf.shape(images)[1] - 1, tf.int32)
        max_x = tf.cast(tf.shape(images)[2] - 1, tf.int32)

        grid = _meshgrid(out_height, out_width)  # normalized -1~1
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_kp]))
        grid = tf.reshape(grid, tf.stack([num_kp, 3, -1]))

        # create 6D affine from scale and orientation
        # [s, 0, 0]   [cos, -sin, 0]
        # [0, s, 0] * [sin,  cos, 0]
        # [0, 0, 1]   [0,    0,   1]

        thetas = tf.eye(2, 3, dtype=tf.float32)
        thetas = tf.tile(thetas[None], [num_kp, 1, 1])

        ones = tf.tile(tf.constant([[[0, 0, 1]]], tf.float32), [num_kp, 1, 1])
        thetas = tf.concat([thetas, ones], axis=1)  # [num_kp, 3,3]

        # Apply transformation to regular grid
        T_g = tf.matmul(thetas, grid)  # [num_kp,3,3] * [num_kp,3,H*W]
        x = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])  # [num_kp,1,H*W]
        y = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])

        # unnormalization [-1,1] --> [-out_size/2,out_size/2]
        x = x * out_width / 2.0
        y = y * out_height / 2.0

        if kpts_xy.dtype != tf.float32:
            kpts_xy = tf.cast(kpts_xy, tf.float32)

        kp_x_ofst = tf.expand_dims(tf.slice(kpts_xy, [0, 0], [-1, 1]), axis=1)  # [B*K,1,1]
        kp_y_ofst = tf.expand_dims(tf.slice(kpts_xy, [0, 1], [-1, 1]), axis=1)  # [B*K,1,1]

        # centerize on keypoints
        x = x + kp_x_ofst
        y = y + kp_y_ofst
        x = tf.reshape(x, [-1])  # num_kp*out_height*out_width
        y = tf.reshape(y, [-1])

        # interpolation
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim2 = width
        dim1 = width * height
        base = tf.tile(batch_inds[:, None], [1, out_height * out_width])  # [B*K,out_height*out_width]
        base = tf.reshape(base, [-1]) * dim1
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        im_flat = tf.reshape(images, tf.stack([-1, C]))  # [B*height*width,C]
        im_flat = tf.cast(im_flat, tf.float32)

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)

        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)

        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        output = tf.reshape(output, tf.stack([num_kp, C]))
        output.set_shape([batch_inds.shape[0], images.shape[-1]])
        return output

