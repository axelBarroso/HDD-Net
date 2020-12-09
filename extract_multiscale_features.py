import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import tensorflow as tf
import numpy as np
from HDDNet.model.hddnet_model import HDDNet, HDDNET_CONF
from HDDNet.utils_hddnet.aux_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def extract_hddnet_features():

    parser = argparse.ArgumentParser(description='HSequences Extract Features')

    parser.add_argument('--list-images', type=str, help='File containing the image paths for extracting features.',
                        required=True)

    parser.add_argument('--results-dir', type=str, default='extracted_features/',
                        help='The output path to save the extracted keypoint.')

    parser.add_argument('--network-version', type=str, default='HDD-Net default',
                        help='The HDD-Net network version name')

    parser.add_argument('--checkpoint-dir', type=str, default='HDDNet/pretrained_model',
                        help='The path to the checkpoint file to load the detector weights.')

    # Multi-Scale Extractor Settings

    parser.add_argument('--extract-MS', type=bool, default=True,
                        help='Set to True if you want to extract multi-scale features.')

    parser.add_argument('--num-points', type=int, default=2048,
                        help='The number of desired features to extract.')

    parser.add_argument('--nms-size', type=int, default=15,
                        help='The NMS size used for extracting keypoints.')

    parser.add_argument('--border-size', type=int, default=15,
                        help='The number of pixels to remove from the score map borders.')

    parser.add_argument('--order-coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use yxsr or xysr.')

    parser.add_argument('--pyramid-levels-extractor', type=int, default=5,
                        help='The number of downsample levels in the pyramid.')

    parser.add_argument('--upsampled-levels', type=int, default=1,
                        help='The number of upsample levels in the pyramid.')

    parser.add_argument('--scale-factor-levels', type=float, default=np.sqrt(2),
                        help='The scale factor between the pyramid levels.')

    # GPU Settings

    parser.add_argument('--gpu-memory-fraction', type=float, default=0.9,
                        help='The fraction of GPU used by the script.')

    parser.add_argument('--gpu-visible-devices', type=str, default="0",
                        help='Set CUDA_VISIBLE_DEVICES variable.')

    args = parser.parse_known_args()[0]

    # remove verbose bits from tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Set CUDA GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible_devices

    version_network_name = args.network_version

    if not args.extract_MS:
        args.pyramid_levels = 0
        args.upsampled_levels = 0

    print('Extract features for : ' + version_network_name)

    check_directory(args.results_dir)
    check_directory(os.path.join(args.results_dir, version_network_name))

    with tf.Graph().as_default():

        tf.set_random_seed(HDDNET_CONF['random_seed'])

        hddnet_model = HDDNet(args)

        init_assign_op, init_feed_dict = tf.contrib.framework.\
            assign_from_checkpoint(tf.train.latest_checkpoint(args.checkpoint_dir), tf.compat.v1.global_variables())

        # GPU Usage
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(init_assign_op, init_feed_dict)

            # read image and extract keypoints and descriptors
            f = open(args.list_images, "r")
            for path_to_image in f:
                path = path_to_image.split('\n')[0]

                if not os.path.exists(path):
                    print('[ERROR]: File {0} not found!'.format(path))
                    return

                create_result_dir(os.path.join(args.results_dir, version_network_name, path))

                im = read_bw_image(path)

                im = im.astype(float) / im.max()

                im_pts, descriptors = hddnet_model.extract_multiscale_features(im, sess)

                file_name = os.path.join(args.results_dir, version_network_name, path)+'.kpt'
                np.save(file_name, im_pts)

                file_name = os.path.join(args.results_dir, version_network_name, path)+'.dsc'
                np.save(file_name, descriptors)

                if len(im_pts) < 0.9 * args.num_points:
                    print('Seems that you may need more features, consider reducing NMS size for extracting more ktps.')

                print('Extracted {} features from {}. '.format(len(im_pts), path))


if __name__ == '__main__':
    extract_hddnet_features()
