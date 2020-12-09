import math
import numpy as np
import tensorflow as tf

def caffe_like_padding(input_tensor, padding):
    """A padding method that has same behavior as Caffe's.
        Implementation code from ASLFeat:
        https://github.com/lzx551402/ASLFeat"""

    def PAD(x): return [x, x]
    if len(input_tensor.get_shape()) == 4:
        padded_input = tf.pad(input_tensor,
                              [PAD(0), PAD(padding), PAD(padding), PAD(0)], "CONSTANT")
    elif len(input_tensor.get_shape()) == 5:
        padded_input = tf.pad(input_tensor,
                              [PAD(0), PAD(padding), PAD(padding), PAD(padding), PAD(0)],
                              "CONSTANT")
    return padded_input


def gaussian_multiple_channels(num_channels, sigma):

    r = 2*sigma
    size = 2*r+1
    size = int(math.ceil(size))
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = r

    gaussian = np.float32((np.exp(-1 * (((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2))))) / ((2 * math.pi * (sigma ** 2))**0.5))
    gaussian = gaussian/gaussian.sum()
    weights = np.zeros((size, size, num_channels, num_channels), dtype=np.float32)

    for i in range(num_channels):
        weights[:, :, i, i] = gaussian

    return weights


def create_derivatives_kernel_det():
    # Sobel derivative 3x3 X
    kernel_filter_dx_3 = np.float32(np.asarray([[-1, 0, 1],
                                                [-2, 0, 2],
                                                [-1, 0, 1]]))
    kernel_filter_dx_3 = kernel_filter_dx_3[..., np.newaxis]
    kernel_filter_dx_3 = kernel_filter_dx_3[..., np.newaxis]

    # Sobel derivative 3x3 Y
    kernel_filter_dy_3 = np.float32(np.asarray([[-1, -2, -1],
                                                [0, 0, 0],
                                                [1, 2, 1]]))
    kernel_filter_dy_3 = kernel_filter_dy_3[..., np.newaxis]
    kernel_filter_dy_3 = kernel_filter_dy_3[..., np.newaxis]

    return kernel_filter_dx_3, kernel_filter_dy_3


def create_derivatives_kernel_desc():

    p = np.asarray([[0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711]], np.float32)
    d = np.asarray([[0.018708, 0.125376, 0.193091, 0.000000, -0.193091, -0.125376, -0.018708]], np.float32)

    farid_h_kernel = d.T * p
    farid_h_kernel = farid_h_kernel[..., np.newaxis]
    farid_h_kernel = farid_h_kernel[..., np.newaxis]

    return farid_h_kernel


def create_circular_kernel_mask(s, num_filters):

    center = [s/2., s/2.]
    radius = min(center[0], center[1], s-center[0], s-center[1])

    Y, X = np.ogrid[:s, :s]
    dist_from_center = np.sqrt((X - center[0] +0.5)**2 + (Y-center[1]+0.5)**2)

    mask = np.where(dist_from_center <= radius, np.ones_like(dist_from_center), np.zeros_like(dist_from_center))
    mask = mask[..., np.newaxis]
    mask = mask[..., np.newaxis]

    mask = np.repeat(mask, num_filters, axis=-1)

    return mask

def local_norm_image(x, k_size, clip, eps=1e-10):
    pad = int(k_size / 2)
    x_pad = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'REFLECT')
    x_mean = tf.nn.avg_pool(x_pad, ksize=[1, k_size, k_size, 1], strides=[1, 1, 1, 1], padding='VALID')
    x2_mean = tf.nn.avg_pool(tf.pow(x_pad, 2.0), ksize=[1, k_size, k_size, 1], strides=[1, 1, 1, 1],
                             padding='VALID')
    x_std = (tf.sqrt(tf.abs(x2_mean - x_mean * x_mean)) + eps)
    x_norm = (x - x_mean) / (1. + x_std)

    if clip:
        return tf.clip_by_value(x_norm, -6., 6.)
    else:
        return x_norm


def define_points_per_level(scale_factor_levels, pyramid_levels, upsampled_levels, num_points):
    point_level = []
    tmp = 0.0
    factor_points = (scale_factor_levels ** 2)
    levels = pyramid_levels + upsampled_levels + 1
    for idx_level in range(levels):
        tmp += factor_points ** (-1 * (idx_level - upsampled_levels))
        point_level.append(num_points * factor_points ** (-1 * (idx_level - upsampled_levels)))

    point_level = np.asarray(list(map(lambda x: int(x / tmp), point_level)))

    return point_level, levels