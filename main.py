import numpy as np




def get_interesting_points(data_sobel, threshold):
    """
    define a function that gets out the interesting points from an orignal sobel dataframe
    in order to estimate the starting position for dtw calculation
    :param data_sobel:
    :param ref_val:
    :return:
    """

    data_sobel_norm = data_sobel / (np.nanargmax(data_sobel))
    data_sobel_abs = np.abs(data_sobel)
    maxima = np.zeros(data_sobel.shape[0])

    for bin in (range(data_sobel.shape[0])):
        maxima[bin] = np.nanargmax(data_sobel_abs[bin, :, :])

    int_points = np.argwhere(data_sobel_abs > threshold)

    return maxima, data_sobel_norm, int_points
