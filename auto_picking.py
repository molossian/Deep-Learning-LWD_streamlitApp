import cv2 as cv
import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtw import dtw
from tqdm import tqdm
import multiprocessing
from functools import partial
################ AGGIUNTE DI MICH ##########################
from math import isinf
from numpy import array, zeros, full, argmin, inf, ndim
def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def manhattan_distance(target_seq, input_seq):
    return np.abs(input_seq - target_seq)

def manhattan_dtw(x, y, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= np.abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, np.max(1, i - w):np.min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (np.max(0, i - w) <= j <= np.min(c, i + w))):
                D1[i, j] = manhattan_distance(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(np.max(0, i - w), np.min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += np.min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def dtw_calc_faster(sobel_data, interesting_points):
    dtw_data = np.zeros((634, 15, 11))
    block_df = np.zeros((634, 10, 15, 12))
    interesting_data = np.zeros((interesting_points.shape[0], 3))

    interesting_data[:, 0] = interesting_points[:, 0]
    interesting_data[:, 1] = interesting_points[:, 2]
    interesting_data[:, 2] = interesting_points[:, 1]

    for depth_bin in tqdm(range(634)):
        for i in range(sobel_data[depth_bin, :, :].shape[1] - 2):
            x, y = sobel_data[depth_bin, 10-5:10+5, i], sobel_data[depth_bin, 10-5:10+5, i + 1]
            d, cost_matrix, acc_cost_matrix, path = manhattan_dtw(x, y)
            dtw_data[depth_bin, i, 0] = d
            block_df[depth_bin, :, i, 0] = x
            block_df[depth_bin, :, i, 1] = y
            y_up = sobel_data[depth_bin, 5:15, i + 1][np.arange(5) - np.arange(5)[:, np.newaxis]]
            d_up, cost_matrix_up, acc_cost_matrix_up, path_up = manhattan_dtw(x, y_up)
            dtw_data[depth_bin, i, 1:6] = d_up
            block_df[depth_bin, :, i, 2:7] = y_up[::-1, :]
            y_down = sobel_data[depth_bin, 0:10, i + 1][np.arange(5) + np.arange(5)[:, np.newaxis]]
            d_down, cost_matrix_down, acc_cost_matrix_down, path_down = manhattan_dtw(x, y_down)
            dtw_data[depth_bin, i, 6:] = d_down
            block_df[depth_bin, :, i, 7:] = y_down

    return dtw_data, block_df, interesting_data.astype('int')

############################################################
def read_las(path, name, plot=False):
    """
    reading las file into the program and
    show a random interval of data to check proper working
    of the function
    """

    data = lasio.read(path + name, engine='normal').data
    data = data[:, :-3]
    a = np.random.randint(0, data.shape[0] // 2)
    if plot:

        plt.figure(figsize=[10, 30])
        plt.imshow(data[a:a + 20, 1:], aspect='auto', cmap='afmhot_r')
        plt.show()
    return data, a


def smoothing(data, a,plot=False ):
    """
    smoothing the raw file in order
    to keep useful and easy some future operations (e.g derivative calculation)
    """

    kernel = np.ones((5, 5), np.float32) / 25
    data_smooth = cv.filter2D(data[:, 1:], -1, kernel)
    if plot:
        plt.figure(figsize=[10, 30])
        plt.imshow(data_smooth[a:a + 20, 1:], aspect='auto', cmap='afmhot_r')
        plt.title('Sample:' + str(a))
        plt.show()
    return data_smooth


def reshape(data_smooth, target_bin, plot=False):
    """
    reshaping data is required in order to bin
    the drilled interval in arbitrarily big windows
    in which to carry out calculations
    """

    data_smooth = data_smooth[:-4, :]

    data_resh = np.reshape(data_smooth, (data_smooth.shape[0] // target_bin, target_bin, data_smooth.shape[1]))
    if plot:
        plt.figure(figsize=[10, 30])
        plt.imshow(data_resh[625, 1:], aspect='auto', cmap='afmhot_r')
        plt.title('Bin 625')
        plt.show()

    return data_resh

def Sobel_calc2(data_resh, target_bin, k_size):
    data_deriv = np.zeros((data_resh.shape[0], target_bin + 2, data_resh.shape[2] + 2))

    for b in range(data_resh.shape[0]):
        data_deriv[b, 1:-1, 1:-1] = data_resh[b, :, :]  # adding periodic boundaries
        data_deriv[b, 1:-1, 0] = data_resh[b, :, -1]
        data_deriv[b, 1:-1, -1] = data_resh[b, :, 0]

    sobel_data = cv.Sobel(data_deriv[i, :, :], cv.CV_64F, 1, 1, ksize=k_size)
    sobel_data[np.isnan(sobel_data)] = 0
    maximum = np.argmax(sobel_data)
    ds_fl = sobel_data.flatten()
    max_pt = ds_fl[maximum]
    threshold = (0.4 * max_pt)

    return sobel_data, threshold, ds_fl, maximum, max_pt

def Sobel_calc(data_resh, target_bin, k_size, a, plot=False):
    """
    define the function through which we calculate derivative of Image log
    to get the sharpest contrasts (or interesting points in image) before starting to
    seek for similarity between numerical series

    :param data_resh: on which to calculate sobel der
    :param target_bin: window size
    :param k_size: kernel size for sobel filter
    :return: new data representing main density contrasts
    """
    data_deriv = np.zeros((data_resh.shape[0], target_bin + 2, data_resh.shape[2] + 2))

    for b in tqdm(range(data_resh.shape[0])):
        data_deriv[b, 1:-1, 1:-1] = data_resh[b, :, :]  # it's like to add periodic boundaries
        data_deriv[b, 1:-1, 0] = data_resh[b, :, -1]
        data_deriv[b, 1:-1, -1] = data_resh[b, :, 0]
    sobel_data = np.zeros(data_deriv.shape)
    for i in tqdm(range(634)):
        sobel_data[i, :, :] = cv.Sobel(data_deriv[i, :, :], cv.CV_64F, 1, 1, ksize=k_size)[:, :]
    sobel_data = sobel_data[:, 1:-1, 1:-1]
    sobel_data[np.isnan(sobel_data)] = 0
    maximum = np.argmax(sobel_data)
    ds_fl = sobel_data.flatten()
    max_pt = ds_fl[maximum]
    threshold = (0.4 * max_pt)
    if plot:
        plt.figure(figsize=[10, 30])
        plt.imshow(sobel_data[625, :], aspect='auto', cmap='viridis')
        plt.title('Sobel 625')
        plt.colorbar()
        plt.show()

    return sobel_data, threshold, ds_fl, maximum, max_pt


def get_interesting_points(sobel_data, threshold):
    """
    define a function that gets out the interesting points from an original sobel dataframe
    in order to estimate the starting position for dtw calculation
    :param sobel_data:
    :param threshold:
    :return:
    """

    data_sobel_norm = sobel_data / (np.nanargmax(sobel_data))
    maxima = np.zeros(sobel_data.shape[0])

    for bin in (range(sobel_data.shape[0])):
        maxima[bin] = np.nanargmax(sobel_data[bin, :, :])

    i_points = np.argwhere(np.abs(sobel_data) > threshold)

    return maxima, data_sobel_norm, i_points


def dtw_calc(sobel_data, interesting_points, guide: bool = False):
    """
    calculate Dynamic Time Warping for sliding windows in numerical series
    in order to find the most similar portions, which are likely to belong
    to the same contrast
    :param interesting_points:
    :param sobel_data: array for dtw computation
    :param guide: if the results are guided by the results of get_interesting_points function
    :return:
    """

    dtw_data = np.zeros((634, 15, 11))
    dtw_data_guided = np.zeros((634, 15, 11))
    block_df = np.zeros((634, 10, 15, 12))
    interesting_data = np.zeros((interesting_points.shape[0], 3))

    bin = np.zeros(interesting_points.shape[0])
    sample = np.zeros(interesting_points.shape[0])
    curve = np.zeros(interesting_points.shape[0])
    x = []

    for coord in range(interesting_points.shape[0]):
        bin[coord] = interesting_points[coord][0]
        sample[coord] = interesting_points[coord][1]
        curve[coord] = interesting_points[coord][2]

    interesting_data[:, 0] = bin
    interesting_data[:, 1] = curve
    interesting_data[:, 2] = sample

    # define a distance measure to guide the computation of DTW in the seek of minimum distance as
    # the best similarity measure
    manhattan_distance = lambda x, y: np.abs(y - x)

    if guide == True:
        for depth_bin in tqdm(range(634)):
            for i in range(sobel_data[depth_bin, :, :].shape[1] - 2):
                for j in range(sobel_data[depth_bin, :, :].shape[0] - 2):
                    if j == 10:
                        x, y = sobel_data[depth_bin, j - 5:j + 5, i], sobel_data[depth_bin, j - 5:j + 5, i + 1]
                        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
                        dtw_data[depth_bin, i, 0] = d

                        block_df[depth_bin, :, i, 0] = x
                        block_df[depth_bin, :, i, 1] = y

                        depth = np.arange(0, len(x))

                        for s in range(1, 6):
                            y_up = sobel_data[depth_bin, (j - s) - 5:(j - s) + 5, i + 1]
                            d_up, cost_matrix_up, acc_cost_matrix_up, path_up = dtw(x, y_up, dist=manhattan_distance)
                            dtw_data[depth_bin, i, s] = d_up
                            block_df[depth_bin, :, i, s + 1] = y_up
                            y_down = sobel_data[depth_bin, (j + s) - 5:(j + s) + 5, i + 1]
                            d_down, cost_matrix_down, acc_cost_matrix_down, path_down = dtw(x, y_down, dist=manhattan_distance)
                            dtw_data[depth_bin, i, s + 5] = d_down
                            block_df[depth_bin, :, i, s + 6] = y_down
                            # for coord in range(int_points.shape[0]):
                            #     if j == sample[coord] and i == curve[coord]:
                            #         dtw_data_guided[depth_bin, i, :] = dtw_data[depth_bin, i, :]

    return dtw_data, dtw_data_guided, block_df, interesting_data.astype('int')


def IntPointsDtw(raw_data, int_points, interesting_data):
    manhattan_distance = lambda target_seq, input_seq: np.abs(input_seq - target_seq)

    wrap_data = np.zeros((raw_data.shape[0], 17, 11))
    c = np.zeros((interesting_data.shape[0], 2, 16))
    x = np.zeros((0))
    y = np.zeros((0))

    for i in tqdm(range(interesting_data.shape[0])):

        j = int(interesting_data[i][1])
        k = (int(interesting_data[i][0]) * 20 + int(interesting_data[i][2]))

        if k > 20:
            target_seq = raw_data[k - 5:k + 5, j]
            c[i, 0, j] = k
            c[i, 1, j] = j

            if j < (raw_data.shape[1] - 2):

                input_seq = raw_data[k - 5:k + 5, j + 1]
            else:

                input_seq = raw_data[k - 5:k + 5, 0]

            d, cost_matrix, acc_cost_matrix, path = dtw(target_seq, input_seq, dist=manhattan_distance)

            wrap_data[k, j + 1, 0] = d

            for s in range(1, 6):
                if j < (raw_data.shape[1] - 2):

                    input_seq_up = raw_data[(k - s) - 5:(k - s) + 5, j + 1]
                    input_seq_down = raw_data[(k + s) - 5:(k + s) + 5, j + 1]
                    d_down, cost_matrix_down, acc_cost_matrix_down, path_down = dtw(target_seq, input_seq_down, dist=manhattan_distance)
                    d_up, cost_matrix_up, acc_cost_matrix_up, path_up = dtw(target_seq, input_seq_up, dist=manhattan_distance)
                    wrap_data[k, j + 1, 5 + s] = d_down
                    wrap_data[k, j + 1, s] = d_up

                else:
                    input_seq_up = raw_data[(k - s) - 5:(k - s) + 5, 0]
                    input_seq_down = raw_data[(k + s) - 5:(k + s) + 5, 0]
                    d_up, cost_matrix_up, acc_cost_matrix_up, path_up = dtw(target_seq, input_seq_up, dist=manhattan_distance)
                    d_down, cost_matrix_down, acc_cost_matrix_down, path_down = dtw(target_seq, input_seq_down, dist=manhattan_distance)
                    wrap_data[k, j + 1, s] = d_up
                    wrap_data[k, j + 1, 5 + s] = d_down

            for j in range(wrap_data.shape[1] - 2):
                u = wrap_data[k, j + 1, :]
                if u[0] != 0:
                    uu = np.argmin(u)
                    uu = int(uu)
                    if uu < 6:
                        k1 = k - uu
                    else:
                        k1 = k + (uu - 5)
                    z = j + 1
                    if z < (raw_data.shape[1] - 2):
                        target_seq = raw_data[k1 - 5:k1 + 5, z]
                        input_seq = raw_data[k1 - 5:k1 + 5, z + 1]
                        c[i, 0, z] = k1
                        c[i, 1, z] = z
                        d, cost_matrix, acc_cost_matrix, path = dtw(target_seq, input_seq, dist=manhattan_distance)
                        wrap_data[k1, z + 1, 0] = d
                    else:
                        target_seq = raw_data[k1 - 5:k1 + 5, 0]
                        input_seq = raw_data[k1 - 5:k1 + 5, 1]
                        c[i, 0, 0] = k1
                        c[i, 1, 0] = 0
                        d, cost_matrix, acc_cost_matrix, path = dtw(target_seq, input_seq, dist=manhattan_distance)
                        wrap_data[k1, 0, 0] = d
                    for s in range(1, 6):
                        if z < (raw_data.shape[1] - 2):
                            input_seq_up = raw_data[(k1 - s) - 5:(k1 - s) + 5, z + 1]
                            input_seq_down = raw_data[(k1 + s) - 5:(k1 + s) + 5, z + 1]

                            d_up, cost_matrix_up, acc_cost_matrix_up, path_up = dtw(target_seq, input_seq_up,
                                                                                    dist=manhattan_distance)
                            d_down, cost_matrix_down, acc_cost_matrix_down, path_down = dtw(target_seq,
                                                                                            input_seq_down,
                                                                                            dist=manhattan_distance)
                            wrap_data[k1, z + 1, s] = d_up
                            wrap_data[k1, z + 1, 5 + s] = d_down
                        else:
                            input_seq_up = raw_data[(k1 - s) - 5:(k1 - s) + 5, 0]
                            input_seq_down = raw_data[(k1 + s) - 5:(k1 + s) + 5, 0]

                            d_up, cost_matrix_up, acc_cost_matrix_up, path_up = dtw(target_seq, input_seq_up,
                                                                                    dist=manhattan_distance)
                            d_down, cost_matrix_down, acc_cost_matrix_down, path_down = dtw(target_seq, input_seq_down, dist=manhattan_distance)
                            wrap_data[k1, 0, s] = d_up
                            wrap_data[k1, 0, 5 + s] = d_down
                    for z in range(wrap_data.shape[1] - 2):
                        u = wrap_data[k1, z + 1, :]
                        if u[0] != 0:
                            uu = int(np.argmin(u))
                            if uu < 6:
                                k2 = k1 - uu
                            else:
                                k2 = k1 + (uu - 5)
                            z1 = z + 1
                            if z1 < (raw_data.shape[1] - 2):
                                target_seq = raw_data[k2 - 5:k2 + 5, z1]
                                input_seq = raw_data[k2 - 5:k2 + 5, z1 + 1]
                                c[i, 0, z1] = k2
                                c[i, 1, z1] = z1
                                d, cost_matrix, acc_cost_matrix, path = dtw(target_seq, input_seq, dist=manhattan_distance)
                                wrap_data[k2, z1 + 1, 0] = d
                            else:
                                target_seq = raw_data[k2 - 5:k2 + 5, 0]
                                input_seq = raw_data[k2 - 5:k2 + 5, 1]
                                c[i, 0, 0] = k2
                                c[i, 1, 0] = 0
                                d, cost_matrix, acc_cost_matrix, path = dtw(target_seq, input_seq, dist=manhattan_distance)
                                wrap_data[k2, 0, 0] = d
                            for s in range(1, 6):
                                if z1 < (raw_data.shape[1] - 2):
                                    input_seq_up = raw_data[(k2 - s) - 5:(k2 - s) + 5, z1 + 1]
                                    input_seq_down = raw_data[(k2 + s) - 5:(k2 + s) + 5, z1 + 1]
                                    d_down, cost_matrix_down, acc_cost_matrix_down, path_down = dtw(target_seq, input_seq_down, dist=manhattan_distance)
                                    d_up, cost_matrix_up, acc_cost_matrix_up, path_up = dtw(target_seq, input_seq_up, dist=manhattan_distance)
                                    wrap_data[k2, z1 + 1, s] = d_up
                                    wrap_data[k2, z1 + 1, 5 + s] = d_down
                                else:
                                    input_seq_up = raw_data[(k2 - s) - 5:(k2 - s) + 5, 0]
                                    input_seq_down = raw_data[(k2 + s) - 5:(k2 + s) + 5, 0]
                                    d_up, cost_matrix_up, acc_cost_matrix_up, path_up = dtw(target_seq, input_seq_up, dist=manhattan_distance)
                                    d_down, cost_matrix_down, acc_cost_matrix_down, path_down = dtw(target_seq, input_seq_down, dist=manhattan_distance)
                                    wrap_data[k2, 0, s] = d_up
                                    wrap_data[k2, 0, 5 + s] = d_down
    return wrap_data, c


###------------------------------------------------------------------------------------------------------------
def manhattan_distance(target_seq, input_seq):
    return np.abs(input_seq - target_seq)

def calcolo_dtw_on_target(target_seq, needle_seq):
    min_distance, cost_matrix, acc_cost_matrix, wrap_path = dtw(target_seq, needle_seq, dist=manhattan_distance)
    return min_distance


def funzione_finale(dati_sm, intr_data):
    # Trasformo la matrice in un df di pandas
    df = pd.DataFrame(data_smooth.T).T
    # Converto le coordinate
    intr_data = np.array([np.array((i[0] * 20 + i[2], i[1])) for i in intr_data])

    for pt in intr_data:
        col = pt[1]
        row = pt[0]
        fetta = df[col][row - 5:row + 6]
        distances = []
        for i in range(16):
            fetta_input = df[col + i][row - 5:row + 6]
            distance = dtw(fetta.to_numpy(), fetta_input.to_numpy(), manhattan_distance)[0]
            distances.append(distance)



if __name__ == '__main__':
    # caricamento dati
    path = 'data/'
    name = 'CORAL_07_DIR_DYNAMIC.las'

    data, a = read_las(path, name)

    data_smooth = smoothing(data, a)
    print(data_smooth.shape)

    data_resh = reshape(data_smooth, 20)
    data_sobel, threshold, ds_fl, maximum, max_pt = Sobel_calc(data_resh, 20, k_size=5, a=a)
    maxima, data_sobel_norm, int_points = get_interesting_points(data_sobel, threshold)

    print(maxima, max_pt, threshold)
    print(data_sobel_norm[625, :, :])
    print(int_points.shape, int_points)
    print(data_sobel[625, :, :])
    # tqdm 1
    dtw_data, dtw_data_guided, block_df, interesting_data = dtw_calc(data_sobel, int_points, guide=True)

    funzione_finale(data_smooth, interesting_data)
    #
    # # tqdm 2
    wrap_data, c = IntPointsDtw(data_smooth, int_points, interesting_data)
    wrap_data, c = IntPointsDtw(data_smooth, int_points, interesting_data)
    print(wrap_data[12505, :, :])
    print(c[7900])

    # scaler = StandardScaler()
    # scaled_df = scaler.fit_transform(block_df_pd)
    # scaled_df = pd.DataFrame(scaled_df, columns = block_df_pd.columns)##

    # kmeans_model = KMeans(n_clusters=2)
    # kmeans_model.fit(scaled_df)
    # centroids = kmeans_model.cluster_centers_
    # print(centroids)

    # block_df_df = np.zeros((block_df.shape[0], block_df.shape[2], block_df.shape[3]))
    # for depth_bin in range(block_df.shape[0]):
    #   for i in range(block_df.shape[2]):
    #      for block in range(block_df.shape[3]):
    #             if block == 0:
    #                block_df_df[depth_bin,i,block] = block_df[depth_bin,:,i,0]
    #               block_df_corr = block_df_df[depth_bin,i,block]
    #              block_df_corr = pd.DataFrame(block_df_corr)
    #             corr_matr = block_df_corr.corr()
    #            n
    #         elif block !=0:
    #            max_corr = np.argmax(corr_matr)
    #           block_df_df[depth_bin, i, block] = block_df[depth_bin,:,i+1,]
