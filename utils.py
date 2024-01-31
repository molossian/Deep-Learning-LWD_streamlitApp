import numpy as np


def resort(i, data, fitnet_CL, fitnet_SL):
    preds = []
    matrix = []
    preds1 = []
    preds2 = []
    preds21 = []
    matrix.append(data[i:i + 40, :])
    matrix = np.asarray(matrix)
    # print("matrix[0].shape", matrix[0].shape)

    p0 = np.array([p[i] for p in fitnet_CL])
    pst = p0.mean(axis=0)[0]
    pst1 = p0.mean(axis=0)[1]

    p1 = np.array([p[i] for p in fitnet_SL])
    pst2 = p1.mean(axis=0)[0]
    pst21 = p1.mean(axis=0)[1]

    m, s = p0.mean(axis=0)[0], p0.std(axis=0)[0]
    m1, s1 = p0.mean(axis=0)[1], p0.std(axis=0)[1]
    m2, s2 = p1.mean(axis=0)[0], p1.std(axis=0)[0]
    m21, s21 = p1.mean(axis=0)[1], p1.std(axis=0)[1]

    preds.append(pst)
    preds1.append(pst1)
    preds = np.asarray(preds)

    preds2.append(pst2)
    preds21.append(pst21)
    preds2 = np.asarray(preds2)

    return preds, preds1, preds2, preds21, matrix, [(m, s), (m1, s1), (m2, s2), (m21, s21)]