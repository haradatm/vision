#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import logging
import os
import random
import sys
import numpy as np

np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
console.setLevel(logging.DEBUG)
logger.addHandler(console)
# logfile = logging.FileHandler(filename="log.txt")
# logfile.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
# logfile.setLevel(logging.DEBUG)
# logger.addHandler(logfile)

import cv2, glob, json, math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import pyproj


def align_umeyama(model, data, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)
    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0 / n * np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0 / n * np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)

    if np.linalg.det(U_svd) * np.linalg.det(V_svd) < 0:
        S[2, 2] = -1

    R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0 / sigma2 * np.trace(np.dot(D_svd, S))

    t = mu_M - s * np.dot(R, mu_D)

    return s, R, t


def plot_trajectory(gt, est, epsg=3857):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for est_p in est:
        ax.scatter(est_p[0], est_p[1], est_p[2], c='red',  marker='.')
    ax.scatter(0., 0., 0., c='black', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Camera 3D")
    plt.legend(['trajectory'], loc="best")
    plt.savefig("plot_camera-3d.png")
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, est_p in enumerate(est):
        ax.scatter(est_p[0], est_p[1], c='red',  marker='.')
    ax.scatter(0., 0., c='black', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title("Camera X-Y")
    plt.legend(['trajectory'], loc="best")
    plt.savefig("plot_camera-xy.png")
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for est_p in est:
        ax.scatter(est_p[0], est_p[2], c='red',  marker='.')
    ax.scatter(0., 0., c='black', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.title("Camera X-Z")
    plt.legend(['trajectory'], loc="best")
    plt.savefig("plot_camera-xz.png")
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for est_p in est:
        ax.scatter(est_p[2], est_p[1], c='red',  marker='.')
    ax.scatter(0., 0., c='black', marker='x')
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    plt.title("Camera Z-Y")
    plt.legend(['trajectory'], loc="best")
    plt.savefig("plot_camera-zy.png")
    plt.show()
    plt.close()

    trans_from = pyproj.Proj('+init=EPSG:%s' % 4326)
    trans_to   = pyproj.Proj('+init=EPSG:%s' % epsg)

    gt_trans = []
    init_x, init_y, init_z = 0., 0., 0.

    for i, gt_p in enumerate(gt):
        # (lan, lon, alt) -> (lon, lan, alt)
        x, y, z = pyproj.transform(trans_from, trans_to, gt_p[1], gt_p[0], gt_p[2])
        if i == 0:
            init_x, init_y, init_z = (x, y, z)
        gt_trans.append((x - init_x, y - init_y, z - init_z))

    gt_trans = np.array(gt_trans,  'f')
    est = np.array(est, 'f')
    s, R, t = align_umeyama(gt_trans, est)

    est_umeyama = []
    for i, est_p in enumerate(est):
        est_p = est_p * s
        est_p = np.dot(R, est_p)
        est_p = est_p + t
        est_umeyama.append(est_p)
    est_umeyama = np.array(est_umeyama, 'f')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, (gt_p, est_p) in enumerate(zip(gt_trans, est_umeyama)):
        ax.scatter( gt_p[0],  gt_p[1],  gt_p[2], c='blue', marker='.')
        ax.scatter(est_p[0], est_p[1], est_p[2], c='red',  marker='.')
    ax.set_xlabel('X (East-West) [meter]')
    ax.set_ylabel('Y (North-South) [meter]')
    ax.set_zlabel('Z (Altitude) [meter]')
    plt.title("EPSG_%s (DE:GK2)" % 31466)
    plt.legend(['ground truth', 'trajectory'], loc="best")
    plt.savefig("plot_EPSG%s-3d.png" % epsg)
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, (gt_p, est_p) in enumerate(zip(gt_trans, est_umeyama)):
        ax.scatter( gt_p[0],  gt_p[1], c='blue', marker='.')
        ax.scatter(est_p[0], est_p[1], c='red',  marker='.')
    ax.set_xlabel('X (East-West) [meter]')
    ax.set_ylabel('Y (North-South) [meter]')
    plt.title("EPSG_%s (DE:GK2)" % 31466)
    plt.legend(['ground truth', 'trajectory'], loc="best")
    plt.savefig("plot_EPSG%s-xy.png" % epsg)
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, (gt_p, est_p) in enumerate(zip(gt_trans, est_umeyama)):
        ax.scatter( gt_p[0],  gt_p[2], c='blue', marker='.')
        ax.scatter(est_p[0], est_p[2], c='red',  marker='.')
    ax.set_xlabel('X (East-West) [meter]')
    ax.set_ylabel('Z (Altitude) [meter]')
    plt.title("EPSG_%s (DE:GK2)" % 31466)
    plt.legend(['ground truth', 'trajectory'], loc="best")
    plt.savefig("plot_EPSG%s-xz.png" % epsg)
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, (gt_p, est_p) in enumerate(zip(gt_trans, est_umeyama)):
        ax.scatter( gt_p[1],  gt_p[2], c='blue', marker='.')
        ax.scatter(est_p[1], est_p[2], c='red',  marker='.')
    ax.set_xlabel('Y (North-South) [meter]')
    ax.set_ylabel('Z (Altitude) [meter]')
    plt.title("EPSG_%s (DE:GK2)" % 31466)
    plt.legend(['ground truth', 'trajectory'], loc="best")
    plt.savefig("plot_EPSG%s-yz.png" % epsg)
    plt.show()
    plt.close()

    return np.hstack([R, t.reshape(3, 1)]), s


def main():
    import argparse
    parser = argparse.ArgumentParser(description='umeyama alignment')
    parser.add_argument('--gt',  default='gt.txt',  help='ground truth potions (.txt)')
    parser.add_argument('--est', default='est.txt', help='estimated potions (.txt)')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    logger.info(json.dumps(args.__dict__, indent=2))
    sys.stdout.flush()

    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if args.gpu >= 0:
    #     torch.cuda.manual_seed_all(seed)

    gt, est = [], []
    for line in open(args.gt):
        cols = line.strip().split(' ')
        gt.append([float(x) for x in cols[:3]])     # (lan, lon, alt)

    for line in open(args.est):
        cols = line.strip().split(' ')
        est.append([float(x) for x in cols[:3]])    # (x, y, z)

    # U, scale = plot_trajectory(gt, est, epsg=3857)    # EPSG  3857: 球面メルカトル図法
    U, scale = plot_trajectory(gt, est, epsg=31466)     # EPSG 31466: DE:ガウスクルーガー, GK 2

    print(
        "R[0,0] R[0,1] R[0,2] t[0] "
        "R[1,0] R[1,1] R[1,2] t[1] "
        "R[2,0] R[2,1] R[2,2] t[2] "
        "s"
    )
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            print("%.6f" % U[i][j], end=" ")
    print("%.6f" % scale)


if __name__ == '__main__':
    main()
