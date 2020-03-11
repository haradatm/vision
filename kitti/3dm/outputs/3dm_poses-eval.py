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
from collections import defaultdict
import itertools


def umeyama_transform(pt, U, scale):
    R, t = U[:, :3], U[:, -1].reshape(3)
    pt = pt.reshape(3) * scale
    pt = np.dot(R, pt)
    pt = pt + t
    return pt


def umeyama_transform_inv(pt, U, scale):
    R, t = U[:, :3], U[:, -1].reshape(3)
    pt = pt - t
    pt = np.dot(np.linalg.inv(R), pt)
    pt = pt / scale
    return pt


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--label_file', default='/Data/haradatm/DATA/KITTI/data_tracking/data_tracking_label_2/training/label_02/0001.txt', help='label file')
    parser.add_argument('--calib_file', default='/Data/haradatm/DATA/KITTI/data_tracking/data_tracking_calib/training/calib/0001.txt', help='calibration data file')
    parser.add_argument('--pose_file', default='frame_trajectory.txt', help='pose file')
    parser.add_argument('--transform_file', default='umeyama.txt', help='transfom matrix file')
    parser.add_argument('--output_dir', default='outputs', help='label file')
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=[], help='moving objects to exclude')
    parser.add_argument('--camera_height', default=1.65, help='camera height')
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

    for i, line in enumerate(open(args.transform_file)):
        if i == 0:
            continue
        line = line.strip()
        cols = line.split(' ')
    U = np.array([float(x) for x in cols[:-1]], dtype=np.float64).reshape(3, 4)
    scale = float(cols[-1])

    params = {}
    for line in open(args.calib_file):
        if line.startswith("P"):
            line = line.strip().split(' ')
            p = line[0].strip().split(':')[0]
            v = np.array([float(s) for s in line[1:]], dtype=np.float64).reshape((3, 4))
            params[p] = v

    camera_param = params['P2'][:, 0:3]
    dist_coef = np.zeros((1, 5), dtype=np.float64)

    logger.info("--- K ----------")
    logger.info(camera_param)
    logger.info("--- dist_coef ----------")
    logger.info(dist_coef)

    poses = defaultdict(list)
    # Rcs, Tcs, Tws, Ps, image_points = [], [], [], [], []
    for frame_id, line in enumerate(open(args.pose_file)):
        line = line.strip()
        cols = line.split(' ')

        timestamp = float(cols[0])
        logger.info("frame: %d, time: %.6f" % (frame_id, timestamp))

        try:
            q = np.array([float(x) for x in cols[4:8]], dtype=np.float64).reshape(1, 4)
            Rw = Rotation.from_quat(q).as_dcm()[0]
            Rc = np.linalg.inv(Rw)
            Tw = np.array([float(x) for x in cols[1:4]], dtype=np.float64).reshape(3, 1)
            Tc = - np.dot(Rc, Tw)

        except np.linalg.LinAlgError as e:
            logger.error("%s (frame: %.6f)" % (str(e), frame_id))
            continue

        logger.info("--- Rc (3x3) ----------")
        logger.info(Rc)
        logger.info("--- Tc (tx, ty, tz) ----------")
        logger.info(Tc)
        logger.info("--- Rw (3x3) ----------")
        logger.info(Rw)
        logger.info("--- Tw (tx, ty, tz) ----------")
        logger.info(Tw)

        P = np.dot(camera_param, np.hstack((Rc, Tc)))
        logger.info("--- P (3x4) ----------")
        logger.info(P)

        poses[frame_id] = {'P':P, "Tw":Tw.T, "Rc":Rc, "Tc":Tc}

    tracks = defaultdict(list)
    for i, line in enumerate(open(args.label_file)):
        line = line.strip()
        cols = line.split(" ")

        # #Values    Name      Description
        # ----------------------------------------------------------------------------
        #    1    frame        Frame within the sequence where the object appearers
        #    1    track id     Unique tracking id of this object within this sequence
        #    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
        #                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
        #                      'Misc' or 'DontCare'
        #    1    truncated    Integer (0,1,2) indicating the level of truncation.
        #                      Note that this is in contrast to the object detection
        #                      benchmark where truncation is a float in [0,1].
        #    1    occluded     Integer (0,1,2,3) indicating occlusion state:
        #                      0 = fully visible, 1 = partly occluded
        #                      2 = largely occluded, 3 = unknown
        #    1    alpha        Observation angle of object, ranging [-pi..pi]
        #    4    bbox         2D bounding box of object in the image (0-based index):
        #                      contains left, top, right, bottom pixel coordinates
        #    3    dimensions   3D object dimensions: height, width, length (in meters)
        #    3    location     3D object location x,y,z in camera coordinates (in meters)
        #    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        #    1    score        Only for results: Float, indicating confidence in
        #                      detection, needed for p/r curves, higher is better.

        frame_id = int(cols[0])
        track_id = int(cols[1])
        object_type = cols[2]
        bbox = list(map(lambda x: float(x), cols[6:10]))
        location = list(map(lambda x: float(x), cols[13:16]))

        if object_type == "DontCare":
            continue

        if track_id in args.exclude_objects:
            continue

        left, top, right, bottom = bbox
        edge_width = 1

        if left < 0 + edge_width:
            continue
        if top < 0 + edge_width:
            continue
        if right > 1241 - edge_width:
            continue
        if bottom > 374 - edge_width:
            continue

        tracks[track_id].append((frame_id, object_type, bbox, location))

    for k, v in sorted(tracks.items(), key=lambda x:x[0]):
        v.sort(key=lambda x:x[0], reverse=False)

    results = defaultdict(list)
    for i, (track_id, objects) in enumerate(tracks.items()):
        for pair in list(itertools.combinations(objects, 2)):
            frame1 = pair[0][0]
            if frame1 not in poses:
                continue
            object_type1 = pair[0][1]
            location1 = pair[0][3]

            frame2 = pair[1][0]
            if frame2 not in poses:
                continue
            object_type2 = pair[1][1]
            location2 = pair[1][3]

            assert object_type1 == object_type2

            pose1 = poses[frame1]['P']
            tw1 = poses[frame1]['Tw']
            rc1 = poses[frame1]['Rc']
            tc1 = poses[frame1]['Tc']
            left1, top1, right1, bottom1 = pair[0][2]
            u1 = left1 + (right1 - left1) / 2
            v1 = top1 + (bottom1 - top1) / 2
            uv1 = np.array([u1, v1], dtype=np.float64)
            uv1 = np.expand_dims(uv1, axis=1)

            logger.info("--- u1, v1 ----------")
            logger.info(uv1)
            logger.info("--- pose1 ----------")
            logger.info(pose1)

            pose2 = poses[frame2]['P']
            tw2 = poses[frame2]['Tw']
            rc2 = poses[frame2]['Rc']
            tc2 = poses[frame2]['Tc']
            left2, top2, right2, bottom2 = pair[1][2]
            u2 = left2 + (right2 - left2) / 2
            v2 = top2 + (bottom2 - top2) / 2
            uv2 = np.array([u2, v2], dtype=np.float64)
            uv2 = np.expand_dims(uv2, axis=1)

            logger.info("--- u2, v2 ----------")
            logger.info(uv2)
            logger.info("--- pose2 ----------")
            logger.info(pose2)

            point_4d_hom = cv2.triangulatePoints(pose1, pose2, uv1, uv2)
            Xw = (point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1)))[:3, :].T

            logger.info("--- Xw ----------")
            logger.info(Xw)

            re_point1, _ = cv2.projectPoints(Xw, rc1, tc1, camera_param, dist_coef)
            re_point1 = re_point1.astype('i')[0][0]
            logger.info("--- origin (u1, v1) ----------")
            logger.info(uv1.reshape(2))
            logger.info("--- re-projection (u1, v1) ----------")
            logger.info(re_point1)
            error1 = np.linalg.norm(uv1.reshape(2) - re_point1)
            logger.info("--- re-projection error (u1, v1) ----------")
            logger.info(error1)

            re_point2, _ = cv2.projectPoints(Xw, rc2, tc2, camera_param, dist_coef)
            re_point2 = re_point2.astype('i')[0][0]
            logger.info("--- origin (u2, v2) ----------")
            logger.info(uv2.reshape(2))
            logger.info("--- re-projection (u2, v2) ----------")
            logger.info(re_point2)
            error2 = np.linalg.norm(uv2.reshape(2) - re_point2)
            logger.info("--- re-projection error (u2, v2) ----------")
            logger.info(error2)

            utw1 = umeyama_transform(tw1, U, scale)
            utw2 = umeyama_transform(tw2, U, scale)
            uXw  = umeyama_transform(Xw,  U, scale)

            lw1 = tw1 + np.array(location1, 'f') / scale
            lw2 = tw2 + np.array(location2, 'f') / scale
            ulw1 = umeyama_transform(lw1, U, scale)
            ulw2 = umeyama_transform(lw2, U, scale)

            results[(track_id, object_type1)].append([
                utw1[0], utw1[1], utw1[2],
                utw2[0], utw2[1], utw2[2],
                uXw[0],  uXw[1],  uXw[2],
                ulw1[0], ulw1[1], ulw1[2],
                ulw2[0], ulw2[1], ulw2[2],
            ])

    results_center = {}
    for (track_id, object_type), coords in results.items():
        coords = np.array(coords, 'f')
        utw1 = np.median(coords[:, : 3], axis=0)
        utw2 = np.median(coords[:, 3: 6], axis=0)
        uXw = np.median(coords[:, 6: 9], axis=0)
        ulw1 = np.median(coords[:, 9:12], axis=0)
        ulw2 = np.median(coords[:, 12:], axis=0)
        results_center[(track_id, object_type)] = (utw1, utw2, uXw, ulw1, ulw2)

    # STDOUT
    print(
        "track_id\tobject_type\t"
        "utw1_x\tutw1_y\tutw1_z\t"
        "utw2 x\tutw2 y\tutw2_z\t"
        "uXw_x\tuXw_y\tuXw_z"
        "ulw1_x\tulw1_y\tulw1_z\t"
        "ulw2 x\tulw2 y\tulw2_z\t"
    )
    for (track_id, object_type), coords in results_center.items():
        utw1, utw2, uXw, ulw1, ulw2 = coords
        print(
            "%d\t%s\t"
            "%.6f\t%.6f\t%.6f\t"
            "%.6f\t%.6f\t%.6f\t"
            "%.6f\t%.6f\t%.6f\t"
            "%.6f\t%.6f\t%.6f\t"
            "%.6f\t%.6f\t%.6f"
            % (
            track_id, object_type1,
            utw1[0], utw1[1], utw1[2],
            utw2[0], utw2[1], utw2[2],
            uXw[0],  uXw[1],  uXw[2],
            ulw1[0], ulw1[1], ulw1[2],
            ulw2[0], ulw2[1], ulw2[2],
            )
        )
        sys.stdout.flush()

    # Plot
    trajectory = []
    for v in poses.values():
        trajectory.append(umeyama_transform(v['Tw'], U, scale))
    trajectory = np.array(trajectory, 'f')

    obj_id, obj_pos = [], []
    for (track_id, object_type), coords in results_center.items():
        _, _, uXw, _, _ = coords
        uXw += [0., 0., args.camera_height]
        obj_id.append(track_id)
        obj_pos.append(uXw.tolist())
    obj_pos = np.array(obj_pos, 'f')

    lab_id, lab_pos = [], []
    for (track_id, object_type), coords in results_center.items():
        _, _, _, ulw1, _ = coords
        ulw1 += [0., 0., args.camera_height]
        lab_id.append(track_id)
        lab_pos.append(ulw1.tolist())
    lab_pos = np.array(lab_pos, 'f')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', marker='.')
    ax.scatter(obj_pos[:, 0], obj_pos[:, 1], obj_pos[:, 2], c='red', marker='x')
    for tid, pos in zip(obj_id, obj_pos):
        ax.text(pos[0], pos[1], pos[2], tid, c='red')
    ax.scatter(lab_pos[:, 0], lab_pos[:, 1], lab_pos[:, 2], c='green', marker='x')
    for tid, pos in zip(lab_id, lab_pos):
        ax.text(pos[0], pos[1], pos[2], tid, c='green')
    ax.set_xlabel('X (East-West) [meter]')
    ax.set_ylabel('Y (North-South) [meter]')
    ax.set_zlabel('Z (Height) [meter]')
    plt.title("EPSG_%s (DE:GK2)" % 31466)
    plt.legend(['trajectory', 'objects', 'gt (uncertainty*)'], loc="best")
    plt.savefig("result-3dm_poses-3d-eval.png")
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(trajectory[:, 0], trajectory[:, 1], c='blue', marker='.')
    ax.scatter(obj_pos[:, 0], obj_pos[:, 1], c='red', marker='x')
    for tid, pos in zip(obj_id, obj_pos):
        ax.annotate(tid, xy=(pos[0], pos[1]), c='red')
    ax.scatter(lab_pos[:, 0], lab_pos[:, 1], c='green', marker='x')
    for tid, pos in zip(lab_id, lab_pos):
        ax.annotate(tid, xy=(pos[0], pos[1]), c='green')
    ax.set_xlabel('X (East-West) [meter]')
    ax.set_ylabel('Y (North-South) [meter]')
    plt.title("EPSG_%s (DE:GK2)" % 31466)
    plt.legend(['trajectory', 'objects', 'gt (uncertainty*)'], loc="best")
    plt.savefig("result-3dm_poses-xy-eval.png")
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(trajectory[:, 0], trajectory[:, 2], c='blue', marker='.')
    ax.scatter(obj_pos[:, 0], obj_pos[:, 2], c='red', marker='x')
    for tid, pos in zip(obj_id, obj_pos):
        ax.annotate(tid, xy=(pos[0], pos[2]), c='red')
    ax.scatter(lab_pos[:, 0], lab_pos[:, 2], c='green', marker='x')
    for tid, pos in zip(lab_id, lab_pos):
        ax.annotate(tid, xy=(pos[0], pos[2]), c='green')
    ax.set_xlabel('X (East-West) [meter]')
    ax.set_ylabel('Z (Height) [meter]')
    plt.title("EPSG_%s (DE:GK2)" % 31466)
    plt.legend(['trajectory', 'objects', 'gt (uncertainty*)'], loc="best")
    plt.savefig("result-3dm_poses-xz-eval.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
