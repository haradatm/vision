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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_dir', default='/Data/haradatm/DATA/KITTI/data_tracking/data_tracking_image_2/training/image_02/0001', help='image file directory')
    parser.add_argument('--label_file', default='/Data/haradatm/DATA/KITTI/data_tracking/data_tracking_label_2/training/label_02/0001.txt', help='label file')
    parser.add_argument('--output_dir', default='outputs', help='label file')
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

    labels = defaultdict(list)
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

        if object_type == "DontCare":
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

        labels[frame_id].append((track_id, object_type, bbox))

    for i, img_file in enumerate(sorted(glob.glob(os.path.join(args.image_dir, "*.png")))):

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_id = int(img_file.split('/')[-1].split('.')[0])

        if frame_id not in labels:
            continue

        dets = labels[frame_id]

        fig = plt.figure(figsize=(18, 12))
        ax = plt.gca()
        plt.imshow(img)
        # plt.axis('off')
        # det_title = plt.title("frame: %04d" % frame_id)
        # plt.setp(det_title, color='b')

        edge_width = 3
        font_size = 18

        for j in range(len(dets)):
            id = dets[j][0]
            name = dets[j][1]
            box  = dets[j][2]

            x = box[0] + edge_width / 2
            y = box[1] + edge_width / 2
            w = box[2] - box[0] - edge_width
            h = box[3] - box[1] - edge_width
            cx = x + w // 2
            cy = y + h // 2
            ax.text(cx, cy - 2, "%s %d" % (name, id), fontsize=font_size, color='white', bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
            ax.add_patch(plt.Rectangle(xy=[x, y], width=w, height=h, fill=False, edgecolor='orange'))
            ax.scatter(cx, cy, color="red", marker="x")

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(output_dir, "bbox-%04d.png" % frame_id), bbox_inches='tight')
        # plt.show()
        plt.close(fig)



if __name__ == '__main__':
    main()
