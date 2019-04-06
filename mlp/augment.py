#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys

reload(sys)
sys.setdefaultencoding('utf-8')
#print sys.getdefaultencoding()

import re
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

import pprint
def pp(obj):
    pp = pprint.PrettyPrinter(indent=1, width=160)
    str = pp.pformat(obj)
    print re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1),16)), str)

import os, sys, math, time
start_time = time.time()

from scipy import misc
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def show_flow_image(datagen, x, show_num):
    i = 0
    for batch in datagen.flow(x, x, batch_size=1):  # second args 'x' is dummy of y
        array = batch[0][0,:,:,:]
        img = array_to_img(array)
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        i += 1
        if i >= show_num:
            break
    plt.show()

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--image', '-i', default='beagle.jpg', type=str, help='input file (.jpg)')
    args = parser.parse_args()

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )

    path = args.image
    base_name, ext = os.path.splitext(os.path.basename(path))

    print('# path: {}'.format(path))
    print('# save_to_dir: {}'.format(base_name))
    print('# save_prefix: {}'.format(base_name))

    if not os.path.exists(base_name):
        os.mkdir(base_name)

    # this is a PIL image
    img = load_img(args.image)

    # this is a Numpy array with shape (3, 150, 150)
    x = img_to_array(img)

    # this is a Numpy array with shape (1, 3, 150, 150)
    x = x.reshape((1,) + x.shape)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=base_name, save_prefix=base_name, save_format='jpg'):
        i += 1
        if i > 100:
            break

    print('time spent:', time.time() - start_time)
