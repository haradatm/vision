#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: test a multi-layer perceptron for image detection

This is a example code to write a feed-forward net for image detection.

usage: python2.7 mlp.py --eval sample.jpg
author: haradatm@hotmail.com

"""
#
# 実験:
# 結果:
#
__version__ = '0.0.1'

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#print sys.getdefaultencoding()

# usage:

import re, math, unicodedata

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

import pprint
def pp(obj):
    pp = pprint.PrettyPrinter(indent=1, width=160)
    str = pp.pformat(obj)
    print re.sub(r"\\u([0-9a-f]{4})", lambda x: unichr(int("0x"+x.group(1),16)), str)

import time, os
start_time = time.time()

import numpy as np
from PIL import Image
import cPickle as pickle
import matplotlib.pyplot as plt
import io
np.set_printoptions(precision=20)   # 印字オプションの変更(デフォルトは8)

from chainer import cuda, FunctionSet, Variable, optimizers, serializers, computational_graph
import chainer.functions as F
from chainer.functions import caffe


in_size = 224

mean_image = np.ndarray((3, 224, 224), dtype=np.float32)
mean_image[0] = 104
mean_image[1] = 117
mean_image[2] = 124


def read_image(path, plot=False):

    if plot:
        plt.figure(figsize=(10, 10))

    # 画像を読み込み,RGB形式に変換する
    image = Image.open(path).convert('RGB')

    # 入力画像サイズの定義
    image_w, image_h = (224, 224)

    # 画像のリサイズ
    w, h = image.size
    if w > h:
        shape = (image_w * w / h, image_h)
    else:
        shape = (image_w, image_h * h / w)
    x = (shape[0] - image_w) / 2
    y = (shape[1] - image_h) / 2
    image = image.resize(shape)

    # 画像のクリップ
    image = image.crop((x, y, x + image_w, y + image_h))

    # pixels は 3次元でそれぞれの軸は (Y座標, X座標, RGB) を表す
    pixels = np.asarray(image).astype(np.float32)

    # 軸を入れ替える -> (RGB, Y座標, X座標)
    pixels = pixels.transpose(2, 0, 1)

    if plot:
        plt.subplot(1, 3, 1)
        plt.title('(RGB, Y座標, X座標)')
        plt.imshow(pixels.astype(np.uint8).transpose(1, 2, 0))

    # RGB から BGR に変換する
    pixels = pixels[::-1, :, :]

    if plot:
        plt.subplot(1, 3, 2)
        plt.title('(BGR, Y座標, X座標)')
        plt.imshow(pixels.astype(np.uint8).transpose(1, 2, 0))

    # 平均画像を引く
    pixels -= mean_image

    if plot:
        plt.subplot(1, 3, 3)
        plt.title('-= mean image')
        plt.imshow(pixels.astype(np.uint8).transpose(1, 2, 0))

    # 4次元 (画像インデックス, BGR, Y座標, X座標) に変換する
    pixels = pixels.reshape((1,) + pixels.shape)

    plt.show()

    return pixels


def get_image(path):
    image = read_image(path, plot=False)
    return image


def load_vgg_model(path):
    print('Loading Caffe model file ...')
    with open(path, 'rb') as f:
        vgg = pickle.load(f)
    return vgg


def load_labels(path):
    labels = []

    for line in open(path, 'rU'):
        line = line.strip()
        if line.startswith('#'):
            continue

        cols = line.split('\t')
        label = cols[1]
        if label not in labels:
            labels.append(label)

    return labels


def load_mlp_model(path):

    n_units  = 1000
    n_labels = 25

    model = FunctionSet(l1=F.Linear(4096, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, n_labels))
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    print('Load model from', path)
    serializers.load_hdf5(path, model)

    return model


labels = load_labels('data/list-25.txt')
model  = load_mlp_model('model/fine-tuning-mlp.model')
vgg    = load_vgg_model('model/vgg19.pkl')


def predict(x_data, gpu=-1):

    xp = cuda.cupy if gpu >= 0 else np
    if gpu >= 0:
        model.to_gpu()
        x_data = cuda.to_gpu(x_data)

    x = Variable(x_data, volatile='on')
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(h1))
    y = model.l3(h2)
    return F.softmax(y), h2


def classify(msgid, N=1, gpu=-1):

    xp = cuda.cupy if gpu >= 0 else np
    if gpu >= 0:
        vgg.to_gpu()

    array = xp.asarray(get_image(msgid))
    x = xp.ascontiguousarray(array)
    h = Variable(x, volatile=True)
    feature, = vgg(inputs={'data': h}, outputs=['fc7'], train=False)
    x = feature.data[0,].reshape((1, 4096))
    y, f = predict(x, gpu=gpu)
    scores = cuda.to_cpu(y.data[0])

    ret = []
    for i, idx in enumerate(np.argsort(scores)[-1::-1]):
        if i >= N:
            break

        ret.append({
            'score': '%0.6f' % scores[idx],
            'label': labels[idx],
        })

    return ret


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: test a multi-layer perceptron for image detection.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--eval', type=unicode, default='shiba.jpeg', help='file for evaluation (.jpg)')
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np

    if args.eval:
        ret = classify(args.eval, N=10, gpu=args.gpu)
        for i, ans in enumerate(ret):
            score = float(ans['score'])
            label = ans['label']
            print '{:>3d} {:>6.2f}% {}'.format(i + 1, score * 100, label)

    print('time spent:', time.time() - start_time)