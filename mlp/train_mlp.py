#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: train a multi-layer perceptron for image detection.

This is a example code to write a feed-forward net for image detection.

usage: python2.7 mlp.py --eval sample.jpg
author: haradatm@hotmail.com

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

import numpy as np
from PIL import Image
import cPickle as pickle
import matplotlib.pyplot as plt
import pydot
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


labels = {}


def load_data():

    global labels
    inv_labels = {}

    # 画像データセットのダウンロード
    print('Loading training dataset ...')
    X = []
    Y = []

    for i, line in enumerate(open('data/list-25.txt', 'r')):
        # if i > 100:
        #     break

        line = line.strip()

        if line.startswith('#'):
            continue

        cols = line.split('\t')
        path = cols[0]
        label = cols[1]
        image = read_image(path)
        X.append(image)

        if label not in inv_labels:
            inv_labels[label] = len(inv_labels)
            labels[len(inv_labels)-1] = label

        Y.append(np.int32(inv_labels[label]))

    x = np.array(X, dtype=np.float32)
    y = np.array(Y, dtype=np.int32)

    print('Loading training dataset ... done.')

    return x, y

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description='Chainer example: train a multi-layer perceptron for image detection.')
    parser.add_argument('--epoch', type=int, default=25, help='number of loops')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--plot', action='store_true', default=False, help='plot results')
    parser.add_argument('--initmodel', default='', help='itialize the model from given file')
    parser.add_argument('--resume', default='', help='resume the optimization from snapshot')
    parser.add_argument('--batchsize', type=int, default=100, help='number of batchsize')
    args = parser.parse_args()

    # 中間層の数
    n_units = 1000

    # 学習の繰り返し回数
    n_epoch = args.epoch

    # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
    batchsize = args.batchsize

    # その他の学習パラメータ
    LEARNIN_RATE = 0.01
    DECAY_FACTOR = 0.97

    print('# GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(n_units))
    print('# minibatch: {}'.format(batchsize))
    print('# epoch: {}'.format(n_epoch))
    print('')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(123)

    # load a new model to be fine-tuned
    if os.path.exists('model/vgg19.pkl'):
        print('Loading Caffe model file: {}'.format('model/vgg19.pkl'))
        with open('model/vgg19.pkl', 'rb') as f:
            vgg = pickle.load(f)
    else:
        print('Loading Caffe model file: {}'.format('model/VGG_ILSVRC_19_layers.caffemodel'))
        vgg = caffe.CaffeFunction('model/VGG_ILSVRC_19_layers.caffemodel')
        pickle.dump(vgg, open('model/vgg19.pkl', 'wb'))

    if args.gpu >= 0:
        vgg.to_gpu()

    X, Y = load_data()
    # x = np.zeros((len(X), 4096), dtype=np.float32)
    x = []

    for i in xrange(0, len(X)):
        pixels = X[i]
        array = xp.asarray(pixels)
        x_data = xp.ascontiguousarray(array)
        if args.gpu >= 0:
            x_data = cuda.to_gpu(x_data)
        h = Variable(x_data, volatile=True)
        feature, = vgg(inputs={'data': h}, outputs=['fc7'], train=False)
        # x[i] = cuda.to_cpu(feature.data[0])
        x.append(cuda.to_cpu(feature.data[0]))
    x = np.array(x, dtype=np.float32)

    # 学習用データを N個, 残りの個数を検証用データに設定
    N = (len(X) * 6 / 7)
    perm = np.random.permutation(len(X))
    train_idx = perm[0:N].copy()
    test_idx  = perm[N:-1].copy()

    x_train = np.array(x[perm[train_idx]], dtype=np.float32)
    x_test  = np.array(x[perm[test_idx]],  dtype=np.float32)
    y_train = np.array(Y[perm[train_idx]], dtype=np.int32)
    y_test  = np.array(Y[perm[test_idx]],  dtype=np.int32)

    x_train_image = np.array(X[perm[train_idx]], dtype=np.float32)

    N = len(x_train)
    N_test = len(x_test)
    print('train: {}, test: {}'.format(N, N_test))
    sys.stdout.flush()

    # Prepare multi-layer perceptron model
    # 多層パーセプトロンモデルの設定
    # 入力 4096次元, 出力 10次元
    model = FunctionSet(l1=F.Linear(4096, n_units),
                        l2=F.Linear(n_units, n_units),
                        l3=F.Linear(n_units, len(labels)))

    # Neural net architecture
    def forward(x_data, y_data, train=True):
        # relu: 負の時は0, 正の時は値をそのまま返す (計算量が小さく学習スピードが速くなることが利点)
        # dropout: ランダムに中間層をドロップ(ないものとする)し,過学習を防ぐ
        x, t = Variable(x_data, volatile=not train), Variable(y_data, volatile=not train)
        h1 = F.dropout(F.relu(model.l1(x)),  train=train)
        h2 = F.dropout(F.relu(model.l2(h1)), train=train)
        y = model.l3(h2)

        # 多クラス分類なので誤差関数としてソフトマックス関数の
        # 交差エントロピー関数を用いて誤差を導出
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


    def predict(x_data, train=False):
        x = Variable(x_data, volatile=not train)
        h1 = F.dropout(F.relu(model.l1(x)),  train=train)
        h2 = F.dropout(F.relu(model.l2(h1)), train=train)
        y = model.l3(h2)

        # ソフトマックス関数で確率値を算出
        return F.softmax(y), h2


    # Setup optimizer (Optimizer の設定)
    # optimizer = optimizers.SGD(LEARNIN_RATE)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # # Init/Resume
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_hdf5(args.initmodel, model)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_hdf5(args.resume, optimizer)

    if args.gpu >= 0:
        model.to_gpu()

    # プロット用に実行結果を保存する
    train_loss = []
    train_acc  = []
    test_loss = []
    test_acc  = []

    # Learning loop
    for epoch in xrange(1, n_epoch+1):

        # training
        # N 個の順番をランダムに並び替える
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0

        # 0..N までのデータをバッチサイズごとに使って学習
        for i in xrange(0, N, batchsize):
            x_batch = x_train[perm[i:i+batchsize]]
            y_batch = y_train[perm[i:i+batchsize]]
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            # 勾配を初期化
            optimizer.zero_grads()

            # 順伝播させて誤差と精度を算出
            loss, acc = forward(x_batch, y_batch, train=True)

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

            # if epoch == 1 and i == 0:
            #     with open('fine-tuning-mlp_graph.dot', 'w') as o:
            #         g = computational_graph.build_computational_graph((loss, ))
            #         o.write(g.dump())
            #         dot = pydot.graph_from_dot_data(g.dump())
            #         dot.set_rankdir('LR')
            #         dot.write_png('fine-tuning-mlp_graph.png', prog='dot')
            #     print('graph generated')

            sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data))  * batchsize

            if (i + 1) % 500 == 0:
                print '{} / {} loss={} accuracy={}'.format(i + 1, len(batchsize), sum_loss / N, sum_accuracy / N)
                sys.stdout.flush()

        # 訓練データの誤差と,正解精度を表示
        print 'epoch: {} done'.format(epoch)
        print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)
        train_loss.append(sum_loss / N)
        train_acc.append(sum_accuracy / N)
        sys.stdout.flush()

        # evaluation
        # テストデータで誤差と正解精度を算出し汎化性能を確認
        sum_accuracy = 0
        sum_loss     = 0
        for i in xrange(0, N_test, batchsize):
            x_batch = x_test[i:i+batchsize]
            y_batch = y_test[i:i+batchsize]
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            # 順伝播させて誤差と精度を算出
            loss, acc = forward(x_batch, y_batch, train=False)

            sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        # テストデータでの誤差と正解精度を表示
        print 'test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)
        test_loss.append(sum_loss / N_test)
        test_acc.append(sum_accuracy / N_test)
        sys.stdout.flush()

        # optimizer.lr *= DECAY_FACTOR

    # model と optimizer を保存する
    print('save the model')
    if args.gpu >= 0: model.to_cpu()
    serializers.save_hdf5('model/fine-tuning-mlp.model', model)
    if args.gpu >= 0: model.to_gpu()
    print('save the optimizer')
    serializers.save_hdf5('model/fine-tuning-mlp.state', optimizer)

    # 精度と誤差をグラフ描画
    if args.plot:

        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.ylim(0., 1.)
        plt.plot(range(len(train_acc)), train_acc)
        plt.plot(range(len(test_acc)), test_acc)
        plt.legend(['train_acc', 'test_acc'], loc=4)
        plt.title('Accuracy of cnn recognition.')
        plt.subplot(1, 2, 2)
        plt.ylim(0., 3.5)
        plt.plot(range(len(train_loss)), train_loss)
        plt.plot(range(len(test_loss)), test_loss)
        plt.legend(['train_loss', 'test_loss'], loc=4)
        plt.title('Loss of cnn recognition.')
        plt.savefig('fine-tuning-mlp_acc-loss.png')
        # plt.show()

    # 答え合わせ
    if args.plot:
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        n = 0
        for idx in np.random.permutation(N)[:20]:
            x = x_train[idx].reshape((1, 4096))
            if args.gpu >= 0:
                x = cuda.to_gpu(x)
            y, f = predict(x, train=False)
            img = x_train_image[idx].reshape((3, 224, 224))
            img = img + mean_image
            img = img[::-1, :, :]
            ans = labels[np.argmax(cuda.to_cpu(y.data[0,]))]

            plt.subplot(4, 5, n+1)
            plt.imshow(img.astype(np.uint8).transpose(1, 2, 0))
            plt.title('{}'.format(ans), size=8)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')

            n += 1
        plt.savefig('fine-tuning-mlp_test.png')
        # plt.show()

print('time spent:', time.time() - start_time)
