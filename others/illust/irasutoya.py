#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

__version__ = '0.0.1'

import sys, time, logging, re
import numpy as np
np.set_printoptions(precision=20)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
# handler = logging.FileHandler(filename="log.txt")
handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

def pp(obj):
    import pprint
    pp = pprint.PrettyPrinter(indent=1, width=160)
    logger.info(pp.pformat(obj))


start_time = time.time()

import re
from bs4 import BeautifulSoup
import requests
import urllib.request


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--keyword', type=str, default='*', help='')
    parser.add_argument('--max_num', type=int, default=99999, help='')
    args = parser.parse_args()

    # 一時的に保存するページのリスト
    url_list = []

    # 検索結果から各ページのリンク先をmaxページ分だけ取得
    for num in range(0, args.max_num, 20):
        url = "http://www.irasutoya.com/search?q={:}&max-results=20&start={:}".format(args.keyword, num)
        # logger.debug(url)

        soup = BeautifulSoup(requests.get(url).text, 'html.parser')

        # Link の箇所を select
        links = soup.select("a[href]")

        for a in links:

            # Link タグの href 属性の箇所を抜き出す
            href = a.attrs['href']

            # 画像データに対応するページのリンク先のみをリストに格納
            if re.search(r"irasutoya.*blog-post.*html$", href):
                if not href in url_list:
                    logger.debug(href)
                    url_list.append(href)

    # 各ページから画像データのリンクを取得して,画像を保存
    for link in url_list:
        res = requests.get(link)
        soup = BeautifulSoup(res.text, 'html.parser')

        # 記事中の画像データを抜き出す

        # class separator -> a の抜き出し
        links = soup.select(".separator > a")

        for a in links:
            try:
                # キャプションの抜き出し
                caption = a.select("img[alta]")[0].get('alt')

                # hrefのデータを取得
                imageLink = 'https:'+a.get('href')

                # ファイル名の取得
                filename = re.search(r".*\/(.*png|.*jpg)$", imageLink)

                # 画像をダウンロードする
                # urllib.request.urlretrieve(imageLink, "download/{}".format(filename.group(1)))

                # デバッグ用にダウンロード先Linkを表示
                print('{:}\t{:}'.format(imageLink, caption))
                sys.stdout.flush()

            except Exception as e:
                logger.exception(e)