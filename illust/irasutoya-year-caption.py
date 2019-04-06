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
    parser.add_argument('--year', type=int, default=2012, help='')
    args = parser.parse_args()

    # 一時的に保存するページのリスト
    url_list = []

    # 記事中の画像データを抜き出す
    for month in range(1, 13):
        logger.debug("https://www.irasutoya.com/{:04d}/{:02d}/blog-post_XXXX.html".format(args.year, month))

        for i in range(1, 1000):

            # url = "http://www.irasutoya.com/search?q={:}&max-results=20&start={:}".format(args.keyword, num)
            url = "https://www.irasutoya.com/{:04d}/{:02d}/blog-post_{:d}.html".format(args.year, month, i)

            soup = BeautifulSoup(requests.get(url).text, 'html.parser')

            # class separator の抜き出し
            links = soup.select(".separator")

            caption = ''
            for a in links:
                try:
                    # キャプションの抜き出し
                    caption = caption + a.text.strip()

                except Exception as e:
                    logger.exception(e)

            # class separator -> a の抜き出し
            links = soup.select(".separator > a")

            for a in links:
                try:
                    # キャプションの抜き出し
                    title = a.select("img[alt]")[0].get('alt')

                    # hrefのデータを取得
                    link = 'https:' + a.get('href')

                    # ファイル名の取得
                    filename = re.search(r".*\/(.*png|.*jpg)$", link)

                    # 画像をダウンロードする
                    # urllib.request.urlretrieve(link, "download/{}".format(filename.group(1)))

                    # デバッグ用にダウンロード先Linkを表示
                    print('{:}\t{:}\t{:}'.format(link, title, caption))
                    sys.stdout.flush()

                except Exception as e:
                    logger.exception(e)
                    continue
