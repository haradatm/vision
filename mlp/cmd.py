#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# 実験:
# 結果:
#
__version__ = '0.0.1'

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#print sys.getdefaultencoding()

# usage: curl -v - -H "Content-type: application/json" -X POST -d '{"text":"","N":"10"}' http://localhost:8088/cmd

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

import subprocess

def search(text, N=10, sort=True):

    cmd1 = 'ls -lt '
    cmd2 = 'head -n {} '.format(N)
    p1 = subprocess.Popen(cmd1.strip().split(' '), stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2.strip().split(' '), stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    output = p2.communicate()[0]

    ret = []
    for i, line in enumerate(output.splitlines()):
        if i >= N:
            break
        cols = re.split(r'[ \t]+', line)
        if len(cols) < 9:
            i -= 1
            continue

        score = float(cols[4])
        label = unicode(cols[2])
        text  = unicode(cols[8])

        ret.append({
            'score': '%0.6f' % score,
            'label': label,
            'text':  text,
        })

    return ret


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--text', type=unicode, default=None, help='text to search')
    args = parser.parse_args()

    results = search(args.text, N=10)
    pp(results)

    print('time spent:', time.time() - start_time)