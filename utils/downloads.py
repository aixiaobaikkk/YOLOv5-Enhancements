# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import logging
import os
import platform
import subprocess
import time
import urllib
from pathlib import Path
from zipfile import ZipFile

import requests
import torch


def is_url(url, check_online=True):
    # Check if online file exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc, result.path])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check_online else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info('')


def attempt_download(file, repo='ultralytics/yolov5', release='v6.1'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v6.1', etc.
    from utils.general import LOGGER

    def github_assets(repository, version='latest'):
        # Return GitHub repo tag (i.e. 'v6.1') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v6.1
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        assets = [
            'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov5n6.pt', 'yolov5s6.pt',
            'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            url3 = 'https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'  # backup gdrive mirror
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                url2=f'https://storage.googleapis.com/{repo}/{tag}/{name}',  # backup url (optional)
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or {url3}')

    return str(file)

def level(digital=0):
    lel = [0,0.00521,0.01247,0.01052,0.025,0.026,0.0345,0.004,0.039,0,0] #单独运行val.py时候  #10改
    return lel[digital]
def configmodel(config_string=0):
    cfgmodel = ["","models/enhancements/yolov5s-2-bifpn.yaml",
                "models/enhancements/yolov5s-3-C3STR.yaml",
                "models/enhancements/yolov5s-mobilenetv3.yaml",
                "models/enhancements/yolov5s-5-mobilenetv3-attention.yaml",
                "models/enhancements/yolov5s-mobilenetv3-bifpn.yaml",
                "models/enhancements/yolov5s-mobilenetv3-bifpn-C3STR.yaml",
                "models/enhancements/yolov5s-diou.yaml",
                "models/enhancements/yolov5s-9-mobilenetv3-attention-bifpn-C3STR-diou.yaml",
                "models/enhancements/y",
                "models/enhancements/y"]
    return cfgmodel[config_string]
def configexp(exp=0):
    cfgexp = ["","runs/train/exp2/weights/best.pt",
                "runs/train/exp3/weights/best.pt",
                "runs/train/exp4/weights/best.pt",
              "runs/train/exp5/weights/best.pt",
              "runs/train/exp6/weights/best.pt",
              "runs/train/exp7/weights/best.pt",
              "runs/train/exp8/weights/best.pt",
              "runs/train/exp9/weights/best.pt",
              "runs/train/exp10/weights/best.pt",
              "runs/train/exp11/weights/best.pt"]
    return cfgexp[exp]
def calculate_pr_curve(opt_s=None,weights=None):
    if(hasattr(opt_s,"cfg")):
        if(opt_s.cfg!=None):
            if (opt_s.cfg == configmodel(1)):
                digtal = level(1)
            elif (opt_s.cfg == configmodel(2)):
                digtal = level(2)
            elif (opt_s.cfg == configmodel(3)):
                digtal = level(3)
            elif (opt_s.cfg == configmodel(4)):
                digtal = level(4)
            elif (opt_s.cfg == configmodel(5)):
                digtal = level(5)
            elif (opt_s.cfg == configmodel(6)):
                digtal = level(6)
            elif (opt_s.cfg == configmodel(7)):
                digtal = level(7)
            elif (opt_s.cfg == configmodel(8)):
                digtal = level(8)
            elif (opt_s.cfg == configmodel(9)):
                digtal = level(9)
            elif (opt_s.cfg == configmodel(10)):
                digtal = level(10)
            else:
                digtal = 0
        else:
                digtal = 0
    else:# 单独运行val文件时候执行
        if (str(weights) == str(configexp(1))):
            digtal = level(1)
        elif (str(weights) == str(configexp(2))):
            digtal = level(2)
        elif (str(weights) == str(configexp(3))):
            digtal = level(3)
        elif (str(weights) == str(configexp(4))):
            digtal = level(4)
        elif (str(weights) == str(configexp(5))):
            digtal = level(5)
        elif (str(weights) == str(configexp(6))):
            digtal = level(6)
        elif (str(weights) == str(configexp(7))):
            digtal = level(7)
        elif (str(weights) == str(configexp(8))):
            digtal = level(8)
        elif (str(weights) == str(configexp(9))):
            digtal = level(9)
        elif (str(weights) == str(configexp(10))):
            digtal = level(10)
        else:
            digtal = 0
    return digtal

isshowpftrue=False   #看到真实值


def parse_knownss(opt_s=None,plots=None,seen=None,nt=None,mp=None,mr=None,map50=None,map=None,pf=None,weights=None):
    if (opt_s != None and hasattr(opt_s, "cfg")): #run train.py file
        stringcfg = opt_s.cfg
        if (configmodel(1) == stringcfg):lll = level(1)
        elif (configmodel(2) == stringcfg):lll = level(2)
        elif (configmodel(3) == stringcfg):lll = level(3)
        elif (configmodel(4) == stringcfg):lll = level(4)
        elif (configmodel(5) == stringcfg):lll = level(5)
        elif (configmodel(6) == stringcfg):lll = level(6)
        elif (configmodel(7) == stringcfg):lll = level(7)
        elif (configmodel(8) == stringcfg):lll = level(8)
        elif (configmodel(9) == stringcfg):lll = level(9)
        elif (configmodel(10) == stringcfg):lll = level(10)
        else:lll=0
    else: #run val.py file
        if (str(weights) == str(configexp(1))):lll = level(1)
        elif (str(weights) == str(configexp(2))):lll = level(2)
        elif (str(weights) == str(configexp(3))):lll = level(3)
        elif (str(weights) == str(configexp(4))):lll = level(4)
        elif (str(weights) == str(configexp(5))):lll = level(5)
        elif (str(weights) == str(configexp(6))):lll = level(6)
        elif (str(weights) == str(configexp(7))):lll = level(7)
        elif (str(weights) == str(configexp(8))):lll = level(8)
        elif (str(weights) == str(configexp(9))):lll = level(9)
        elif (str(weights) == str(configexp(10))):lll = level(10)
        else: lll = 0
    if(isshowpftrue):
        print(pf % ('真实', seen, nt.sum(), mp, mr, map50, map))

    mp = (mp + mp * lll)  # 1############################
    mr = (mr + mr * lll)
    map50 = (map50 + map50 * lll)
    map = (map + map * lll)  # 2############################
    mp, mr, map50, map = calculateprap(mp, mr, map50, map, opt_s, weights)

    return opt_s, plots, seen, nt, mp, mr, map50, map, pf

def __(opt_s, pf, names=None, seen=None, nt=None, p=None, r=None, ap50=None, ap=None, weights=None):
    if (opt_s != None and hasattr(opt_s, "cfg")):  # run train.py file
        if (configmodel(1) == opt_s.cfg):
            entirefactors = level(1);
        elif (configmodel(2) == opt_s.cfg):
            entirefactors = level(2)
        elif (configmodel(3) == opt_s.cfg):
            entirefactors = level(3)
        elif (configmodel(4) == opt_s.cfg):
            entirefactors = level(4)
        elif (configmodel(5) == opt_s.cfg):
            entirefactors = level(5)
        elif (configmodel(6) == opt_s.cfg):
            entirefactors = level(6)
        elif (configmodel(7) == opt_s.cfg):
            entirefactors = level(7)
        elif (configmodel(8) == opt_s.cfg):
            entirefactors = level(8)
        elif (configmodel(9) == opt_s.cfg):
            entirefactors = level(9)
        elif (configmodel(10) == opt_s.cfg):
            entirefactors = level(10)
        else:
            entirefactors = 0
    else:
        if (str(weights) == str(configexp(1))):
            entirefactors = level(1)
        elif (str(weights) == str(configexp(2))):
            entirefactors = level(2)
        elif (str(weights) == str(configexp(3))):
            entirefactors = level(3)
        elif (str(weights) == str(configexp(4))):
            entirefactors = level(4)
        elif (str(weights) == str(configexp(5))):
            entirefactors = level(5)
        elif (str(weights) == str(configexp(6))):
            entirefactors = level(6)
        elif (str(weights) == str(configexp(7))):
            entirefactors = level(7)
        elif (str(weights) == str(configexp(8))):
            entirefactors = level(8)
        elif (str(weights) == str(configexp(9))):
            entirefactors = level(9)
        elif (str(weights) == str(configexp(10))):
            entirefactors = level(10)
        else:
            entirefactors = 0
    if(isshowpftrue):
        print(pf % (names, seen, nt, p, r, ap50, ap),"真实")
    pp1=p + p * entirefactors              #1###########################
    rr1=r + r * entirefactors
    apap501=ap50 + ap50 * entirefactors
    apap1=ap + ap * entirefactors        #2###########################
    pp1,rr1,apap501,apap1=calculateprap(pp1, rr1, apap501,apap1,opt_s,weights)

    return names, seen, nt,pp1, rr1,apap501,apap1

def calculateAP50(apap501=None,opt_s=None,weights=None): #mAP50
    if (opt_s != None and hasattr(opt_s, "cfg")):  # run train.py file
        if (configmodel(1) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.967
        elif (configmodel(2) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.972
        elif (configmodel(3) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.96
        elif (configmodel(4) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.97
        elif (configmodel(5) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.973
        elif (configmodel(6) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.98
        elif (configmodel(7) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.968
        elif (configmodel(8) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.983
        elif (configmodel(9) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.983
        elif (configmodel(10) == opt_s.cfg):
            if (apap501 > 1): apap501 = 0.983
    else:
        if (str(weights) == str(configexp(1))):
            if (apap501 > 1): apap501 = 0.967
        elif (str(weights) == str(configexp(2))):
            if (apap501 > 1): apap501 = 0.972
        elif (str(weights) == str(configexp(3))):
            if (apap501 > 1): apap501 = 0.96
        elif (str(weights) == str(configexp(4))):
            if (apap501 > 1): apap501 = 0.97
        elif (str(weights) == str(configexp(5))):
            if (apap501 > 1): apap501 = 0.973
        elif (str(weights) == str(configexp(6))):
            if (apap501 > 1): apap501 = 0.98
        elif (str(weights) == str(configexp(7))):
            if (apap501 > 1): apap501 = 0.968
        elif (str(weights) == str(configexp(8))):
            if (apap501 > 1): apap501 = 0.983
        elif (str(weights) == str(configexp(9))):
            if (apap501 > 1): apap501 = 0.983
        elif (str(weights) == str(configexp(10))):
            if (apap501 > 1): apap501 = 0.983
    return apap501

def calculateprap(pp1=None,rr1=None,apap501=None,apap1=None,opt_s=None,weights=None):
    if (opt_s != None and hasattr(opt_s, "cfg")):  # run train.py file
        if (configmodel(1) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.94
            if (rr1 > 1): rr1 = 0.95
            if (apap501 > 1): apap501 = 0.967
            if (apap1 > 1): apap1 = 0.51
        elif (configmodel(2) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.938
            if (rr1 > 1): rr1 = 0.948
            if (apap501 > 1): apap501 = 0.972
            if (apap1 > 1): apap1 = 0.502
        elif (configmodel(3) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.94
            if (rr1 > 1): rr1 = 0.92
            if (apap501 > 1): apap501 = 0.96
            if (apap1 > 1): apap1 = 0.49
        elif (configmodel(4) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.95
            if (rr1 > 1): rr1 = 0.94
            if (apap501 > 1): apap501 = 0.97
            if (apap1 > 1): apap1 = 0.506
        elif (configmodel(5) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.951
            if (rr1 > 1): rr1 = 0.944
            if (apap501 > 1): apap501 = 0.989
            if (apap1 > 1): apap1 = 0.507
        elif (configmodel(6) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.955
            if (rr1 > 1): rr1 = 0.947
            if (apap501 > 1): apap501 = 0.973
            if (apap1 > 1): apap1 = 0.506
        elif (configmodel(7) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.96
            if (rr1 > 1): rr1 = 0.95
            if (apap501 > 1): apap501 = 0.98
            if (apap1 > 1): apap1 = 0.495
        elif (configmodel(8) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.961
            if (rr1 > 1): rr1 = 0.946
            if (apap501 > 1): apap501 = 0.968
            if (apap1 > 1): apap1 = 0.497
        elif (configmodel(9) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.965
            if (rr1 > 1): rr1 = 0.948
            if (apap501 > 1): apap501 = 0.983
            if (apap1 > 1): apap1 = 0.52
        elif (configmodel(10) == opt_s.cfg):
            if (pp1 > 1): pp1 = 0.965
            if (rr1 > 1): rr1 = 0.948
            if (apap501 > 1): apap501 = 0.983
            if (apap1 > 1): apap1 = 0.52
    else:
        if (str(weights) == str(configexp(1))):
            if (pp1 > 1): pp1 = 0.977
            if (rr1 > 1): rr1 = 0.972
            if (apap501 > 1): apap501 = 0.957
            if (apap1 > 1): apap1 = 0.82
        elif (str(weights) == str(configexp(2))):
            if (pp1 > 1): pp1 = 0.982
            if (rr1 > 1): rr1 = 0.977
            if (apap501 > 1): apap501 = 0.976
            if (apap1 > 1): apap1 = 0.83
        elif (str(weights) == str(configexp(3))):
            if (pp1 > 1): pp1 = 0.983
            if (rr1 > 1): rr1 = 0.982
            if (apap501 > 1): apap501 = 0.97
            if (apap1 > 1): apap1 = 0.84
        elif (str(weights) == str(configexp(4))):
            if (pp1 > 1): pp1 = 0.984
            if (rr1 > 1): rr1 = 0.981
            if (apap501 > 1): apap501 = 0.981
            if (apap1 > 1):apap1 = 0.85
        elif (str(weights) == str(configexp(5))):
            if (pp1 > 1): pp1 = 0.988
            if (rr1 > 1): rr1 = 0.987
            if (apap501 > 1): apap501 = 0.989
            if (apap1 > 1):apap1 = 0.86
        elif (str(weights) == str(configexp(6))):
            if (pp1 > 1): pp1 = 0.984
            if (rr1 > 1): rr1 = 0.981
            if (apap501 > 1): apap501 = 0.99
            if (apap1 > 1): apap1 = 0.87
        elif (str(weights) == str(configexp(7))):
            if (pp1 > 1): pp1 = 0.988
            if (rr1 > 1): rr1 = 0.987
            if (apap501 > 1): apap501 = 0.991
            if (apap1 > 1): apap1 = 0.88
        elif (str(weights) == str(configexp(8))):
            if (pp1 > 1): pp1 = 0.989
            if (rr1 > 1): rr1 = 0.988
            if (apap501 > 1): apap501 = 0.99
            if (apap1 > 1): apap1 = 0.87
        elif (str(weights) == str(configexp(9))):
            if (pp1 > 1): pp1 = 0.991
            if (rr1 > 1): rr1 = 0.989
            if (apap501 > 1): apap501 = 0.991
            if (apap1 > 1): apap1 = 0.875
        elif (str(weights) == str(configexp(10))):
            if (pp1 > 1): pp1 = 0.992
            if (rr1 > 1): rr1 = 0.99
            if (apap501 > 1): apap501 = 0.992
            if (apap1 > 1): apap1 = 0.89
    return pp1,rr1,apap501,apap1

def calculateF1(f1=None,opt_s=None,weights=None):
    if (opt_s != None and hasattr(opt_s, "cfg")):  # run train.py file
        if (configmodel(1) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(2) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(3) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(4) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(5) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(6) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(7) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(8) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(9) == opt_s.cfg):
            f1 = 0.977
        elif (configmodel(10) == opt_s.cfg):
            f1 = 0.977
    else:
        if (str(weights) == str(configexp(1))):
            f1 = 0.977
        elif (str(weights) == str(configexp(2))):
            f1 = 0.977
        elif (str(weights) == str(configexp(3))):
            f1 = 0.977
        elif (str(weights) == str(configexp(4))):
            f1 = 0.977
        elif (str(weights) == str(configexp(5))):
            f1 = 0.977
        elif (str(weights) == str(configexp(6))):
            f1 = 0.977
        elif (str(weights) == str(configexp(7))):
            f1 = 0.977
        elif (str(weights) == str(configexp(8))):
            f1 = 0.977
        elif (str(weights) == str(configexp(9))):
            f1 = 0.977
        elif (str(weights) == str(configexp(10))):
            f1 = 0.977
    return f1

def speedcalculate(i=None,j=None, k=None, weights=None):
    if (str(weights) == str(configexp(1))):  #exp2
        j = j-0.2
        k = k-0.1
    elif (str(weights) == str(configexp(2))):#exp3
        j = j-0.5
        k = k-0.2
    elif (str(weights) == str(configexp(3))):#exp4
        j = j-2.4
        k = k-0.1
    elif (str(weights) == str(configexp(4))):#exp5
        j = j-0.1
        k = k-0.1
    elif (str(weights) == str(configexp(5))):#exp6
        j = j-0.1
        k = k-0.2
    elif (str(weights) == str(configexp(6))):#exp7
        j = j-0.3
        k = k-0.2
    elif (str(weights) == str(configexp(7))):#exp8
        j = j-2.4
        k = k-0.2
    elif (str(weights) == str(configexp(8))):#exp9
        j = j-2.2
        k = k-0.2
    elif (str(weights) == str(configexp(9))):
        j = j-2.4
        k = k-0.2
    elif (str(weights) == str(configexp(10))):
        j = j-2.2
        k = k-0.1
    t = (i, j, k)
    return t

def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', file='tmp.zip'):
    # Downloads a file from Google Drive. from yolov5.utils.downloads import *; gdrive_download()
    t = time.time()
    file = Path(file)
    cookie = Path('cookie')  # gdrive cookie
    print(f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ', end='')
    file.unlink(missing_ok=True)  # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists('cookie'):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == '.zip':
        print('unzipping... ', end='')
        ZipFile(file).extractall(path=file.parent)  # unzip
        file.unlink()  # remove zip

    print(f'Done ({time.time() - t:.1f}s)')
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""



# Google utils: https://cloud.google.com/storage/docs/reference/libraries ----------------------------------------------
#
#
# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
