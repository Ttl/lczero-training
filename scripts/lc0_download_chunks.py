#!/usr/bin/python3
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from os import listdir, makedirs
import os.path
import subprocess
from subprocess import PIPE
import multiprocessing

location = 'http://data.lczero.org/files/'
tar_dir = './tars/'
rescore_dir = './rescored/'
rescorer = '../../lc0/build/release/rescorer'
syzygy_paths = ''
rescore = False

basename = 'training-run2-'
startdate = datetime(2019, 2, 17, 6, 00)

def download_chunk(filename):
    link = location + filename
    folder_name = filename.split('.')[0]
    folders = [f for f in listdir(tar_dir) if not os.path.isfile(os.path.join(tar_dir, f))]
    if folder_name in folders:
        # Folder names keep track of what chunks we have. It's important not to
        # delete empty folders in tar_dir.
        #
        # If we have folder we have downloaded and untarred this chunk.
        # If we don't have folder we either haven't downloaded it yet or have
        # incomplete download.
        print("Already downloaded {}".format(filename))
    else:
        print('Downloading', link)
        r = subprocess.run(["wget", "-c", link], cwd=tar_dir, stdout=PIPE, stderr=PIPE)
        if r.returncode != 0:
            print(r.stderr)
            raise ValueError("Wget failed")
        print('Extracting tar')
        subprocess.run(["tar", "xf", filename], cwd=tar_dir, stdout=PIPE, stderr=PIPE)
        if r.returncode != 0:
            print(r.stderr)
            raise ValueError("tar failed")

    if rescore:
        in_dir = filename.split('.')[0]
        in_dir = os.path.join(tar_dir, in_dir)
        # Check if there is anything to rescore
        files = [f for f in listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
        if len(files) > 0:
            print("Rescoring {}".format(folder_name))
            out_dir = os.path.join(rescore_dir, folder_name)
            makedirs(out_dir, exist_ok=True)
            r = subprocess.run([rescorer, "rescore", "--syzygy-paths={}".format(syzygy_paths), "--input={}".format(in_dir), "--output={}".format(out_dir)], stdout=PIPE, stderr=PIPE)
            if r.returncode != 0:
                print(r.stderr)
                raise ValueError("rescorer failed")

page = requests.get(location)

soup = BeautifulSoup(page.content, 'html.parser')
links = list(soup.find_all('a', href=True))

tarfiles = [f for f in listdir(tar_dir) if os.path.isfile(os.path.join(tar_dir, f)) and f.endswith('.tar')]
tarfiles = set(tarfiles)

todo_list = []
for link in links:
    if not link.text.startswith(basename):
        continue
    link_date = link.text[len(basename):].split('.')[0]
    link_date = datetime.strptime(link_date, '%Y%m%d-%H%M')
    filename = link['href']
    if link_date >= startdate:
        todo_list.append(filename)

pool = multiprocessing.Pool(processes=4)
results = pool.map_async(download_chunk, todo_list).get()
