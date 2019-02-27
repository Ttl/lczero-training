#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import yaml
import sys
import glob
import gzip
import random
import multiprocessing as mp
import tensorflow as tf
from tfprocess import TFProcess
from chunkparser import ChunkParser

SKIP = 32


def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")


def get_latest_chunks(path, num_chunks):
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)

    if len(chunks) < num_chunks:
        print("Not enough chunks {}".format(len(chunks)))
        sys.exit(1)

    print("sorting {} chunks...".format(len(chunks)), end='')
    chunks.sort(key=os.path.getmtime, reverse=True)
    print("[done]")
    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]), os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks

def get_window_chunks(path, num_chunks, train_ratio, start):
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)

    if len(chunks) < num_chunks:
        print("Not enough chunks {}".format(len(chunks)))
        sys.exit(1)

    get_folder = lambda x: os.path.split(x)[0].split(os.sep)[-1]

    chunks.sort(key=get_folder)

    train_chunks = []
    test_chunks = []
    for i in range(len(chunks)):
        if len(train_chunks) <= i * train_ratio:
            train_chunks.append(chunks[i])
        else:
            test_chunks.append(chunks[i])

    train_start = int(train_ratio * start)
    num_train_chunks = int(train_ratio * num_chunks)
    test_start = int((1 - train_ratio) * start)
    num_test_chunks = int((1 - train_ratio) * num_chunks)
    train_chunks = train_chunks[train_start:train_start + num_train_chunks]
    test_chunks = test_chunks[test_start:test_start + num_test_chunks]

    print("{} - {}".format(train_chunks[-1], train_chunks[0]))
    print("{} - {}".format(test_chunks[-1], test_chunks[0]))
    random.shuffle(train_chunks)
    random.shuffle(test_chunks)
    return train_chunks, test_chunks

class FileDataSrc:
    """
        data source yielding chunkdata from chunk files.
    """
    def __init__(self, chunks):
        self._chunks = []
        self.done = chunks

    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        self._chunks = []
        self.done = value

    @chunks.getter
    def chunks(self, value):
        return self._chunks

    def next(self):
        if not self._chunks:
            self._chunks, self.done = self.done, self._chunks
            random.shuffle(self._chunks)
        if not self._chunks:
            return None
        while len(self._chunks):
            filename = self._chunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    self.done.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))

def get_train_test_chunks(cfg, chunks_start):
    num_chunks = cfg['dataset']['num_chunks']
    train_ratio = cfg['dataset']['train_ratio']
    num_train = int(num_chunks * train_ratio)
    num_test = num_chunks - num_train
    if 'input_test' in cfg['dataset']:
        train_chunks, _ = get_window_chunks(cfg['dataset']['input_train'], num_train, 1, chunks_start)
        test_chunks = get_latest_chunks(cfg['dataset']['input_test'], num_test)
    else:
        train_chunks, test_chunks = get_window_chunks(cfg['dataset']['input'], num_chunks, train_ratio, chunks_start)
    return train_chunks, test_chunks

def replay(cmd, cfg):
    steps_per_net = cfg['replay']['steps_per_net']
    games_per_net = cfg['replay']['games_per_net']

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    steps = 0
    chunks_start = 0
    if os.path.exists(os.path.join(root_dir, 'checkpoint')):
        cp = tf.train.latest_checkpoint(root_dir)
        steps = int(cp.split('-')[-1])
        chunks_start = (steps // steps_per_net) * games_per_net

    train_chunks, test_chunks = get_train_test_chunks(cfg, chunks_start)

    shuffle_size = cfg['training']['shuffle_size']
    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits
    # Load data with split batch size, which will be combined to the total batch size in tfprocess.
    ChunkParser.BATCH_SIZE = split_batch_size

    num_chunks = cfg['dataset']['num_chunks']
    train_ratio = cfg['dataset']['train_ratio']
    num_train = int(num_chunks * train_ratio)
    num_test = num_chunks - num_train

    train_parser = ChunkParser(FileDataSrc(train_chunks),
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        train_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    train_iterator = dataset.make_one_shot_iterator()

    shuffle_size = int(shuffle_size*(1.0-train_ratio))
    test_parser = ChunkParser(FileDataSrc(test_chunks),
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        test_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    test_iterator = dataset.make_one_shot_iterator()

    tfprocess = TFProcess(cfg)
    tfprocess.init(dataset, train_iterator, test_iterator)

    if os.path.exists(os.path.join(root_dir, 'checkpoint')):
        tfprocess.restore(cp)

    # If number of test positions is not given
    # sweeps through all test chunks statistically
    # Assumes average of 10 samples per test game.
    # For simplicity, testing can use the split batch size instead of total batch size.
    # This does not affect results, because test results are simple averages that are independent of batch size.
    num_evals = cfg['training'].get('num_test_positions', num_test * 10)
    num_evals = max(1, num_evals // ChunkParser.BATCH_SIZE)
    print("Using {} evaluation batches".format(num_evals))

    real_total_steps = cfg['training']['total_steps']

    num_chunks = cfg['dataset']['num_chunks']
    train_ratio = cfg['dataset']['train_ratio']
    num_train = int(num_chunks * train_ratio)

    while True:
        steps = tfprocess.session.run(tfprocess.global_step)
        cfg['training']['total_steps'] = steps + steps_per_net
        tfprocess.process_loop(total_batch_size, num_evals, batch_splits=batch_splits)
        if steps + steps_per_net >= real_total_steps:
            break
        chunks_start += games_per_net
        train_chunks, test_chunks = get_train_test_chunks(cfg, chunks_start)
        if len(train_chunks) < 0.9 * num_train:
            print("Ran out of training chunks")
            break
        train_parser.new_chunks(train_chunks)
        test_parser.new_chunks(test_chunks)

    if cmd.output is not None:
        tfprocess.save_leelaz_weights(cmd.output)

    tfprocess.session.close()
    train_parser.shutdown()
    test_parser.shutdown()

def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    rl_replay = False
    try:
        rl_replay = cfg['replay']['replay']
    except:
        pass

    if rl_replay:
        return replay(cmd, cfg)

    num_chunks = cfg['dataset']['num_chunks']
    train_ratio = cfg['dataset']['train_ratio']
    num_train = int(num_chunks*train_ratio)
    num_test = num_chunks - num_train
    if 'input_test' in cfg['dataset']:
        train_chunks = get_latest_chunks(cfg['dataset']['input_train'], num_train)
        test_chunks = get_latest_chunks(cfg['dataset']['input_test'], num_test)
    else:
        chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks)
        train_chunks = chunks[:num_train]
        test_chunks = chunks[num_train:]

    shuffle_size = cfg['training']['shuffle_size']
    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits
    # Load data with split batch size, which will be combined to the total batch size in tfprocess.
    ChunkParser.BATCH_SIZE = split_batch_size

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    train_parser = ChunkParser(FileDataSrc(train_chunks),
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        train_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    train_iterator = dataset.make_one_shot_iterator()

    shuffle_size = int(shuffle_size*(1.0-train_ratio))
    test_parser = ChunkParser(FileDataSrc(test_chunks),
            shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
        test_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
    dataset = dataset.map(ChunkParser.parse_function)
    dataset = dataset.prefetch(4)
    test_iterator = dataset.make_one_shot_iterator()

    tfprocess = TFProcess(cfg)
    tfprocess.init(dataset, train_iterator, test_iterator)

    if os.path.exists(os.path.join(root_dir, 'checkpoint')):
        cp = tf.train.latest_checkpoint(root_dir)
        tfprocess.restore(cp)

    # If number of test positions is not given
    # sweeps through all test chunks statistically
    # Assumes average of 10 samples per test game.
    # For simplicity, testing can use the split batch size instead of total batch size.
    # This does not affect results, because test results are simple averages that are independent of batch size.
    num_evals = cfg['training'].get('num_test_positions', num_test * 10)
    num_evals = max(1, num_evals // ChunkParser.BATCH_SIZE)
    print("Using {} evaluation batches".format(num_evals))

    tfprocess.process_loop(total_batch_size, num_evals, batch_splits=batch_splits)

    if cmd.output is not None:
        tfprocess.save_leelaz_weights(cmd.output)

    tfprocess.session.close()
    train_parser.shutdown()
    test_parser.shutdown()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg', type=argparse.FileType('r'),
        help='yaml configuration with training parameters')
    argparser.add_argument('--output', type=str,
        help='file to store weights in')

    mp.set_start_method('spawn')
    main(argparser.parse_args())
    mp.freeze_support()
