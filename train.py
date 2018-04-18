# -*- coding: utf-8 -*-
import argparse

import numpy as np
import chainer
from chainer import training, iterators, optimizers
from chainer.datasets import TupleDataset
from chainer.training import extensions

from utility import load_wordid_text
from network import NStepEncDec, EncDecUpdater


def main():
    src_index_text_filepath = args.src_index_sentence_text
    dst_index_text_filepath = args.dst_index_sentence_text

    print("src_index_data: ", src_index_text_filepath)
    print("dst_index_data: ", dst_index_text_filepath)

    src_dataset = load_wordid_text(src_index_text_filepath, reverse=False)
    dst_dataset = load_wordid_text(dst_index_text_filepath, reverse=False)

    n_words_src = max([max(x) for x in src_dataset]) + 1
    n_words_dst = max([max(x) for x in dst_dataset]) + 1

    n_layers = 2
    n_dim = 500

    if args.resume != '':
        model = NStepEncDec(n_layers, n_words_src, n_words_dst, n_dim)
        chainer.serializers.load_hdf5(args.resume, model)
        print(model)
    else:
        model = NStepEncDec(n_layers, n_words_src, n_words_dst, n_dim)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(0).use()
        chainer.cuda.check_cuda_available()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train = TupleDataset(src_dataset, dst_dataset)
    train_iter = iterators.SerialIterator(train, args.batch_size, shuffle=False)
    updater = EncDecUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')

    snapshot_interval = (10, 'epoch')
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}.hdf', chainer.serializers.save_hdf5), trigger=snapshot_interval)
    # trainer.extend(extensions.LogReport())
    # trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.run()

    model.to_cpu()
    chainer.serializers.save_hdf5("result/enc-dec_transmodel.hdf5", model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning RNN Language Model')

    parser.add_argument('src_index_sentence_text', action='store', nargs=None, const=None, default=None,
                        type=str, choices=None, help='Source separated text data for learning',
                        metavar=None)
    parser.add_argument('dst_index_sentence_text', action='store', nargs=None, const=None, default=None,
                        type=str, choices=None, help='Destination separated text data for learning',
                        metavar=None)

    parser.add_argument('--batch_size', '-b', type=int, default=200,
                        help='Number of sentences in each mini-batch.')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train.')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='Model file to resume learning.')

    args = parser.parse_args()

    if args.gpu >= 0:
        import cupy as cp
    else:
        cp = np

    main()
