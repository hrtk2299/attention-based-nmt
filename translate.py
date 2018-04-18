# -*- coding: utf-8 -*-
import argparse

import chainer

from utility import load_wordidmap
from network import NStepEncDec


def main():
    src_index_text_filepath = args.src_index_words_text
    dst_index_text_filepath = args.dst_index_words_text

    print("src_index_words_data: ", src_index_text_filepath)
    print("dst_index_words_data: ", dst_index_text_filepath)

    src_index2word, _ = load_wordidmap(src_index_text_filepath)
    dst_index2word, _ = load_wordidmap(dst_index_text_filepath)

    n_words_src = len(src_index2word)
    n_words_dst = len(dst_index2word)

    src_word2index = {value: key for key, value in src_index2word.items()}

    n_layers = 2
    n_dim = 500
    model = NStepEncDec(n_layers, n_words_src, n_words_dst, n_dim)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(0).use()
        chainer.cuda.check_cuda_available()
        model.to_gpu()

    chainer.serializers.load_hdf5(args.model_filepath, model)

    xs = []
    with open(args.target_text, 'r') as f:
        for line in f:
            tmp_line = line.strip().strip('.').strip(' ').split(' ')
            idx_line = [0] + [
                src_word2index[x] if x in src_word2index else src_word2index['<eos>'] for x in tmp_line] + [1]
            # idx_line = idx_line[::-1]
            xs.append(xp.asarray(idx_line))

    result = model.translate(xs)

    xs = [chainer.cuda.to_cpu(x) for x in xs]
    for no, (x, y) in enumerate(zip(xs, result), start=1):
        sentence_src = ' '.join([src_index2word[word_id] for word_id in x[1:-1]])
        sentence_dst = ' '.join([dst_index2word[word_id] for word_id in y])
        print(f'{no:>4}: {sentence_src:<70}\n   -> {sentence_dst}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural machine translator')

    parser.add_argument('model_filepath', type=str, help='Pre-trained model filepath.')
    parser.add_argument('src_index_words_text', type=str, help='Source separated text data for learning')
    parser.add_argument('dst_index_words_text', type=str, help='Destination separated text data for learning')
    parser.add_argument('target_text', type=str, help='Target text data')

    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')

    args = parser.parse_args()

    if args.gpu >= 0:
        import cupy as xp
    else:
        import numpy as xp

    main()
