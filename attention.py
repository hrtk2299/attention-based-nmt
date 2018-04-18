# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class BahdanauAttention(chainer.Chain):
    def __init__(self, dim: int):
        super(BahdanauAttention, self).__init__()
        with self.init_scope():
            self.query_layer = L.Linear(dim, dim, nobias=True)
            self.memory_layer = L.Linear(dim, dim, nobias=True)
            self.v = L.Linear(dim, 1, nobias=True)

    def __call__(self, query: list, memory: list) -> list:
        # query: batch * (len, dim)
        processed_query = [self.query_layer(x).data for x in query]

        alignment_list = []
        # processed_query: batch * (dst_len, dim)
        # memory: batch * (src_len, dim)
        for i in range(len(processed_query)):
            alignment = []
            for j in range(len(processed_query[i])):
                alignment.append(self.v(F.tanh(processed_query[i][j] + self.memory_layer(memory[i]))).data.T)
            alignment_list.append(self.xp.concatenate(alignment))

        return alignment_list


class AttentionWrapper(chainer.Chain):
    def __init__(self, lstm_cell, attention_mechanism):
        super(AttentionWrapper, self).__init__()
        self.lstm_cell = lstm_cell
        self.attention_mechanism = attention_mechanism

        with self.init_scope():
            self.proj = L.Linear(self.lstm_cell.out_size * 2, self.lstm_cell.out_size, nobias=True)

    def __call__(self, hx, cx, xs, memory: list):
        alignment_list = self.attention_mechanism(xs, memory)
        alignment_list = [F.softmax(alignment).data for alignment in alignment_list]
        attention = [x @ y for x, y in zip(alignment_list, memory)]
        cell_input = [self.xp.concatenate((x.data, y), axis=1)
                      for x, y in zip(xs, attention)]
        cell_input = [F.tanh(self.proj(x)) for x in cell_input]
        hx, cx, ys = self.lstm_cell(hx, cx, cell_input)
        return hx, cx, ys


def test():
    import numpy as xp

    # BahdanauAttention test
    in_batch = 10
    out_batch = 20
    attention_mechanism = BahdanauAttention(256)
    query = [xp.ones((15, 256)).astype(xp.float32) for _ in range(in_batch)]
    memory = [xp.ones((25, 256)).astype(xp.float32) for _ in range(out_batch)]
    ys = attention_mechanism(query, memory)
    assert ys[0].shape == (15, 25)
    print('BahdanauAttention test: OK')

    # AttentionWrapper test
    lstm_cell = L.NStepLSTM(2, 256, 256, 0.1)
    attention_mechanism = BahdanauAttention(256)
    attention_wrapper = AttentionWrapper(lstm_cell, attention_mechanism)

    in_batch = 10
    out_batch = 20
    xs = [xp.ones((15, 256)).astype(xp.float32) for _ in range(in_batch)]
    memory = [xp.ones((25, 256)).astype(xp.float32) for _ in range(out_batch)]

    hx, cx, ys = attention_wrapper(None, None, xs, memory)
    assert ys[0].shape == (15, 256)
    print('AttentionWrapper test: OK')


if __name__ == '__main__':
    test()
