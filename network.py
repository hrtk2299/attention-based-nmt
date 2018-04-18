# -*- coding: utf-8 -*-
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training

from attention import BahdanauAttention, AttentionWrapper

BOS = 0
EOS = 1


class NStepEncDec(chainer.Chain):
    def __init__(self, n_layers, n_word_src, n_word_dst, n_units):
        super(NStepEncDec, self).__init__()
        with self.init_scope():
            self.embed_src = L.EmbedID(n_word_src, n_units)
            self.embed_dst = L.EmbedID(n_word_dst, n_units)
            self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.attention_mechanism = BahdanauAttention(n_units)
            self.decoder_with_attn = AttentionWrapper(self.decoder, self.attention_mechanism)
            self.fc = L.Linear(n_units, n_word_dst)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        ys_in = [y[:-1] for y in ys]
        ys_out = [y[1:] for y in ys]

        exs = [self.embed_src(x) for x in xs]
        eys = [self.embed_dst(y) for y in ys_in]

        hx, cx, memory = self.encoder(None, None, exs)

        memory_dim = memory[0].shape[1]
        memory = [x.data[:, :memory_dim // 2] + x.data[:, memory_dim // 2:] for x in memory]

        hx = self.xp.dstack([hx.data[i] + hx.data[i + 1]
                             for i in range(0, 2 * self.n_layers, 2)]).transpose(2, 0, 1)
        cx = self.xp.dstack([cx.data[i]
                             for i in range(0, 2 * self.n_layers, 2)]).transpose(2, 0, 1)

        _, _, os = self.decoder_with_attn(hx, cx, eys, memory)

        n_batch = len(xs)
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)

        loss = F.sum(F.softmax_cross_entropy(
            self.fc(concat_os), concat_ys_out, reduce='no')) / n_batch

        return loss

    def translate(self, xs, max_length=100):
        n_batch = len(xs)

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            exs = [self.embed_src(x) for x in xs]

            h, c, memory = self.encoder(None, None, exs)

            memory_dim = memory[0].shape[1]
            memory = [x.data[:, :memory_dim // 2] + x.data[:, memory_dim // 2:] for x in memory]

            h = self.xp.dstack([h.data[i] + h.data[i + 1]
                                for i in range(0, 2 * self.n_layers, 2)]).transpose(2, 0, 1)
            c = self.xp.dstack([c.data[i]
                                for i in range(0, 2 * self.n_layers, 2)]).transpose(2, 0, 1)
            ys = self.xp.full(n_batch, BOS, 'i')

            result = []
            for _ in range(max_length):
                eys = self.embed_dst(ys)
                eys = F.split_axis(eys, n_batch, 0)

                h, c, ys = self.decoder_with_attn(h, c, eys, memory)

                cys = F.concat(ys, axis=0)
                wy = self.fc(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')

                result.append(ys)

        result = cuda.to_cpu(self.xp.vstack(result)).T

        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


class EncDecUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(EncDecUpdater, self).__init__(
            train_iter,
            optimizer,
            device=device
        )

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        model = optimizer.target

        x = train_iter.__next__()
        xs = [model.xp.array(s[0]) for s in x]
        ys = [model.xp.array(s[1]) for s in x]

        loss = model(xs, ys)

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()
