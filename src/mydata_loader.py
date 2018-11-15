#!/usr/bin/env python3
# coding: utf-8

"""
data loader for DeFN
"""

from __future__ import division, absolute_import, print_function, unicode_literals

import pandas as pd
import numpy as np
from sklearn.utils import shuffle


class MyDataLoader:
    """
    My data loader
    """
    def __init__(self, file_prefix, batchsize=100):
        self.file_prefix = file_prefix
        self.sets = ["train", "validation", "test"]
        # self.sets = ["validation"]
        self.xys = ["x", "y"]
        self.data = {}          # train/validation/test
        self.batchsize = batchsize
        self.current_batch = 0
        self.num_batches = {}
        self.xdim = 0
        self.ydim = 0

        for name in self.sets:
            xfile = self.file_prefix + "_" + name + "_feat.csv.gz"
            yfile = self.file_prefix + "_" + name + "_label.csv.gz"
            text = "Opening ({0:s}, {1:s})"
            print(text.format(xfile, yfile))

            x = pd.read_csv(xfile, sep=",", header=None)
            y = pd.read_csv(yfile, sep=",", header=None)

            # shuffle
            if name == "train":
                x, y = shuffle(x, y, random_state=0)

            xsize = len(x)
            ysize = len(y)
            if xsize != ysize:
                print("error: |y| != |x|")

            if xsize < batchsize:
                print("error: |x| is smaller than batchsize")

            self.xdim = x.shape[1]
            self.ydim = y.shape[1]

            self.num_batches[name] = int(xsize / self.batchsize)
            tmp = self.num_batches[name]

            batch_xs = np.split(x[0:tmp * self.batchsize], tmp)
            batch_ys = np.split(y[0:tmp * self.batchsize], tmp)

            dat = {"x": x, "y": y, "batch_xs": batch_xs, "batch_ys": batch_ys}
            self.data[name] = dat

    def get_xdim(self):
        return self.xdim

    def get_ydim(self):
        return self.ydim

    def get_data_sets(self, setname, xyname):
        if setname in self.sets and xyname in self.xys:
            return self.data[setname][xyname]
        else:
            print("error")

    def next_batch(self, setname):
        """
        batch_xs, batch_ys = next_batch("validation", 100)
        """
        if setname not in self.sets:
            print("error")
            return None
        next_xs = self.data[setname]["batch_xs"][self.current_batch]
        next_ys = self.data[setname]["batch_ys"][self.current_batch]

        if self.current_batch == self.num_batches[setname] - 1:
            self.current_batch = 0
        else:
            self.current_batch = self.current_batch + 1
        return next_xs, next_ys


def main():
    """
    main
    """
    dataloader = MyDataLoader(file_prefix="../data/charval2017X_M24", batchsize=50)
    x = dataloader.get_data_sets("validation", "x")
    y = dataloader.get_data_sets("validation", "y")

    # batch_xs, batch_ys = dataloader.next_batch("validation")
    print(dataloader.get_xdim())
    print(dataloader.get_ydim())

    for i in range(2000):
        batch_xs, batch_ys = dataloader.next_batch("train")
        # print(batch_xs[0][0])
    l = len(batch_xs)
    print("%d" % l)


if __name__ == "__main__":
    main()


# Local Variables:
# coding: utf-8
# End:
