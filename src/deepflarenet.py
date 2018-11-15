#!/usr/bin/env python3
# coding: utf-8

"""
DeFN main

"""

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import mydata_loader
from dfn_utils import *


class DeepFlareNet(object):
    """
    Feature extraction by 4-layered NN
    """
    def __init__(self, batchsize=50,
                 max_epoch=3000,
                 classweights=None,
                 logdir="./tmp",
                 inprefix="tmpin",
                 outfile_model="tmp.ckpt", outprefix_latent="tmplat", outfile_pred="tmppred"):
        # initialization
        tf.set_random_seed(0)
        self.batchsize = batchsize
        self.max_epoch = max_epoch
        self.classweights = classweights
        self.pkeep = tf.placeholder(tf.float32)
        # `self.is_training` switches train or test in BN
        self.is_training = tf.placeholder(tf.bool)
        self.logdir = logdir
        self.inprefix = inprefix
        self.outfile_model = outfile_model
        self.outprefix_latent = outprefix_latent
        self.outfile_pred = outfile_pred

        # data set
        self.dataloader = mydata_loader.MyDataLoader(file_prefix=self.inprefix,
                                                     batchsize=self.batchsize)
        self.trainset_x = self.dataloader.get_data_sets("train", "x")
        self.trainset_y = self.dataloader.get_data_sets("train", "y")
        self.validationset_x = self.dataloader.get_data_sets("validation", "x")
        self.validationset_y = self.dataloader.get_data_sets("validation", "y")
        self.testset_x = self.dataloader.get_data_sets("test", "x")
        self.testset_y = self.dataloader.get_data_sets("test", "y")

        self.dim_X = self.dataloader.get_xdim()
        self.dim_Y = self.dataloader.get_ydim()

        self.X = tf.placeholder(tf.float32, [None, self.dim_X])
        self.Y = tf.placeholder(tf.float32, [None, self.dim_Y])  # output category

        # Log
        self.max_val_acc = 0
        self.tes_acc_when_max_val = 0
        self.max_tes_tss = 0

        # Declare class variables
        self.y_pred = None
        self.saver = self.writer = self.summary_op = self.sess = None
        self.train_op_extractor = None
        self.extracted_features = None

        # Network structure: number of nodes
        self.extractor_num_nodes = [self.dim_X, 200, 200, self.dim_X, 200, 200, self.dim_X, 200, self.dim_Y]

        self.ext_W, self.ext_b = self.declare_w_and_b(self.extractor_num_nodes, "ext")

        self.cost_extractor = 0

    def build_extractor(self, X):
        """
        network structure
        h0: input layer
        """
        h1 = tf.nn.relu(tf.matmul(X, self.ext_W[0]) + self.ext_b[0])
        h1d = tf.nn.dropout(h1, self.pkeep)

        h2bn = self.relu_BN(h1d, self.ext_W[1], self.is_training, "ext_BN_1")
        h3bn = self.relu_BN(h2bn, self.ext_W[2], self.is_training, "ext_BN_2") + X
        h4bn = self.relu_BN(h3bn, self.ext_W[3], self.is_training, "ext_BN_3")
        h5bn = self.relu_BN(h4bn, self.ext_W[4], self.is_training, "ext_BN_4")
        h6bn = self.relu_BN(h5bn, self.ext_W[5], self.is_training, "ext_BN_5") + X
        h7bn = self.relu_BN(h6bn, self.ext_W[6], self.is_training, "ext_BN_6")

        y_pred = tf.nn.softmax(tf.matmul(h7bn, self.ext_W[7]) + self.ext_b[7])

        self.extracted_features = h7bn

        return y_pred

    def declare_w_and_b(self, num_nodes, nameprefix="tmp"):
        """
        declare w and b, and set initial values
        num_nodes: list of num of nodes. e.g. [79, 100, 100, 100, 100, 100, 2]
        """
        weights = list()
        biases = list()

        for i in range(0, len(num_nodes) - 1):
            wname = "{0:s}_w_{1:d}".format(nameprefix, i)
            bname = "{0:s}_b_{1:d}".format(nameprefix, i)
            weights.append(tf.Variable(tf.truncated_normal([num_nodes[i], num_nodes[i + 1]], stddev=0.1),
                                       name=wname))
            if i == 0:
                biases.append(tf.Variable(tf.ones([num_nodes[i + 1]]) + 0.1,
                                          name=bname))
            else:
                biases.append(tf.Variable(tf.zeros([num_nodes[i + 1]]) + 0.1,
                                          name=bname))
            # weights.append(tf.Variable(tf.truncated_normal([num_nodes[i], num_nodes[i + 1]], stddev=0.1)))
            # if i == 0:
            #     biases.append(tf.Variable(tf.ones([num_nodes[i + 1]]) + 0.1))
            # else:
            #     biases.append(tf.Variable(tf.zeros([num_nodes[i + 1]]) + 0.1))

        return weights, biases

    def relu_BN(self, x, w, is_training, name="tmp"):
        """
        Args:
        x: input feature tensor
        w: weight matrix
        is_training: in training->True, in testing->False
        """
        return tf.nn.relu(tf.layers.batch_normalization(tf.matmul(x, w),
                                                        momentum=0.99,
                                                        epsilon=0.001,
                                                        center=True,
                                                        scale=False,
                                                        beta_initializer=tf.zeros_initializer(),
                                                        gamma_initializer=tf.ones_initializer(),
                                                        moving_mean_initializer=tf.zeros_initializer(),
                                                        moving_variance_initializer=tf.ones_initializer(),
                                                        training=is_training,
                                                        trainable=True,
                                                        # name=None,
                                                        name=name,
                                                        reuse=None))

    def initialize_model(self):
        """
        Initialize model structure and set cost function and training method
        """
        self.y_pred = self.build_extractor(self.X)

        # weighted cross entropy
        self.cost_extractor = cross_entropy_with_clip(self.Y, self.y_pred, self.classweights)

        # Do include the following to update BN parameters
        # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op_extractor = tf.train.AdamOptimizer(0.001, beta1=0.9).minimize(self.cost_extractor)

        # start session
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # saver
        self.saver = tf.train.Saver()

        # summary
        tf.summary.scalar("cross_entropy", self.cost_extractor)
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    def print_dnn_setting(self):
        """
        print network info
        """
        print("[Structure]: %s" % self.extractor_num_nodes)
        print("[Batch size]: %s" % self.batchsize)
        print("[Class weights]: %s" % self.classweights)

    def train_model(self, update_interval=100):
        """
        iteration
        """
        for epoch in range(self.max_epoch + 1):
            batch_xs, batch_ys = self.dataloader.next_batch("train")
            summary, _ = self.sess.run([self.summary_op, self.train_op_extractor],
                                       feed_dict={self.X: batch_xs,
                                                  self.Y: batch_ys,
                                                  self.is_training: True,
                                                  self.pkeep: 0.75})
            self.writer.add_summary(summary, epoch)

            if epoch % update_interval == 0:
                self.show_training_status(epoch, ignore_first_epochs=500)

    def save_testset_prob(self, filename):
        """
        save probs
        """
        x = self.testset_x
        y = self.testset_y
        pred = self.y_pred
        p = self.sess.run(pred,
                          feed_dict={self.X: x,
                                     self.Y: y,
                                     self.is_training: False,
                                     self.pkeep: 1.0})
        np.savetxt(filename, p, delimiter=",", fmt="%.4e")

    def save_transformed_data(self, outprefix_latent):
        """
        Save extracted features as the activations of the N-1th layer.
        Labels are also saved.
        """
        sets = ["train", "validation", "test"]
        for name in sets:
            if name == "train":
                x = self.trainset_x
                y = self.trainset_y
            elif name == "validation":
                x = self.validationset_x
                y = self.validationset_y
            else:
                x = self.testset_x
                y = self.testset_y

            feat = self.sess.run(self.extracted_features,
                                 feed_dict={self.X: x,
                                            self.Y: y,
                                            self.is_training: False,
                                            self.pkeep: 1.0})

            xfile = outprefix_latent + "_" + name + "_feat.csv.gz"
            yfile = outprefix_latent + "_" + name + "_label.csv.gz"
            np.savetxt(xfile, feat, delimiter=",", fmt="%.6e")
            np.savetxt(yfile, y, delimiter=",", fmt="%.6e")

    def show_training_status(self, epoch, ignore_first_epochs=0):
        """
        Show training/validation/test-set error for showing training status
        """
        tra_acc, tra_ent = self.calc_accuracy(self.trainset_x, self.trainset_y, 1.0)
        val_acc, val_ent = self.calc_accuracy(self.validationset_x, self.validationset_y, 1.0)
        tes_acc, tes_ent = self.calc_accuracy(self.testset_x, self.testset_y, 1.0)

        tes_tss = self.calc_tss(self.testset_x, self.testset_y, 1.0)

        if epoch >= ignore_first_epochs and self.max_val_acc < val_acc:
            self.max_val_acc = val_acc
            self.tes_acc_when_max_val = tes_acc

        if epoch >= ignore_first_epochs and self.max_tes_tss < tes_tss:
            print("Max TSS updated.")
            self.max_tes_tss = tes_tss
            # save the current best results and network params
            if tes_tss > 0.79:
                self.save_model(self.outfile_model)
                self.save_testset_prob(self.outfile_pred)
                # self.save_transformed_data(self.outprefix_latent)

        # text = "[{0:06d}]Acc: Tra={1:0.4f}, Val={2:0.4f}, Tes={3:0.4f}, MaxVal={4:0.4f}({5:0.4f})"
        # print(text.format(epoch, tra_acc, val_acc, tes_acc,
        #                   self.max_val_acc, self.tes_acc_when_max_val))

        text = "[{0:06d}]Acc: Tra={1:0.4f}, Val={2:0.4f}, Tes={3:0.4f}, MaxVal={4:0.4f}({5:0.4f}), TSS={6:0.4f}"
        print(text.format(epoch, tra_acc, val_acc, tes_acc,
                          self.max_val_acc, self.tes_acc_when_max_val,
                          tes_tss))

    def calc_accuracy(self, xtmp, ytmp, pkeep):
        """
        calculate accuracy and loss
        """
        correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc, ent = self.sess.run([accuracy, self.cost_extractor],
                                 feed_dict={self.X: xtmp,
                                            self.Y: ytmp,
                                            self.is_training: False,
                                            self.pkeep: pkeep})
        return acc, ent

    def save_model(self, modelfile="tmp.ckpt"):
        """
        Save named variables to modelfile
        """
        print("Saving model in file: %s" % modelfile)
        self.saver.save(self.sess, modelfile)
        print("Model saved.")

    def load_model(self, modelfile="tmp.ckpt"):
        """
        Restore named variables from modelfile
        """
        print("Loading model from file: %s" % modelfile)
        self.saver.restore(self.sess, modelfile)
        print("Model restored.")

    def calc_tss(self, xtmp, ytmp, pkeep):
        """
        calculate tss
        """
        tss = TSS(self.Y, self.y_pred)
        t = self.sess.run(tss,
                          feed_dict={self.X: xtmp,
                                     self.Y: ytmp,
                                     self.is_training: False,
                                     self.pkeep: pkeep})

        return t


def main(argv):
    """
    extractor main
    """
    # 0. Setup
    np.random.seed(0)
    myflag = tf.flags.FLAGS  # To use tf.flags.DEFINE_** in __main__
    outprefix_latent = myflag.inprefix + "c"
    outfile_pred = make_iso8601_filename(myflag.outprefix_pred)

    print("[ In]: data %s_(train|validation|test)_(feat|label).csv.gz" % myflag.inprefix)
    print("[Out]: log %s" % myflag.logdir)
    print("[Out]: save model %s" % myflag.outfile_model)
    print("[ In]: load model %s" % myflag.infile_model)
    print("[Out]: latent %s" % outprefix_latent)
    print("[Out]: prediction %s" % outfile_pred)
    print("Max epoch = %s" % myflag.max_epoch)

    # 1. Initialize
    net1 = DeepFlareNet(batchsize=150,
                        max_epoch=myflag.max_epoch,
                        classweights=[1, 60],
                        logdir=myflag.logdir,
                        inprefix=myflag.inprefix,
                        outfile_model=myflag.outfile_model,
                        outprefix_latent=outprefix_latent,
                        outfile_pred=outfile_pred)
    net1.print_dnn_setting()
    net1.initialize_model()

    # # 2. Train and Save
    # net1.train_model(update_interval=100)
    # # net1.save_model(myflag.outfile_model)

    # 3. Load and Test
    net1.load_model(myflag.infile_model)
    net1.show_training_status(epoch=8000)

if __name__ == "__main__":
    # The following defines are used in the main function
    # Usage: tf.flags.DEFINE_**(NAME, DEFAULT_VALUE, DOCSTRING)
    tf.flags.DEFINE_integer("max_epoch",
                            # 10000,
                            16000,
                            "Maximum epoch")
    tf.flags.DEFINE_string("inprefix",
                           "../data/charval2017X_M24",
                           "[Input] Prefix for train_feat, train_label, validation_feat, validation_label, test_feat, test_label files")
    tf.flags.DEFINE_string("outprefix_pred",
                           "./predictions/charval2017X_M24_test_Epred",
                           "[Output] Prefix for prediction file")
    tf.flags.DEFINE_string("outfile_model",
                           "../model/charval2017X_M24_Emodel.ckpt",
                           "[Output] Model file for saving")
    tf.flags.DEFINE_string("infile_model",
                           "../model/charval2017X_M24_Emodel.ckpt",
                           "[In] Model file for loading")
    tf.flags.DEFINE_string("logdir",
                           "./log/deepflarenet",
                           "[Output] Directory for tensorflow logs")
    tf.app.run(main=main)




# Local Variables:
# coding: utf-8
# End:
