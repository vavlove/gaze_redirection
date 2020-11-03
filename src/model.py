# Model for Training & testing

from __future__ import division

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib.pyplot import imsave

from src.archs import *
from src.data_loader import ImageData
from utils.ops import l1_loss, content_loss, style_loss, angular_error


class Model(object):
    """
    Main model.
    @author: Zhe He
    @contact: zhehe@student.ethz.ch
    """
    def __init__(self, params):
        """init

        Parameters
        ----------
        params: dict.
        """

        self.params = params
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.lr = tf.Variable(0, dtype=tf.float32, trainable=False, name='learning_rate')

        (self.train_data, self.valid_data, self.test_data) = self.data_loader()

        # building graph
        # (self.x_r, self.angles_r, self.labels, self.x_t,
         # self.angles_g) = self.train_iter.get_next()

        self.x_r = keras.Input(shape=(self.params.image_size, self.params.image_size, 3), name="x_r")
        self.x_t = keras.Input(shape=(self.params.image_size, self.params.image_size, 3), name="x_t")
        self.angles_r = keras.Input(shape=(2), name="angles_r")
        self.angles_g = keras.Input(shape=(2), name="angles_g")
        self.labels = keras.Input(shape=(1), name="labels")

        # (self.x_valid_r, self.angles_valid_r, self.labels_valid,
        #  self.x_valid_t, self.angles_valid_g) = self.valid_iter.get_next()

        # (self.x_test_r, self.angles_test_r, self.labels_test,
        #  self.x_test_t, self.angles_test_g) = self.test_iter.get_next()

        self.x_g = generator(self.x_r, self.angles_g)
        self.x_recon = generator(self.x_g, self.angles_r, reuse=True)

        # self.angles_valid_g = tf.random_uniform([params.batch_size, 2], minval=-1.0, maxval=1.0)

        # self.x_valid_g = generator(self.x_valid_r, self.angles_valid_g,
        #                            reuse=True)

        # reconstruction loss
        self.recon_loss = l1_loss(self.x_r, self.x_recon)

        # content loss and style loss
        # self.c_loss, self.s_loss = self.feat_loss()

        # regression losses and adversarial losses
        (self.d_loss, self.g_loss, self.reg_d_loss,
         self.reg_g_loss, self.gp) = self.adv_loss()

        # update operations for generator and discriminator
        self.d_op, self.g_op = self.add_optimizer()

        # adding summaries
        self.summary = self.add_summary()

        # initialization operation
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

    def data_loader(self):
        """ load traing and testing dataset """

        hps = self.params

        image_data_class = ImageData(load_size=hps.image_size,
                                     channels=3,
                                     data_path=hps.data_path,
                                     ids=hps.ids)
        image_data_class.preprocess()

        train_dataset_num = len(image_data_class.train_images)
        test_dataset_num = len(image_data_class.test_images)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (image_data_class.train_images,
             image_data_class.train_angles_r,
             image_data_class.train_labels,
             image_data_class.train_images_t,
             image_data_class.train_angles_g))
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (image_data_class.test_images,
             image_data_class.test_angles_r,
             image_data_class.test_labels,
             image_data_class.test_images_t,
             image_data_class.test_angles_g))

        train_dataset = train_dataset.shuffle(8).repeat(train_dataset_num).map(image_data_class.image_processing).batch(hps.batch_size)
        valid_dataset = test_dataset.shuffle(8).repeat(train_dataset_num).map(image_data_class.image_processing).batch(hps.batch_size)
        test_dataset = test_dataset.shuffle(8).repeat(train_dataset_num).map(image_data_class.image_processing).batch(hps.batch_size)

        return (train_dataset, valid_dataset, test_dataset)

    def adv_loss(self):
        """Build sub graph for discriminator and gaze estimator

        Returns
        -------
        d_loss: scalar, adversarial loss for training discriminator.
        g_loss: scalar, adversarial loss for training generator.
        reg_loss_d: scalar, MSE loss for training gaze estimator
        reg_loss_g: scalar, MSE loss for training generator
        gp: scalar, gradient penalty
        """

        hps = self.params

        gan_real, reg_real = discriminator(hps, self.x_r)
        gan_fake, reg_fake = discriminator(hps, self.x_g, reuse=True)

        self.d_real = gan_real

        eps = tf.random.uniform(shape=[hps.batch_size, 1, 1, 1], minval=0.,maxval=1.)


        with tf.GradientTape() as g:
            g.watch(eps)
            interpolated = eps * self.x_r + (1. - eps) * self.x_g
            gan_inter, _ = discriminator(hps, interpolated, reuse=True)

        grad = g.gradient(gan_inter, interpolated)

        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean(tf.square(slopes - 1.))

        d_loss = (-tf.reduce_mean(gan_real) +
                  tf.reduce_mean(gan_fake) + 10. * gp)
        g_loss = -tf.reduce_mean(gan_fake)

        reg_loss_d = tf.losses.mean_squared_error(self.angles_r, reg_real)
        reg_loss_g = tf.losses.mean_squared_error(self.angles_g, reg_fake)

        return d_loss, g_loss, reg_loss_d, reg_loss_g, gp

    # def feat_loss(self):
    #     """
    #     build the sub graph of perceptual matching network

    #     Returns
    #     -------
    #     c_loss: scalar, content loss
    #     s_loss: scalar, style loss
    #     """

    #     content_layers = ["vgg_16/conv5/conv5_3"]
    #     style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
    #                     "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]

    #     _, endpoints_mixed = vgg_16(
    #         tf.concat([self.x_g, self.x_t], 0))

    #     c_loss = content_loss(endpoints_mixed, content_layers)
    #     s_loss = style_loss(endpoints_mixed, style_layers)

    #     return c_loss, s_loss

    def optimizer(self, lr):
        """Return an optimizer

        Parameters
        ----------
        lr: learning rate.

        Returns
        -------
        tensorflow Optimizer instance.
        """

        hps = self.params

        if hps.optimizer == 'sgd':
            return tf.keras.optimizers.SGD(lr)
        if hps.optimizer == 'adam':
            return tf.keras.optimizers.Adam(lr,
                                          beta_1=hps.adam_beta1,
                                          beta_2=hps.adam_beta2)
        raise AttributeError("attribute 'optimizer' is not assigned!")

    def add_optimizer(self):
        """Add an optimizer.

        Returns
        -------
        g_op: update operation for generator.
        d_op: update operation for discriminator.
        """

        ### Pass in the models/layers here?
        # g_vars = tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        # d_vars = tf.get_collection(
        #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        print(self.x_g)
        g_vars = self.x_g.trainable_variables
        d_vars = self.d_real.trainable_variables

        g_opt = self.optimizer(self.lr)
        d_opt = self.optimizer(self.lr)

        g_loss = (self.g_loss + 5.0 * self.reg_g_loss +
                  50.0 * self.recon_loss)
                  # 100.0 * self.s_loss +  100.0 * self.c_loss)### TODO Perceptual loss 
                 
        d_loss = self.d_loss + 5.0 * self.reg_d_loss

        g_op = g_opt.minimize(loss=g_loss, var_list=g_vars)
        d_op = d_opt.minimize(loss=d_loss, var_list=d_vars)

        return d_op, g_op

    def add_summary(self):
        """Add summary operation.

        Return
        ------
        summary_op: tf summary.
        """

        tf.summary.scalar('recon_loss', self.recon_loss)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('reg_d_loss', self.reg_d_loss)
        tf.summary.scalar('reg_g_loss', self.reg_g_loss)
        tf.summary.scalar('gp', self.gp)
        tf.summary.scalar('lr', self.lr)
        tf.summary.scalar('c_loss', self.c_loss)
        tf.summary.scalar('s_loss', self.s_loss)

        tf.summary.image('real', (self.x_r + 1) / 2.0, max_outputs=5)
        tf.summary.image('fake', tf.clip_by_value(
            (self.x_g + 1) / 2.0, 0., 1.), max_outputs=5)
        tf.summary.image('recon', tf.clip_by_value(
            (self.x_recon + 1) / 2.0, 0., 1.), max_outputs=5)

        tf.summary.image('x_test', tf.clip_by_value(
            (self.x_valid_r + 1) / 2.0, 0., 1.), max_outputs=5)
        tf.summary.image('x_test_fake', tf.clip_by_value(
            (self.x_valid_g + 1) / 2.0, 0., 1.), max_outputs=5)

        summary_op = tf.summary.merge_all()

        return summary_op

    def train(self):
        """Train the model and save checkpoints.
        """

        hps = self.params

        num_epoch = hps.epochs
        batch_size = hps.batch_size
        learning_rate = hps.lr

        summary_dir = os.path.join(hps.log_dir, 'summary')
        model_path = os.path.join(hps.log_dir, 'model.ckpt')

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:

            # init
            sess.run([self.init_op])

            summary_writer = tf.summary.FileWriter(summary_dir,
                                                   graph=sess.graph)

            saver = tf.train.Saver(max_to_keep=3)

            variables_to_restore = slim.get_variables_to_restore(
                include=['vgg_16'])
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, hps.vgg_path)

            try:

                for epoch in range(num_epoch):

                    print("Epoch: %d" % epoch)

                    if epoch >= hps.epochs / 2:

                        learning_rate = (2. - 2. * epoch / hps.epochs) * hps.lr

                    for it in range(num_iter):

                        feed_d = {self.lr: learning_rate}

                        sess.run([self.d_op], feed_dict=feed_d)

                        if it % 5 == 0:
                            sess.run(self.g_op, feed_dict=feed_d)

                        if it % hps.summary_steps == 0:

                            summary, global_step = sess.run(
                                [self.summary, self.global_step],
                                feed_dict=feed_d)
                            summary_writer.add_summary(summary, global_step)
                            summary_writer.flush()
                            saver.save(sess, model_path,
                                       global_step=global_step)

            except KeyboardInterrupt:
                print("stop training")

    def eval(self):
        """ Evaluation. """
        hps = self.params

        checkpoint = tf.train.latest_checkpoint(hps.log_dir)

        x_fake = generator(self.x_test_r, self.angles_test_g, reuse=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        with tf.Session(config=tf_config) as test_sess:

            with test_sess.graph.as_default():
                saver.restore(test_sess, checkpoint)

                imgs_dir = os.path.join(hps.log_dir, 'eval')
                if not os.path.exists(imgs_dir):
                    os.mkdir(imgs_dir)

                tar_dir = os.path.join(imgs_dir, 'targets')
                gene_dir = os.path.join(imgs_dir, 'genes')
                real_dir = os.path.join(imgs_dir, 'reals')
                os.makedirs(tar_dir)
                os.makedirs(gene_dir)
                os.makedirs(real_dir)

                try:
                    i = 0
                    while True:
                        (real_imgs, target_imgs, fake_imgs,
                         a_r, a_t) = test_sess.run(
                            [self.x_test_r, self.x_test_t, x_fake,
                             self.angles_test_r, self.angles_test_g])
                        a_t = a_t * np.array([15, 10])
                        a_r = a_r * np.array([15, 10])
                        delta = angular_error(a_t, a_r)

                        for j in range(real_imgs.shape[0]):
                            imsave(os.path.join(
                                tar_dir,
                                '%d_%d_%.3f_H%d_V%d.jpg' % (
                                    i, j, delta[j], a_t[j][0],
                                    a_t[j][1])), target_imgs[j])
                            imsave(os.path.join(
                                gene_dir,
                                '%d_%d_%.3f_H%d_V%d.jpg' % (
                                    i, j, delta[j], a_t[j][0],
                                    a_t[j][1])), fake_imgs[j])
                            imsave(os.path.join(
                                real_dir,
                                '%d_%d_%.3f_H%d_V%d.jpg' % (
                                    i, j, delta[j], a_t[j][0],
                                    a_t[j][1])), real_imgs[j])

                        i = i + 1
                except tf.errors.OutOfRangeError:
                    logging.info("quanti_eval finished.")
