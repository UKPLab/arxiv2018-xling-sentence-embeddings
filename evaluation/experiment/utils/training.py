from __future__ import division

import os
import shutil
#import tempfile
from backports import tempfile

import numpy as np
import tensorflow as tf
import experiment


class DefaultTraining(experiment.Training):
    def __init__(self, config, config_global, logger):
        super(DefaultTraining, self).__init__(config, config_global, logger)

        self.n_epochs = self.config['epochs']
        self.batchsize = self.config['batchsize']
        self.dropout_keep_prob = 1.0 - self.config.get('dropout', 0.0)

        # checkpointing and weight restoring
        self.save_folder = self.config.get('save_folder')
        if not self.save_folder:
            self.save_folder = tempfile.TemporaryDirectory().name

        self.run_recorded_epochs = 0  # number of recorded epochs in this run
        self.state = TrainState(self.save_folder, less_is_better=False, logger=self.logger)
        self.early_stopping_patience = self.config.get('early_stopping_patience', self.n_epochs)

    def remove_checkpoints(self):
        self.state.clear()

    def start(self, model, data, sess, evaluation):
        return 0, 0

    def record_epoch(self, sess, score):
        self.run_recorded_epochs += 1
        previous_score = self.state.best_score
        self.state.record(sess, score)

        if previous_score != self.state.best_score:
            self.logger.debug('Validation score improved from {:.6f} to {:.6f}'.format(
                previous_score, self.state.best_score
            ))
        else:
            self.logger.debug('Validation score did not improve ({:.6f}; best={:.6f})'.format(
                score, self.state.best_score
            ))

    def restore_best_weights(self, sess):
        self.state.load(sess, weights='best')

    def is_early_stopping(self):
        return self.state.recorded_epochs - self.state.best_epoch >= self.early_stopping_patience


class BatchedTraining(DefaultTraining):
    """This is a simple training method that runs over the training data in a linear fashion, just like in keras."""

    def __init__(self, config, config_global, logger):
        super(BatchedTraining, self).__init__(config, config_global, logger)
        self.initial_learning_rate = self.config.get('initial_learning_rate', 1.1)
        self.dynamic_learning_rate = self.config.get('dynamic_learning_rate', False)

        self.epoch_learning_rate = self.initial_learning_rate

        self.batch_i = 0

    def get_feed_dict(self, model, data, sess):
        raise NotImplementedError()

    def prepare_next_epoch(self, model, data, sess, epoch):
        """Prepares the next epoch, especially the batches"""
        raise NotImplementedError()

    def start(self, model, data, sess, evaluation):
        super(BatchedTraining, self).start(model, data, sess, evaluation)

        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer_name = self.config.get('optimizer', 'sgd')
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        else:
            raise Exception('No such optimizer: {}'.format(optimizer_name))

        train = optimizer.minimize(model.loss)

        self.logger.debug('Initializing all variables')
        sess.run(tf.global_variables_initializer())

        self.state.load(sess, weights='last')
        start_epoch = self.state.recorded_epochs + 1
        end_epoch = self.n_epochs + 1

        if self.state.recorded_epochs > 0:
            self.logger.debug('Loaded the weights of last epoch {} with score={}'.format(
                self.state.recorded_epochs, self.state.scores[-1]
            ))
            if not self.is_early_stopping() and start_epoch < end_epoch:
                self.logger.debug('Now calculating validation score (to verify the restoring success)')
                valid_score = evaluation.start(model, data, sess, valid_only=True)['valid']
                self.logger.debug('Score={:.4f}'.format(valid_score))

        self.logger.debug('Running from epoch {} to epoch {}'.format(start_epoch, end_epoch - 1))

        for epoch in range(start_epoch, end_epoch):
            if self.is_early_stopping():
                self.logger.debug('Early stopping (no improvement in the last {} epochs)'.format(
                    self.state.recorded_epochs - self.state.best_epoch
                ))
                break

            self.logger.debug('Epoch {}/{}'.format(epoch, self.n_epochs))

            self.logger.debug('Preparing epoch')
            self.prepare_next_epoch(model, data, sess, epoch)

            train_losses = []  # used to calculate the epoch train loss
            recent_train_losses = []  # used to calculate the display loss

            self.logger.debug('Training')
            for _ in range(self.get_n_batches()):
                feed_dict = self.get_feed_dict(model, data, sess)
                feed_dict[learning_rate] = self.epoch_learning_rate
                _, loss, loss_individual = sess.run(
                    [train, model.loss, model.loss_individual],
                    feed_dict=feed_dict
                )
                recent_train_losses = ([loss] + recent_train_losses)[:20]
                train_losses.append(loss)
            self.logger.debug('train loss={:.6f}'.format(np.mean(train_losses)))

            self.logger.debug('Now calculating validation score')
            valid_score = evaluation.start(model, data, sess, valid_only=True)['valid']

            # if the validation score is better than the best observed previous loss, create a checkpoint
            self.record_epoch(sess, valid_score)

        if self.state.best_epoch < self.state.recorded_epochs:
            self.logger.debug('Restoring the weights of the best epoch {} with score {}'.format(
                self.state.best_epoch, self.state.best_score
            ))
            self.restore_best_weights(sess)

            self.logger.debug('Now calculating validation score (to verify the restoring success)')
            valid_score = evaluation.start(model, data, sess, valid_only=True)['valid']
            self.logger.debug('Score={:.4f}'.format(valid_score))

        self.remove_checkpoints()
        return self.state.best_epoch, self.state.best_score

    def get_n_batches(self):
        raise NotImplementedError()


class TrainState(object):
    def __init__(self, path, less_is_better, logger):
        """Represents the a training state

        :param path: the folder where the checkpoints should be written to
        :param less_is_better: True if a smaller validation score is desired
        """
        self.path = path
        self.logger = logger
        self.less_is_better = less_is_better
        self._saver = None

        self.initialize()

    def initialize(self):
        self.scores = []
        self.best_score = -1 if not self.less_is_better else 2
        self.best_epoch = 0
        self.recorded_epochs = 0
        if self.path and not os.path.exists(self.path):
            os.mkdir(self.path)

    def load(self, session, weights='last'):
        """

        :param session:
        :param weights: 'last' or 'best'
        :return:
        """
        if os.path.exists(self.scores_file):
            scores = []
            with open(self.scores_file, 'r') as f:
                for line in f:
                    scores.append(float(line))

            self.scores = scores
            op = max if not self.less_is_better else min
            self.best_score = op(scores)
            self.best_epoch = scores.index(self.best_score) + 1
            self.recorded_epochs = len(scores)

        restore_path = '{}-{}'.format(
            self.checkpoint_file,
            self.recorded_epochs if weights == 'last' else self.best_epoch
        )
        if os.path.exists(restore_path) or os.path.exists('{}.index'.format(restore_path)):
            self.saver.restore(session, restore_path)
        else:
            self.logger.debug('Could not restore weights. Path does not exist: {}'.format(restore_path))

    def record(self, session, score):
        self.recorded_epochs += 1
        self.scores.append(score)
        with open(self.scores_file, 'a') as f:
            f.write('{}\n'.format(score))
        self.saver.save(session, self.checkpoint_file, global_step=self.recorded_epochs)

        if (not self.less_is_better and score > self.best_score) or (self.less_is_better and score < self.best_score):
            self.best_score = score
            self.best_epoch = self.recorded_epochs

    def clear(self):
        shutil.rmtree(self.path)
        self._saver = None
        self.initialize()

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=None)
        return self._saver

    @property
    def scores_file(self):
        return os.path.join(self.path, 'scores.txt')

    @property
    def checkpoint_file(self):
        return os.path.join(self.path, 'model-checkpoint')
