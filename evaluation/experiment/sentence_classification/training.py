import math

import numpy as np

from experiment.utils.training import BatchedTraining


class SentenceClassificationBatchedTraining(BatchedTraining):
    def __init__(self, config, config_global, logger):
        super(SentenceClassificationBatchedTraining, self).__init__(config, config_global, logger)
        self.n_batches = None
        self.data = None
        self.batch_i = 0
        self.epoch_shuffle_indices = None

    def get_feed_dict(self, model, data, sess):
        batch_sents, batch_labels = self.get_next_batch(model, data, sess)
        return {
            model.input_sent: batch_sents,
            model.input_label: batch_labels,
            model.dropout_keep_prob: self.dropout_keep_prob
        }

    def prepare_next_epoch(self, model, data, sess, epoch):
        self.epoch_learning_rate = self.initial_learning_rate
        if self.dynamic_learning_rate:
            self.epoch_learning_rate /= float(epoch)

        self.n_batches = int(math.ceil(len(data.train) / float(self.batchsize)))
        if self.data is None:
            self.data = data.train

        self.epoch_shuffle_indices = np.random.permutation(len(self.data))
        self.batch_i = 0

    def get_n_batches(self):
        return self.n_batches

    def get_next_batch(self, model, data, sess):
        """Return the training data for the next batch

        :return: questions, good answers, bad answers
        :rtype: list, list, list
        """
        indices = self.epoch_shuffle_indices[self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]
        batch_data = [self.data[i] for i in indices]
        self.batch_i += 1

        # transpose of zip(batch_data)
        return zip(*batch_data)


component = SentenceClassificationBatchedTraining
