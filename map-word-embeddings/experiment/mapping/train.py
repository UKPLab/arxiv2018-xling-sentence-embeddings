import math

import numpy as np

from experiment.utils.training import BatchedTraining


class FixSpaceTraining(BatchedTraining):
    def __init__(self, config, config_global, logger):
        super(FixSpaceTraining, self).__init__(config, config_global, logger)
        self.n_batches = None
        self.batch_i = 0
        self.epoch_shuffle_indices = None

    def get_feed_dict(self, model, data, sess):
        source, translation, random_other = self.get_next_batch(model, data, sess)
        return {
            model.input_source: source,
            model.input_translation: translation,
            model.input_random_other: random_other,
            model.dropout_keep_prob: self.dropout_keep_prob
        }

    def prepare_next_epoch(self, model, data, sess, epoch):
        self.epoch_learning_rate = self.initial_learning_rate
        if self.dynamic_learning_rate:
            self.epoch_learning_rate /= float(epoch)

        if self.n_batches is None:
            self.n_batches = int(math.ceil(len(data.train_data) / float(self.batchsize)))

        self.epoch_shuffle_indices = np.random.permutation(len(data.train_data))
        self.epoch_shuffle_indices_other = np.random.permutation(len(data.train_data))
        self.batch_i = 0

    def get_n_batches(self):
        return self.n_batches

    def get_next_batch(self, model, data, sess):
        """Return the training data for the next batch

        :return: sentences, sentence lengths, labels
        :rtype: list, list, list
        """
        indices = self.epoch_shuffle_indices[self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]
        indices_other = self.epoch_shuffle_indices_other[
                        self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]

        batch_data = [data.train_data[i] for i in indices]
        batch_sents_source = [d[0] for d in batch_data]
        batch_sents_translation = [d[1] for d in batch_data]
        batch_sents_other = [data.train_data[i][1] for i in indices_other]

        self.batch_i += 1
        return batch_sents_source, batch_sents_translation, batch_sents_other


component = FixSpaceTraining
