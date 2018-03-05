import tensorflow as tf

from experiment import Model


class SentenceClassificationModel(Model):
    def __init__(self, config, config_global, logger):
        super(SentenceClassificationModel, self).__init__(config, config_global, logger)
        self.__summary = None

    def build_input(self, data, sess):
        self.input_sent = tf.placeholder(tf.float32, [None, data.embedding_size])
        self.input_label = tf.placeholder(tf.float32, [None, len(data.classes)])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def create_outputs(self, prediction):
        self.loss_individual = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_label, logits=prediction)

        reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(self.loss_individual) + reg

        self.predict = tf.nn.softmax(prediction)
        tf.summary.scalar('Loss', self.loss)

    @property
    def summary(self):
        if self.__summary is None:
            self.__summary = tf.summary.merge_all(key='summaries')
        return self.__summary
