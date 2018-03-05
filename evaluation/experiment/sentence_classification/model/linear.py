import tensorflow as tf

from experiment.sentence_classification.model import SentenceClassificationModel
from experiment.utils.variables import weight_variable, bias_variable


class LinearModel(SentenceClassificationModel):
    def build(self, data, sess):
        self.build_input(data, sess)
        W1 = weight_variable('W1', [data.embedding_size, len(data.classes)])
        b1 = bias_variable('b1', [len(data.classes)])

        input_dropout = tf.nn.dropout(self.input_sent, self.dropout_keep_prob)
        prediction = tf.nn.xw_plus_b(input_dropout, W1, b1)

        self.create_outputs(prediction)


component = LinearModel
