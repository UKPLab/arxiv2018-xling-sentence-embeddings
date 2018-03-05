import tensorflow as tf

from experiment.sentence_classification.model import SentenceClassificationModel
from experiment.utils.variables import weight_variable, bias_variable


class FeedForwardModel(SentenceClassificationModel):
    def build(self, data, sess):
        self.build_input(data, sess)
        input_dropout = tf.nn.dropout(self.input_sent, self.dropout_keep_prob
                                      )
        W1 = weight_variable('W1', [data.embedding_size, data.embedding_size])
        b1 = bias_variable('b1', [data.embedding_size])
        layer_1 = tf.nn.relu(tf.nn.xw_plus_b(input_dropout, W1, b1))

        W2 = weight_variable('W2', [data.embedding_size, len(data.classes)])
        b2 = bias_variable('b2', [len(data.classes)])
        prediction = tf.nn.xw_plus_b(layer_1, W2, b2)
        self.create_outputs(prediction)


component = FeedForwardModel
