import tensorflow as tf

from experiment import Model
from experiment.utils.variables import weight_variable, bias_variable


class FixSpaceModel(Model):
    def __init__(self, config, config_global, logger):
        super(FixSpaceModel, self).__init__(config, config_global, logger)

    def build(self, data, sess):
        self.build_input(data, sess)

        W1 = weight_variable('W1', [data.embedding_size, data.embedding_size])
        b1 = bias_variable('b1', [data.embedding_size])

        source_rep = tf.nn.tanh(tf.nn.xw_plus_b(self.input_source, W1, b1))
        translation_rep = tf.nn.tanh(tf.nn.xw_plus_b(self.input_translation, W1, b1))
        random_other_rep = tf.nn.tanh(tf.nn.xw_plus_b(self.input_random_other, W1, b1))

        self.create_outputs(source_rep, translation_rep, random_other_rep)

    def build_input(self, data, sess):
        self.input_source = tf.placeholder(tf.float32, [None, data.embedding_size])
        self.input_translation = tf.placeholder(tf.float32, [None, data.embedding_size])
        self.input_random_other = tf.placeholder(tf.float32, [None, data.embedding_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def create_outputs(self, source, translation, other):
        # We apply dropout before similarity. This only works when we dropout the same indices in question and answer.
        # Otherwise, the similarity would be heavily biased (in case of angular/cosine distance).
        dropout_multiplicators = tf.nn.dropout(source * 0.0 + 1.0, self.dropout_keep_prob)

        source_dropout = source * dropout_multiplicators
        translation_dropout = translation * dropout_multiplicators
        other_dropout = other * dropout_multiplicators

        self.similarity_source_translation = cosine_similarity(
            source_dropout,
            translation_dropout,
        )
        self.similarity_source_other = cosine_similarity(
            source_dropout,
            other_dropout,
        )

        self.loss_individual = hinge_loss(
            self.similarity_source_translation,
            self.similarity_source_other,
            self.config['margin']
        )

        self.loss = tf.reduce_mean(self.loss_individual)

        self.source_transformed = source
        self.translation_transformed = translation
        self.predict = self.similarity_source_translation
        self.predict_orig = cosine_similarity(
            self.input_source,
            self.input_translation,
        )


def cosine_similarity(a, b):
    with tf.device('/cpu:0'):
        normalize_a = tf.nn.l2_normalize(a, 1)
        normalize_b = tf.nn.l2_normalize(b, 1)

        # Reduce sum is not deterministic on GPU!
        # https://github.com/tensorflow/tensorflow/issues/3103
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)

        return cos_similarity


def hinge_loss(similarity_good_tensor, similarity_bad_tensor, margin):
    return tf.maximum(
        0.0,
        tf.add(
            tf.subtract(
                margin,
                similarity_good_tensor
            ),
            similarity_bad_tensor
        )
    )


component = FixSpaceModel
