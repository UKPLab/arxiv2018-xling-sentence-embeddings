import tensorflow as tf

from experiment.mapping.model.transform_both import FixSpaceModel
from experiment.utils.variables import weight_variable, bias_variable


class FixSpaceSeparateTransformationModel(FixSpaceModel):
    def __init__(self, config, config_global, logger):
        super(FixSpaceSeparateTransformationModel, self).__init__(config, config_global, logger)

    def build(self, data, sess):
        self.build_input(data, sess)

        W1_src = weight_variable('W1_src', [data.embedding_size, data.embedding_size])
        b1_src = bias_variable('b1_src', [data.embedding_size])

        W1_target = weight_variable('W1_target', [data.embedding_size, data.embedding_size])
        b1_target = bias_variable('b1_target', [data.embedding_size])

        source_rep = tf.nn.tanh(tf.nn.xw_plus_b(self.input_source, W1_src, b1_src))
        translation_rep = tf.nn.tanh(tf.nn.xw_plus_b(self.input_translation, W1_target, b1_target))
        random_other_rep = tf.nn.tanh(tf.nn.xw_plus_b(self.input_random_other, W1_target, b1_target))

        self.create_outputs(source_rep, translation_rep, random_other_rep)


component = FixSpaceSeparateTransformationModel
