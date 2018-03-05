import tensorflow as tf

from experiment.mapping.model.transform_both import FixSpaceModel
from experiment.utils.variables import weight_variable, bias_variable


class FixSpaceTranslationOnlyModel(FixSpaceModel):
    def __init__(self, config, config_global, logger):
        super(FixSpaceTranslationOnlyModel, self).__init__(config, config_global, logger)

    def build(self, data, sess):
        self.build_input(data, sess)

        W1 = weight_variable('W1', [data.embedding_size, data.embedding_size])
        b1 = bias_variable('b1', [data.embedding_size])

        source_rep = self.input_source
        translation_rep = tf.nn.tanh(tf.nn.xw_plus_b(self.input_translation, W1, b1))
        random_other_rep = tf.nn.tanh(tf.nn.xw_plus_b(self.input_random_other, W1, b1))

        self.create_outputs(source_rep, translation_rep, random_other_rep)


component = FixSpaceTranslationOnlyModel
