class ComponentBase(object):
    def __init__(self, config, config_global, logger):
        """This is a simple base object for all experiment components

        :type config: dict
        :type config_global: dict
        :type logger: logging.Logger
        """
        self.config = config or dict()
        self.config_global = config_global or dict()
        self.logger = logger


class Data(ComponentBase):
    def setup(self):
        pass

    def reshuffle(self, repetition):
        """Will be called when we run multiple experiments to reshuffle data. e.g. for random subsample validation"""
        pass


class Model(ComponentBase):
    def __init__(self, config, config_global, logger):
        super(Model, self).__init__(config, config_global, logger)

    def build(self, data, sess):
        raise NotImplementedError()


class Training(ComponentBase):
    def start(self, model, data, sess, evaluation):
        """

        :param model:
        :type model: Model
        :param data:
        :type data: Data
        :type evaluation: Evaluation
        """
        raise NotImplementedError()


class Evaluation(ComponentBase):
    def start(self, model, data, sess, valid_only=False):
        """

        :type model: Model
        :type data: Data
        :type sess: tensorflow.Session
        :type valid_only: bool
        :return: scores for all tested runs (data split, language combinations, ...)
        :rtype: OrderedDict[basestring, float]
        """
        raise NotImplementedError()
