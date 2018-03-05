import importlib

import click
import tensorflow as tf

from experiment.config import load_config
from experiment.utils import logging_utils


@click.command()
@click.argument('config_file')
def run(config_file):
    """This program is the starting point for every experiment. It pulls together the configuration and all necessary
    experiment classes to load

    """
    config = load_config(config_file)
    config_global = config['global']
    logger = logging_utils.setup(config)

    # Allow the gpu to be used in parallel
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        # We are now fetching all relevant modules. It is strictly required that these modules contain a variable named
        # 'component' that points to a class which inherits from experiment.Data, experiment.Experiment,
        # experiment.Trainer or experiment.Evaluator
        data_module = config['data-module']
        model_module = config['model-module']
        training_module = config['training-module']
        evaluation_module = config.get('evaluation-module', None)

        # The modules are now dynamically loaded
        DataClass = importlib.import_module(data_module).component
        ModelClass = importlib.import_module(model_module).component
        TrainingClass = importlib.import_module(training_module).component
        EvaluationClass = importlib.import_module(evaluation_module).component

        # We then wire together all the modules and start training
        data = DataClass(config['data'], config_global, logger)
        model = ModelClass(config['model'], config_global, logger)
        training = TrainingClass(config['training'], config_global, logger)
        evaluation = EvaluationClass(config['evaluation'], config_global, logger)

        # setup the data (validate, create generators, load data, or else)
        logger.info('Setting up the data')
        data.setup()
        # build the model (e.g. compile it)
        logger.info('Building the model')
        model.build(data, sess)
        # start the training process
        logger.info('Starting the training process')
        training.start(model, data, sess, evaluation)

        # perform evaluation, if required
        if not config['evaluation'].get('skip', False):
            logger.info('Evaluating')
            evaluation.start(model, data, sess, valid_only=False)
        else:
            logger.info('Skipping evaluation')

        logger.info('DONE')


if __name__ == '__main__':
    run()
