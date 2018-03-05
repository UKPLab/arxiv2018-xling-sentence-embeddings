import importlib
import json
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
import xlsxwriter
from backports.tempfile import TemporaryDirectory
from shutil import rmtree

from experiment.config import load_config, read_config
from experiment.utils import logging_utils


class DataWriter(object):
    def __init__(self, path, logger):
        self.path = path
        self.logger = logger
        self.workbook = None
        self.worksheet = None
        self.row = None
        self.titles = None

        self.title_format = None

    def add_task(self, task_name):
        if self.workbook is not None:
            self.workbook.close()

        self.workbook = xlsxwriter.Workbook('{}-{}.xlsx'.format(
            self.path.replace('.xlsx', ''),
            task_name
        ))
        self.title_format = self.workbook.add_format({'bold': True})
        self.worksheet = self.workbook.add_worksheet(task_name)

        self.row = 0
        self.titles = []

    def add_data_all(self, data_all):
        all_items = set()
        for run_name, run_results in data_all:
            all_items |= set(run_results.keys())
        all_items = list(all_items)

        for run_name, run_results in data_all:
            for it in all_items:
                if it not in run_results:
                    run_results[it] = [0.0] * len(list(run_results.values())[0])
            self.add_data(run_name, run_results)

    def add_data(self, run_name, data):
        row_data = OrderedDict()
        for key, values in data.items():
            for i, value in enumerate(values, start=1):
                row_data['{} ({})'.format(key, i)] = value
        n_hidden_cols = len(row_data)
        for key, values in data.items():
            row_data['{} mean'.format(key)] = np.mean(values)
            row_data['{} std'.format(key)] = np.std(values)

        if self.row == 0:
            self.titles = ['Run'] + list(row_data.keys())
            self.worksheet.write_row(0, 0, self.titles, cell_format=self.title_format)
            self.worksheet.set_column(1, n_hidden_cols, width=20, options={'hidden': 1})
            self.worksheet.set_column(n_hidden_cols + 1, len(self.titles), width=20)
            self.row = 1

        self.worksheet.write_string(self.row, 0, run_name, cell_format=self.title_format)
        self.worksheet.set_column(self.row, 0, width=40)
        for key, value in row_data.items():
            try:
                col = self.titles.index(key)
                self.worksheet.write_number(self.row, col, value)
            except ValueError:
                self.logger.error('No comparable values of previous runs for key {}'.format(key))

        self.row += 1

    def finish(self):
        self.workbook.close()


@click.command()
@click.argument('config_file')
def run(config_file):
    """This program is the starting point for every experiment. It pulls together the configuration and all necessary
    experiment classes to load

    """
    config = load_config(config_file)
    logger = logging_utils.setup(config)
    data_writer = DataWriter(config['report'], logger)
    try:
        for task in config['tasks']:
            task_name = task['name']
            repetitions = task['repetitions']
            with open(task['config'], 'r') as f:
                task_config = f.read()
            data_writer.add_task(task_name)
            logger.info('Task {}'.format(task_name))

            results = []
            for run in task['runs']:
                run_config = read_config(task_config.format(**run))
                run_config_global = run_config['global']
                run_name = run_config.get('name', json.dumps(run))
                logger.info('Run {}'.format(run_name))

                data_module = run_config['data-module']
                model_module = run_config['model-module']
                training_module = run_config['training-module']
                evaluation_module = run_config.get('evaluation-module')
                DataClass = importlib.import_module(data_module).component
                ModelClass = importlib.import_module(model_module).component
                TrainingClass = importlib.import_module(training_module).component
                EvaluationClass = importlib.import_module(evaluation_module).component

                data = DataClass(run_config['data'], run_config_global, logger)
                logger.debug('Setting up the data')
                data.setup()

                best_lr = 0
                best_dev_score = 0
                best_dev_results = []
                for lr in [0.05, 0.01, 0.005, 0.001, 0.0005]:
                    train_conf = dict(run_config['training'])
                    train_conf['initial_learning_rate'] = lr

                    run_results = OrderedDict()
                    for i in range(repetitions):
                        logger.info('Repetition {}'.format(i))
                        data.reshuffle(i)  # e.g. random subsample

                        sess_config = tf.ConfigProto()
                        sess_config.gpu_options.allow_growth = True

                        tf.reset_default_graph()
                        with TemporaryDirectory() as tmp_dir:
                            train_conf['save_folder'] = tmp_dir
                            with tf.Session(config=sess_config) as sess:
                                model = ModelClass(run_config['model'], run_config_global, logger)
                                training = TrainingClass(train_conf, run_config_global, logger)
                                evaluation = EvaluationClass(run_config['evaluation'], run_config_global, logger)

                                # build the model (e.g. compile it)
                                logger.debug('Building the model')
                                model.build(data, sess)
                                # start the training process
                                logger.debug('Starting the training process')
                                training.start(model, data, sess, evaluation)
                                logger.debug('Evaluating')
                                result = evaluation.start(model, data, sess, valid_only=False)
                                logger.info('Got results for {}: {}'.format(run_name, json.dumps(result)))

                                for key, value in result.items():
                                    if key not in run_results:
                                        run_results[key] = []
                                    run_results[key].append(value)

                    aggregated_dev = np.mean(run_results['valid'])
                    if aggregated_dev > best_dev_score:
                        best_dev_score = aggregated_dev
                        best_dev_results = run_results
                        best_lr = lr

                logger.info('best result {} for run with learning rate {}'.format(best_dev_score, best_lr))
                results.append((run_name, best_dev_results))

            data_writer.add_data_all(results)
    finally:
        data_writer.finish()

    logger.info('DONE')


if __name__ == '__main__':
    run()
