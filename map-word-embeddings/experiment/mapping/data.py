import gzip
import os
from collections import defaultdict

import numpy as np

import experiment


class FixSpaceData(experiment.Data):
    def __init__(self, config, config_global, logger):
        super(FixSpaceData, self).__init__(config, config_global, logger)
        self.train_path = config['train_path']
        self.valid_path = config.get('valid_path', None)
        self.test_path = config.get('test_path', None)
        self.valid_ratio = config['valid_ratio'] if self.valid_path is None else None
        self.test_ratio = config['test_ratio'] if self.test_path is None else None

        self.vector_space = config.get('vector_space', None)  # first one found
        self.lang_a = config['lang_a']
        self.lang_b = config['lang_b']
        self.max_load = config.get('max_load', None)

        self.train_data, self.valid_data, self.test_data = None, None, None
        self.classes = []
        self.embedding_size = None

    def setup(self):
        full_train_data = self.load_data(self.train_path)
        full_train_data_size = len(next(iter(full_train_data.values())))

        if self.valid_path is None:
            valid_start = int((1 - self.valid_ratio - self.test_ratio) * full_train_data_size)
            test_start = int((1 - self.test_ratio) * full_train_data_size)
            self.train_data = list(zip(full_train_data[self.lang_a][:valid_start], full_train_data[self.lang_b][:valid_start]))
            self.valid_data = list(
                zip(full_train_data[self.lang_a][valid_start:test_start], full_train_data[self.lang_b][valid_start:test_start]))
            self.test_data = list(zip(full_train_data[self.lang_a][test_start:], full_train_data[self.lang_b][test_start:]))
        else:
            self.train_data = list(zip(full_train_data[self.lang_a], full_train_data[self.lang_b]))
            full_valid_data = self.load_data(self.valid_path)
            self.valid_data = list(zip(full_valid_data[self.lang_a], full_valid_data[self.lang_b]))
            full_test_data = self.load_data(self.test_path)
            self.test_data = list(zip(full_test_data[self.lang_a], full_test_data[self.lang_b]))

        self.embedding_size = len(self.train_data[0][0])

        self.logger.debug('Embedding Size: {}'.format(self.embedding_size))
        self.logger.debug('Training Examples: {}'.format(len(self.train_data)))
        self.logger.debug('Validation Items: {}'.format(len(self.valid_data)))
        self.logger.debug('Test Items: {}'.format(len(self.test_data)))

    def load_data(self, data_path):
        """
        :return: dict[language, list[embedding]]
        """
        data = defaultdict(lambda: list())

        filenames = [f for f in os.listdir(data_path) if not f.startswith('.')]

        for file in filenames:
            file_no_gz = file[:-3] if file.endswith('.gz') else file
            split = file_no_gz.split('.')
            language = split[-2]
            vector_space = split[-1]

            if self.vector_space is None:
                self.vector_space = vector_space

            if vector_space == self.vector_space:
                file_path = os.path.join(data_path, file)
                if not file.endswith('.gz'):
                    f = open(file_path, 'r')
                else:
                    f = gzip.open(file_path, 'r')
                try:
                    for line in f:
                        if len(data[language]) >= self.max_load:
                            break
                        line = line.strip().split()
                        if line:
                            embedding = np.asarray([float(s) for s in line], np.float32)
                            data[language].append(embedding)
                finally:
                    f.close()
        return data


component = FixSpaceData
