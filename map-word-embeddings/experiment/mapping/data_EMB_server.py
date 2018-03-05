import io
import json
from math import ceil

import numpy as np
import requests

import experiment


class FixSpaceData(experiment.Data):
    def __init__(self, config, config_global, logger):
        super(FixSpaceData, self).__init__(config, config_global, logger)
        self.embeddings_webserver_url = config['embeddings_webserver_url']
        self.word_embedding_alias = config['word_embedding_alias']

        self.train_path_a = config['train_path_a']
        self.train_path_b = config['train_path_b']
        self.valid_ratio = config['valid_ratio']
        self.test_ratio = config['test_ratio']

        self.max_load = config.get('max_load', None)

        self.train_data, self.valid_data, self.test_data = None, None, None
        self.classes = []
        self.embedding_size = None

    def setup(self):
        full_train_data_a = self.load_data(self.train_path_a)
        full_train_data_b = self.load_data(self.train_path_b)

        self.embedding_size = len(full_train_data_a[0])
        full_train_data_size = len(full_train_data_a)

        valid_start = int((1 - self.valid_ratio - self.test_ratio) * full_train_data_size)
        test_start = int((1 - self.test_ratio) * full_train_data_size)
        self.train_data = list(zip(full_train_data_a[:valid_start], full_train_data_b[:valid_start]))
        self.valid_data = list(
            zip(full_train_data_a[valid_start:test_start], full_train_data_b[valid_start:test_start])
        )
        self.test_data = list(zip(full_train_data_a[test_start:], full_train_data_b[test_start:]))

        self.logger.debug('Embedding Size: {}'.format(self.embedding_size))
        self.logger.debug('Training Examples: {}'.format(len(self.train_data)))
        self.logger.debug('Validation Items: {}'.format(len(self.valid_data)))
        self.logger.debug('Test Items: {}'.format(len(self.test_data)))

    def load_data(self, data_path):
        """
        :return: list[embedding]
        """
        sentences = []
        with io.open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(sentences) >= self.max_load:
                    break
                line = line.strip()
                if line:
                    sentences.append(line)

        embeddings = []
        for batch in range(int(ceil(len(sentences) / float(512)))):
            sents_batch = sentences[batch * 512: (batch + 1) * 512]
            config = {
                'sentences': sents_batch,
                'embedding_types': [[self.word_embedding_alias, ['mean']]]
                # 'embedding_types': [['bivcd_en_de', ['mean']]]
            }
            sentvec_strings = requests.post('{}/convert'.format(self.embeddings_webserver_url), data={
                'conversion': json.dumps(config)
            }).text

            for sentvec_string in sentvec_strings.split('\n')[:-1]:
                try:
                    embeddings.append(np.fromstring(sentvec_string.strip(), sep=' '))
                except:
                    print('ERROR: {}'.format(sentvec_string))

        return embeddings


component = FixSpaceData
