from collections import OrderedDict

import numpy as np

import experiment
from experiment.sentence_classification.data.generic import binarize


class GenericRandomSubsampleValidationSentenceClassificationData(experiment.Data):
    def __init__(self, config, config_global, logger):
        super(GenericRandomSubsampleValidationSentenceClassificationData, self).__init__(config, config_global, logger)

        self.dev_ratio = config['dev_ratio']
        self.test_ratio = config['test_ratio']
        self.train_language = config['train_language']

        self.evaluate_all_cross_language = self.config.get('evaluate-all-cross-language', False)
        self.logger.debug('Evaluating all cross language? {}'.format(self.evaluate_all_cross_language))

        self.classes = None
        self.raw_data = None
        # structure of raw data: dict[language, list[(embedding, label)]]
        self.len_raw_data = None
        self.len_dev = None
        self.len_test = None

    def setup(self):
        self.raw_data = OrderedDict()
        for embeddings_config in self.config['all_embeddings']:
            embeddings_path = embeddings_config['embeddings']
            language = embeddings_path.split('/')[-1].split('.')[-2]
            split = self.load_split(embeddings_path, embeddings_config['labels'])
            if language not in self.raw_data:
                self.raw_data[language] = split
            else:
                # append
                self.raw_data[language] += split

        self.len_raw_data = len(list(self.raw_data.values())[0])
        self.len_dev = int(self.len_raw_data * self.dev_ratio)
        self.len_test = int(self.len_raw_data * self.test_ratio)

        self.reshuffle(None)

        self.logger.debug('Train examples: {}'.format(len(self.train)))
        self.logger.debug('Valid examples: {}'.format(len(self.valid)))
        self.logger.debug('Test examples: {}'.format(len(list(self.test.values())[0])))

        self.embedding_size = len(self.train[0][0])

    def get_shuffles(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if self.config.get('per-language-shuffle', False):
            self.logger.info('Shuffling with per language shuffling')
            result = dict([(l, np.random.permutation(len(v))) for (l, v) in self.raw_data.items()])
        else:
            self.logger.debug('Shuffling for all languages at once')
            shuffle = np.random.permutation(self.len_raw_data)
            result = dict([(l, shuffle) for l in self.raw_data.keys()])
        return result

    def reshuffle(self, repetition):
        self.logger.debug('Reshuffling and selecting data for random subsample validation')

        split_shuffles = self.get_shuffles(repetition)

        dev_indices = split_shuffles[self.train_language][:self.len_dev]
        train_indices = split_shuffles[self.train_language][self.len_dev + self.len_test:]
        self.valid = [self.raw_data[self.train_language][i] for i in dev_indices]
        self.train = [self.raw_data[self.train_language][i] for i in train_indices]

        self.test = OrderedDict()
        for language, language_data in self.raw_data.items():
            if self.evaluate_all_cross_language and language != self.train_language:
                test_indices = split_shuffles[language]
            else:
                test_indices = split_shuffles[language][self.len_dev:self.len_dev + self.len_test]

            self.test[language] = [language_data[i] for i in test_indices]

    def load_split(self, embeddings_path, labels_path):
        embeddings = []
        labels = []

        with open(embeddings_path, 'r') as f_embeddings, open(labels_path, 'r') as f_labels:
            for line in f_embeddings:
                embedding = np.fromstring(line.strip(), sep=' ')
                embeddings.append(embedding)
                label = next(f_labels).strip()
                labels.append(label)

        classes = sorted(list(set(labels)))
        if self.classes is None:
            self.classes = classes
        assert classes == self.classes

        labels = binarize(labels, self.classes)
        return list(zip(embeddings, labels))


component = GenericRandomSubsampleValidationSentenceClassificationData
