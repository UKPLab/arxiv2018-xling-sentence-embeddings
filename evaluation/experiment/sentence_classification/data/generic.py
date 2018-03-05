from collections import OrderedDict

import numpy as np

import experiment


class GenericSentenceClassificationData(experiment.Data):
    def __init__(self, config, config_global, logger):
        super(GenericSentenceClassificationData, self).__init__(config, config_global, logger)
        self.classes = None

    def setup(self):
        # structure of train data: list[(embedding, label)]
        self.train = self.load_split(self.config['train_embeddings'], self.config['train_labels'])
        # np.random.shuffle(self.train)

        assert 'valid_embeddings' not in self.config or 'valid_split' not in self.config
        if 'valid_embeddings' in self.config:
            self.valid = self.load_split(self.config['valid_embeddings'], self.config['valid_labels'])
        else:
            n_valid = int(len(self.train) * self.config['valid_split'])
            self.valid = self.train[:n_valid]
            self.train = self.train[n_valid:]

        # structure of test data: dict[language, list[(embedding, label)]]
        self.test = OrderedDict()
        for embeddings_config in self.config['test_embeddings']:
            embeddings_path = embeddings_config['embeddings']
            language = embeddings_path.split('/')[-1].split('.')[-2]
            self.test[language] = self.load_split(embeddings_path, embeddings_config['labels'])

        self.logger.debug('Train examples: {}'.format(len(self.train)))
        self.logger.debug('Valid examples: {}'.format(len(self.valid)))
        self.logger.debug('Test examples: {}'.format(len(list(self.test.values())[0])))

        self.embedding_size = len(self.train[0][0])

    def load_split(self, embeddings_path, labels_path):
        embeddings = []
        labels = []

        with open(embeddings_path, 'r') as f_embeddings, open(labels_path, 'r') as f_labels:
            for line in f_embeddings:
                embedding = [float(s) for s in line.strip().split(' ')]
                embeddings.append(embedding)
                label = next(f_labels).strip()
                labels.append(label)

        classes = sorted(list(set(labels)))
        if self.classes is None:
            self.classes = classes
        assert classes == self.classes

        self.logger.debug('Class distribution {}'.format(
            [len([l for l in labels if l == c]) / float(len(labels)) for c in self.classes]))

        labels = binarize(labels, self.classes)
        return list(zip(embeddings, labels))


def binarize(labels, classes):
    results = []
    for label in labels:
        val = np.zeros(len(classes), dtype=np.float32)
        val[classes.index(label)] = 1.0
        results.append(val)
    return results


component = GenericSentenceClassificationData
