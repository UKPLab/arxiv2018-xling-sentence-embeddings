from __future__ import division

import math
from collections import OrderedDict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import experiment


class SentenceClassificationEvaluation(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(SentenceClassificationEvaluation, self).__init__(config, config_global, logger)
        self.batchsize_test = self.config.get('batchsize_test', 512)
        self.report_accuracy = self.config.get('report_accuracy', False)

    def start(self, model, data, sess, valid_only=False):
        results = OrderedDict()

        valid_accuracy, valid_f1 = self.evaluate(model, sess, data.valid)
        self.logger.debug('Results Valid: Accuracy={}, F1={}'.format(valid_accuracy, valid_f1))
        results['valid'] = valid_f1

        if not valid_only:
            # also validate on test
            for language, test in data.test.items():
                test_accuracy, test_f1 = self.evaluate(model, sess, test)
                results['test-{}'.format(language)] = test_accuracy if self.report_accuracy else test_f1
                self.logger.debug('Results Test ({}): Accuracy={}, F1={}'.format(language, test_accuracy, test_f1))

        return results

    def evaluate(self, model, sess, split_data):
        embeddings, labels = zip(*split_data)
        predictions = []
        for test_batch in range(int(math.ceil(len(embeddings) / float(self.batchsize_test)))):
            test_batch_indices = self.batchsize_test * test_batch, self.batchsize_test * (test_batch + 1)
            test_batch_embeddings = embeddings[test_batch_indices[0]:test_batch_indices[1]]
            prediction, = sess.run([model.predict], feed_dict={
                model.input_sent: test_batch_embeddings,
                model.dropout_keep_prob: 1.0,
            })
            predictions += prediction.tolist()

        labels_flat = np.argmax(labels, axis=1)
        predictions_flat = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels_flat, predictions_flat)
        assert len(labels_flat) == len(predictions_flat)

        labels = sorted(list(set(labels_flat)), reverse=True)
        pos_label = labels[0]  # the biggest one. if 0.0 and 1.0 it is 1.0
        n_classes = len(labels)
        # average = 'macro' if n_classes > 2 else 'binary'
        average = 'macro'
        f1 = f1_score(labels_flat, predictions_flat, average=average, pos_label=pos_label)

        # print(confusion_matrix(labels_flat, predictions_flat))

        return accuracy, f1


component = SentenceClassificationEvaluation
