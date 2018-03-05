from __future__ import division

import os

import numpy as np
import json
import experiment


class FixSpaceEvaluation(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(FixSpaceEvaluation, self).__init__(config, config_global, logger)
        self.batchsize_test = self.config.get('batchsize_test', 512)

    def start(self, model, data, sess, valid_only=False):
        results = dict()

        evaluations = [('valid', data.valid_data)]
        if not valid_only:
            evaluations += [('test', data.test_data)]

        for split, data in evaluations:
            results_fix, results_orig = self.evaluate(model, sess, data, with_orig=not valid_only)
            self.logger.info('Accuracy ({}) FIX: {}'.format(split, results_fix))
            self.logger.info('Accuracy ({}) ORIG: {}'.format(split, results_orig))
            results[split] = results_fix
            results['{}-orig'.format(split)] = results_orig

        if not valid_only and 'transform' in self.config:
            self.logger.info('Now transforming embeddings of files')
            folder_in = self.config['transform']['in']
            folder_out = self.config['transform']['out']

            for filename in [n for n in os.listdir(folder_in) if not n.startswith('.')]:
                path_in = os.path.join(folder_in, filename)
                path_out = os.path.join(folder_out, filename)

                if filename.split('.')[-2] == self.config['transform']['source_lang']:
                    input_field = model.input_source
                    transformation_field = model.source_transformed
                else:
                    input_field = model.input_translation
                    transformation_field = model.translation_transformed

                with open(path_in, 'r') as f_in, open(path_out, 'w') as f_out:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            f_out.write('\n')
                        else:
                            embedding = np.array([float(x) for x in line.split()])
                            embedding_transformed, = sess.run([transformation_field], feed_dict={
                                input_field: [embedding],
                                model.dropout_keep_prob: 1.0,
                            })
                            f_out.write(' '.join(['{}'.format(x) for x in embedding_transformed[0]]) + '\n')

        if not valid_only and 'webserver' in self.config:
            from flask import Flask
            from flask import request
            app = Flask(__name__)

            @app.route("/convert", methods=['POST'])
            def convert():
                conversion_data = json.loads(request.form.get('conversion'))

                sentences = conversion_data['sentences']
                chosen_embedding_types = conversion_data['target']
                # 'lang_a' or 'lang_b'

                if chosen_embedding_types == 'lang_a':
                    input_field = model.input_source
                    transformation_field = model.source_transformed
                else:
                    input_field = model.input_translation
                    transformation_field = model.translation_transformed

                result = ''

                embeddings = [np.fromstring(s, sep=' ') for s in sentences]
                embeddings_transformed, = sess.run([transformation_field], feed_dict={
                    input_field: embeddings,
                    model.dropout_keep_prob: 1.0
                })
                for e in embeddings_transformed:
                    result += ' '.join([str(e) for e in e]) + '\n'

                return result

            print('Running webserver')
            app.run(host='0.0.0.0', port=self.config['webserver']['port'])

        return results

    def evaluate(self, model, sess, split_data, with_orig=False):
        results_fix = []
        results_orig = []

        all_sents_lang_b = [s[1] for s in split_data]
        for i, (sent_a, sent_b) in enumerate(split_data):
            fd = {
                model.input_source: [sent_a] * len(all_sents_lang_b),
                model.input_translation: all_sents_lang_b,
                model.dropout_keep_prob: 1.0,
            }
            if with_orig:
                predictions_fix, predictions_orig = sess.run([model.predict, model.predict_orig], feed_dict=fd)
                results_orig.append(1.0 if i == np.argmax(predictions_orig) else 0.0)
            else:
                predictions_fix,  = sess.run([model.predict], feed_dict=fd)
                results_orig.append(0.0)
            results_fix.append(1.0 if i == np.argmax(predictions_fix) else 0.0)

        return np.mean(results_fix), np.mean(results_orig)


component = FixSpaceEvaluation
