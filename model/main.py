import json
import logging
import os
import sys

import click
import numpy as np
from flask import Flask, render_template
from flask import request

from sentence_embeddings import get_sentence_embedding, operations
from word_embeddings import embeddings


def start_webserver(model_name, embeddings, host, port, logger):
    embedding_types_dict = dict([(e.alias, e) for e in embeddings])

    app = Flask('concatenated-p-means-embeddings-{}'.format(model_name))

    @app.route("/")
    def index():
        return render_template(
            'index.html',
            model_name=model_name, embeddings=embeddings, operations=operations.keys(), host=request.host
        )

    @app.route("/embed", methods=['POST'])
    def convert():
        try:
            output = ''
            conversion_data = json.loads(request.form.get('conversion'))

            sentences = conversion_data['sentences']
            chosen_embedding_types = conversion_data['embedding_types']
            # [('glove', ['mean'], ...)

            for i, sentence in enumerate(sentences):
                embs = []
                tokens = sentence.split()
                tokens_lower = sentence.lower().split()
                for embedding_type_name, operation_names in chosen_embedding_types:
                    embeddings = embedding_types_dict[embedding_type_name]
                    embs.append(get_sentence_embedding(
                        tokens_lower if embeddings.lowercased else tokens,
                        embeddings,
                        operation_names
                    ))
                concat_emb = np.concatenate(embs, axis=0)
                output += ' '.join([str(e) for e in concat_emb]) + '\n'
            return output
        except BaseException as e:
            logger.exception('Error while processing sentence embedding conversion request')
            return 'There was an error while processing sentence embedding conversion request (logged). \n' \
                   + 'Usually this is related to malformed json payload.', 500

    print('Starting server...')
    app.run(host=host, port=port, debug=False)


@click.command()
@click.option('--model', default='en-de', help='en-de, en-fr, or monolingual')
@click.option('--embeddings-folder', default='data', help='path where the word embeddings will be stored')
@click.option('--webserver-host', default='0.0.0.0', help='For private (host-only) access set this to "127.0.0.1"')
@click.option('--webserver-port', default=8080, help='port of the webserver')
def run(model, embeddings_folder, webserver_host, webserver_port):
    logger = logging.getLogger('xling_sentence_embeddings')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel('DEBUG')
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)
    logger.setLevel('DEBUG')

    if not os.path.exists(embeddings_folder):
        os.mkdir(embeddings_folder)

    for emb in embeddings[model]:
        emb.download_file(embeddings_folder, logger)
    for emb in embeddings[model]:
        emb.load_vectors(embeddings_folder, logger)

    logger.info('Finished loading data. The model is ready!')
    logger.info('Now starting the webserver (open in browser for more information!)...')

    start_webserver(model, embeddings[model], webserver_host, webserver_port, logger)


if __name__ == '__main__':
    run()
