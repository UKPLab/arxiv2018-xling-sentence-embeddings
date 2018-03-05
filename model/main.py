import json
import logging
import os
import sys

import click
import numpy as np
from flask import Flask, render_template
from flask import request

from sentence_embeddings import get_sentence_embedding, operations
from word_embeddings import WordEmbeddings

ukp_embeddings_share_url = 'https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings'
ukp_xling_embeddings_share_url = '{}/xling-wordembeddings'.format(ukp_embeddings_share_url)
ukp_monoling_embeddings_share_url = '{}/monolingual-wordembeddings'.format(ukp_embeddings_share_url)

embeddings = {
    'en-de': [
        WordEmbeddings(
            'mapped_bivcd_en_de', ukp_xling_embeddings_share_url, 'mapped_bivcd_en_de.txt.gz',
            approximate_filesize=101035275, file_n_lines=86761, lowercased=True
        ),
        WordEmbeddings(
            'mapped_attract_repel_en_de', ukp_xling_embeddings_share_url, 'mapped_attract_repel_en_de.txt.gz',
            approximate_filesize=270155631, file_n_lines=234036, lowercased=True
        ),
        WordEmbeddings(
            # this is the small FT version. See full version in the comments below
            'mapped_fasttext_en_de', ukp_xling_embeddings_share_url, 'mapped_fasttext_300k_en_de.txt.gz',
            approximate_filesize=680901729, file_n_lines=599959, lowercased=True
        ),
        # WordEmbeddings(
        #    'mapped_fasttext_en_de', ukp_xling_embeddings_share_url, 'mapped_fasttext_en_de.txt.gz',
        #    approximate_filesize=5561825768, file_n_lines=4794000, lowercased=True
        # ),
    ],
    'en-fr': [
        WordEmbeddings(
            'mapped_bivcd_en_fr', ukp_xling_embeddings_share_url, 'mapped_bivcd_en_fr.txt.gz',
            approximate_filesize=173658707, file_n_lines=149255, lowercased=True
        ),
        WordEmbeddings(
            'mapped_attract_repel_en_fr', ukp_xling_embeddings_share_url, 'mapped_attract_repel_en_fr.txt.gz',
            approximate_filesize=318510281, file_n_lines=276513, lowercased=True
        ),
        WordEmbeddings(
            # this is the small FT version. See full version in the comments below
            'mapped_fasttext_en_fr', ukp_xling_embeddings_share_url, 'mapped_fasttext_300k_en_fr.txt.gz',
            approximate_filesize=678257779, file_n_lines=598857, lowercased=True
        ),
        # WordEmbeddings
        #    'mapped_fasttext_en_fr', ukp_xling_embeddings_share_url, 'mapped_fasttext_en_fr.txt.gz',
        #    approximate_filesize=4238398397, file_n_lines=3661392, lowercased=True
        # ),
    ],
    'monolingual': {
        WordEmbeddings(
            'glove', ukp_monoling_embeddings_share_url, 'glove.840B.300d.txt.gz',
            approximate_filesize=2176768669, file_n_lines=8175104, lowercased=False
        ),
        WordEmbeddings(
            'gnews', ukp_monoling_embeddings_share_url, 'G.utf8.gz',
            approximate_filesize=8407252992, file_n_lines=2930018, lowercased=False
        ),
        WordEmbeddings(
            'komninos_attract_repel', ukp_monoling_embeddings_share_url, 'Komninos.attract-repel.gz',
            approximate_filesize=220019840, file_n_lines=857908, lowercased=True
        ),
        WordEmbeddings(
            'morph_specialized', ukp_monoling_embeddings_share_url, 'english-morph-fitted.txt.simple.gz',
            approximate_filesize=1356373227, file_n_lines=5250747, lowercased=True
        ),
    }
}


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
