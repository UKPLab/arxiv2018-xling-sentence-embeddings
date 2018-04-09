import gzip
import io
import os
from collections import OrderedDict

import numpy as np
import requests
from progressbar import DataTransferBar, Percentage, Bar, Timer, AdaptiveETA, ProgressBar


class WordEmbeddings(object):
    def __init__(self, alias, base_url, embeddings_filename, approximate_filesize, file_n_lines, lowercased):
        self.alias = alias
        self.base_url = base_url
        self.embeddings_filename = embeddings_filename
        self.approximate_filesize = approximate_filesize
        self.file_n_lines = file_n_lines
        self.lowercased = lowercased

        self.vectors = None
        self.embeddings_dimensionality = None

    def download_file(self, embeddings_folder, logger):
        logger.info('Downloading word embeddings file for "{}"...'.format(self.alias))
        out_path = os.path.join(embeddings_folder, self.embeddings_filename)
        if os.path.exists(out_path):
            logger.info('Already downloaded -> skipping!')
        else:
            url = '{}/{}'.format(self.base_url, self.embeddings_filename)
            r = requests.get(url, stream=True)
            total_size = int(
                r.headers.get('content-length', self.approximate_filesize)
            )  # size of the embeddings file (bytes)
            chunk_size = 4 * 1024 * 1024  # 4 MB

            bar = DataTransferBar(max_value=total_size).start()

            completed_bytes = 0
            try:
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            completed_bytes += len(chunk)
                            if completed_bytes > bar.max_value:
                                bar.max_value = completed_bytes
                            bar.update(completed_bytes)
            except BaseException as e:
                os.unlink(out_path)
                raise e
            bar.finish()
            logger.info('Done!')

    def load_vectors(self, embeddings_folder, logger):
        logger.info('Loading "{}" word embeddings into memory...'.format(self.alias))
        embeddings_path = os.path.join(embeddings_folder, self.embeddings_filename)
        self.vectors = OrderedDict()

        widgets = [Percentage(), ' of {}K'.format(int(self.file_n_lines / 1000)),
                   ' ', Bar(), ' ', Timer(), ' ', AdaptiveETA()]

        bar = ProgressBar(widgets=widgets, max_value=self.file_n_lines).start()

        if self.embeddings_filename.endswith('.gz'):
            f = gzip.open(embeddings_path, 'rt')
        else:
            f = io.open(embeddings_path, 'rt', encoding='utf-8')

        try:
            for i, line in enumerate(f):
                if i % 100 == 0:
                    if i > bar.max_value:
                        bar.max_value = i
                    bar.update(i)

                line = line.strip()
                if line:
                    word, vec = line.split(' ', 1)
                    try:
                        np_vec = np.fromstring(vec, sep=' ')
                        if self.embeddings_dimensionality is None:
                            if len(np_vec) < 10:
                                logger.debug("Skipping header")
                                continue
                            else:
                                self.embeddings_dimensionality = len(np_vec)
                        if len(np_vec) == self.embeddings_dimensionality:
                            self.vectors[word] = np_vec
                    except:
                        if logger is not None:
                            logger.debug("Embeddings reader: Could not convert line: {}".format(line))
        finally:
            f.close()
            bar.finish()

        logger.info('Done!')


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
    'monolingual': [
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
    ]
}
