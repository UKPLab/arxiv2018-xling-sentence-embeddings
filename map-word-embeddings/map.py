#
# Maps all word embeddings of a word embeddings file using the mapping HTTP API
#

import io
import json
from collections import defaultdict
from math import ceil

import click
import numpy as np
import requests

from experiment.utils.data import read_embeddings


def map_embeddings(embeddings, target, map_url):
    sentences = [' '.join([str(s) for s in embedding.tolist()]) for embedding in embeddings]
    config = {'sentences': sentences, 'target': target}
    sentvec_strings = requests.post('{}/convert'.format(map_url),
                                    data={'conversion': json.dumps(config)}).text

    result = []
    for sentvec_string in sentvec_strings.split('\n')[:-1]:
        result.append(np.fromstring(sentvec_string.strip(), sep=' '))
    return result


@click.command()
@click.option('--embeddings-path', prompt="Path of word embeddings", help="Path of the word embeddings to map. Tokens must be post-fixed with '_language'")
@click.option('--out-path', prompt="Output path", help="Path of the output word embeddings file")
@click.option('--map-url', prompt="URL of mapping API", help="URL of the mapping webservice (as shown in the log of the model training)")
@click.option('--lang-a', prompt="lang_a of your experiment config", help="Must be set to lang_a of the config file of the trained model")
@click.option('--batchsize', default=2000, help="Number of tokens to map in parallel")
def map_all_embeddings(embeddings_path, out_path, map_url, lang_a, batchsize):
    """Maps all word embeddings of a word embeddings file using the mapping HTTP API. This will produce a word
    embeddings file with the same tokens, but mapped vectors."""
    embeddings_dict = read_embeddings(embeddings_path, is_gzipped=False, skip_first=False)

    items = embeddings_dict.items()
    items_languages = defaultdict(list)
    for token, value in items:
        lang = token.split('_')[-1]
        items_languages[lang].append((token, value))

    print('Languages in embeddings file: {}'.format(', '.join(items_languages.keys())))

    with io.open(out_path, 'w', encoding='utf-8') as f:
        for language, embs in items_languages.items():
            n_batches = int(ceil(len(embs) / float(batchsize)))
            for batch in range(n_batches):
                print('batch {}/{}'.format(batch, n_batches))
                batch_embs = embs[batch * batchsize: (batch + 1) * batchsize]
                toks, vecs = zip(*batch_embs)
                mapped_vecs = map_embeddings(vecs, 'lang_a' if language == lang_a else 'lang_b', map_url)
                for tok, mapped_vec in zip(toks, mapped_vecs):
                    f.write(u'{} {}\n'.format(tok, ' '.join([str(d) for d in mapped_vec])))

    print('DONE')


if __name__ == '__main__':
    map_all_embeddings()
