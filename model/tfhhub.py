import logging
import os
import shutil
import sys
import tempfile

import click
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from word_embeddings import embeddings


def non_zero_tokens(tokens):
    """Receives a batch of vectors of tokens (float) which are zero-padded. Returns a vector of the same size, which has
    the value 1.0 in positions with actual tokens and 0.0 in positions with zero-padding.

    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tf.maximum(tokens, 1.0), [1], keepdims=True))


def p_mean(values, p, n_toks):
    n_toks = tf.cast(tf.maximum(tf.constant(1.0), n_toks), tf.complex64)
    p_tf = tf.constant(float(p), dtype=tf.complex64)
    values = tf.cast(values, dtype=tf.complex64)
    res = tf.pow(
        tf.reduce_sum(
            tf.pow(values, p_tf),
            axis=1,
            keepdims=False
        ) / n_toks,
        1.0 / p_tf
    )
    return tf.real(res)


def mean(token_embeddings, non_oov_tokens, non_oov_count):
    return tf.reduce_sum(token_embeddings, axis=1, keepdims=False) / tf.maximum(non_oov_count, 1.0)


def max(token_embeddings, non_oov_tokens, non_oov_count):
    non_oov_tokens = tf.expand_dims(non_oov_tokens, -1)
    # the last term ensures that we do not set OOV embeddings to very low values to ignore in the maximum if there is
    # ONLY OOV values
    token_embeddings = token_embeddings + ((non_oov_tokens - 1) * 10) * tf.minimum(1.0, tf.expand_dims(non_oov_count, -1))
    return tf.reduce_max(token_embeddings, axis=1, keepdims=False)


def min(token_embeddings, non_oov_tokens, non_oov_count):
    non_oov_tokens = tf.expand_dims(non_oov_tokens, -1)
    token_embeddings = token_embeddings + ((non_oov_tokens - 1) * -10) * tf.minimum(1.0, tf.expand_dims(non_oov_count, -1))
    return tf.reduce_min(token_embeddings, axis=1, keepdims=False)


operations = dict([
    ('mean', mean),
    ('max', max),
    ('min', min),
    ('p_mean_3', lambda token_embeddings, non_oov_tokens, non_oov_count: p_mean(token_embeddings, p=3, n_toks=non_oov_count)),
])


def make_module_spec(model, p_means, names, vocab_files, vocab_sizes):
    def module_fn():
        """Spec function for a token embedding module."""
        sentences = tf.placeholder(shape=[None], dtype=tf.string, name='sentences')
        tokens = tf.string_split(sentences, ' ')

        averaged_embeddings = []
        for i, (name, vocab_file, vocab_size) in enumerate(zip(names, vocab_files, vocab_sizes)):
            embs = embeddings[model][i]
            assert embs.alias == name
            embeddings_var = tf.get_variable(
                initializer=tf.zeros([vocab_size + 1, embs.embeddings_dimensionality]),
                name='embeddings_{}'.format(name),
                dtype=tf.float32)

            lookup_table = tf.contrib.lookup.index_table_from_file(
                vocabulary_file=vocab_file,
                num_oov_buckets=1,
            )

            tokens_dense = tf.sparse_tensor_to_dense(tokens, default_value='')
            token_ids = lookup_table.lookup(tokens_dense)
            non_oov_tokens = non_zero_tokens(tf.abs(tf.to_float(token_ids) - vocab_size))
            non_oov_count = tf.reduce_sum(non_oov_tokens, axis=1, keepdims=True)
            token_embeddings = tf.nn.embedding_lookup(params=embeddings_var, ids=token_ids)

            for p_mean in p_means:
                op_embedding = operations[p_mean](token_embeddings, non_oov_tokens, non_oov_count)
                averaged_embeddings.append(op_embedding)

        sentence_embedding = tf.concat(averaged_embeddings, axis=1)

        hub.add_signature("default", {"sentences": sentences},
                          {"default": sentence_embedding})

    return hub.create_module_spec(module_fn)


def export(export_path, model, p_means, names, vocabs, vecs, logger):
    tmpdir = tempfile.mkdtemp()

    vocab_files, vocab_sizes = [], []
    for name, vocab in zip(names, vocabs):
        vocab_file = os.path.join(tmpdir, '{}.txt'.format(name))
        with tf.gfile.GFile(vocab_file, "w") as f:
            f.write("\n".join(vocab))

        vocab_files.append(vocab_file)
        vocab_sizes.append(len(vocab))

    spec = make_module_spec(model, p_means, names, vocab_files, vocab_sizes)

    try:
        with tf.Graph().as_default():
            m = hub.Module(spec)
            # see https://github.com/tensorflow/hub/blob/r0.1/examples/text_embeddings/export.py
            load_embeddings = []
            feed_dict = {}
            for name, vec in zip(names, vecs):
                p_embeddings = tf.placeholder(tf.float32)
                load_embeddings.append(tf.assign(m.variable_map['embeddings_{}'.format(name)], p_embeddings))
                feed_dict[p_embeddings] = vec

            with tf.Session() as sess:
                sess.run(load_embeddings, feed_dict=feed_dict)
                m.export(export_path, sess)
    finally:
        shutil.rmtree(tmpdir)


def export_hub_module(export_path, model, p_means, embeddings_folder, logger):
    logger.info('Now loading all embeddings')
    names, vocabs, vecs = [], [], []
    for emb in embeddings[model]:
        emb.download_file(embeddings_folder, logger)
    for emb in embeddings[model]:
        emb.load_vectors(embeddings_folder, logger)
        names.append(emb.alias)
        vocabs.append(list(emb.vectors.keys()))
        vecs.append(list(emb.vectors.values()))
        vecs[-1].append(np.zeros(len(vecs[-1][0]), dtype=np.float32))

    # Export the embedding vectors into a TF-Hub module.
    export(export_path, model, p_means, names, vocabs, vecs, logger)


@click.command()
@click.option('--export-path', help='export path of the tf hub module')
@click.option('--model', default='en-de', help='en-de, en-fr, or monolingual')
@click.option('--p-means', default='mean,max,min,p_mean_3', help='comma separated p-means/operations')
@click.option('--embeddings-folder', default='data', help='path where the word embeddings will be stored')
def main(export_path, model, p_means, embeddings_folder):
    logger = logging.getLogger('xling_sentence_embeddings')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel('DEBUG')
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)
    logger.setLevel('DEBUG')

    if not os.path.exists(embeddings_folder):
        os.mkdir(embeddings_folder)

    p_means = p_means.split(',')

    export_hub_module(export_path, model, p_means, embeddings_folder, logger)


if __name__ == "__main__":
    main()
