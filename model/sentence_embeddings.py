import numpy as np


def gen_mean(vals, p):
    p = float(p)
    return np.power(
        np.mean(
            np.power(
                np.array(vals, dtype=complex),
                p),
            axis=0),
        1 / p
    )


operations = dict([
    ('mean', (lambda word_embeddings: [np.mean(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('max', (lambda word_embeddings: [np.max(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('min', (lambda word_embeddings: [np.min(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('p_mean_2', (lambda word_embeddings: [gen_mean(word_embeddings, p=2.0).real], lambda embeddings_size: embeddings_size)),
    ('p_mean_3', (lambda word_embeddings: [gen_mean(word_embeddings, p=3.0).real], lambda embeddings_size: embeddings_size)),
])


def get_sentence_embedding(sentence, embeddings, chosen_operations):
    word_embeddings = []
    for tok in sentence:
        vec = embeddings.vectors.get(tok)
        if vec is not None:
            word_embeddings.append(vec)

    if not word_embeddings:
        print('No word embeddings for sentence:\n{}'.format(sentence))
        size = 0
        for o in chosen_operations:
            size += operations[o][1](embeddings.embeddings_dimensionality)
        sentence_embedding = np.zeros(size)
    else:
        concat_embs = []
        for o in chosen_operations:
            concat_embs += operations[o][0](word_embeddings)
        sentence_embedding = np.concatenate(
            concat_embs,
            axis=0
        )

    return sentence_embedding
