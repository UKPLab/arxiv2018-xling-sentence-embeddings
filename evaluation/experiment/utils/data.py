from collections import OrderedDict

import numpy as np


def merge_dicts(dict_a, dict_b):
    """Performs a recursive merge of two dicts dict_a and dict_b, wheras dict_b always overwrites the values of dict_a

    :param dict_a: the first dictionary. This is the weak dictionary which will always be overwritten by dict_b (dict_a
                   therefore is a default dictionary type
    :param dict_b: the second dictionary. This is the strong dictionary which will always overwrite vales of dict_a
    :return:
    """
    merge_result = dict_a.copy()
    for key in dict_b:
        if key in dict_a and isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
            merge_result[key] = merge_dicts(dict_a[key], dict_b[key])
        else:
            merge_result[key] = dict_b[key]
    return merge_result


def replace_dict_values(source, replacements):
    """Creates a copy of the source dictionary and replaces all values specified in a replacement list

    :param source: The source dictionary
    :param replacements: The replacements. This is a list of key/value tuples, where dots in the key describe the
    hierarchy (e.g. my.property)
    :return: a copy of source with the replacements
    """
    result = dict(source)
    for key, value in replacements:
        if '.' in key:
            split = key.split('.')
            result[split[0]] = replace_dict_values(
                result[split[0]],
                [('.'.join(split[1:]), value)]
            )
        else:
            result[key] = value
    return result


def read_embeddings(path, vocab=None, logger=None):
    """Reads an embeddings file and returns a dictionary with the mapping from word to vector. For reproducibility and
    debugging reasons we are returning an ordered dict here.

    :param path: path of the embeddings file
    :type path: str
    :param vocab: the vocabulary to keep
    :type vocab: list[str] | set[str]
    :rtype: OrderedDict[str, list[float]]
    :return:
    """
    if isinstance(vocab, list):
        vocab = set(vocab)  # __contains__ for sets is much faster

    result = OrderedDict()
    with open(path, 'r') as f:
        for line in f:
            if line:
                tv = line.strip().split()
                word = tv[0]
                if vocab is None or word in vocab:
                    try:
                        vector = np.array([float(v) for v in tv[1:]])
                        result[word] = vector
                    except:
                        message = "Embeddings reader: Could not convert line: {}".format(line)
                        if logger is not None:
                            logger.debug(message)
                        else:
                            print(message)

    return result


def unique_items(seq):
    """A function that returns all unique items of a list in an order-preserving way"""
    id_fn = lambda x: x
    seen_items = {}
    result = []
    for item in seq:
        marker = id_fn(item)
        if marker in seen_items: continue
        seen_items[marker] = 1
        result.append(item)
    return result
