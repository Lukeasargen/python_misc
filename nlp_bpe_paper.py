"""
Neural Machine Translation of Rare Words with Subword Units
https://arxiv.org/pdf/1508.07909.pdf
"""

import re
import collections


def get_stats(vocab):
    """ count the frequency of pairs of symbols """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    """ combine the pair into a single symbol """
    v_out = {}
    bigram = re.escape(' '.join(pair))  # combine with return
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')  # create re pattern
    for word in v_in:
        w_out = p.sub(''.join(pair), word)  # replace with the pair joined
        v_out[w_out] = v_in[word]  # remember the frequency of the word
    return v_out


if __name__ == "__main__":


    vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
    'n e w e s t </w>':6, 'w i d e s t </w>':3}

    num_merges = 10
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        print("best :", best)
        vocab = merge_vocab(best, vocab)
        print("vocab :", vocab)
        
