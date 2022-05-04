import numpy as np
import pickle
import os
import csv
import argparse
from scipy.stats import spearmanr
from gensim.models import word2vec
from tqdm import tqdm
from multiprocessing import Pool
from time import time

test_file = {
        "wordsim353" : './datasets/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt',
        "men" : './datasets/MEN/MEN_dataset_natural_form_full',
        "simlex" : './datasets/SimLex-999/SimLex-999.txt',
    }

def get_emb(vec_file):
    f = open(vec_file, 'r', errors='ignore')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = np.array([float(ele) for ele in vec])
        word_emb[word] = np.array(vec)
        vocabulary[word] = i
        vocabulary_inv[i] = word

    return word_emb, vocabulary, vocabulary_inv


def read_sim_test(test="wordsim353"):
    f = open(test_file[test])
    if test == 'wordsim353':
        csv_reader = csv.reader(f, delimiter='\t')
        tests = {}
        for i, row in enumerate(csv_reader):
            if i > 0:
                word_pair = (row[0], row[1])
                tests[word_pair] = float(row[2])
    elif test == 'men':
        tests = {}
        for line in f:
            tmp = line.split(" ")
            if len(tmp) != 3:
                continue
            word_pair = (tmp[0], tmp[1])
            tests[word_pair] = float(tmp[2])
    elif test == 'simlex':
        tests = {}
        for i, line in enumerate(f):
            if i == 0:
                continue
            tmp = line.split("\t")
            if len(tmp) != 10:
                continue
            word_pair = (tmp[0], tmp[1])
            tests[word_pair] = float(tmp[3])
    return tests


def calc_sim(w1, w2):
    return np.dot(w1, w2)/np.linalg.norm(w1)/np.linalg.norm(w2)


def test_sim(word_emb, tests):
    pool = Pool(20)
    real_tests = {}
    for word_pair in tests:
        w1 = word_pair[0]
        w2 = word_pair[1]
        if w1 in word_emb and w2 in word_emb:
            real_tests[word_pair] = tests[word_pair]
    print(f'{len(real_tests)}/{len(tests)} actual test cases!')
    t0 = time()
    args = [(word_emb[word_pair[0]], word_emb[word_pair[1]]) for word_pair in real_tests.keys()]
    res = pool.starmap(calc_sim, args)
    truth = list(real_tests.values())
    rho = spearmanr(truth, res)[0]
    print(f'Spearman coefficient: {rho}')
    return rho


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='wiki')
    parser.add_argument('--emb_file', default='jose.txt')

    args = parser.parse_args()
    print(args)

    test_cases = []
    print(f"Reading embedding from {os.path.join(args.dataset, args.emb_file)}")
    word_emb, vocabulary, vocabulary_inv = get_emb(vec_file=os.path.join(args.dataset, args.emb_file))
    rho = np.zeros(3)
    cnt = 0
    for key in test_file:
        print(f'### Test: {key} ###')
        tests = read_sim_test(test=key)
        rho[cnt] = test_sim(word_emb, tests)
        cnt = cnt + 1
        print('\n')

    print(rho.mean())