#!/usr/bin/env python
"""Random utils for doc retrieval experiment

  /|_/\
=( °w° )=
  )   (  //
 (__ __)//

"""
import argparse
import re
import math
import json

import config

__author__ = ['chaonan99', 'yixin1']


def reverse_convert_brc(string):
    string = re.sub(r'\(', '-LRB-',   string)
    string = re.sub(r'\)', '-RRB-',   string)
    string = re.sub(r'\[', '-LSB-',   string)
    string = re.sub(r'\]', '-RSB-',   string)
    string = re.sub(r'{',  '-LCB-',   string)
    string = re.sub(r'}',  '-RCB-',   string)
    string = re.sub(r':',  '-COLON-', string)
    string = re.sub(r' ',  '_',       string)
    return string


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def get_default_tfidf_ranker_args():
    args = argparse.Namespace(ngram=2,
                              hash_size=int(math.pow(2, 24)),
                              num_workers=4)
    return args


class DocIDTokenizer:
    """DocIDTokenizer is used for tokenizing doc ID

    Example
    -------
    >>> docid_tokenizer = DocIDTokenizer(case_insensitive=True)
    >>> tokens, lemmas = docid_tokenizer.tokenize_docid('Barack_Obama')

    """
    class __DocIDTokenizer:
        def __init__(self, case_insensitive=False):
            self.tokenized_docid_dict = json.load(open(config.TOKENIZED_DOC_ID,
                                                        encoding='utf-8',
                                                        mode='r'))
            if case_insensitive:
                self.tokenized_docid_dict = {k.lower(): v for k, v in \
                    self.tokenized_docid_dict.items()}

        def tokenize_docid(self, doc_id):
            return self.tokenized_docid_dict[doc_id]['words'], \
                   self.tokenized_docid_dict[doc_id]['lemmas']

    instance = None
    case_insensitive = None

    def __init__(self, case_insensitive=False):
        if DocIDTokenizer.instance is None or \
               case_insensitive != DocIDTokenizer.case_insensitive:
            print("Reload tokenizer dictionary")
            DocIDTokenizer.case_insensitive = case_insensitive
            DocIDTokenizer.instance = \
                DocIDTokenizer.__DocIDTokenizer(case_insensitive)

        ## I don't know why I need the followings but the code does not work
        ## in Python 3.6 w/o them
        self.instance = DocIDTokenizer.instance
        self.case_insensitive = case_insensitive

    @classmethod
    def clean_instance(cls):
        DocIDTokenizer.instance = None

    def __getattr__(self, name):
        return getattr(self.instance, name)


def get_docids_from_sds(sds):
    all_ids = []
    for k, s in sds.items():
        all_ids.extend([it[0] for it in s])
    return set(all_ids)


def get_docids_from_ssi(ssi):
    return set([it[0].split('<SENT_LINE>')[0] for it in ssi])


def get_docids_from_sds_prio(sds):
    max_score = -100
    max_key = None
    for k, s in sds.items():
        if len(s) == 0:
            continue
        if s[0][1] > max_score:
            max_score = s[0][1]
            max_key = k
    if max_key is None:
        return set()
    else:
        return set([it[0] for it in sds[max_key]])


def get_docids_from_pdo(pdo):
    return set([it[0] for it in pdo])


def get_docids_from_evi(evi):
    return set([iii for i in evi for ii in i for iii in ii if type(iii) == str])


def get_docids_from_evi_common(evi):
    return set.intersection(*[set([iii for ii in i for iii in ii \
                                   if type(iii) == str]) for i in evi])


def get_sentids_from_evi(evi):
    return set(['<SENT_LINE>'.join([ii[2], ii[3]]) for i in evi for ii in i])


def get_docids_from_predicted_evi(evi):
    return
