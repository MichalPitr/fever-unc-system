import copy
import json

import numpy
import torch

import config
from utils import fever_db, text_clean
from tqdm import tqdm


class DocIdDict(object):
    def __init__(self):
        self.tokenized_doc_id_dict = None

    def load_dict(self):
        if self.tokenized_doc_id_dict is None:
            self.tokenized_doc_id_dict = json.load(open(config.TOKENIZED_DOC_ID, encoding='utf-8', mode='r'))

    def clean(self):
        self.tokenized_doc_id_dict = None


global_doc_id_object = DocIdDict()


def e_tokenize(text, tok):
    return tok.tokenize(text_clean.normalize(text))


def save_jsonl(d_list, filename):
    print("Save to Jsonl:", filename)
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item) + '\n')


def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def load_model(model, model_path, device):
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cpu'}))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device)


def tokenize_doc_id(doc_id, tokenizer):
    # path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    # print(path_stanford_corenlp_full_2017_06_09)
    #
    # drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    # tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])

    doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
    tokenized_doc_id = e_tokenize(doc_id_natural_format, tokenizer)
    t_doc_id_natural_format = tokenized_doc_id.words()
    lemmas = tokenized_doc_id.lemmas()
    return t_doc_id_natural_format, lemmas


def doc_id_to_tokenized_text(doc_id, including_lemmas=False):
    # global tokenized_doc_id_dict
    global_doc_id_object.load_dict()
    tokenized_doc_id_dict = global_doc_id_object.tokenized_doc_id_dict

    if tokenized_doc_id_dict is None:
        tokenized_doc_id_dict = json.load(open(config.TOKENIZED_DOC_ID, encoding='utf-8', mode='r'))

    if including_lemmas:
        return tokenized_doc_id_dict[doc_id]['words'], tokenized_doc_id_dict[doc_id]['lemmas']

    return ' '.join(tokenized_doc_id_dict[doc_id]['words'])


def merge_sent_results(sent_r_list):
    r_len = len(sent_r_list[0])
    for sent_r in sent_r_list:
        assert len(sent_r) == r_len

    new_list = copy.deepcopy(sent_r_list[0])
    for i in range(r_len):
        prob_list = []
        score_list = []
        for sent_r in sent_r_list:
            assert sent_r[i]['selection_id'] == new_list[i]['selection_id']
            prob_list.append(sent_r[i]['prob'])
            score_list.append(sent_r[i]['score'])
        # assert len(prob_list) ==
        new_list[i]['prob'] = float(numpy.mean(prob_list))
        new_list[i]['score'] = float(numpy.mean(score_list))

    return new_list


def delete_unused_evidence(d_list):
    for item in d_list:
        if item['predicted_label'] == 'NOT ENOUGH INFO':
            item['predicted_evidence'] = []
