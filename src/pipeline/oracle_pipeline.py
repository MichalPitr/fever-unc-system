import os
from argparse import ArgumentParser
from pathlib import Path

from chaonan_src.doc_retrieval_experiment import DocRetrievalExperimentSpiral, DocRetrievalExperiment, \
    DocRetrievalExperimentTwoStep
from chaonan_src._doc_retrieval.item_rules_spiral import ItemRuleBuilderSpiral, ItemRuleBuilderNoPageview
from chaonan_src._doc_retrieval.item_rules_test import ItemRuleBuilderTest
from sentence_retrieval.simple_nnmodel import get_score_multihop
from utils import common
import utils
import config
from utils.tokenize_fever import tokenized_claim
from utils import c_scorer
from typing import Dict
from sentence_retrieval import simple_nnmodel
from simi_sampler_nli_v0 import simi_sampler
import nli.mesim_wn_simi_v1_1
import nli.mesim_wn_simi_v1_2
import nli.mesim_wn_simi_v1_3
import copy
import json
import numpy as np
import nn_doc_retrieval.disabuigation_training as disamb
from nn_doc_retrieval import nn_doc_model
from log_helper import LogHelper

LogHelper.setup()
logs  = LogHelper.get_logger("pipeline")

default_model_path_dict: Dict[str, str] = {
    'sselector': config.PRO_ROOT / 'saved_models/saved_sselector/i(57167)_epoch(6)_(tra_score:0.8850885088508851|raw_acc:1.0|pr:0.3834395939593578|rec:0.8276327632763276|f1:0.5240763176570098)_epoch',
    'sselector_1': config.PRO_ROOT / 'saved_models/saved_sselector/i(77083)_epoch(7)_(tra_score:0.8841384138413841|raw_acc:1.0|pr:0.3964771477147341|rec:0.8262076207620762|f1:0.5358248492912955)_epoch',
    'sselector_2': config.PRO_ROOT / 'saved_models/saved_sselector/i(58915)_epoch(7)_(tra_score:0.8838383838383839|raw_acc:1.0|pr:0.39771352135209675|rec:0.8257575757575758|f1:0.5368577222846761)_epoch',

    'nn_doc_selector': config.PRO_ROOT / 'saved_models/nn_doc_selector/i(9000)_epoch(1)_(tra_score:0.9212421242124212|pr:0.4299679967996279|rec:0.8818631863186318|f1:0.5780819247968391)',
    'no_doc_nli': config.PRO_ROOT / 'saved_models/saved_nli_m/i(77000)_epoch(11)_dev(0.6601160116011601)_loss(1.1138329989302813)_seed(12)',

    'nli_ema_0': config.PRO_ROOT / "saved_models/ema_nli/ema_i(64500)_epoch(9)_dev(0.6601160116011601)_lacc(0.6968696869686969)_seed(12)"
}


# Sentence selection ensemble
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
        new_list[i]['prob'] = float(np.mean(prob_list))
        new_list[i]['score'] = float(np.mean(score_list))

    return new_list


id2label = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO",
}


# NLI ensemble
def merge_nli_results(nli_r_list):
    r_len = len(nli_r_list[0])
    for nli_r in nli_r_list:
        assert len(nli_r) == r_len

    new_list = copy.deepcopy(nli_r_list[0])
    logits_list = []
    for i in range(r_len):
        logits_current_logits_list = []
        for nli_r in nli_r_list:
            assert nli_r[i]['id'] == new_list[i]['id']
            logits_current_logits_list.append(np.asarray(nli_r[i]['logits'], dtype=np.float32))  # [(3)]
        logits_current_logits = np.stack(logits_current_logits_list, axis=0)  # [num, 3]
        current_mean_logits = np.mean(logits_current_logits, axis=0)  # [3]
        logits_list.append(current_mean_logits)

    logits = np.stack(logits_list, axis=0)  # (len, 3)
    y_ = np.argmax(logits, axis=1)  # (len)
    assert y_.shape[0] == len(new_list)

    for i in range(r_len):
        new_list[i]['predicted_label'] = id2label[y_[i]]

    return new_list


default_steps = {
    's1.tokenizing': {
        'do': True,
        'out_file': 'None'  # if false, we will directly use the out_file, This out file for downstream
    },
    's2.1doc_retri': {
        'do': True,
        'out_file': 'None'  # if false, we will directly use the out_file, for downstream
    },

    's2.2.1doc_nn_retri': {
        'do': True,
        'out_file': 'None'
    },

    's3.1sen_select': {
        'do': True,
        'out_file': 'None',
        'ensemble': True,
    },

    's4.2doc_retri': {
        'do': True,
        'out_file': 'None'
    },
    's5.2sen_select': {
        'do': True,
        'out_file': 'None'
    },
    's6.nli': {
        'do': True,
        'out_file': config.RESULT_PATH / "pipeline_r_aaai_doc/2018_09_02_17:11:42_r/nli_r_shared_task_dev_no_doc_scale:0.05.jsonl"
    }
}


class HAONAN_DOCRETRI_OBJECT:
    def __init__(self):
        self.instance = None


def init_haonan_docretri_object(object, method='pageview'):
    item_rb_selector = {
        'word_freq': ItemRuleBuilderTest,
        'pageview': ItemRuleBuilderSpiral,
        'nopageview': ItemRuleBuilderNoPageview,
    }[method]
    if object.instance is None:
        object.instance = DocRetrievalExperimentTwoStep(item_rb_selector())


def pipeline(in_file,
             out_file,
             working_dir,
             model_path_dict,
             steps):
    """
    :param in_file: The raw input file.
    :param eval_file: Whether to provide evaluation along the line.
    :return:
    """

    logs = LogHelper.get_logger("unc-pipeline")

    logs.info("Start pipeline")


    sentence_retri_2_scale_prob = 0.9
    sent_retri_2_top_k = 1

    enhance_retri_1_scale_prob = -1



    working_dir = Path(working_dir)

    if not working_dir.exists():
        working_dir.mkdir()

    logs.info("Current Result Root: {0}".format(working_dir))
    in_file = Path(in_file)
    in_file_stem = in_file.stem
    tokenized_file = working_dir / f"t_{in_file_stem}.jsonl"

    ## Tokenize claims for documents
    logs.info("Step 1. Tokenizing.")
    tokenized_claim(in_file, tokenized_file)

    logs.info("Step 6. NLI")
    sentences = common.load_jsonl(in_file)
    sentence_retri_1_scale_prob = 0.1
    logs.info("Threshold: {0}".format( sentence_retri_1_scale_prob))
    sent_select_results_list_1 = simi_sampler.threshold_sampler_insure_unique(tokenized_file, sentences,
                                                                              sentence_retri_1_scale_prob, top_n=5)


    nli_results = nli.mesim_wn_simi_v1_2.pipeline_nli_run(tokenized_file,
                                                          sent_select_results_list_1,
                                                          [sentences],
                                                          model_path_dict['no_doc_nli'],
                                                          with_logits=True,
                                                          with_probs=True,
                                                          load_from_dict=False)


    nli_results_file = working_dir / f"single_sent_nli_r_{in_file_stem}_with_doc_scale:{sentence_retri_1_scale_prob}_e0.jsonl"
    common.save_jsonl(nli_results, nli_results_file)

    for item in nli_results:
        del item['label']

    build_submission_file(nli_results, out_file)
    output_file = working_dir / "predictions.jsonl"
    build_submission_file(nli_results, output_file)


def pipeline_tokenize(in_file, out_file):
    tokenized_claim(in_file, out_file)


def first_doc_retrieval(retri_object, in_file, method='pageview', top_k=100):
    # doc_exp = DocRetrievalExperiment()
    logger = LogHelper.get_logger("doc-retrieval-1")
    logger.info("First doc")
    init_haonan_docretri_object(retri_object, method=method)
    d_list = common.load_jsonl(in_file)
    logger.info("Sample evidence")
    retri_object.instance.sample_answer_with_priority(d_list, top_k=top_k)
    return d_list


def second_doc_retrieval(retri_object, upstream_sent_file, additiona_d_list):
    # doc_exp = DocRetrievalExperimentSpiral()
    init_haonan_docretri_object(retri_object)
    # additiona_d_list = common.load_jsonl(in_file)
    retri_object.instance.feed_sent_file(upstream_sent_file)
    retri_object.instance.find_sent_link_with_priority(additiona_d_list, predict=True)
    return additiona_d_list


def nli_f(upstream_dev_file, additional_sent_file_list):
    pass


def append_hidden_label(d_list):
    for item in d_list:
        item['label'] = 'hidden'
    return d_list


def build_submission_file(d_list, filename):
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            instance_item = dict()
            instance_item['id'] = item['id']
            instance_item['predicted_label'] = item['predicted_label']
            instance_item['predicted_evidence'] = item['predicted_evidence']
            out_f.write(json.dumps(instance_item) + "\n")


# New method added
def delete_unused_evidence(d_list):
    for item in d_list:
        if item['predicted_label'] == 'NOT ENOUGH INFO':
            item['predicted_evidence'] = []


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--in-file",type=str, required=True)
    parser.add_argument("--out-file",type=str, required=True)
    parser.add_argument("--working-dir",type=str, required=True)
    args = parser.parse_args()
    pipeline(args.in_file,
             args.out_file,
             args.working_dir,
             model_path_dict=default_model_path_dict,
             steps=default_steps)
