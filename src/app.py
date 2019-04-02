import datetime
from argparse import ArgumentParser

import torch
from flask import json
from tqdm import tqdm

import config
import logging

from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from fever.api.web_server import fever_web_api
from doc_retrieval.item_rules_spiral import ItemRuleBuilderSpiral
from nli import mesim_wn_simi_v1_2
from nli import simi_sampler
from nn_doc_retrieval import nn_doc_model
from nn_doc_retrieval.disabuigation_training import item_remove_old_rule, item_resorting, \
    enforce_disabuigation_into_retrieval_result_v2
from sentence_retrieval import simple_nnmodel
from sentence_retrieval.simple_nnmodel import get_score_multihop_list
from utils import text_clean, fever_db
from utils.common import load_model, merge_sent_results, delete_unused_evidence
from utils.data_utils.exvocab import load_vocab_embeddings
from utils.data_utils.fever_reader_with_wn_simi import WNSIMIReader
from utils.drqa.tokenizers import set_default
from utils.drqa.tokenizers.corenlp_tokenizer import CoreNLPTokenizer
from utils.wn_featurizer import wn_persistent_api

from logging.config import dictConfig


def fever_app(caller):
    logger = logging.getLogger()
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        },
        'allennlp': {
            'level': 'INFO',
            'handlers': ['wsgi']
        },
    })

    logger.info("Set up flask app")

    nn_doc_retri_threshold = 0.00001
    top_k = 100
    nn_doc_top_k = 10
    sent_prob_for_2doc = 0.1
    sent_topk_for_2doc = 5
    sentence_retri_1_scale_prob = 0.05
    sentence_retri_2_scale_prob = 0.9
    sent_retri_2_top_k = 1
    enhance_retri_1_scale_prob = -1

    def predict_pipeline(claims):
        # Step 1: Tokenization
        logger.info('Step 1')
        logger.info('Start: ' + str(datetime.datetime.now().time()))

        tokenized_list = []
        for idx, claim in enumerate(claims):
            claim_tok = ' '.join(tok.tokenize(text_clean.normalize(claim["claim"])).words())
            item_tokenized = {'id': idx, 'claim': claim_tok}
            tokenized_list.append(item_tokenized)
        logger.info('End: ' + str(datetime.datetime.now().time()))

        # Step 2: 1st Doc retrieval
        logger.info('Step 2')
        logger.info('Start: ' + str(datetime.datetime.now().time()))

        for item in tokenized_list:
            item_doc_retrieval = item
            item_rb.first_only_rules(item_doc_retrieval)
            item_doc_retrieval['predicted_docids'] = list(
                set([k for k, v in sorted(item_doc_retrieval['prioritized_docids'],
                                          key=lambda x: (-x[1], x[0]))][:top_k]))

        doc_retrieval_list = tokenized_list
        item_remove_old_rule(doc_retrieval_list)
        item_resorting(doc_retrieval_list)

        nn_doc_list = nn_doc_model.pipeline_function_list(doc_retrieval_list, doc_retrieval_model, vocab, cursor)
        enforce_disabuigation_into_retrieval_result_v2(nn_doc_list, doc_retrieval_list, prob_sh=nn_doc_retri_threshold)
        logger.info('End: ' + str(datetime.datetime.now().time()))

        # Step 3: 1st Sentence selection
        logger.info('Step 3')
        logger.info('Start: ' + str(datetime.datetime.now().time()))
        dev_sent_list_1_e0 = simple_nnmodel.pipeline_first_sent_selection_list(tokenized_list, doc_retrieval_list,
                                                                               sent_selector_model, vocab,
                                                                               top_k=nn_doc_top_k, cursor=cursor)
        dev_sent_list_1_e1 = simple_nnmodel.pipeline_first_sent_selection_list(tokenized_list, doc_retrieval_list,
                                                                               sent_selector_model_1, vocab,
                                                                               top_k=nn_doc_top_k, cursor=cursor)
        dev_sent_list_1_e2 = simple_nnmodel.pipeline_first_sent_selection_list(tokenized_list, doc_retrieval_list,
                                                                               sent_selector_model_2, vocab,
                                                                               top_k=nn_doc_top_k, cursor=cursor)
        dev_sent_list_1 = merge_sent_results([dev_sent_list_1_e0, dev_sent_list_1_e1, dev_sent_list_1_e2])
        filtered_dev_instance_1_for_doc2 = simi_sampler.threshold_sampler_insure_unique_list(tokenized_list,
                                                                                             dev_sent_list_1,
                                                                                             sent_prob_for_2doc,
                                                                                             top_n=sent_topk_for_2doc)
        dev_sent_1_list = simi_sampler.threshold_sampler_insure_unique_list(doc_retrieval_list, dev_sent_list_1,
                                                                            sentence_retri_1_scale_prob,
                                                                            top_n=sent_topk_for_2doc)
        logger.info('End: ' + str(datetime.datetime.now().time()))

        # Step 4: 2nd Doc retrieval
        logger.info('Step 4')
        logger.info('Start: ' + str(datetime.datetime.now().time()))
        item_rb.preext_sent_dict = {item['id']: item for item in filtered_dev_instance_1_for_doc2}

        for item in dev_sent_1_list:
            item_rb.second_only_rules(item)
            pids = [it[0] for it in item['prioritized_docids']]
            item['prioritized_docids_aside'] = [it for it in item['prioritized_docids_aside'] if it[0] not in pids]
            porg = set([k for k, v in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0]))][:top_k])
            paside = set([k for k, v in sorted(item['prioritized_docids_aside'], key=lambda x: (-x[1], x[0]))][:top_k])
            item['predicted_docids'] = list(porg | paside)
            item['predicted_docids_origin'] = list(porg)
            item['predicted_docids_aside'] = list(paside)

        logger.info('End: ' + str(datetime.datetime.now().time()))

        # Step 5: 2nd Sentence selection
        logger.info('Step 5')
        logger.info('Start: ' + str(datetime.datetime.now().time()))
        dev_sent_list_2 = get_score_multihop_list(tokenized_list, dev_sent_1_list, sent_selector_2_model, vocab, cursor)
        logger.info('End: ' + str(datetime.datetime.now().time()))

        # Step 6: NLI
        logger.info('Step 6')
        logger.info('Start: ' + str(datetime.datetime.now().time()))
        sentence_retri_nli_scale_prob = 0.1
        sent_select_results_list_1 = simi_sampler.threshold_sampler_insure_unique_list(tokenized_list, dev_sent_list_1,
                                                                                       sentence_retri_nli_scale_prob,
                                                                                       top_n=5)
        nli_results = mesim_wn_simi_v1_2.pipeline_nli_run_list(tokenized_list,
                                                               sent_select_results_list_1,
                                                               [dev_sent_list_1, dev_sent_list_2],
                                                               nli_model, vocab, dev_fever_data_reader, cursor)
        delete_unused_evidence(nli_results)

        nli_results = simi_sampler.threshold_sampler_insure_unique_merge(nli_results, dev_sent_list_2,
                                                                         sentence_retri_2_scale_prob,
                                                                         top_n=5, add_n=sent_retri_2_top_k)
        delete_unused_evidence(nli_results)

        nli_results = simi_sampler.threshold_sampler_insure_unique_merge(nli_results, dev_sent_list_1,
                                                                         enhance_retri_1_scale_prob,
                                                                         top_n=100, add_n=100)
        delete_unused_evidence(nli_results)

        predictions = []
        for final_item in nli_results:
            sentences = []
            for evidence in final_item['predicted_evidence']:
                sentences.extend((evidence[0], evidence[1]))
            prediction = final_item['predicted_label'].upper()
            predictions.append({"predicted_label":prediction,"predicted_evidence":sentences})
        logger.info('End: ' + str(datetime.datetime.now().time()))
        return predictions

    cursor = fever_db.get_cursor()


    tok = CoreNLPTokenizer(annotators=['pos', 'lemma'])
    item_rb = ItemRuleBuilderSpiral(tokenizer=tok, cursor=cursor)
    p_dict = wn_persistent_api.persistence_load()
    model_path_dict = {
        'sselector': config.DATA_ROOT / 'models/sent_selector',
        'sselector_1': config.DATA_ROOT / 'models/sent_selector_1',
        'sselector_2': config.DATA_ROOT / 'models/sent_selector_2',
        'nn_doc_selector': config.DATA_ROOT / 'models/nn_doc_selector',
        'no_doc_nli': config.DATA_ROOT / 'models/nli',
    }
    # Preload the NN models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")

    doc_retrieval_model = nn_doc_model.Model(weight=weight_dict['glove.840B.300d'],
                                             vocab_size=vocab.get_vocab_size('tokens'),
                                             embedding_dim=300, max_l=160, num_of_class=2)
    load_model(doc_retrieval_model, model_path_dict['nn_doc_selector'], device)

    sent_selector_model = simple_nnmodel.Model(weight=weight_dict['glove.840B.300d'],
                                               vocab_size=vocab.get_vocab_size('tokens'),
                                               embedding_dim=300, max_l=300, num_of_class=2)
    load_model(sent_selector_model, model_path_dict['sselector'], device)
    sent_selector_model_1 = simple_nnmodel.Model(weight=weight_dict['glove.840B.300d'],
                                                 vocab_size=vocab.get_vocab_size('tokens'),
                                                 embedding_dim=300, max_l=300, num_of_class=2)
    load_model(sent_selector_model_1, model_path_dict['sselector_1'], device)
    sent_selector_model_2 = simple_nnmodel.Model(weight=weight_dict['glove.840B.300d'],
                                                 vocab_size=vocab.get_vocab_size('tokens'),
                                                 embedding_dim=300, max_l=300, num_of_class=2)
    load_model(sent_selector_model_2, model_path_dict['sselector_2'], device)

    sent_selector_2_model = simple_nnmodel.Model(weight=weight_dict['glove.840B.300d'],
                                                 vocab_size=vocab.get_vocab_size('tokens'),
                                                 embedding_dim=300, max_l=300, num_of_class=2)
    load_model(sent_selector_2_model, model_path_dict['sselector'], device)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }
    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=True, wn_p_dict=p_dict, max_l=420)
    nli_model = mesim_wn_simi_v1_2.Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                                                      1024 + 450 + dev_fever_data_reader.wn_feature_size),
                                         rnn_size_out=(450, 450),
                                         weight=weight_dict['glove.840B.300d'],
                                         vocab_size=vocab.get_vocab_size('tokens'),
                                         mlp_d=900, embedding_dim=300, max_l=400)

    load_model(nli_model, model_path_dict['no_doc_nli'], device)
    logger.info('Finished loading models.')
    return caller(predict_pipeline)



def web():
    return fever_app(fever_web_api)


if __name__ == "__main__":
    call_method = None

    def cli_method(predict_function):
        global call_method
        call_method = predict_function

    def cli():
        return fever_app(cli_method)

    cli()

    parser = ArgumentParser()
    parser.add_argument("--in-file")
    parser.add_argument("--out-file")
    args = parser.parse_args()

    claims = []

    with open(args.in_file,"r") as in_file:
        for text_line in in_file:
            line = json.loads(text_line)
            claims.append(line)

    ret = call_method(claims)

    with open(args.out_file,"w+") as out_file:
        for prediction in ret:
            out_file.write(json.dumps(prediction)+"\n")