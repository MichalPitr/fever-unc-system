import datetime
import torch
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from flask import Flask, request, jsonify

from fever.api.web_server import fever_web_api

import config
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
from utils.fever_db import create_db, build_sentences_table, create_sent_db, save_wiki_pages, check_document_id
from utils.wn_featurizer import wn_persistent_api


def fever():
    print("Start building wiki document database. This might take a while.")
    create_db(str(config.FEVER_DB))
    save_wiki_pages(str(config.FEVER_DB))
    create_sent_db(str(config.FEVER_DB))
    build_sentences_table(str(config.FEVER_DB))
    check_document_id(str(config.FEVER_DB))
    print("Wiki document database is ready.")


    nn_doc_retri_threshold = 0.00001
    top_k = 100
    nn_doc_top_k = 10
    sent_prob_for_2doc = 0.1
    sent_topk_for_2doc = 5
    sentence_retri_1_scale_prob = 0.05
    sentence_retri_2_scale_prob = 0.9
    sent_retri_2_top_k = 1
    enhance_retri_1_scale_prob = -1

    def predict_pipeline(claim):
        # Step 1: Tokenization
        print('Step 1')
        print('Start: ' + str(datetime.datetime.now().time()))
        claim_tok = ' '.join(tok.tokenize(text_clean.normalize(claim)).words())
        item_tokenized = {'id': 0, 'claim': claim_tok}
        tokenized_list = [item_tokenized]
        print('End: ' + str(datetime.datetime.now().time()))

        # Step 2: 1st Doc retrieval
        print('Step 2')
        print('Start: ' + str(datetime.datetime.now().time()))
        item_doc_retrieval = {'id': 0, 'claim': claim_tok}
        item_rb.first_only_rules(item_doc_retrieval)
        item_doc_retrieval['predicted_docids'] = list(
            set([k for k, v in sorted(item_doc_retrieval['prioritized_docids'],
                                      key=lambda x: (-x[1], x[0]))][:top_k]))
        doc_retrieval_list = [item_doc_retrieval]
        item_remove_old_rule(doc_retrieval_list)
        item_resorting(doc_retrieval_list)

        nn_doc_list = nn_doc_model.pipeline_function_list(doc_retrieval_list, doc_retrieval_model, vocab, cursor)
        enforce_disabuigation_into_retrieval_result_v2(nn_doc_list, doc_retrieval_list, prob_sh=nn_doc_retri_threshold)
        print('End: ' + str(datetime.datetime.now().time()))

        # Step 3: 1st Sentence selection
        print('Step 3')
        print('Start: ' + str(datetime.datetime.now().time()))
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
        print('End: ' + str(datetime.datetime.now().time()))

        # Step 4: 2nd Doc retrieval
        print('Step 4')
        print('Start: ' + str(datetime.datetime.now().time()))
        item_rb.preext_sent_dict = {item['id']: item for item in filtered_dev_instance_1_for_doc2}
        item = dev_sent_1_list[0]
        item_rb.second_only_rules(item)
        pids = [it[0] for it in item['prioritized_docids']]
        item['prioritized_docids_aside'] = [it for it in item['prioritized_docids_aside'] if it[0] not in pids]
        porg = set([k for k, v in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0]))][:top_k])
        paside = set([k for k, v in sorted(item['prioritized_docids_aside'], key=lambda x: (-x[1], x[0]))][:top_k])
        item['predicted_docids'] = list(porg | paside)
        item['predicted_docids_origin'] = list(porg)
        item['predicted_docids_aside'] = list(paside)
        dev_doc_2_list = [item]
        print('End: ' + str(datetime.datetime.now().time()))

        # Step 5: 2nd Sentence selection
        print('Step 5')
        print('Start: ' + str(datetime.datetime.now().time()))
        dev_sent_list_2 = get_score_multihop_list(tokenized_list, dev_doc_2_list, sent_selector_2_model, vocab, cursor)
        print('End: ' + str(datetime.datetime.now().time()))

        # Step 6: NLI
        print('Step 6')
        print('Start: ' + str(datetime.datetime.now().time()))
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

        final_item = nli_results[0]
        sentences = []
        for evidence in final_item['predicted_evidence']:
            _, e_text, _ = fever_db.get_evidence(cursor, evidence[0], evidence[1])
            sentences.append((evidence, e_text))
        prediction = final_item['predicted_label'].upper()

        print('End: ' + str(datetime.datetime.now().time()))
        return prediction, sentences

    def batch_predict(instances):
        predictions = []
        for instance in instances:
            prediction, sentences = predict_pipeline(instance['claim'])
            # [(page, lineId), ...]
            prediction.append({"predicted_label": prediction, "predicted_evidence": sentences})
        return predictions

    cursor = fever_db.get_cursor()

    path_stanford_corenlp_full_2017_06_09 = str(config.DATA_ROOT / 'stanford-corenlp/*')
    set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
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

    print('Finished loading models.')

    return fever_web_api(batch_predict)


