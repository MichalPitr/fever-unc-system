from utils import common
from utils import fever_db


def item_resorting(d_list, top_k=None):
    for item in d_list:
        item['predicted_docids'] = []

        # Reset Exact match
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        for k, it in enumerate(item['prioritized_docids']):
            if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
                item['prioritized_docids'][k] = [it[0], 5.0]
                item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids']:
                item['predicted_docids'].append(it[0])

        if top_k is not None and len(item['predicted_docids']) > top_k:
            item['predicted_docids'] = item['predicted_docids'][:top_k]


def item_remove_old_rule(d_list):
    for item in d_list:
        for i, (doc_id, priority) in enumerate(item['prioritized_docids']):
            if '-LRB-' in doc_id:  # Only use for disamb
                item['prioritized_docids'][i] = [doc_id, 1.0]


def trucate_item(d_list, top_k=None):
    for item in d_list:
        if top_k is not None and len(item['predicted_docids']) > top_k:
            item['predicted_docids'] = item['predicted_docids'][:top_k]


def sample_disamb_inference(d_list, cursor, contain_first_sentence=False):
    inference_list = []
    for item in d_list:
        inference_list.extend(inference_build(item, cursor, contain_first_sentence=contain_first_sentence))
    return inference_list


def inference_build(item, cursor, contain_first_sentence=False):
    doc_t_list = [it[0] for it in item['prioritized_docids']]
    # evidence_group = check_sentences.check_and_clean_evidence(item)
    t_claim = ' '.join(item['claim_tokens'])
    eid = item['id']

    b_list = []
    for doc_id in doc_t_list:
        if '-LRB-' in doc_id and common.doc_id_to_tokenized_text(doc_id) not in t_claim:
            item = dict()
            item['selection_id'] = str(eid) + '###' + str(doc_id)
            example = common.doc_id_to_tokenized_text(doc_id)
            description_sent = ''
            if contain_first_sentence:
                r_list, id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
                for sent, sent_id in zip(r_list, id_list):
                    if int(sent_id.split('(-.-)')[1]) == 0:
                        description_sent = sent

            item['query'] = example + ' ' + description_sent
            item['text'] = t_claim
            item['selection_label'] = 'hidden'

            b_list.append(item)

    return b_list


# with prob # important we used this function for final selection
def enforce_disabuigation_into_retrieval_result_v2(disabuigation_r_list, r_list, prob_sh=0.5):
    # Index by id and doc_id
    disabuigation_dict = dict()
    for item in disabuigation_r_list:
        disabuigation_dict[item['selection_id']] = item

    for item in r_list:
        the_id = item['id']
        for i, (doc_id, priority) in enumerate(item['prioritized_docids']):
            if '-LRB-' in doc_id:  # Only use for disamb
                query_id = str(the_id) + '###' + doc_id
                if query_id in disabuigation_dict:
                    query_selection = disabuigation_dict[query_id]
                    item['prioritized_docids'][i] = [doc_id, query_selection['prob']]

        # Reset Exact match
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        for k, it in enumerate(item['prioritized_docids']):
            if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
                item['prioritized_docids'][k] = [it[0], 5.0]
                if it[0] not in item['predicted_docids']:
                    item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids'] and it[1] >= prob_sh:
                item['predicted_docids'].append(it[0])
