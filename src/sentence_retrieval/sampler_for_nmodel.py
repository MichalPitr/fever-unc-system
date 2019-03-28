"""
This file contains scripts to build or sample data for neural sentence selector.

Neural sentence selector aimed to fine-select sentence for NLI models since NLI models are sensitive to data.
"""

import itertools

from nn_doc_retrieval.disabuigation_training import trucate_item
from utils import fever_db, common
from utils.sentence_utils import check_and_clean_evidence, SENT_LINE


def get_full_list_from_list_d(tokenized_data_file, additional_data_file, cursor, pred=False, top_k=None):
    """
    This method will select all the sentence from upstream doc retrieval and label the correct evident as true
    :param tokenized_data_file: Remember this is tokenized data with original format containing 'evidence'
    :param additional_data_file:    This is the data after document retrieval.
                                    This file need to contain *"predicted_docids"* field.
    :return:
    """
    d_list = tokenized_data_file

    additional_d_list = additional_data_file

    if top_k is not None:
        trucate_item(additional_d_list, top_k=top_k)

    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    full_data_list = []

    for item in d_list:
        doc_ids = additional_data_dict[item['id']]["predicted_docids"]

        if not pred:
            if item['evidence'] is not None:
                e_list = check_and_clean_evidence(item)
                all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))
            else:
                all_evidence_set = None
            # print(all_evidence_set)
            r_list = []
            id_list = []

            if all_evidence_set is not None:
                for doc_id, ln in all_evidence_set:
                    _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
                    r_list.append(text)
                    id_list.append(doc_id + '(-.-)' + str(ln))

        else:            # If pred, then reset to not containing ground truth evidence.
            all_evidence_set = None
            r_list = []
            id_list = []

        for doc_id in doc_ids:
            cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
            # Merging to data list and removing duplicate
            for i in range(len(cur_r_list)):
                if cur_id_list[i] in id_list:
                    continue
                else:
                    r_list.append(cur_r_list[i])
                    id_list.append(cur_id_list[i])

        assert len(id_list) == len(set(id_list))  # check duplicate
        assert len(r_list) == len(id_list)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        # sorted(evidences_set, key=lambda x: (x[0], x[1]))
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set)
        cur_id = item['id']
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
            sent_item['query'] = item['claim']
            full_data_list.append(sent_item)

    return full_data_list


def get_additional_list_list(d_list, additional_d_list, cursor, item_key='prioritized_docids_aside', top_k=6):
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[int(add_item['id'])] = add_item

    full_data_list = []

    for item in d_list:
        doc_ids_p_list = additional_data_dict[int(item['id'])][item_key]
        doc_ids = list(set([k for k, v in sorted(doc_ids_p_list, key=lambda x: (-x[1], x[0]))][:top_k]))

        all_evidence_set = None
        r_list = []
        id_list = []

        for doc_id in doc_ids:
            cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
            # Merging to data list and removing duplicate
            for i in range(len(cur_r_list)):
                if cur_id_list[i] in id_list:
                    continue
                else:
                    r_list.append(cur_r_list[i])
                    id_list.append(cur_id_list[i])

        assert len(id_list) == len(set(id_list))  # check duplicate
        assert len(r_list) == len(id_list)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set)
        cur_id = item['id']
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
            # selection_id is '[item_id<##>[doc_id]<SENT_LINE>[line_number]'
            sent_item['query'] = item['claim']
            full_data_list.append(sent_item)

    return full_data_list


def convert_to_formatted_sent(zipped_s_id_list, evidence_set):
    sent_list = []
    for sent, sid in zipped_s_id_list:
        sent_item = dict()

        cur_sent = sent
        doc_id, ln = sid.split('(-.-)')[0], int(sid.split('(-.-)')[1])

        t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

        if ln != 0 and t_doc_id_natural_format.lower() not in sent.lower():
            cur_sent = f"{t_doc_id_natural_format} <t> " + sent

        sent_item['text'] = cur_sent
        sent_item['sid'] = doc_id + SENT_LINE + str(ln)
        # sid is '[doc_id]<SENT_LINE>[line_number]'
        if evidence_set is not None:
            if (doc_id, ln) in evidence_set:
                sent_item['selection_label'] = "true"
            else:
                sent_item['selection_label'] = "false"
        else:
            sent_item['selection_label'] = "hidden"

        sent_list.append(sent_item)

    return sent_list

