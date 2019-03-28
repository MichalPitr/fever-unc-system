import copy
import json
import random

from utils import common
from utils import fever_db
from utils.sentence_utils import SENT_LINE, check_and_clean_evidence, Evidences


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def convert_evidence2scoring_format(predicted_sentids):
    e_list = predicted_sentids
    pred_evidence_list = []
    for i, cur_e in enumerate(e_list):
        doc_id = cur_e.split(SENT_LINE)[0]
        ln = cur_e.split(SENT_LINE)[1]
        pred_evidence_list.append([doc_id, int(ln)])
    return pred_evidence_list


def paired_selection_score_dict(sent_list, selection_dict=None):
    if selection_dict is None:
        selection_dict = dict()

    for item in sent_list:
        selection_id: str = item['selection_id']
        item_id: int = int(selection_id.split('<##>')[0])
        sentid: str = selection_id.split('<##>')[1]
        doc_id: str = sentid.split(SENT_LINE)[0]
        ln: int = int(sentid.split(SENT_LINE)[1])

        score: float = float(item['score'])
        prob: float = float(item['prob'])
        claim: str = item['query']

        ssid = (item_id, doc_id, ln)
        if ssid in selection_dict:
            assert claim == selection_dict[ssid]['claim']
            error_rate_prob = prob - float(selection_dict[ssid]['prob'])
            assert error_rate_prob < 0.01
        else:
            selection_dict[ssid] = dict()
            selection_dict[ssid]['score'] = score
            selection_dict[ssid]['prob'] = prob
            selection_dict[ssid]['claim'] = claim

    return selection_dict


def threshold_sampler_insure_unique_list(org_data_file, full_sent_list, prob_threshold=0.5, top_n=5):
    """
    Providing samples to the Training set by a probability threshold on the upstream selected sentences.
    """
    d_list = org_data_file
    augmented_dict = dict()
    for sent_item in full_sent_list:
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        remain_str = selection_id.split('<##>')[1]
        if org_id in augmented_dict:
            if remain_str not in augmented_dict[org_id]:
                augmented_dict[org_id][remain_str] = sent_item
            else:
                print("Exist")
        else:
            augmented_dict[org_id] = {remain_str: sent_item}

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # print("Potential error?")
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])].values()
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _, _ in item['scored_sentids']][:top_n]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def threshold_sampler_insure_unique_merge(org_data_file, full_sent_list, prob_threshold=0.5,
                                          top_n=5, add_n=1):
    """
    Providing samples to the Training set by a probability threshold on the upstream selected sentences.
    """
    if not isinstance(org_data_file, list):
        d_list = common.load_jsonl(org_data_file)
    else:
        d_list = org_data_file
    augmented_dict = dict()
    for sent_item in full_sent_list:
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        remain_str = selection_id.split('<##>')[1]
        if org_id in augmented_dict:
            if remain_str not in augmented_dict[org_id]:
                augmented_dict[org_id][remain_str] = sent_item
        else:
            augmented_dict[org_id] = {remain_str: sent_item}

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # print("Potential error?")
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])].values()
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        cur_predicted_sentids = cur_predicted_sentids[:add_n]

        # if item['scored_sentids']
        if len(item['predicted_sentids']) >= 5:
            continue
        else:
            item['predicted_sentids'].extend(
                [sid for sid, _, _ in cur_predicted_sentids if sid not in item['predicted_sentids']])
            item['predicted_sentids'] = item['predicted_sentids'][:top_n]
            item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])

        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def sample_additional_data_for_item_v1_0(item, additional_data_dictionary):
    res_sentids_list = []
    flags = []

    if item['verifiable'] == "VERIFIABLE":
        assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
        e_list = check_and_clean_evidence(item)
        current_id = item['id']
        assert current_id in additional_data_dictionary
        additional_data = additional_data_dictionary[current_id]['predicted_sentids']
        # additional_data_with_score = additional_data_dictionary[current_id]['scored_sentids']

        # print(len(additional_data))

        for evidences in e_list:
            # print(evidences)
            new_evidences = copy.deepcopy(evidences)
            n_e = len(evidences)
            if n_e < 5:
                current_sample_num = random.randint(0, 5 - n_e)
                random.shuffle(additional_data)
                for sampled_e in additional_data[:current_sample_num]:
                    doc_ids = sampled_e.split(SENT_LINE)[0]
                    ln = int(sampled_e.split(SENT_LINE)[1])
                    new_evidences.add_sent(doc_ids, ln)

            if new_evidences != evidences:
                flag = f"verifiable.non_eq.{len(new_evidences) - len(evidences)}"
                flags.append(flag)
                pass
            else:
                flag = "verifiable.eq.0"
                flags.append(flag)
                pass
            res_sentids_list.append(new_evidences)

        assert len(res_sentids_list) == len(e_list)

    elif item['verifiable'] == "NOT VERIFIABLE":
        assert item['label'] == 'NOT ENOUGH INFO'

        e_list = check_and_clean_evidence(item)
        current_id = item['id']
        additional_data = additional_data_dictionary[current_id]['predicted_sentids']
        # print(len(additional_data))
        random.shuffle(additional_data)
        current_sample_num = random.randint(2, 5)
        raw_evidences_list = []
        for sampled_e in additional_data[:current_sample_num]:
            doc_ids = sampled_e.split(SENT_LINE)[0]
            ln = int(sampled_e.split(SENT_LINE)[1])
            raw_evidences_list.append((doc_ids, ln))
        new_evidences = Evidences(raw_evidences_list)

        if len(new_evidences) == 0:
            flag = f"verifiable.eq.0"
            flags.append(flag)
            pass
        else:
            flag = f"not_verifiable.non_eq.{len(new_evidences)}"
            flags.append(flag)

        assert all(len(e) == 0 for e in e_list)
        res_sentids_list.append(new_evidences)
        assert len(res_sentids_list) == 1

    assert len(res_sentids_list) == len(flags)

    return res_sentids_list, flags


def evidence_list_to_text_list(cursor, evidences, contain_head=True):
    # One evidence one text and len(evidences) == len(text_list)
    current_evidence_text_list = []
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))

    cur_head = 'DO NOT INCLUDE THIS FLAG'

    for doc_id, line_num in evidences:

        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)

        cur_text = ""

        if contain_head and cur_head != doc_id:
            cur_head = doc_id

            t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

            if line_num != 0:
                cur_text = f"{t_doc_id_natural_format} <t> "

        # Important change move one line below: July 16
        # current_evidence_text.append(e_text)
        cur_text = cur_text + e_text

        current_evidence_text_list.append(cur_text)

    assert len(evidences) == len(current_evidence_text_list)
    return current_evidence_text_list


def select_sent_with_prob_for_eval_list(input_file, additional_file, prob_dict_file, cursor):
    """
    This method select sentences with upstream sentence retrieval.

    :param input_file: This should be the file with 5 sentences selected.
    :return:
    """

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = load_data(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    d_list = input_file

    for item in d_list:
        e_list = additional_data_dict[item['id']]['predicted_sentids']
        assert additional_data_dict[item['id']]['id'] == item['id']

        pred_evidence_list = []
        for i, cur_e in enumerate(e_list):
            doc_id = cur_e.split(SENT_LINE)[0]
            ln = int(cur_e.split(SENT_LINE)[1])  # Important changes Bugs: July 21
            pred_evidence_list.append((doc_id, ln))

        pred_evidence = Evidences(pred_evidence_list)

        evidence_text_list = evidence_list_to_text_list(cursor, pred_evidence, contain_head=True)

        evidences = sorted(pred_evidence, key=lambda x: (x[0], x[1]))
        item_id = int(item['id'])

        evidence_text_list_with_prob = []
        for text, (doc_id, ln) in zip(evidence_text_list, evidences):
            ssid = (item_id, doc_id, int(ln))
            if ssid not in prob_dict_file:
                print("Some sentence pair don't have 'prob'.")
                prob = 0.5
            else:
                prob = prob_dict_file[ssid]['prob']
                assert item['claim'] == prob_dict_file[ssid]['claim']

            evidence_text_list_with_prob.append((text, prob))

        item['evid'] = evidence_text_list_with_prob
        item['predicted_evidence'] = convert_evidence2scoring_format(e_list)
        item['predicted_sentids'] = e_list
        # This change need to be saved.
        # item['predicted_label'] = additional_data_dict[item['id']]['label']

    return d_list
