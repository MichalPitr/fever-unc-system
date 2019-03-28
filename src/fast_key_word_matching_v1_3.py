import copy
import json

from tqdm import tqdm

from utils import common
from utils import text_clean

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``",
    ',', '.', ':'
}


def load_keyword_dict_v1_3(in_filename, filtering=False):
    # COLON cleaned
    id_to_key_dict = dict()
    with open(in_filename, encoding='utf-8', mode='r') as in_f:
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            if filtering and text_clean.filter_document_id(item['docid']):
                continue
            # id_to_key_dict[item['docid']] = item['keys']
            # This is a list of keys:
            id_to_key_dict[item['docid']] = [common.doc_id_to_tokenized_text(item['docid'])]

    return id_to_key_dict


def set_priority(id_to_key_dict, priority):
    """
    {'document_id': [key_trigger_words]}
    -> {'document_id': [(key_trigger_words, priority)]}
    :param priority: The priority
    :param id_to_key_dict:
    :return:
    """
    prioritied_dict = dict()
    for doc_id, keys in id_to_key_dict.items():
        prioritied_dict[doc_id] = [(key, priority) for key in keys]

    return prioritied_dict


def build_flashtext_processor_with_prioritized_kw_dict(keyword_processor, keyword_dict):
    """
    This method convert keyword dictionary to a flashtext consumeable format and build the keyword_processor
    :param keyword_processor:
    :param keyword_dict:
    { 'document_id' : [(key_trigger_words, priority)]

    flashtext consumeable
    { 'key_trigger_word', set((document_id, priority))}
    :return:
    """
    for doc_id, kwords in tqdm(keyword_dict.items()):
        for kw, priority in kwords:
            if kw in keyword_processor:
                # If doc_id exist:
                found = False

                for exist_doc_id, exist_priority in keyword_processor[kw]:
                    if exist_doc_id == doc_id:
                        # Update the priority by remove the old and add the new
                        keyword_processor[kw].remove((exist_doc_id, exist_priority)) # Remove original
                        keyword_processor[kw].add((doc_id, max(exist_priority, priority))) # Add new
                        found = True
                        break

                # If doc_id not found
                if not found:
                    keyword_processor[kw].add((doc_id, priority))

            else:
                keyword_processor.add_keyword(kw, {(doc_id, priority)})


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list

parentheses_dict = [
    ('-LRB-', '-RRB-'),
    ('-LSB-', '-RSB-'),
    ('-LCB-', '-RCB-'),
    ('-lrb-', '-rrb-'),
    ('-lsb-', '-rsb-'),
    ('-lcb-', '-rcb-')
]


def get_words_inside_parenthese(seq):
    r_list = []
    stacks = [[] for _ in parentheses_dict]

    for t in seq:
        jump_to_next = False
        for i, (l_s, r_s) in enumerate(parentheses_dict):
            if t == l_s:
                stacks[i].append(l_s)
                jump_to_next = True
            elif t == r_s:
                stacks[i].pop()
                jump_to_next = True

        if not jump_to_next and any([len(stack) != 0 for stack in stacks]):
            r_list.append(t)

    return r_list


def check_parentheses(seq):
    stacks = [[] for _ in parentheses_dict]
    for t in seq:
        for i, (l_s, r_s) in enumerate(parentheses_dict):
            if t == l_s:
                stacks[i].append(l_s)
            elif t == r_s:
                if len(stacks[i]) <= 0:
                    # print(seq)
                    return False
                stacks[i].pop()

    valid = True
    for stack in stacks:
        if len(stack) != 0:
            valid = False
            break

    return valid


def remove_parentheses(seq):
    new_seq = []
    stacks = [[] for _ in parentheses_dict]

    for t in seq:
        jump_to_next = False
        for i, (l_s, r_s) in enumerate(parentheses_dict):
            if t == l_s:
                stacks[i].append(l_s)
                jump_to_next = True
            elif t == r_s:
                stacks[i].pop()
                jump_to_next = True

        if not jump_to_next and all([len(stack) == 0 for stack in stacks]):
            new_seq.append(t)

    if new_seq == seq:
        return []

    return new_seq


def id_dict_key_word_expand(id_to_key_dict, create_new_key_word_dict=False):
    """
    :param id_to_key_dict: Original key word dictionary:
    { 'document_id' : [key_trigger_words] }
    :param create_new_key_word_dict: Whether to create a new dictionary or just expand the original one.

    :return: { 'document_id' : [key_trigger_words] } with key word without parentheses in the list.
    """
    if not create_new_key_word_dict:
        for k, v in tqdm(id_to_key_dict.items()):
            if 'disambiguation' in k:   # Removing all disambiguation pages
                continue

            org_keys = copy.deepcopy(v)
            for o_key in org_keys:
                key_t_list = o_key.split(' ')

                if not check_parentheses(key_t_list):
                    pass
                    # print("Pass:", key_t_list)
                    # print()
                else:
                    new_key_t_list = remove_parentheses(key_t_list)
                    if len(new_key_t_list) != 0:
                        id_to_key_dict[k].append(' '.join(new_key_t_list))

            if len(id_to_key_dict[k]) > 1:
                # pass
                # if verbose:
                print(k, id_to_key_dict[k])

        return None
    else:
        new_kw_dict = dict()

        for k, v in tqdm(id_to_key_dict.items()):
            if 'disambiguation' in k:   # Removing all disambiguation pages
                continue

            org_keys = copy.deepcopy(v)
            for o_key in org_keys:
                key_t_list = o_key.split(' ')

                if not check_parentheses(key_t_list):
                    pass
                    # print("Pass:", key_t_list)
                    # print()
                else:
                    new_key_t_list = remove_parentheses(key_t_list)
                    if len(new_key_t_list) != 0:
                        new_kw_dict[k] = [' '.join(new_key_t_list)]
                        # print(k, new_kw_dict[k])

        return new_kw_dict

def check_inside_paretheses_overlap(doc_id_tokens, doc_id_lemmas, claim_tokens, claim_lemmas):
    p_did_tokens = get_words_inside_parenthese(doc_id_tokens)
    p_did_tokens_set = set([t.lower() for t in p_did_tokens]) - set(STOPWORDS)
    p_did_lemmas = get_words_inside_parenthese(doc_id_lemmas)
    p_did_lemmas_set = set([t.lower() for t in p_did_lemmas]) - set(STOPWORDS)

    claim_tokens_set = set([t.lower() for t in claim_tokens]) - set(STOPWORDS)
    claim_lemmas_set = set([t.lower() for t in claim_lemmas]) - set(STOPWORDS)

    if len(set.intersection(p_did_tokens_set, claim_tokens_set)) > 0:
        return 3.0
    elif len(set.intersection(p_did_lemmas_set, claim_lemmas_set)) > 0:
        return 2.0
    else:
        return 1.0

