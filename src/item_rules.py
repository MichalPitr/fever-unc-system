import unicodedata
from copy import copy

import inflection
import numpy as np
from flashtext import KeywordProcessor

import config
from doc_retrieval.doc_utils import DocIDTokenizer
from doc_retrieval.fast_key_word_matching_v1_3 import \
    build_flashtext_processor_with_prioritized_kw_dict as build_processor
from doc_retrieval.fast_key_word_matching_v1_3 import get_words_inside_parenthese as extract_par
from doc_retrieval.fast_key_word_matching_v1_3 import id_dict_key_word_expand, set_priority, \
    check_inside_paretheses_overlap, load_keyword_dict_v1_3
from doc_retrieval.google_querier import GoogleQuerier
from doc_retrieval.wiki_pageview_utils import WikiPageviews
from utils.fever_db import convert_brc
from utils.text_clean import STOPWORDS

__author__ = ['chaonan99', 'yixin1']


class KeywordRuleBuilder(object):
    """KeywordRuleBuilder applies post processing rules on keyword processor """
    @classmethod
    def __essential_remove(cls, keyword_processor, remove_str):
        remove_set = copy(keyword_processor[remove_str])
        if remove_set is not None:
            result_set = set([it for it in remove_set if it[0] != remove_str])
            if len(result_set) == 0:
                keyword_processor.remove_keyword(remove_str)
            else:
                keyword_processor[remove_str] = result_set

    @classmethod
    def eliminate_pure_digits_in_place(cls, keyword_processor):
        for i in range(100000):
            cls.__essential_remove(keyword_processor, str(i))

    @classmethod
    def eliminate_ordinals_in_place(cls, keyword_processor):
        for i in range(1000):
            cls.__essential_remove(keyword_processor, inflection.ordinalize(i))

    @classmethod
    def eliminate_stop_words_in_place(cls, keyword_processor):
        for w in STOPWORDS:
            cls.__essential_remove(keyword_processor, w)
            cls.__essential_remove(keyword_processor, w.title())


class DocIDRuleBuilder(object):
    """DocIDRuleBuilder contains docid based rules, including parentheses
    overlapping, enhancement for film, TV series, ... and others
    """
    def __init__(self, claim_tokens, claim_lemmas):
        self.claim_tokens = claim_tokens
        self.claim_lemmas = claim_lemmas

    def tokenize_docid(self, id_prio_tuple, docid_tokenizer):
        self.doc_id, self.priority = id_prio_tuple
        self.doc_id_tokens, self.doc_id_lemmas = \
            docid_tokenizer.tokenize_docid(self.doc_id.lower())
        return self

    def parentheses_overlap_rule(self):
        if self.priority == 1.0:
            self.priority = check_inside_paretheses_overlap(self.doc_id_tokens,
                                                            self.doc_id_lemmas,
                                                            self.claim_tokens,
                                                            self.claim_lemmas)
        return self

    def common_word_rule(self):
        addup_score = 0.2 if 'film'  in extract_par(self.doc_id_tokens) \
                 else 0.1 if 'album' in extract_par(self.doc_id_tokens) \
                          or 'TV'    in extract_par(self.doc_id_tokens) \
                 else 0
        self.priority += addup_score
        return self

    @property
    def id_prio_tuple(self):
        return (self.doc_id, self.priority)


class ItemRuleBuilderBase:
    """ItemRuleBuilderBase is the base class for item rule builder """
    def __init__(self, tokenizer, keyword_processor=None):
        self.tokenizer = tokenizer
        self.keyword_processor = self.build_kp() if keyword_processor is None else keyword_processor

    def _build_kp(self, case_sensitive=True):
        ## Prepare tokenizer and flashtext keyword processor
        keyword_processor = KeywordProcessor(case_sensitive=case_sensitive)
        id_to_key_dict = load_keyword_dict_v1_3(config.DATA_ROOT / "id_dict.jsonl", filtering=True)
        exact_match_rule_dict = set_priority(id_to_key_dict, priority=5.0)
        noisy_key_dict = id_dict_key_word_expand(id_to_key_dict, create_new_key_word_dict=True)
        noisy_parenthese_rule_dict = set_priority(noisy_key_dict, priority=1.0)

        build_processor(keyword_processor, exact_match_rule_dict)
        build_processor(keyword_processor, noisy_parenthese_rule_dict)

        ## Change priorities of digital numbers
        KeywordRuleBuilder.eliminate_pure_digits_in_place(keyword_processor)
        KeywordRuleBuilder.eliminate_ordinals_in_place(keyword_processor)
        KeywordRuleBuilder.eliminate_stop_words_in_place(keyword_processor)

        return keyword_processor

    def build_kp(self, case_sensitive=True):
        return self._build_kp(case_sensitive)

    def _keyword_match(self, claim, raw_set=False, custom_kp=None):
        kp = self.keyword_processor if custom_kp is None else custom_kp
        finded_keys = kp.extract_keywords(claim)
        if isinstance(finded_keys, list) and len(finded_keys) != 0:
            finded_keys = set.union(*finded_keys)
        return finded_keys

    def normalize(self, text):
        """Resolve different type of unicode encodings."""
        return unicodedata.normalize('NFD', text)

    def get_token_lemma_from_claim(self, claim):
        claim_norm = self.normalize(claim)
        claim_tok_r = self.tokenizer.tokenize(claim_norm)
        claim_tokens = claim_tok_r.words()
        claim_lemmas = claim_tok_r.lemmas()
        return claim_tokens, claim_lemmas

    @classmethod
    def get_all_docid_in_evidence(cls, evidence):
        return [iii for i in evidence for ii in i for iii in ii if type(iii) == str]

    @property
    def rules(self):
        return lambda x: x


class ItemRuleBuilder(ItemRuleBuilderBase):
    """ItemRuleBuilder contains basic document retrieval rules """
    def __init__(self, tokenizer, keyword_processor=None):
        super().__init__(tokenizer, keyword_processor)
        self.docid_tokenizer = DocIDTokenizer(case_insensitive=True)
        self.google_querier = GoogleQuerier(self.keyword_processor)

    def exact_match_rule(self, item):
        claim_tokens, claim_lemmas = \
            self.get_token_lemma_from_claim(item['claim'])
        claim = ' '.join(claim_tokens)

        finded_keys = self._keyword_match(claim)

        item['prioritized_docids'] = list(finded_keys)
        item['claim_lemmas'] = claim_lemmas
        item['claim_tokens'] = claim_tokens
        item['processed_claim'] = claim
        self.item = item
        return self

    def docid_based_rule(self):
        item = self.item
        assert 'prioritized_docids' in item, 'Apply exact match rule first!'
        for i, id_prio_tuple in enumerate(item['prioritized_docids']):
            docid_rule_builder = DocIDRuleBuilder(item['claim_tokens'],
                                                  item['claim_lemmas'])
            docid_rule_builder.tokenize_docid(id_prio_tuple,
                                              self.docid_tokenizer)\
                              .parentheses_overlap_rule()\
                              .common_word_rule()
            item['prioritized_docids'][i] = docid_rule_builder.id_prio_tuple
        return self

    def eliminate_the_rule(self):
        return self.eliminate_start_words_rule(starts=['The'])

    def eliminate_articles_rule(self):
        return self.eliminate_start_words_rule(starts=['The', 'A', 'An'])

    def eliminate_start_words_rule(self, starts=['The'], modify_pdocid=False):
        item = self.item
        claim_tokens = copy(item['claim_tokens'])
        finded_keys  = item['prioritized_docids']
        if claim_tokens[0] in starts:
            claim_tokens[1] = claim_tokens[1].title()
            claim = ' '.join(claim_tokens[1:])
            fk_new = self._keyword_match(claim)
            finded_keys = set(finded_keys) | set(fk_new)
            item['prioritized_docids'] = list(finded_keys)
        return self

    def singularize_rule(self):
        """Singularize words
        """
        item = self.item
        if len(item['prioritized_docids']) < 1:
            claim_tokens = item['claim_tokens']
            # finded_keys  = item['prioritized_docids']
            claim_tokens = [inflection.singularize(c) for c in claim_tokens]
            claim = ' '.join(claim_tokens)
            fk_new = self._keyword_match(claim)
            # finded_keys = set(finded_keys) | set(fk_new)
            item['prioritized_docids'] = list(fk_new)
        return self

    def google_query_rule(self):
        item = self.item
        docid_dict = {k: v for k, v in item['prioritized_docids']}
        esn = sum([v > 1 for v in docid_dict.values()])
        if (len(item['prioritized_docids']) > 15 and esn < 5) or len(item['prioritized_docids']) < 1:
            self.google_querier.get_google_docid(item)
            for k in item['google_docids']:
                docid_dict[k] = 6.0
            item['prioritized_docids'] = [(k, v) for k, v in docid_dict.items()]
        return self


class ItemRuleBuilderRawID(ItemRuleBuilder):
    def __init__(self, tokenizer, keyword_processor=None):
        super(ItemRuleBuilderRawID, self).__init__(tokenizer, keyword_processor)
        self.wiki_pv = WikiPageviews()

    def build_kp(self, case_sensitive=False):
        return self._build_kp(case_sensitive)

    def _recursive_key_matcher(self, claim, fkd=None):
        fkd = {} if fkd is None else fkd
        finded_keys_dict = self.google_querier.get_keywords(claim)

        for key, value in finded_keys_dict.items():
            if key.lower() in STOPWORDS:
                continue

            ## First letter is case sensitive
            value = [v for v in value if v[0][0] == key[0]]

            if len(value) == 0:
                ## This seems to be a bug, for example, for claim
                ## "The hero of the Odyssey is Harry Potter.", it will first
                ## match "the Odyssey" and failed to match first letter
                ## "The_Odyssey" and "the Odyssey" will be ignored.
                ## But 2 actually performs best on dev so we just keep it here
                key_tokens = key.split(' ')
                if len(key_tokens) > 2:
                    key_tokens[1] = key_tokens[1].title()
                    self._recursive_key_matcher(' '.join(key_tokens[1:]), fkd)
            else:
                fkd.update({key:value})

        return fkd

    def _keyword_match(self, claim, raw_set=False, custom_kp=None):
        kp = self.keyword_processor if custom_kp is None else custom_kp
        if not raw_set:
            finded_keys = kp.extract_keywords(claim)
            if isinstance(finded_keys, list) and len(finded_keys) != 0:
                finded_keys = set.union(*finded_keys)
            return finded_keys
        else:
            finded_keys_dict = self.google_querier.get_keywords(claim)
            finded_keys_dict = self._recursive_key_matcher(claim)
            finded_keys = finded_keys_dict.values()
            finded_keys = set([i for ii in finded_keys for i in ii]) \
                          if len(finded_keys) > 0 else set(finded_keys)

            return finded_keys_dict, finded_keys

    def google_query_rule(self):
        item = self.item
        if len(item['prioritized_docids']) > 40:
            # all_keys = item['structured_docids'].keys()
            item['google_docids'] = []
            matched_docid = self.google_querier\
                .google_it(item['processed_claim'])
            # item['prioritized_docids'].append((matched_docid, 6.0))
            # Consume redundent keywords
            print(item['processed_claim'])
            print(matched_docid)
            if matched_docid is not None:
                fkd_new = {}
                key_remains = []
                for key, value in item['structured_docids'].items():
                    key_tokens = key.split(' ')
                    if not np.all(list(map(lambda x: x in matched_docid,
                                           key_tokens))):
                        key_remains.append(key)
                        fkd_new.update({key: value})
                item['structured_docids'] = fkd_new
                finded_keys = fkd_new.values()
                finded_keys = set([i for ii in finded_keys for i in ii]) \
                              if len(finded_keys) > 0 else set(finded_keys)
                item['prioritized_docids'] = list(finded_keys)
                item['prioritized_docids'].append((matched_docid, 6.0))
        return self

    def test_recursive_match(self, claim):
        return self._recursive_key_matcher(claim)

    def pageview_rule(self):
        """Assign high priority to frequently viewed pages
        """
        if not hasattr(self, 'wiki_pv'):
            print("Reload wiki pageview dict")
            self.wiki_pv = WikiPageviews()

        item = self.item
        docid_groups = [[i[0] for i in it] \
                        for _, it in item['structured_docids'].items()]
        changed = False
        for key, group_prio_docids in item['structured_docids'].items():
            group_docids = [it[0] for it in group_prio_docids]
            if len(group_docids) > 1:
                changed = True
                all_scores = map(lambda x: self.wiki_pv[convert_brc(x)],
                                 group_docids)
                all_scores = np.array(list(all_scores))
                prios = np.argsort(all_scores)[::-1]
                new_gpd = []
                for i, p in enumerate(prios):
                    # new_gpd.append((group_prio_docids[p][0],
                    #                 group_prio_docids[p][1] + \
                    #                     max(1.0 - i*0.2, 0)))
                    new_gpd.append((group_prio_docids[p][0],
                                    max(1.0 - i*0.2, 0)))
                item['structured_docids'][key] = new_gpd

        if changed:
            finded_keys = item['structured_docids'].values()
            finded_keys = set([i for ii in finded_keys for i in ii]) \
                          if len(finded_keys) > 0 else set(finded_keys)
            item['prioritized_docids'] = list(finded_keys)
        return self

    def eliminate_start_words_rule(self, starts=['The']):
        item = self.item
        claim_tokens = copy(item['claim_tokens'])
        finded_keys  = item['prioritized_docids']
        if claim_tokens[0] in starts:
            claim_tokens[1] = claim_tokens[1].title()
            claim = ' '.join(claim_tokens[1:])
            fkd_new, fk_new = self._keyword_match(claim, raw_set=True)
            finded_keys = set(finded_keys) | set(fk_new)
            item['structured_docids'].update(fkd_new)
            item['prioritized_docids'] = list(finded_keys)
        return self

    def singularize_rule(self):
        item = self.item
        if len(item['prioritized_docids']) < 1:
            claim_tokens = item['claim_tokens']
            # finded_keys  = item['prioritized_docids']
            claim_tokens = [inflection.singularize(c) for c in claim_tokens]
            claim = ' '.join(claim_tokens)
            fkd_new, fk_new = self._keyword_match(claim, raw_set=True)
            # finded_keys = set(finded_keys) | set(fk_new)
            item['prioritized_docids'] = list(fk_new)
            item['structured_docids'] = fkd_new
        return self
