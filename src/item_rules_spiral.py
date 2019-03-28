import json

import numpy as np

from doc_retrieval.item_rules import ItemRuleBuilderRawID
from doc_retrieval.wiki_pageview_utils import WikiPageviews
from utils import fever_db

__author__ = ['chaonan99']


class ItemRuleBuilderSpiral(ItemRuleBuilderRawID):
    def __init__(self, tokenizer, cursor, keyword_processor=None):
        super(ItemRuleBuilderSpiral, self).__init__(tokenizer, keyword_processor)
        self.cursor = cursor

    def initialize_item(self, item):
        self.item = item
        return self

    def exact_match_rule(self):
        item = self.item
        claim_tokens, claim_lemmas = self.get_token_lemma_from_claim(item['claim'])
        claim = ' '.join(claim_tokens)

        finded_keys_dict, finded_keys = self._keyword_match(claim, raw_set=True)

        item['prioritized_docids'] = list(finded_keys)
        item['structured_docids'] = finded_keys_dict
        item['claim_lemmas'] = claim_lemmas
        item['claim_tokens'] = claim_tokens
        item['processed_claim'] = claim
        self.item = item
        return self

    def expand_from_preext_sent_rule(self):
        item = self.item

        # if len(item['prioritized_docids']) < 5:
        new_pdocids = []
        structured_docids_sent = {}
        sent_ids = self.preext_sent_dict[item['id']]['scored_sentids']
        for sent_id, score, probability in sent_ids:
            docid, sent_ind = sent_id.split('<SENT_LINE>')
            sent_ind = int(sent_ind)
            id_list, sent_list, sent_links = fever_db.get_evidence(self.cursor, docid, sent_ind)
            sent_links = json.loads(sent_links)
            all_links = np.array(sent_links)
            all_links = np.array(all_links)
            all_links = all_links.reshape(-1, 2)[:, 1]
            all_links = list(map(fever_db.reverse_convert_brc, all_links))
            all_links = list(map(lambda x: x.replace(' ', '_'), all_links))
            prio_docids = [(id_link, score) for id_link in all_links]
            new_pdocids.extend(prio_docids)
            structured_docids_sent.update({sent_id: prio_docids})
        item['prioritized_docids_sent'] = new_pdocids
        item['structured_docids_sent'] = structured_docids_sent
        return self

    def spiral_high_priority_aside_rule(self):
        item = self.item
        sds = item['structured_docids_sent']
        item['structured_docids_aside'] = {}
        item['prioritized_docids_aside'] = []
        max_score = -100
        max_key = None
        for k, s in sds.items():
            if len(s) == 0:
                continue
            if s[0][1] > max_score:
                max_score = s[0][1]
                max_key = k
        if max_key is not None:
            item['structured_docids_aside'] = {max_key: sds[max_key]}
            item['prioritized_docids_aside'] = sds[max_key]
        return self

    def pageview_spiral_aside_rule(self):
        if not hasattr(self, 'wiki_pv'):
            print("Reload wiki pageview dict")
            self.wiki_pv = WikiPageviews()

        item = self.item
        changed = False
        for key, group_prio_docids in item['structured_docids_aside'].items():
            group_docids = [it[0] for it in group_prio_docids]
            if len(group_docids) > 1:
                changed = True
                all_scores = map(lambda x: self.wiki_pv[fever_db.convert_brc(x)],
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
                item['structured_docids_aside'][key] = new_gpd

        if changed:
            finded_keys = item['structured_docids_aside'].values()
            finded_keys = set([i for ii in finded_keys for i in ii]) \
                          if len(finded_keys) > 0 else set(finded_keys)
            item['prioritized_docids_aside'] = list(finded_keys)
        return self

    @property
    def first_only_rules(self):
        return lambda x: self.initialize_item(x)\
                             .exact_match_rule()\
                             .docid_based_rule()\
                             .singularize_rule()\
                             .eliminate_articles_rule()\
                             .pageview_rule()\

    @property
    def second_only_rules(self):
        return lambda x: self.initialize_item(x)\
                             .expand_from_preext_sent_rule()\
                             .spiral_high_priority_aside_rule()\
                             .pageview_spiral_aside_rule()\
