SENT_LINE = '<SENT_LINE>'

class Evidences(object):
    # Evidences is a list of docid and sentences line number
    def __init__(self, evidences):
        evidences_set = set()
        for doc_id, line_num in evidences:
            if doc_id is not None and line_num is not None:
                evidences_set.add((doc_id, line_num))

        evidences_list = sorted(evidences_set, key=lambda x: (x[0], x[1]))
        # print(evidences_list)
        self.evidences_list = evidences_list

    def add_sent(self, sent, ln):
        o_set = set(self.evidences_list)
        o_set.add((sent, ln))
        o_set = sorted(o_set, key=lambda x: (x[0], x[1]))
        self.evidences_list = list(o_set)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Evidences):
            return False

        if len(o.evidences_list) != len(self.evidences_list):
            return False

        is_eq = True
        for o, _e in zip(o.evidences_list, self.evidences_list):
            if o != _e:
                is_eq = False
                break

        return is_eq

    def __hash__(self) -> int:
        hash_str_list = []
        for doc_id, line_num in self.evidences_list:
            hash_str_list.append(f'{doc_id}###{line_num}')
        hash_str = '@'.join(hash_str_list)
        return hash_str.__hash__()

    def __repr__(self):
        return '{Evidences: ' + self.evidences_list.__repr__() + '}'

    def __len__(self):
        return self.evidences_list.__len__()

    def __iter__(self):
        return self.evidences_list.__iter__()


def check_and_clean_evidence(item):
    whole_annotators_evidences = item['evidence']
    # print(evidences)
    evidences_list_set = set()
    for one_annotator_evidences_list in whole_annotators_evidences:
        cleaned_one_annotator_evidences_list = []
        for evidence in one_annotator_evidences_list:
            docid, sent_num = evidence[-2], evidence[-1]
            # print(docid, sent_num)
            cleaned_one_annotator_evidences_list.append((docid, sent_num))

        one_annotator_evidences = Evidences(cleaned_one_annotator_evidences_list)
        evidences_list_set.add(one_annotator_evidences)

    return evidences_list_set
