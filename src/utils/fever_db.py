import sqlite3
from utils import text_clean
from tqdm import tqdm
import config
import json
import re


# Write some standard API for query information

def get_cursor(save_path=str(config.FEVER_DB)):
    conn = sqlite3.connect(save_path, check_same_thread=False)
    cursor = conn.cursor()
    return cursor


def get_evidence(cursor, doc_id, line_num):
    key = f'{doc_id}(-.-){line_num}'
    # print("SELECT * FROM sentences WHERE id = \"%s\"" % key)
    cursor.execute("SELECT * FROM sentences WHERE id=?", (key,))
    fetched_data = cursor.fetchone()
    if fetched_data is not None:
        _id, text, h_links, doc_id = fetched_data
    else:
        _id, text, h_links, doc_id = None, None, None, None
    return _id, text, h_links


def get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False):
    cursor.execute("SELECT * FROM sentences WHERE doc_id=?", (doc_id, ))
    fetched_data = cursor.fetchall()
    r_list = []
    id_list = []
    h_links_list = []
    for id, text, h_links, doc_id in fetched_data:
        # print(id, text, h_li
        # nks, doc_id)
        r_list.append(text)
        id_list.append(id)
        h_links_list.append(json.loads(h_links))

    if with_h_links:
        return r_list, id_list, h_links_list
    else:
        return r_list, id_list
# API Ends


    # convert_special
def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string


def reverse_convert_brc(string):
    string = re.sub('\(', '-LRB-', string)
    string = re.sub('\)', '-RRB-', string)
    string = re.sub('\[', '-LSB-', string)
    string = re.sub(']', '-RSB-', string)
    string = re.sub('{', '-LCB-', string)
    string = re.sub('}', '-RCB-', string)
    string = re.sub(':', '-COLON-', string)
    return string


def create_db(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text, lines_json);")


def create_sent_db(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE sentences (id PRIMARY KEY, text, h_links, doc_id);")
    c.execute("CREATE INDEX doc_id_index ON sentences(doc_id);")


def insert_many(cursor, items):
    cursor.executemany("INSERT INTO documents VALUES (?,?,?)", items)


def iter_over_db(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("SELECT * from documents")
    count = 0
    for pid, text, lines in tqdm(c, total=5416537):
        pid_words = pid.strip().split('_')
        print(pid_words, len(pid_words))

        if len(text) > 1:
            lines_items = json.loads(lines)
            for line in lines_items:
                # print(line['sentences'])
                if line['sentences']:
                    count += 1

    print(count)


def insert_one_sent(cursor, item):
    cursor.execute("INSERT INTO sentences VALUES (?,?,?,?)", item)


def check_document_id(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("SELECT * from documents")
    count = 0
    for pid, text, lines in tqdm(c, total=5416537):
        pid_words = pid.strip().replace('_', ' ')
        match = re.search('[a-zA-Z]', pid_words)
        if match is None:
            print(pid_words)
        elif text_clean.check_arabic(pid_words):
            print('arabic:', pid_words)
        else:
            count += 1

    print(count)
