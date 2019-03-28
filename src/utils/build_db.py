import config
from utils.fever_db import create_db, create_sent_db, check_document_id, save_wiki_pages, build_sentences_table

print("Start building wiki document database. This might take a while.")
create_db(str(config.FEVER_DB))
save_wiki_pages(str(config.FEVER_DB))
create_sent_db(str(config.FEVER_DB))
build_sentences_table(str(config.FEVER_DB))
check_document_id(str(config.FEVER_DB))
print("Wiki document database is ready.")