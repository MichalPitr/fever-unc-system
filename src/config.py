import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PRO_ROOT = SRC_ROOT.parent

DATA_ROOT = PRO_ROOT / "nsmn-data"
FEVER_DB = DATA_ROOT / "fever.db"
TOKENIZED_DOC_ID = DATA_ROOT / "tokenized_doc_id.json"

WN_FEATURE_CACHE_PATH = DATA_ROOT / "wn_feature_p"
