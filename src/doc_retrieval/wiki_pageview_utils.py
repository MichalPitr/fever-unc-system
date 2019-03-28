import pickle
from collections import defaultdict

import config

__author__ = ['chaonan99']


class WikiPageviews(object):
    """WikiPageviews"""
    pageview_path = str(config.DATA_ROOT) + "/pageviews.pkl"

    def __init__(self):
        pv_dict_raw = pickle.load(open(self.pageview_path, 'rb'))
        self.pageview_dict = defaultdict(lambda: 0, pv_dict_raw)

    def __getitem__(self, ind):
        return self.pageview_dict[ind]
