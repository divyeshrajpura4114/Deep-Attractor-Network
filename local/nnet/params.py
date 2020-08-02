#!/usr/bin/env python

from . import libraries
from .libraries import *

class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

class Hparam(DictAsMember):
    def __init__(self, config_path = '/home/divraj/divyesh/danet/conf/conf.yaml'):
        super(Hparam, self).__init__()
        self.config_path = config_path
    
    def load_hparam(self):
        stream = open(self.config_path, 'r')
        hp_dict = yaml.load(stream, Loader = yaml.FullLoader)
        hp_dotdict = DictAsMember(hp_dict)
        return hp_dotdict