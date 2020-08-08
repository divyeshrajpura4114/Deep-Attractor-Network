#!/usr/bin/env python

import common
from common import libraries, params
from common.libraries import *

class visdomvisulization:
    def __init__(self, env_name, title, xlabel, ylabel):
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        #+ "_" + str(datetime.now().strftime("%d-%m %Hh%M"))
    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win = self.loss_win,
            update = 'append' if self.loss_win else None,
            opts = dict(xlabel = self.xlabel, ylabel = self.ylabel, title = self.title)
        )
