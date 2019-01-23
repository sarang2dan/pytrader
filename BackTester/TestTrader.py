# -*- coding: utf-8 -*-
from __future__ import division
import os,sys
import numpy as np
import pandas as pd


from MainModule.DataTypes import *


class MessTrader(object):
    def __init__(self):
        self.items = {}
        return

    def setPortfolio(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def add(self, model, code, row_index, position):
        if self.item.get(code) is None:
            self.items[code] = []

        a_item = TradeItem(model, code, row_index, position)
        self.items[code].append( a_item )
        return

    def assessSignal(self, column, df, row_index, position):
        #print "row=%s, len=%s" % (row_index,df.shape[0])
        if (row_index) >= (df.shape[0]-1):
            return None

        current_price = df.loc[row_index,column]
        next_price = df.loc[row_index+1,column]
        diff = next_price - current_price

        #print "row_index=%s, position=%s, price=%s, next=%s, diff=%s" % (row_index,position,current_price,next_price,diff)

        if np.sign(diff) > 0:
            if position==POSITION.LONG:
                return True
        elif np.sign(diff) < 0:
            if position == POSITION.SHORT:
                return True
        else:
            return True

        return False


    def simulate(self):
        result = {}

        self.portfolio.dump()

        for model in self.portfolio.items.keys():
            for a_item in self.portfolio.items[model].values():
                #for key in self.items.keys():

                count_correct = 0

                for a_trade in self.items[a_item.code]:
                    result = self.assessSignal(a_item.column,
                                               a_item.df,
                                               a_trade.row_index,
                                               a_trade.position)
                    if result is not None:
                        if result:
                            count_correct += 1

                a_item.hit_ratio = count_correct/(a_item.df.shape[0])
                print("model=%s, code=%s , count_correct=%s, hit_ratio=%0.2f" % (model,a_item.code,count_correct,a_item.hit_ratio))


    def dump(self):
        print(">>> %s.%s <<<" % (self.__class__.__name__, getMethodName())
        for key in self.items.keys():
            for a_item in self.items[key]:
                print "... model=%s, code=%s, row_index=%s, position=%s" % (a_item.model,a_item.code,a_item.row_index,a_item.position)

        print "--- Done ---"
