# todo: 동적으로 코드를 사용자로부터 받아서 실행할 수 있도록 구성가능
# todo: zipline을 이용하는 것과 모의투자?를 이용할 수 있겠지만, 모의투자는 Algorithm 트레이더에 넣어야 할 것 같다.

import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MainModule.DataTypes import *
from MainModule.AlphaModel.MachineLearningModel.MachineLearningModel import Estimator
from scipy.stats import randint as sp_randint
from Utils.DBManager import DBManager
from sklearn.metrics import confusion_matrix,classification_report,roc_curve, auc


def mmGetConfusionMatrix(y_true, y_pred):
    cfs_matrix = confusion_matrix(y_true, y_pred)
    return cfs_matrix

def mmClassificationReport(y_true, y_pred, target_names):
    return classification_report(y_true, y_pred, target_names=target_names)

def mmDrawROC(y_true, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('Sensitivity')
    plt.xlabel('Fall-out')
    plt.show()
    return

class MeanReversionEvaluator:
    def __init__(self):
        BaseBackTester.__init__(self)
        self.model = services.get('mean_reversion_model')

    def setWindowSize(self, size):
        self.window_size = size

    def setThreshold(self,threshold):
        self.threshold = threshold


    def doTest(self,model,portfolio,start_date,end_date):
        self.loadDataFrames(model,portfolio,start_date,end_date)

        for a_item in portfolio.items[model]:
            for row_index in range(a_item.df.shape[0]):
                if (row_index+1)>self.window_size:
                    position = self.determinePosition(a_item.df,a_item.column,row_index)
                    if position!=HOLD:
                        self.trader.add(model,a_item.code,row_index,position)

        #self.trader.dump()

    def getHitRatio(self,name, code, start_date,end_date,lags_count=5):
        a_predictor = self.predictor.get(code,name)

        df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count )
        df_x_test = df_dataset[ [self.config.get('input_column')] ]
        df_y_true = df_dataset[ [self.config.get('output_column')] ]


        self.loadDataFrames(model,portfolio,start_date,end_date)

        for a_item in portfolio.items[model]:
            for row_index in range(a_item.df.shape[0]):
                if (row_index+1)>self.window_size:
                    position = self.determinePosition(a_item.df,a_item.column,row_index)
                    if position!=HOLD:
                        self.trader.add(model,a_item.code,row_index,position)


        pred = classifier.predict(x_test)

        hit_count = 0
        total_count = len(y_test)
        for index in range(total_count):
            if (pred[index]) == (y_test[index]):
                hit_count = hit_count + 1

        hit_ratio = hit_count/total_count
        score = classifier.score(x_test, y_test)
        #print "hit_count=%s, total=%s, hit_ratio = %s" % (hit_count,total_count,hit_ratio)

        return hit_ratio, score
        # Output the hit-rate and the confusion matrix for each model

        #print("%s\n" % confusion_matrix(pred, y_test))


class MachineLearningEvaluator:
    def showROC(evctx: ModelEvaluatorContext):
        mmDrawROC(evctx.df_y_true, evctx.df_y_pred)
        return

    def getConfusionMatrix(evctx: ModelEvaluatorContext):
        cfs_matrix = mmGetConfusionMatrix(evctx.df_y_true, evctx.df_y_pred)
        logger.info('\n-- ConfusionMatrix --\n%s' % str(cfs_matrix))
        return

    def printClassificationReport(evctx: ModelEvaluatorContext):
        report = mmClassificationReport(evctx.df_y_true, evctx.df_y_pred, ['Down','Up'])
        logger.info('\n-- Classification Report --\n%s' % str(report))
        return

    def getHitRatio(evctx: ModelEvaluatorContext):
        hit_count = 0
        total_count = len(evctx.df_y_true)

        for row_index in range(total_count):
            if evctx.df_y_pred[row_index] == evctx.df_y_true[row_index]:
                hit_count += 1

        hit_count / total_count

        return hit_count, total_count

    def drawHitRatio(evctx: ModelEvaluatorContext):
        # todo: fix!! 제대로 동작하지 않는 것 같다..
        ax = evctx.df[[evctx.input_col_name]].plot()
        hit_count = 0
        total_count = len(evctx.df_y_true)
        for i in range(total_count):
            if evctx.df_y_pred[i] is evctx.df_y_true.values[i]:
                hit_count += 1
                ax.annotate('Yes',
                            xy=(i, evctx.df.ix[i][evctx.input_col_name]),
                            xytext=(10, 30),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='-|>'))

        hit_ratio = hit_count / total_count
        legend = plt.legend(loc='upper left')
        legend.set_title('hit rate: %.4f' % hit_ratio)

        plt.show()

        return hit_count, total_count

    def drawDrawdown(evctx, mmctx):
        MachineLearningEvaluator.__drawPosition(evctx, mmctx, draw_always=True)
        return
    def drawPosition(evctx, mmctx):
        MachineLearningEvaluator.__drawPosition(evctx, mmctx, draw_always=False)
        return

    def __drawPosition(evctx: ModelEvaluatorContext,
                       mmctx: MainModuleContext,
                       draw_always):

        ax = evctx.df[[evctx.input_col_name]].plot()

        for i in range(len(evctx.df_y_true)):
            if (draw_always == True) or ((i+1) > mmctx.time_lags):
                position = mmctx.machine_learning_model.determinePosition(evctx.code,
                                                                          evctx.df,
                                                                          evctx.input_col_name,
                                                                          i,
                                                                          verbose=True)
                if position == POSITION.LONG:
                    ax.annotate('Long',
                                xy=(i, evctx.df.ix[i, evctx.input_col_name]),
                                xytext=(10, -30),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle='-|>'))
                elif position == POSITION.SHORT:
                    ax.annotate('Short',
                                xy=(i, evctx.df.ix[i, evctx.input_col_name]),
                                xytext=(10, 30),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle='-|>'))
        plt.show()
        # Output the hit-rate and the confusion matrix for each model
        return
