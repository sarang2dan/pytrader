from MainModule.AlphaModel.AlphaModel import AlphaModel
from MainModule.DataTypes import *
from Utils.DataTypes import *
from Utils.DataCrawler.StockCodeCrawler import *
from Utils.DBManager import DBManager

from logger import logger

import matplotlib.pyplot as plt

from sklearn import svm, linear_model, ensemble
from sklearn.metrics import confusion_matrix,classification_report,roc_curve, auc


# from sklearn.svm import LinearSVC, SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression


import pandas as pd
import numpy as np
import datetime

def test_estimator(estimator, x_test, y_test):
    predict_result = estimator.predict(x_test)
    hit_count = 0
    total_count = len(x_test)
    for i in range(total_count):
        if predict_result[i] == y_test[i]:
            hit_count += 1

    hit_ratio = hit_count / total_count
    score = estimator.score(x_test, y_test)

    return hit_ratio, score

class Estimator:
    # todo: machine_learning 들의 derived class가 다르다..
    # todo: 따라서 아래의 코드들은 위험하다.. 검토 후 변경하는 것이 좋겠다..
    def __init__(self, name, machine):
        self.name = name
        self.machine = machine

    def train(self, x_train, y_train):
        # self.machine.fit(x_train, y_train)
        self.machine.fit(x_train.reshape(len(x_train), 1), y_train)
        # return self.machine.score(x_train, y_train)
        return

    def predict(self, x_test: np.ndarray, with_probability=True):
        pred_res = self.machine.predict(x_test)
        pred_probability = self.machine.predict_proba(x_test)
        return pred_res, pred_probability

    def score(self, x_test, y_test):
        return self.machine.score(x_test.reshape(len(x_test), 1), y_test)

class Predictors(object):
    def __init__(self):
        self.items = {}
        return

    def createEstimator(self, estimator_type):
        if ESTIMATOR_TYPE.svm.name == estimator_type:
            return Estimator(estimator_type,
                             svm.SVC(C=1.0, #kernel='linear',
                                     gamma='auto',
                                     probability=True))
        elif ESTIMATOR_TYPE.logistic.name == estimator_type:
            return Estimator(estimator_type,
                             linear_model.LogisticRegression(C=1.0))
        elif ESTIMATOR_TYPE.random_forest.name == estimator_type:
            return Estimator(estimator_type,
                             ensemble.RandomForestClassifier())

        raise ValueError('%s is not valid value' % estimator_type)

    # todo: 함수 다시 봐서 정리하기.....
    def trainAll(self,
                 mmctx: MainModuleContext):

        self.random_stock_codes = mmctx.random_stock_codes

        test_result = {'code': [],
                       'company': []}
        for ctype in ESTIMATOR_TYPE.__members__:
            test_result[ctype] = []

        cnt = 0
        total_cnt = len(mmctx.random_stock_codes)

        for code in mmctx.random_stock_codes:
            cnt += 1
            item = mmctx.stockInfoItems[code]

            df = DBManager.loadStockPriceToDataFrame(makePriceTableName(code, row_unit=mmctx.row_unit),
                                                     mmctx.start_date,
                                                     mmctx.end_date)

            lagged_df = makeLaggedDataFrame(df,
                                            mmctx.input_col_name,
                                            mmctx.output_col_name,
                                            time_lags=mmctx.time_lags)
            # todo: 50개 보다 적으면 학습포기, 150보다 정확한 숫자를 찾아보자..
            if len(lagged_df) < 50:
                if mmctx.verbose == True:
                    logger.info('[%d/%d][(%s) %s] does not have enough data.' %
                                (cnt, total_cnt, code, StockInfoItem.getCompanyName(item)))
                continue

            test_result['code'].append(code)
            test_result['company'].append(StockInfoItem.getCompanyName(item))

            x_train, x_test, y_train, y_test = \
                splitDataSet(lagged_df,
                             mmctx.input_col_name,
                             mmctx.output_col_name,
                             mmctx.split_ratio)

            if mmctx.verbose == True:
                msg = '[(%s) %-10s] [scores]' % (code, StockInfoItem.getCompanyName(item))

            for ctype in ESTIMATOR_TYPE.__members__:
                estimator = self.createEstimator(ctype)

                self.addEstimator(code, ctype, estimator)

                estimator.train(x_train, y_train)

                score = estimator.score(x_test, y_test)

                test_result[ctype].append(score)

                if mmctx.verbose == True:
                    msg += '[%s: %.4f]' % (ctype, score)
                pass # for predictor_type

            if mmctx.verbose == True:
                logger.info(msg)

            pass # for code in rows_code:

        df_result = pd.DataFrame(test_result)

        return df_result
    def addEstimator(self, code: str, estimator_type: str, estimator):
        if code not in self.items:
            self.items[code] = {}

        self.items[code][estimator_type] = estimator

        return
    def getEstimator(self, code: str):
        return self.items.get(code)

    pass

class MachineLearningModel(AlphaModel):
    def __init__(self, predictors: Predictors):
        self.predictors = predictors
        return

    def calcScore(self,
                  mmctx: MainModuleContext):
        return self.predictors.trainAll(mmctx)

    def determinePosition(self, code, df, col_name, row_index, verbose=False):
        position = POSITION.HOLD

        if (row_index-1) < 0:
            position = POSITION.HOLD
        else:
            current_price = df.ix[row_index][col_name]
            prediction_result = 0
            estimators = self.predictors.getEstimator(code)

            for etype in ESTIMATOR_TYPE.__members__:
                estimator = estimators[etype]
                pred, pred_probability = estimator.predict(current_price)
                # 3개의 의사를 통해, 매수매도 결정
                if verbose == True:
                    logger.info('[code:%s] [(%s) pos: %s] [predict probability: %s]' %
                                (code, etype, POSITION.valueToName(pred[0]), str(pred_probability[0])))

                prediction_result += pred[0]

            if prediction_result > 1:
                position = POSITION.LONG
            else:
                position = POSITION.SHORT
                pass  # if prediction_result > 1:
            pass # if (row_index-1) < 0:

        if verbose == True:
            logger.info('[code:%s] [pos: %s]' % (code, position.name))

        return position

    pass