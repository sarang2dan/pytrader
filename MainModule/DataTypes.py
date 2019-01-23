from enum import Enum
from logger import logger
import pandas
import datetime
import numpy as np
from Utils.DataTypes import getMethodName
import Utils.DataTypes as UDataType
from Utils.DBManager import DBManager


mmMakePriceTableName = UDataType.makePriceTableName


# todo: 파일이름을 common.py로 변경하자..
def makeLaggedDataFrame(df: pandas.DataFrame,
                        input_col_name,
                        output_col_name,
                        time_lags=5):
    lag_col_name = '%s_Lag%d' % (input_col_name, time_lags)
    percent_col_name = lag_col_name + '_Change'

    df[lag_col_name] = df[input_col_name].shift(time_lags)
    df[percent_col_name] = df[lag_col_name].pct_change() * 100.0

    # todo: 왜 '-1에서 0으로 세팅하는 것'으로 변경했을까??
    # df_lag[output_col_name] = np.sign(df_lag['%s_Lag%s_Change' % (input_col_name,time_lags)])
    df[output_col_name] = np.where(df[percent_col_name] > 0, 1, 0)

    # todo: Volume_indicator가 필요한지 의문을 가져보자..
    vol_col_name = 'Volume'
    lag_vol_col_name = '%s_Lag%s' % (vol_col_name, time_lags)
    percent_vol_col_name = lag_vol_col_name + '_Change'

    df[lag_vol_col_name] = df[vol_col_name].shift(time_lags)
    df[percent_vol_col_name] = df[lag_vol_col_name].pct_change() * 100.0
    df['Volume_indicator'] = np.sign(df[percent_vol_col_name])

    return df.dropna(how='any')

def splitDataSet(df: pandas.DataFrame,
                 input_col_name,
                 output_col_name,
                 split_ratio):
    # df is lagged
    def getPivotIndexByRatio(df, split_ratio):
        pivot_ix = int(len(df) * split_ratio)
        return pivot_ix

    pivot_index = getPivotIndexByRatio(df, split_ratio)

    # df[['a']] 를 하면 결과적으로 1개의 컬럼이 있다고 명시된다.
    # df['a'] 이러면 shape[1] 즉 컬럼의 개수가 세팅이 되지 않는다.

    # Create training and test sets
    x_train = df[input_col_name][0:pivot_index]
    x_test = df[input_col_name][pivot_index:]
    y_train = df[output_col_name][0:pivot_index]
    y_test = df[output_col_name][pivot_index:]

    # x_train.reshape(len(x_train), 1)
    # x_test.reshape(len(x_test), 0)

    return x_train, x_test, y_train, y_test


class ALPHA_MODELS(Enum):
    stationarity = 1
    machine_learning = 2

    def valueToName(num):
        return ALPHA_MODELS.__dict__['_value2member_map_'][num].name
    pass

class POSITION(Enum):
    SHORT = -1
    HOLD = 0
    LONG = 1
    def valueToName(num):
        return POSITION.__dict__['_value2member_map_'][num].name
    pass

class ESTIMATOR_TYPE(Enum):
    logistic = 1
    random_forest = 2
    svm = 3

    def valueToName(num):
        return ESTIMATOR_TYPE.__dict__['_value2member_map_'][num].name

    def getMemberList():
        return ESTIMATOR_TYPE.__dict__['_member_names_'];

    pass

class PortfolioItem:
    def __init__(self, column, code, company):
        self.code = code
        self.company = company
        self.column = column
        self.df = None
        self.score = 0

# todo: 이 클래스는 정리하는 게 옳을 듯..
class Portfolio:
    def __init__(self):
        self.models = {}
        for model in ALPHA_MODELS.__members__:
            self.models[model] = {}
        return

    def clear(self):
        self.models.clear()
        self.__init__()
        return

    def findCode(self, model, code):
        if code in self.models[model]:
            return self.models[model][code]
        return None

    def add(self, column, model, code, company):
        item = PortfolioItem(column, code, company)
        self.models[model][code] = item
        return

    def makeUniverse(self, column, model, stock_dict):
        for code in stock_dict.keys():
            self.add(column, model, code, stock_dict[code])
        return

    def dump(self):
        logger.info("Start %s" % getMethodName())
        for model in self.models:
            logger.info('-' * 50)
            logger.info("{0:-<50s}".format("'- [model: %s] " % model))
            for code in self.models[model]:
                i = self.models[model][code]
                logger.info("[code: %s][company: %s][column: %s]" % \
                            (i.code, i.company, i.column))
        logger.info("--- Done ---")


class TradeItem:
    def __init__(self, model, code, row_index, position):
        self.model = model
        self.code = code
        self.row_index = row_index
        self.position = position

class MainModuleContext(object):
    def __init__(self,
                 mean_reversion_model: object,
                 machine_learning_model: object,
                 predictors: object,
                 stockInfoItems: dict,
                 random_stock_codes: list,
                 start_date: datetime.datetime,
                 end_date: datetime.datetime,
                 input_col_name: str,
                 output_col_name: str,
                 time_lags=10,
                 split_ratio=0.75,
                 window_size=20,
                 threshold=1.5,
                 row_unit='day',
                 verbose=False):
        self.mean_reversion_model = mean_reversion_model  # MeanReversionModel(mmctx.window_size, mmctx.threshold)
        self.machine_learning_model = machine_learning_model  # MachineLearningModel(mmctx.predictors)
        self.predictors = predictors
        self.stockInfoItems = stockInfoItems
        self.random_stock_codes = random_stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.input_col_name = input_col_name
        self.output_col_name = output_col_name
        self.time_lags = time_lags
        self.split_ratio = split_ratio
        self.threshold = threshold
        self.window_size = window_size
        self.row_unit = row_unit
        self.verbose = verbose
        return
    pass

class BackTesterContext(object):
    def __init__(self,
                 predictors: object,
                 stockInfoItems: dict,
                 random_stock_codes: list,
                 start_date: datetime.datetime,
                 end_date: datetime.datetime,
                 input_col_name: str,
                 output_col_name: str,
                 time_lags=10,
                 split_ratio=0.75,
                 window_size=20,
                 threshold=1.5,
                 row_unit='day',
                 verbose=False):
        self.predictors = predictors
        self.stockInfoItems = stockInfoItems
        self.random_stock_codes = random_stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.input_col_name = input_col_name
        self.output_col_name = output_col_name
        self.time_lags = time_lags
        self.split_ratio = split_ratio
        self.threshold = threshold
        self.window_size = window_size
        self.row_unit = row_unit
        self.verbose = verbose
        return
    pass



class ModelEvaluatorContext(object):
    def __init__(self,
                 code,
                 input_col_name,
                 output_col_name,
                 split_ratio,
                 estimator,
                 df,
                 df_lagged,
                 df_x_test,
                 df_y_true,
                 df_y_pred,
                 df_y_pred_proba):
        self.code = code
        self.input_col_name = input_col_name
        self.output_col_name = output_col_name
        self.split_ratio = split_ratio
        self.estimator = estimator
        self.df = df
        self.df_lagged = df_lagged
        self.df_x_test = df_x_test
        self.df_y_true = df_y_true
        self.df_y_pred = df_y_pred
        self.df_y_pred_proba = df_y_pred_proba
        return

    def createNewContext(code, predictors, estimator_type, mmctx: MainModuleContext):
        estimator = predictors.getEstimator(code)[estimator_type]
        df = DBManager.loadStockPriceToDataFrame(mmMakePriceTableName(code, mmctx.row_unit),
                                                 mmctx.start_date,
                                                 mmctx.end_date)

        df_lagged = makeLaggedDataFrame(df.copy(),
                                        mmctx.input_col_name,
                                        mmctx.output_col_name,
                                        mmctx.time_lags)

        df_x_test = df_lagged[[mmctx.input_col_name]]
        df_y_true = df_lagged[[mmctx.output_col_name]]

        df_y_pred, df_y_pred_proba = estimator.predict(df_x_test.values)

        return ModelEvaluatorContext(code,
                                     mmctx.input_col_name,
                                     mmctx.output_col_name,
                                     mmctx.split_ratio,
                                     estimator,
                                     df,
                                     df_lagged,
                                     df_x_test,
                                     df_y_true,
                                     df_y_pred,
                                     df_y_pred_proba)


