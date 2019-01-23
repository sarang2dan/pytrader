from Utils.DataTypes import *
from Utils.DataCrawler.StockCodeCrawler import *
from MainModule.AlphaModel.MathmaticalModel.MeanReversionModel import *
from MainModule.AlphaModel.MachineLearningModel.MachineLearningModel import *
from logger import logger
from Utils.DBManager import *

from MainModule.ModelEvaluator.ModelEvaluator import *
from MainModule.AlphaModel.MachineLearningModel.ParameterOptimizer import *


# todo: 전반적인 코드 분석 및 정리

class PortfolioBuilder(object):
    def __init__(self, mmctx: MainModuleContext):
        self.mean_reversion_model = mmctx.mean_reversion_model
        self.machine_learning_model = mmctx.machine_learning_model
        return

    def doStationarityTest(self, mmctx: MainModuleContext):
        # 아래의 코드를 stockInfo로 대체
        test_result = {'code': [],
                       'company': [],
                       'adf_statistics': [],
                       'adf_1': [],
                       'adf_5': [],
                       'adf_10': [],
                       'hurst_exp': [],
                       'halflife': [] }

        cnt = 0
        total_cnt = len(mmctx.random_stock_codes)

        if mmctx.verbose == True:
            logger.info("---- Testing Stationary ----")

        for code in mmctx.random_stock_codes:
            cnt += 1
            item = mmctx.stockInfoItems[code]

            df = DBManager.loadStockPriceToDataFrame(makePriceTableName(code, row_unit=mmctx.row_unit),
                                                     mmctx.start_date,
                                                     mmctx.end_date)
            df_col = df[mmctx.input_col_name]

            if len(df_col) > 0:
                try:
                    adf_statistic, adf_1, adf_5, adf_10 = \
                        self.mean_reversion_model.calcADF(df_col)

                    hurst_exp = self.mean_reversion_model.calcHurstExponent(df_col,
                                                                            mmctx.time_lags)
                    halflife = self.mean_reversion_model.calcHalfLife(df_col)

                    test_result['code'].append(code)
                    test_result['company'].append(StockInfoItem.getCompanyName(item))
                    test_result['adf_statistics'].append(adf_statistic)
                    test_result['adf_1'].append(adf_1)
                    test_result['adf_5'].append(adf_5)
                    test_result['adf_10'].append(adf_10)
                    test_result['hurst_exp'].append(hurst_exp)
                    test_result['halflife'].append(halflife)
                except Exception as e:
                    logger.warn("[(%s) %s][row count: %d][Error: %s]" %
                                (code, StockInfoItem.getCompanyName(item), len(df_col), str(e)))

                pass # if len(df_col) > 0:
            pass # for code in rows_code:

            df_stationarity_result = pd.DataFrame(data=test_result,
                                                  # index=test_result['code'],
                                                  columns=test_result.keys())

        return df_stationarity_result

    def rankStationarity(self, df: pd.DataFrame, rank_col_name):
        df[rank_col_name] = 0
        df['rank_adf'] = 0
        df['rank_hurst'] = 0
        df['rank_halflife'] = 0

        halflife_percentile = np.percentile(df['halflife'],
                                            np.arange(0, 100, 10))
        if mmctx.verbose == True:
            logger.info('halflife_percentile\n' + str(halflife_percentile))

        for i in range(len(df)):
            df.ix[i, 'rank_adf'] = self.assessADF(df.ix[i]['adf_statistics'],
                                                  df.ix[i]['adf_1'],
                                                  df.ix[i]['adf_5'],
                                                  df.ix[i]['adf_10'])

            df.ix[i, 'rank_hurst'] = self.assessHurst(df.ix[i]['hurst_exp'])

            df.ix[i, 'rank_halflife'] = self.assessHalflife(halflife_percentile,
                                                            df.ix[i]['halflife'],
                                                            mmctx.verbose)
            if mmctx.verbose == True:
                code = df.ix[i, 'code']
                company_name = df.ix[i, 'company']
                msg = "[(%s) %-10s][rank_adf: %d][hurst: %.4f][halflife: %.4f]" % \
                            (code, company_name, df.ix[i]['rank_adf'], df.ix[i]['hurst_exp'], df.ix[i]['halflife'])
                logger.info(msg)

        df[rank_col_name] = df['rank_adf'] + df['rank_hurst'] + df['rank_halflife']
        return df

    def buildUniverse(self, df: pd.DataFrame, col_name, high_rank_ratio):
        """high_rank_ratio: 0.80이면 80% 이상, 즉 상위 20% 만 종목만을 리턴한다."""
        percentile_array = np.percentile(df[col_name],
                                          np.arange(0, 100, 1))
        # if verbose == True:
        logger.info(percentile_array)

        base_value = percentile_array[int(high_rank_ratio * 100)]
        universe = {}
        for i in range(len(df)):
            if df.ix[i][col_name] >= base_value:
                universe[df.ix[i]['code']] = df.ix[i]['company']

        # #ratio_idx = np.trunc(high_rank_ratio * len(percentile_array))
        # ratio_index = int(high_rank_ratio * 100)
        # universe = {}
        #
        # for i in range(len(df)):
        #     percentile_idx = self.getPercentileIndex(percentile_array,
        #                                              df.ix[i][col_name])
        #     # todo: 의미를  분석하기..
        #     if percentile_idx >= ratio_idx:
        #         universe[df.ix[i]['code']] = df.ix[i]['company']

        return universe

    def getPercentileIndex(self, percentile_arr, value):
        for index in range(len(percentile_arr)):
            if value <= percentile_arr[index]:
                return index
        return len(percentile_arr)

    # todo: 분석. 특히 assessMachineLearning() 함수
    def rankMachineLearning(self, df_ml, rank_col_name):
        ml_models = ESTIMATOR_TYPE.__members__
        percentile = {}

        for ml in ml_models:
            df_ml['%s_%s' % (rank_col_name, ml)] = 0
            percentile[ml] = np.percentile(df_ml[ml],
                                           np.arange(0, 100, 1))
            if mmctx.verbose == True:
                msg = '\n-- percentile : [%s] --\n' % ml
                msg += str(percentile[ml])
                logger.info(msg)

        for ml in ml_models:
            for i in range(len(df_ml)):
                df_ml.ix[i, '%s_%s' % (rank_col_name, ml)] = \
                    self.assessMachineLearning(percentile[ml],
                                               df_ml.ix[i, ml], verbose=False) # mmctx.verbose)

        rank_col_name_list = ['%s_%s' % (rank_col_name, i) for i in ml_models]
        df_ml[rank_col_name] = df_ml[rank_col_name_list].sum(axis=1)

        return df_ml

    def assessADF(self, adf_statistics, adf_1, adf_5, adf_10):
        # logger.info('as:%f a1:%f a5:%f a10:%f' % (adf_statistics, adf_1, adf_5, adf_10))
        if adf_statistics < adf_1:
            return 3
        if adf_statistics < adf_5:
            return 2
        if adf_statistics < adf_10:
            return 1
        return 0

    def assessHurst(self, hurst):
        # 0.5 라면 GBM이다.
        # 0.0 이라면 평균회귀한다.
        # 1.0 이라면 발산하는 형태의 추세를 보인다.
        if hurst < 0.1:
            return 3
        if hurst < 0.2:
            return 2
        if hurst < 0.3:
            return 1
        if hurst >= 0.3:
            return 0

    def assessHalflife(self, percentile, halflife, verbose=False):
        rank_value = 0
        for i in range(6, len(percentile)):
            if halflife <= percentile[i]:
                rank_value = i - 6
                break
        assert rank_value >= 0
        assert rank_value < 4
        return rank_value

    def doMachineLearningTest(self,
                              mmctx: MainModuleContext):
        return self.machine_learning_model.calcScore(mmctx)

    def assessMachineLearning(self, percentile, score, verbose=False):
        if verbose == True:
            logger.info("[%s][score: %.4f]" % (getMethodName(), score))

        if score <= percentile[50]:
            return 0
        elif score <= percentile[60]:
            return np.power(1.5, 0)
        elif score <= percentile[70]:
            return np.power(1.5, 1)
        elif score <= percentile[80]:
            return np.power(1.5, 2)
        elif score <= percentile[90]:
            return np.power(1.5, 3)
        elif score <= percentile[99]:
            return np.power(1.5, 4)

    pass # class Portfolio():


if __name__ == "__main__":

    # services.register('dbhandler',DataHandler())
    # services.register('dbwriter',DataWriter())
    # services.register('dbreader',DataReader())
    # services.register('charter',Charter())
    # services.register('configurator',Configurator())
    #
    # services.register('predictors',Predictors())
    # services.register('trader',MessTrader())
    # services.register('mean_reversion_model',MeanReversionModel())
    # services.register('machine_learning_model',MachineLearningModel())


    DBManager.init()

    stockInfo = StockInfo()
    stockInfo.loadFromDatabase(DBManager.getStockInfoDBConnection(),
                               MARKET_TYPE.KOSPI)

    # todo: 머신러닝에 사용할 종목을 로딩. 로딩되는 아이템은 비어있을 수 있다.. 아이템 수를 높이는 방향으로 그냥 넘어갈까??
    random_stock_codes = stockInfo.copyRandomStockCodes(item_count=500)

    predictors = Predictors()
    window_size = 10
    threshold = 1.5

    machine_learning_model = MachineLearningModel(predictors)
    mean_reversion_model = MeanReversionModel(window_size, threshold)

    mmctx = MainModuleContext(mean_reversion_model=mean_reversion_model,
                              machine_learning_model=machine_learning_model,
                              predictors=predictors,
                              stockInfoItems=stockInfo.items,
                              random_stock_codes=random_stock_codes,
                              start_date=datetime.datetime(2010, 1, 1, 0, 0, 0),
                              end_date=datetime.datetime(2016, 10, 1, 0, 0, 0),
                              input_col_name='Close',
                              output_col_name='Close_indicator',
                              time_lags=10,
                              split_ratio=0.75,
                              window_size=10,
                              threshold=1.5,
                              row_unit='day',
                              verbose=True)

    pfBuilder = PortfolioBuilder(mmctx)
    universe = Portfolio()

    #mean_backtester = MeanReversionBackTester()
    #machine_learning_evaluator = MachineLearningBackTester()

    #crawler.updateAllCodes()
    #crawler.updateAllStockData(1,2010,1,1,2015,12,1,start_index=90)

    #finder.setTimePeriod('20150101','20151130')

    df_stationarity = pfBuilder.doStationarityTest(mmctx)
    df_stationary_rank = pfBuilder.rankStationarity(df_stationarity, 'rank')
    stationarity_codes = pfBuilder.buildUniverse(df_stationary_rank, 'rank', high_rank_ratio=0.90)

    logger.info('*' * 50)
    logger.info(len(df_stationary_rank))
    logger.info(len(stationarity_codes))
    logger.info('*'*50)
    for i in range(len(df_stationary_rank)):
        if df_stationary_rank.ix[i]['rank'] > 3:
            logger.info(df_stationary_rank.ix[i]['code'])
    logger.info('*'*50)

    df_machine_result = pfBuilder.doMachineLearningTest(mmctx)
    df_machine_rank = pfBuilder.rankMachineLearning(df_machine_result, 'rank')
    machine_codes = pfBuilder.buildUniverse(df_machine_rank, 'rank', high_rank_ratio=0.90)

    if mmctx.verbose == True:
        print(df_machine_result)
        print(df_machine_rank)

    #print services.get('predictors').dump()
    #print df_machine_rank
    #print machine_codes

    universe.clear()

    universe.makeUniverse(mmctx.input_col_name,
                          ALPHA_MODELS.stationarity.name,
                          stationarity_codes)

    universe.makeUniverse(mmctx.input_col_name,
                          ALPHA_MODELS.machine_learning.name,
                          machine_codes)

    universe.dump()

    logger.info('-' * 70)
    logger.info('{0:-^70s}'.format(' start evaluating '))

    mmctx.start_date = datetime.datetime(2016, 1, 1, 0, 0, 0)
    mmctx.end_date = datetime.datetime(2016, 10, 1, 0, 0, 0)
    mmctx.time_lags = 5

    sdf = df_machine_result.head(1)
    for code in sdf['code']:
        evctx = ModelEvaluatorContext.createNewContext(code,
                                                       predictors,
                                                       ESTIMATOR_TYPE.random_forest.name,
                                                       mmctx)

        MachineLearningEvaluator.getConfusionMatrix(evctx)
        MachineLearningEvaluator.printClassificationReport(evctx)
        MachineLearningEvaluator.showROC(evctx)
        # hit_count, total_count = MachineLearningEvaluator.drawHitRatio(evctx)
        # logger.info("hit_ratio: %3d/%-3d = %.4f" % (hit_count, total_count, hit_count / total_count))
        # MachineLearningEvaluator.drawPosition(evctx, mmctx)
        # MachineLearningEvaluator.drawDrawdown(evctx, mmctx)
        # ParameterOptimizer.doGridSearch(evctx)
        # ParameterOptimizer.doRandomSearch(evctx)

        # MachineLearningEvaluator.showROC(evctx, lags_count=5)
        # MachineLearningEvaluator.getHitRatio('rf','006650','20151101','20151130',lags_count=1)
        # MachineLearningEvaluator.getHitRatio('rf','006650','20151101','20151130',lags_count=5)
        # MachineLearningEvaluator.getHitRatio('rf','006650','20151101','20151130',lags_count=10)
        # MachineLearningEvaluator.drawHitRatio('rf','006650','20151101','20151130',lags_count=5)

        #optimizeHyperparameter()
        #optimizeHyperparameterByRandomSearch('rf','006650','20150101','20151130',lags_count=5)

    # mean_backtester.setThreshold(1.5)
    # mean_backtester.setWindowSize(20)


    #estimator = self.predictors.getEstimator(code)[estimator_type]
    #df, df_x_test, df_y_true = self.getTestDataset(code, estimator, mmctx)
    #df_y_pred, df_y_pred_proba = estimator.predict(df_x_test.values)


    #mean_backtester.doTest('stationarity',universe,'20150101','20151130')

    """
    services.get('trader').setPortfolio(universe)
    services.get('trader').simulate()
    """

    #services.get('trader').dump()


    #services.get('charter').drawStationarityTestHistogram(df)
    #services.get('charter').drawStationarityTestBoxPlot(df)
    #services.get('charter').drawStationarityRankHistogram(df_stationary_rank)

    DBManager.finalize()
