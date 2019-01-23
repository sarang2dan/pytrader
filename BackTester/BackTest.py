# # todo: 동적으로 코드를 사용자로부터 받아서 실행할 수 있도록 구성가능
# # todo: zipline을 이용하는 것과 모의투자?를 이용할 수 있겠지만, 모의투자는 Algorithm 트레이더에 넣어야 할 것 같다.
#
# import os,sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from MainModule.DataTypes import *
# from MainModule.AlphaModel.MachineLearningModel.MachineLearningModel import Estimator
# from scipy.stats import randint as sp_randint
# from Utils.DBManager import DBManager
#
# class MeanReversionBackTester(object):
#     def __init__(self, trader: object, mmctx: MainModuleContext):
#         super()
#         self.trader = trader,
#         self.model = mmctx.mean_reversion_model
#         self.start_date = mmctx.start_date
#         self.end_date = mmctx.end_date
#         self.window_size = mmctx.window_size
#         self.threshold = mmctx.threshold
#         return
#
#     def doTest(self, portfolio: dict,model_type, start_date, end_date):
#         for pitem in portfolio.models[model_type].values():
#             pitem.df = DBManager.loadStockPriceToDataFrame(pitem.code,
#                                                            start_date,
#                                                            end_date)
#
#         for pitem in portfolio.items[model_type].values():
#             for row_index in range(pitem.df.shape[0]):
#                 if (row_index+1) > self.window_size:
#                     position = self.model.determinePosition(pitem.df,
#                                                             pitem.column,
#                                                             row_index)
#                     if position!= POSITION.HOLD:
#                         self.trader.add(model_type,
#                                         pitem.code,
#                                         row_index,
#                                         position)
#
#         #self.trader.dump()
#
#     def getHitRatio(self, name, code, start_date,end_date,lags_count=5):
#         a_predictor = self.predictor.get(code,name)
#
#         df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count )
#         df_x_test = df_dataset[ [self.config.get('input_column')] ]
#         df_y_true = df_dataset[ [self.config.get('output_column')] ]
#
#
#         self.loadDataFrames(model,portfolio,start_date,end_date)
#
#         for a_item in portfolio.items[model]:
#             for row_index in range(a_item.df.shape[0]):
#                 if (row_index+1)>self.window_size:
#                     position = self.determinePosition(a_item.df,a_item.column,row_index)
#                     if position!=HOLD:
#                         self.trader.add(model,a_item.code,row_index,position)
#
#
#         pred = classifier.predict(x_test)
#
#         hit_count = 0
#         total_count = len(y_test)
#         for index in range(total_count):
#             if (pred[index]) == (y_test[index]):
#                 hit_count = hit_count + 1
#
#         hit_ratio = hit_count/total_count
#         score = classifier.score(x_test, y_test)
#         #print "hit_count=%s, total=%s, hit_ratio = %s" % (hit_count,total_count,hit_ratio)
#
#         return hit_ratio, score
#         # Output the hit-rate and the confusion matrix for each model
#
#         #print("%s\n" % confusion_matrix(pred, y_test))
#
#
#
# class MachineLearningBackTester(object):
#     def __init__(self, trader: object, mmctx: MainModuleContext):
#         self.trader = trader,
#         self.model = mmctx.machine_learning_model
#         self.start_date = mmctx.start_date
#         self.end_date = mmctx.end_date
#         self.window_size = mmctx.window_size
#         self.threshold = mmctx.threshold
#         self.predictors = mmctx.predictors
#
#     def getTestDataset(self, code, estimator: Estimator, mmctx: MainModuleContext):
#
#         df = DBManager.loadStockPriceToDataFrame(mmMakePriceTableName(code, mmctx.row_unit),
#                                                  mmctx.start_date,
#                                                  mmctx.end_date)
#
#         df_dataset = makeLaggedDataFrame(df,
#                                          mmctx.input_col_name,
#                                          mmctx.output_col_name,
#                                          mmctx.time_lags)
#
#         df_x_test = df_dataset[[mmctx.input_col_name]]
#         df_y_true = df_dataset[[mmctx.output_col_name]]
#
#         return df, df_x_test, df_y_true
#
#
#     def showROC(self, code, estimator_type, mmctx: MainModuleContext):
#         estimator = self.predictors.getEstimator(code)[estimator_type]
#
#         df, df_x_test, df_y_true = self.getTestDataset(code, estimator, mmctx)
#
#         df_y_pred, df_y_pred_proba = estimator.predict(df_x_test.values)
#
#         estimator.drawROC(df_y_true,df_y_pred)
#         return
#
#     def getConfusionMatrix(self, code, estimator_type, mmctx: MainModuleContext):
#         estimator = self.predictors.getEstimator(code)[estimator_type]
#         df, df_x_test, df_y_true = self.getTestDataset(code, estimator, mmctx)
#         df_y_pred, df_y_pred_proba = estimator.predict(df_x_test.values)
#
#         estimator.confusionMatrix(df_y_true, df_y_pred)
#
#         return
#
#     def printClassificationReport(self, code, estimator_type, mmctx: MainModuleContext):
#         estimator = self.predictors.getEstimator(code)[estimator_type]
#         df, df_x_test, df_y_true = self.getTestDataset(code, estimator, mmctx)
#         df_y_pred, df_y_pred_proba = estimator.predict(df_x_test.values)
#         estimator.classificationReport(df_y_true,df_y_pred,['Down','Up'])
#
#         return
#
#     def getHitRatio(self, code, estimator_type, mmctx: MainModuleContext):
#         estimator = self.predictors.getEstimator(code)[estimator_type]
#         df, df_x_test, df_y_true = self.getTestDataset(code, estimator, mmctx)
#         df_y_pred, df_y_pred_proba = estimator.predict(df_x_test.values)
#
#         hit_count = 0
#         total_count = len(df_y_true)
#         for row_index in range(total_count):
#             if df_y_pred[row_index] == df_y_true[row_index]:
#                 hit_count += 1
#
#         hit_ratio = hit_count / total_count
#         logger.info("hit_ratio: %3d/%-3d = %.4f" % \
#                         (hit_count, total_count, hit_ratio))
#         plt.show()
#         return hit_ratio
#
#         # Output the hit-rate and the confusion matrix for each model
#
#     def drawHitRatio(self, code, estimator_type, mmctx: MainModuleContext):
#         estimator = self.predictors.getEstimator(code)[estimator_type]
#         df, df_x_test, df_y_true = self.getTestDataset(code, estimator, mmctx)
#         df_y_pred, df_y_pred_proba = estimator.predict(df_x_test.values)
#
#         ax = df[[mmctx.input_col_name]].plot()
#
#         for row_index in range(df_y_true.shape[0]):
#             if df_y_pred[row_index] == df_y_true[row_index]:
#                 ax.annotate('Yes',
#                             xy=(row_index, df.ix[row_index][mmctx.input_col_name]),
#                             xytext=(10,30),
#                             textcoords='offset points',
#                             arrowprops=dict(arrowstyle='-|>'))
#         plt.show()
#         return
#
#
#     def drawDrawdown(self,name, code, start_date,end_date,lags_count=5):
#         a_predictor = self.predictors.get(code, name)
#
#         df_dataset = self.predictors.makeLaggedDataset(code, start_date, end_date, self.config.get('input_column'), self.config.get('output_column'), lags_count)
#         df_x_test = df_dataset[ [self.config.get('input_column')] ]
#         df_y_true = df_dataset[ [self.config.get('output_column')] ].values
#
#
#         df_y_pred,df_y_pred_probability = a_predictor.predict(df_x_test)
#
#
#         ax = df_dataset[ [self.config.get('input_column')] ].plot()
#
#         for row_index in range(df_y_true.shape[0]):
#
#             position = self.model.determinePosition(code,df_dataset,self.config.get('input_column'),row_index)
#             if position==LONG:
#                 ax.annotate('Long', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,-30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
#             elif position==SHORT:
#                 ax.annotate('Short', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
#
#             if (df_y_pred[row_index] == df_y_true[row_index]):
#                 ax.annotate('Yes', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
#
#
#         plt.show()
#         # Output the hit-rate and the confusion matrix for each model
#
#
#
#     def drawPosition(self,name, code, start_date,end_date,lags_count=5):
#         a_predictor = self.predictors.get(code, name)
#
#         df_dataset = self.predictors.makeLaggedDataset(code, start_date, end_date, self.config.get('input_column'), self.config.get('output_column'), lags_count)
#         df_x_test = df_dataset[ [self.config.get('input_column')] ]
#         df_y_true = df_dataset[ [self.config.get('output_column')] ].values
#
#
#         df_y_pred,df_y_pred_probability = a_predictor.predict(df_x_test)
#
#
#         ax = df_dataset[ [self.config.get('input_column')] ].plot()
#
#         for row_index in range(df_y_true.shape[0]):
#             if (row_index+1)>lags_count:
#
#                 #determinePosition(self,code,df,column,row_index,verbose=False):
#
#                 position = self.model.determinePosition(code,df_dataset,self.config.get('input_column'),row_index)
#                 if position==LONG:
#                     ax.annotate('Long', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,-30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
#                 elif position==SHORT:
#                     ax.annotate('Short', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
#
#
#         plt.show()
#         # Output the hit-rate and the confusion matrix for each model
#
#
#     def optimizeHyperparameter(self,name, code, start_date,end_date,lags_count=5):
#         a_predictor = self.predictors.get(code, name)
#
#         df_dataset = self.predictors.makeLaggedDataset(code, start_date, end_date, self.config.get('input_column'), self.config.get('output_column'), lags_count)
#
#         X_train,X_test,Y_train,Y_test = self.predictors.splitDataset(df_dataset, 'price_date', [self.config.get('input_column')], self.config.get('output_column'), split_ratio=0.8)
#
#         param_grid = {"max_depth": [3, None],
#                     "min_samples_split": [1, 3, 10],
#                     "min_samples_leaf": [1, 3, 10],
#                     "bootstrap": [True, False],
#                     "criterion": ["gini", "entropy"]}
#
#         a_predictor.doGridSearch(X_train.values,Y_train.values,param_grid)
#
#
#     def optimizeHyperparameterByRandomSearch(self,name, code, start_date,end_date,lags_count=5):
#         a_predictor = self.predictors.get(code, name)
#
#         df_dataset = self.predictors.makeLaggedDataset(code, start_date, end_date, self.config.get('input_column'), self.config.get('output_column'), lags_count)
#
#         X_train,X_test,Y_train,Y_test = self.predictors.splitDataset(df_dataset, 'price_date', [self.config.get('input_column')], self.config.get('output_column'), split_ratio=0.8)
#
#         param_dist = {"max_depth": [3, None],
#                     "min_samples_split": sp_randint(1, 11),
#                     "min_samples_leaf": sp_randint(1, 11),
#                     "bootstrap": [True, False],
#                     "criterion": ["gini", "entropy"]}
#
#         a_predictor.doRandomSearch(X_train.values,Y_train.values,param_dist,20)
#         print sp_randint(1, 11)
