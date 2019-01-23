from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from MainModule.DataTypes import *
from logger import logger


class ParameterOptimizer:
    def __doGridSearch(estimator, x_train, y_train, param_grid):
        grid_search = GridSearchCV(estimator, param_grid=param_grid)
        grid_search.fit(x_train, y_train)
        logger.info('{0:-^70s}'.format(' %s ' % getMethodName()))
        for params, mean_score, scores in grid_search.grid_scores_:
            logger.info("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        return

    def __doRandomSearch(estimator, x_train, y_train, param_dist, iter_count):
        random_search = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=iter_count)
        random_search.fit(x_train, y_train)

        logger.info('{0:-^70s}'.format(' %s ' % getMethodName()))
        for params, mean_score, scores in random_search.grid_scores_:
            logger.info("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
        return

    def doGridSearch(evctx: ModelEvaluatorContext):
        """Hyper-Parameter Optimization"""
        param_grid = {"max_depth": [3, None],
                      "min_samples_split": [1, 3, 10],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}

        x_train, x_test, y_train, y_test = splitDataSet(evctx.df_lagged,
                                                        evctx.input_col_name,
                                                        evctx.output_col_name,
                                                        evctx.split_ratio)

        ParameterOptimizer.__doGridSearch(evctx.estimator.machine,
                                          x_train.reshape(len(x_train), 1),
                                          y_train,
                                          param_grid)
        return


    def doRandomSearch(evctx: ModelEvaluatorContext):
        param_dist = {"max_depth": [3, None],
                      "min_samples_split": sp_randint(1, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}

        x_train, x_test, y_train, y_test = splitDataSet(evctx.df_lagged,
                                                        evctx.input_col_name,
                                                        evctx.output_col_name,
                                                        evctx.split_ratio)

        ParameterOptimizer.__doRandomSearch(evctx.estimator.machine,
                                            x_train.reshape(len(x_train), 1),
                                            y_train,
                                            param_dist,
                                            iter_count=20)
        return