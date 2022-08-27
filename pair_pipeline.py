import pandas as pd
import numpy as np
import yfinance as yf
import itertools
from tqdm import tqdm

import statsmodels.api as sm

from itertools import combinations

from arch.unitroot.cointegration import phillips_ouliaris
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano as th
import seaborn as sns

#python 3.7
from typing_extensions import Literal

from typing import (TYPE_CHECKING,
                    Any,
                    Callable,
                    Dict,
                    Hashable,
                    List,
                    Optional,
                    Tuple,
                    TypeVar,
                    Union)

#Typing Hints:
Array = Any
ClusteringOutput = Tuple[List[Tuple[str, str]],
                         List[Dict[str, str]],
                         Dict[int, Dict[str, 
                                        Union[Array,
                                              List[Tuple[str, str]],
                                              List[Dict[str, str]]]]]]
                                              

class PairSelection():
    """
    Implementation of pair selection methods for kalman filter based pairs trading.
    Supports cointegration only, via `coint_johansen` or clustering based pair 
    identification via `OPTICS` or `DBSCAN` followed by cointegration and spread
    mean-reversion identification. 
    
    Data is acquired from `Yahoo Finance` through `yfinance` and then adjusted.
    
    Data will be downloaded and adjusted at `__init__`
    
    Please see docstrings of `get_clustered_pairs`
    and `get_cointegrated_pairs` for the structure of returns
    -------------
    :param tickers: list of ticker symbols to define the universe.
    :param fromdate: backtest start date. string in 'YYYY-MM-DD' format.
    :param todate: backtest end date. string in 'YYYY-MM-DD' format.
    :param min_usd_vol: mean daily volume in USD threshold to apply.
    """
    def __init__(self, 
                 tickers: List[str], 
                 fromdate, 
                 todate,
                 min_usd_vol: int=None,
                 save_ohlc: bool=False,
                 data_path: str=None):
        
        self.tickers = tickers
        self.fromdate = fromdate
        self.todate = todate
        #if save_ohlc_path:
        self.save_ohlc = save_ohlc
        
        if data_path:  #re-use saved data, avoid downloading
            self.data_path = data_path
            print('Loading Data...')
            data = pd.read_csv(self.data_path, 
                               header=[0,1], 
                               index_col=0, 
                               parse_dates=True,)
        
        elif data_path == None:  #download
            data = self.download_data(tickers_list=self.tickers,
                                      start_date=self.fromdate,
                                      end_date=self.todate,
                                      save=self.save_ohlc,)
        
        #On certain dates, yfinance returns `NaN`s for the last row of data
        #check to see if that's the case before dropping columns with
        #any`NaN` values
        lastrow = data.iloc[-1].isna().to_numpy()
        if lastrow.all() == True:
            data.drop(data.tail(1).index,inplace=True)
        #now drop `NaN`s
        data.dropna(axis=1, inplace=True) 
        self.ohlc = data
        
        #apply volume threshold if needed
        if min_usd_vol:
            self.closes = self._usd_vol_threshold(min_dollar_vol=min_usd_vol, 
                                                  data=self.ohlc)
            self.min_usd_vol = min_usd_vol
        else:
            self.closes = data['Close']
 
        #we will need the returns data later on for PCA and clustering
        self.returns = self.closes.pct_change().dropna()

    def download_data(self,
                      tickers_list: List[str],
                      start_date,
                      end_date,
                      chunk_size: int=1000,
                      save: bool=False,) -> pd.DataFrame:
        """
        Downloads OHLCV data of securities from `yfinance` and 
        into a single `pd.DataFrame`.
        Will also adjust the Open, High and Low columns.
        ---------------------------------------------------------------
        :param tickers_list: list of ticker names
        :param start_date: datetime in 'YYYY-MM-DD' format, 
                           date where OHLCV in-sample data starts
        :param end_date: datetime in 'YYYY-MM-DD' format, 
                         date where OHLCV in-sample data ends
        :param chunk_size: number of tickers in each chunk, 
                           to help with downloading
        :param save_path: optional, path to save the combined OHLCV dataframe to.
        ---------------------------------------------------------------
        Returns: two dataframes, OHLCV for in sample and OHLCV for out of sample.
        """
        ticker_chunks = [self.tickers[i*chunk_size:(i+1)*chunk_size] 
                         for i in range((len(self.tickers)+chunk_size-1)//chunk_size)]

        all_tickers_dict = {}
        for i in range(len(ticker_chunks)):
            print('Downloading chunk {0}/{1}...'.format(i+1, len(ticker_chunks)))
            chunk = yf.download(ticker_chunks[i],
                                start=self.fromdate,
                                end=self.todate,
                                show_errors=False,)

            #print('Adjusting...')
            #chunk = adj_ohlc(chunk)
            #save to dictionary
            all_tickers_dict['chunk_{0}'.format(i)] = chunk

        #combine into a single dataframe
        tickers_master = pd.concat(list(all_tickers_dict.values()), axis=1)
        print('Adjusting Prices...')
        tickers_master = self._adj_ohlc(tickers_master)
        #if needed save it for future reference
        if save:
            print('Saving data...')
            tickers_master.to_csv('data/sec_masters/sec_masters_{0}-{1}.csv'.format(self.fromdate,
                                                                    self.todate))

        return tickers_master 

    def get_clustered_pairs(self,
                            n_pca_components: int,
                            cluster_alg: Literal['OPTICS', 'DBSCAN'],
                            max_halflife: int,
                            pca_kwargs: dict={},
                            cluster_kwargs: dict={},
                            coint_significance: float=0.10,
                            max_hurst_exp: float = 0.5,
                            eps: float=0.5,) -> ClusteringOutput:
        """
        Executes the pair selection algorithm
        on `self.closes`
        The algorithm is as follows:
        1. Reduce dimensionality of returns data with PCA
        2. Use clustering to identify clusters of ETFs
        3. Within each cluster, keep pairs that satisfy:
            -Cointegration (Phillips-Ouliaris test)
            -Hurst Exponent < 0.5
            -Mean-Reversion halflife < threshold       
        ----------------
        :param n_pca_components: number of principal components
                                 to use in PCA
        :param cluster_alg: 'OPTICS' or 'DBSCAN', if DBSCAN, eps needs 
                            to be specified.
                            see docs on `sklearn.cluster.OPTICS` 
                            and `sklearn.cluster.DBSCAN` for more info
        :param max_halflife: mean-reversion halflife threshold
        :param coint_significance: p-value threshold
        :param max_hurst_exp: hurst exponent threshold
        :param pca_kwargs: additional keyword arguments to pass onto 
                           `sklearn.decomposition.PCA`
                           pass as a dict like {'whiten': True, 'tol': 0.5} etc.
        :param cluster_args: arguments to pass onto the `sklearn.cluster.OPTICS`
                             or `sklearn.cluster.DBSCAN` instances. 
                             Has to include `eps` for DBSCAN
        :param cluster_kwargs: keyword arguments to pass onto the `sklearn.cluster.OPTICS`
                               or `sklearn.cluster.DBSCAN` instances.
                               pass as a dict like {'min_samples': 10, 'max_eps': 0.5} etc.
        ----------------
        Returns: `list_pairs`, `list_pairs_dict` and `cluster_dict`:
                   `list_pairs' includes all pairs from all clusers
                   and has the following structure:
                     [('ticker_1', 'ticker_2'),
                      ('ticker_3', 'ticker_4'),...]
                    
                   `list_pairs_dict` includes all pairs from all clusters
                   and has the following structure:
                     [{'ticker_1': 'ticker', 
                       'ticker_2': 'some_other_ticker'}, 
                      {'ticker_1': 'yet_another_ticker', 
                       'ticker_2': 'yet_other_another_ticker'},...]
                 
                   `cluster_dict` is a nested dictionary with
                   highest level keys 
                   for the cluster labels: [0,1,...,num_clusters-1].
                   for each cluster label key, the second level 
                   dictionaries have the following keys and structure:
                     -'coint_score': 2D `np.array` of cointegration scores
                                     where [i,j]th element is the score
                                     for the data.keys()[i], data.keys()[j]
                                     pair
                     -'coint_pvalue': 2D `np.array` of cointegration pvalues
                                     where [i,j]th element is the pvalue
                                     for the data.keys()[i], data.keys()[j]
                                     pair
                     -'hurst_exp': 1D `np.array` of hurst exponents, [i]th
                                 element is the hurst exponent for
                                 data.keys()[i]
                     -'halflife': 1D `np.array` of mean reversion halflifes, 
                                  [i]th element is the halflife for
                                  data.keys()[i]
                     -'pairs': list of pairs in the following format:
                               [(ticker_1, ticker_2), (ticker_3, ticker_2), ...]
                     -'pairs_dict': list of pairs in the following format:
                                    [{'ticker_1': 'ticker', 
                                      'ticker_2': 'some_other_ticker'}, 
                                     {'ticker_1': 'yet_another_ticker', 
                                      'ticker_2': 'yet_other_another_ticker'},...]
        """
        returns = self.closes.pct_change().dropna()
        
        X_pca = self._fit_PCA(returns=returns, 
                              n_components=n_pca_components, 
                              **pca_kwargs)
        
        #save pca result, useful for plotting
        self.X_pca = X_pca
                    
        
        (clustered_series, 
         labels) = self._cluster(data=X_pca, 
                                 cluster_alg=cluster_alg,
                                 eps=eps,
                                 **cluster_kwargs,)
        
        #save clustering result as well, useful for plotting
        self.clustered_series = clustered_series
        self.cluster_labels = labels
        
        (list_pairs,
         list_pairs_dict, 
         cluster_dict) = self._select_pairs(closes_data=self.closes,
                                            clusters=clustered_series,
                                            coint_significance=coint_significance,
                                            max_halflife=max_halflife,
                                            max_hurst=max_hurst_exp,)
        
        self.pairs_list = list_pairs
        self.pairs_list_dict = list_pairs_dict
        self.cluster_dict = cluster_dict
        
        return list_pairs, list_pairs_dict, cluster_dict
    
    def get_cointegrated_pairs(self,
                               confidence_level: int,) -> List[Dict[str, str]]:
        """
        Tests pairs for (Johansen) cointegration and 
        returns a list of pairs that cointegrate 
        above the confidence level
        -----------------------------------------------------------
        :param conf_level: either 90, 95 or 99, confidence level
        -----------------------------------------------------------
        Returns: a list of dictionaries with ticker ids
                 [{'ticker_1': 'some_ticker', 
                   'ticker_2': 'some_other_ticker'}, 
                  {'ticker_1': 'yet_another_ticker', 
                   'ticker_2': 'yet_other_another_ticker'},...]
        """        
        pairs = self._get_coint_pairs(closes=self.closes,
                                      conf_level=confidence_level,)
        
        self.coint_pairs = pairs
        return pairs
                                      
    def _get_clustered_pairs(self, 
                             data: pd.DataFrame,
                             n_pca_components: int,
                             cluster_alg: str,
                             max_halflife: int,
                             pca_kwargs={},
                             cluster_kwargs={},
                             coint_significance: float=0.10,
                             max_hurst_exp: float=0.5,
                             eps: float=0.5,) -> ClusteringOutput:
        """
        Executes the pair selection algorithm.
        The algorithm is as follows:
        1. Reduce dimensionality of returns data with PCA
        2. Use clustering to identify clusters of ETFs
        3. Within each cluster, keep pairs that satisfy:
            -Cointegration (Phillips-Ouliaris test)
            -Hurst Exponent < 0.5
            -Mean-Reversion halflife < threshold       
        ----------------
        :param data: `pd.DataFrame` of closes
        :param n_pca_components: number of principal components
                                 to use in PCA
        :param cluster_alg: 'OPTICS' or 'DBSCAN', if DBSCAN, eps needs 
                            to be specified.
                            see docs on `sklearn.cluster.OPTICS` 
                            and `sklearn.cluster.DBSCAN` for more info
        :param max_halflife: mean-reversion halflife threshold
        :param coint_significance: p-value threshold
        :param max_hurst_exp: hurst exponent threshold
        :param pca_kwargs: additional keyword arguments to pass onto 
                           `sklearn.decomposition.PCA`
                           pass as a dict like {'whiten': True, 'tol': 0.5} etc.
        :param cluster_args: arguments to pass onto the `sklearn.cluster.OPTICS`
                             or `sklearn.cluster.DBSCAN` instances. 
                             Has to include `eps` for DBSCAN
        :param cluster_kwargs: keyword arguments to pass onto the `sklearn.cluster.OPTICS`
                               or `sklearn.cluster.DBSCAN` instances.
                               pass as a dict like {'min_samples': 10, 'max_eps': 0.5} etc.
        ----------------
        Returns: `list_pairs_dict` and `cluster_dict`:
                   `list_pairs_dict` includes all pairs from all clusters
                   and has the following structure:
                     [{'ticker_1': 'ticker', 
                       'ticker_2': 'some_other_ticker'}, 
                      {'ticker_1': 'yet_another_ticker', 
                       'ticker_2': 'yet_other_another_ticker'},...]
                 
                   `cluster_dict` is a nested dictionary with
                   highest level keys 
                   for the cluster labels: [0,1,...,num_clusters-1].
                   for each cluster label key, the second level 
                   dictionaries have the following keys and structure:
                     -'coint_score': 2D `np.array` of cointegration scores
                                     where [i,j]th element is the score
                                     for the data.keys()[i], data.keys()[j]
                                     pair
                     -'coint_pvalue': 2D `np.array` of cointegration pvalues
                                     where [i,j]th element is the pvalue
                                     for the data.keys()[i], data.keys()[j]
                                     pair
                     -'hurst_exp': 1D `np.array` of hurst exponents, [i]th
                                 element is the hurst exponent for
                                 data.keys()[i]
                     -'halflife': 1D `np.array` of mean reversion halflifes, 
                                  [i]th element is the halflife for
                                  data.keys()[i]
                     -'pairs': list of pairs in the following format:
                               [(ticker_1, ticker_2), (ticker_3, ticker_2), ...]
                     -'pairs_dict': list of pairs in the following format:
                                    [{'ticker_1': 'ticker', 
                                      'ticker_2': 'some_other_ticker'}, 
                                     {'ticker_1': 'yet_another_ticker', 
                                      'ticker_2': 'yet_other_another_ticker'},...]
        """
        returns = data.pct_change().dropna()
        
        X_pca = self._fit_PCA(returns=returns, 
                              n_components=n_pca_components, 
                              **pca_kwargs,)
        clustered_series = self._cluster(data=X_pca, 
                                         cluster_alg=cluster_alg,
                                         eps=eps,
                                         **cluster_kwargs,)
        (list_pairs,
         list_pairs_dict, 
         cluster_dict) = self._select_pairs(closes_data=data,
                                            clusters=clustered_series,
                                            coint_significance=coint_significance,
                                            max_halflife=max_halflife,
                                            max_hurst=max_hurst_exp,)
        
        return list_pairs, list_pairs_dict, cluster_dict
                               
    def _get_coint_pairs(self, 
                         closes: pd.DataFrame, 
                         conf_level: int,) -> List[Dict[str, str]]:
        """
        Tests pairs for cointegration and returns a list of pairs that cointegrate 
        above the confidence level
        -----------------------------------------------------------
        :param closes: `pd.DataFrame` of closes
        :param conf_level: either 90, 95 or 99, confidence level
        -----------------------------------------------------------
        Returns: a list of dictionaries with ticker ids
                 [{'ticker_1': 'some_ticker', 'ticker_2': 'some_other_ticker'}, 
                  {'ticker_1': 'yet_another_ticker', 'ticker_2': 'yet_other_another_ticker'},...]
        """
        if conf_level not in [90, 95, 99]:
            raise ValueError(f"Confidence level parameter `conf_level` must be one of [90, 95, 99]")
        cointegrating_pairs = []
        tickers = closes.keys().tolist()
        pairs = list(combinations(tickers, 2))
        pbar = tqdm(pairs)
        for i, (ticker_1, ticker_2) in enumerate(pbar):
            pbar.set_description(f"Confidence Level={conf_level}%")
            pair_closes = closes[[ticker_1, ticker_2]]

            #second and third parameters indicate constant term, with a lag of 1
            result = coint_johansen(pair_closes, 0, 1)
            # the 90%, 95%, and 99% confidence levels for the trace statistic and maximum 
            # eigenvalue statistic are stored in the first, second, and third column of 
            # cvt and cvm, respectively
            confidence_level_cols = {90: 0,
                                     95: 1,
                                     99: 2,}
            confidence_level_col = confidence_level_cols[conf_level]
            trace_crit_value = result.cvt[:, confidence_level_col]
            eigen_crit_value = result.cvm[:, confidence_level_col]

            # The trace statistic and maximum eigenvalue statistic are stored in lr1 and lr2;
            # see if they exceeded the confidence threshold
            if np.all(result.lr1 >= trace_crit_value) and np.all(result.lr2 >= eigen_crit_value):
                cointegrating_pairs.append(dict(ticker_1=ticker_1,
                                                ticker_2=ticker_2,))
        print('There are {0} cointegrating pairs'.format(len(cointegrating_pairs)), 
              'at {0}% confidence level'.format(conf_level))

        return cointegrating_pairs
 
    def _adj_ohlc(self, 
                  data: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts 'Open', 'High', 'Low' data using the adjustment factor.
        Works for multi-indexed `pd.DataFrame`
        Yahoo Finance data provides `Adj Close` but does not provide
        adjusted values for other prices. 
        -------------------
        :param data: multi-indexed `pd.DataFrame` with tickers at level 1.
        -------------------
        Returns: multi-indexed `pd.DataFrame` with adjusted 'Open', 'High', 'Low' columns
        """
        adj_factor = data['Adj Close']/data['Close']
        data['Open'] = adj_factor*data['Open']
        data['High'] = adj_factor*data['High']
        data['Low'] = adj_factor*data['Low']
        data['Close'] = data['Adj Close']
        data.drop('Adj Close', axis=1, inplace=True)
        return data

    def _usd_vol_threshold(self,
                           min_dollar_vol: int, 
                           data: pd.DataFrame,) -> pd.DataFrame:
        """
        Selects securities with average daily volume in dollars larger than 
        the threshold value
        ---------------------------------------------------------
        :param min_dollar_vol: threshold USD volume, int
        :param master_data: securities master dataframe
        ---------------------------------------------------------
        Returns: a `pd.DataFrame` with closing prices for tickers having average 
                 daily dollar volumes larger than the threshold dollar volume value
                 indexed by date.
        """
        closes = data['Close']
        volume = data['Volume']
        mean_dollar_vol = (closes*volume).mean()
        adequate_dollar_vol = mean_dollar_vol[mean_dollar_vol >= min_dollar_vol]
        adq_dollar_vol_tickers = adequate_dollar_vol.keys().tolist()
        print('There are {0} tickers with adequate'.format(len(adq_dollar_vol_tickers)), 
              'average daily USD volume of {0}'.format(min_dollar_vol))
        return closes[adq_dollar_vol_tickers] 
    
    def _fit_PCA(self,
                 returns: pd.DataFrame, 
                 n_components: int, 
                 **kwargs,) -> np.array:
        """
        Performs PCA on closing prices data. 
        Performs standard scaling before fitting.
        -----------------------------------
        :param returns: `pd.DataFrame` of returns for tickers in universe
        :param n_components: Number of components to keep
        :param kwargs: additional keyword arguments to pass onto 
                       `sklearn.decomposition.PCA`
        ------------------------------------
        Returns: `np.array` of PCA components
        """
        scaler = preprocessing.StandardScaler()
        scaled_returns = pd.DataFrame(scaler.fit_transform(returns),
                                      columns = returns.columns,
                                      index = returns.index,)
        
        pca = PCA(n_components=n_components, **kwargs)
        pca.fit(scaled_returns)
        X = pca.components_.T  
        return X
    
    def _cluster(self,
                 data: np.array, 
                 cluster_alg: str,
                 eps:float=0.5, 
                 **kwargs,) -> pd.Series:
        """
        Performs clustering on PCA transformed returns data
        ----------
        :param data: `np.array` of PCA reduced returns data
        :param cluster_alg: 'OPTICS' or 'DBSCAN', if DBSCAN, eps needs 
                            to be specified.
                            see docs on `sklearn.cluster.OPTICS` 
                            and `sklearn.cluster.DBSCAN` for more info
        :param eps: `eps` for DBSCAN
        :param kwargs: keyword arguments to pass onto the `sklearn.cluster.OPTICS`
                       or `sklearn.cluster.DBSCAN` instances.
        ----------
        Returns: A `pd.Series` with tickers as index and corresponding cluster label
        """
        if cluster_alg == 'OPTICS':
            clst = OPTICS(**kwargs)
        elif cluster_alg == 'DBSCAN':
            clst = DBSCAN(eps=eps, **kwargs)
        else:
            raise ValueError('Only OPTICS or DBSCAN is supported')
        clusters = clst.fit(data)
        labels = clusters.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print ("\nClusters discovered: %d" % n_clusters_)
        clustered_series = pd.Series(index=self.returns.columns, data=labels.flatten())
        clustered_series = clustered_series[clustered_series != -1]
        return clustered_series, labels
    
    def _select_pairs(self, 
                      closes_data: pd.DataFrame, 
                      clusters: pd.Series, 
                      coint_significance: float, 
                      max_halflife: int, 
                      max_hurst=0.5,) -> ClusteringOutput:
        """
        Selects ticker pairs in the following way:
        1. Check for cointegration using the Phillips Ouliaris test
        2. Calculate the Hurst exponent of the spread
           and check if its less than 0.5 (default)
        3. Calculate the mean-reversion half-life and check if 
           it is less than the threshold
        ----------------
        :param data: closes data for the ETFs
        :param clusters: `pd.Series` with ETF tickers as index and cluster labels 
        :param coint_significance: p-value threshold
        :param max_halflife: mean-reversion halflife threshold
        :param max_hurst: hurst exponent threshold
        ----------------
        Returns: `list_pairs_dict` and `cluster_dict`:
                   `list_pairs_dict` includes all pairs from all clusters
                   and has the following structure:
                     [{'ticker_1': 'ticker', 
                       'ticker_2': 'some_other_ticker'}, 
                      {'ticker_1': 'yet_another_ticker', 
                       'ticker_2': 'yet_other_another_ticker'},...]
                 
                   `cluster_dict` is a nested dictionary with
                   highest level keys 
                   for the cluster labels: [0,1,...,num_clusters-1].
                   for each cluster label key, the second level 
                   dictionaries have the following keys and structure:
                     -'coint_score': 2D `np.array` of cointegration scores
                                     where [i,j]th element is the score
                                     for the data.keys()[i], data.keys()[j]
                                     pair
                     -'coint_pvalue': 2D `np.array` of cointegration pvalues
                                     where [i,j]th element is the pvalue
                                     for the data.keys()[i], data.keys()[j]
                                     pair
                     -'hurst_exp': 1D `np.array` of hurst exponents, [i]th
                                 element is the hurst exponent for
                                 data.keys()[i]
                     -'halflife': 1D `np.array` of mean reversion halflifes, 
                                  [i]th element is the halflife for
                                  data.keys()[i]
                     -'pairs': list of pairs in the following format:
                               [(ticker_1, ticker_2), (ticker_3, ticker_2), ...]
                     -'pairs_dict': list of pairs in the following format:
                                    [{'ticker_1': 'ticker', 
                                      'ticker_2': 'some_other_ticker'}, 
                                     {'ticker_1': 'yet_another_ticker', 
                                      'ticker_2': 'yet_other_another_ticker'},...]
        """
        def check_pairs(data: pd.DataFrame, 
                        significance: float, 
                        max_mr_halflife: int, 
                        max_hurst_exp: float,) -> dict:
            
            n = data.shape[1]
            score_matrix = np.zeros((n, n))
            pvalue_matrix = np.ones((n, n))
            hurst_matrix = np.ones((n,n))
            halflife_matrix = np.ones((n,n))
            keys = data.keys()
            pairs_dict = []
            pairs = []
            pair_indices = list(combinations(range(n), 2))
            pbar = tqdm(pair_indices)
            for k, (i, j) in enumerate(pbar):
                pbar.set_description("Pair {0}-{1}".format(keys[i],
                                                           keys[j]))
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                spread = S1 - S2
                hurst_exponent = _get_hurst_exponent(spread, 
                                                        max_lag=20)
                halflife = _get_mean_reversion_halflife(spread)
                coint_result_1 = phillips_ouliaris(S1, S2, trend='ct')
                coint_score_1 = coint_result_1.stat
                coint_pvalue_1 = coint_result_1.pvalue
                coint_result_2 = phillips_ouliaris(S2, S1, trend='ct')
                coint_pvalue_2 = coint_result_2.pvalue
                score_matrix[i, j] = coint_score_1
                pvalue_matrix[i, j] = coint_pvalue_1
                hurst_matrix[i, j] = hurst_exponent
                halflife_matrix[i, j] = halflife
                if ((coint_pvalue_1 <= significance) & 
                    (coint_pvalue_2 <= significance) & 
                    (hurst_exponent <= max_hurst_exp) & 
                    (halflife <= max_mr_halflife)):
                    pairs_dict.append(dict(ticker_1=keys[i],
                                            ticker_2=keys[j]))
                    pairs_dict.append(dict(ticker_1=keys[j],
                                            ticker_2=keys[i]))                    
                    pairs.append((keys[i], keys[j]))
                    pairs.append((keys[j], keys[i]))
            result = {'coint_score': score_matrix,
                      'coint_pvalue': pvalue_matrix,
                      'hurst_exp': hurst_matrix,
                      'halflife': halflife_matrix,
                      'pairs': pairs,
                      'pairs_dict': pairs_dict}
            return result    
    
        counts = clusters.value_counts()
        ticker_count_reduced = counts[(counts>1) & (counts<=9999)]
        cluster_dict = {}
        for i, which_clust in enumerate(ticker_count_reduced.index):
            tickers = clusters[clusters == which_clust].index
            results = check_pairs(data=closes_data[tickers].dropna(),
                                  significance=coint_significance,
                                  max_mr_halflife=max_halflife,
                                  max_hurst_exp=max_hurst)
            cluster_dict[which_clust] = {}
            cluster_dict[which_clust]['score_matrix'] = results['coint_score']
            cluster_dict[which_clust]['pvalue_matrix'] = results['coint_pvalue']
            cluster_dict[which_clust]['hurst_exponent'] = results['hurst_exp']
            cluster_dict[which_clust]['halflife'] = results['halflife']
            cluster_dict[which_clust]['pairs'] = results['pairs']
            cluster_dict[which_clust]['pairs_dict'] = results['pairs_dict']

        list_pairs = []
        for clust in cluster_dict.keys():
            list_pairs.extend(cluster_dict[clust]['pairs'])
        list_pairs_dict = []
        for clust in cluster_dict.keys():
            list_pairs_dict.extend(cluster_dict[clust]['pairs_dict'])

        return list_pairs, list_pairs_dict, cluster_dict
    
    #Plotting methods. TSNE plotting might require fine-tuning parameters
    #Better to leave the function arguments exposed opposed to setting them
    #during class init.
    
    def plot_cluster_TSNE(self, 
                          tsne_kwargs: dict={}, 
                          save_path:str=None):
        """
        Plots the TSNE plot of clustered tickers and saves it
        --------------------
        :param tsne_kwargs: kwargs to pass onto the `sklearn.manifold.TSNE` 
                            instance in dict format. 
        :param save_path: path to output
        """
        clustered_tickers = list(self.clustered_series.index)
        X_pca_df = pd.DataFrame(index=self.returns.T.index,
                                data=self.X_pca)
        X_clust_tickers = X_pca_df.loc[clustered_tickers]
        tsne = TSNE(n_components=2, **tsne_kwargs)
        
        X_tsne_tick = tsne.fit_transform(X_clust_tickers)
        
        plt.figure(figsize=(8, 8), facecolor='white')
        plt.clf()
        plt.axis('off')

        plt.scatter(X_tsne_tick[:, 0],
                    X_tsne_tick[:, 1],
                    s=100,
                    c=self.cluster_labels[self.cluster_labels != -1],
                    cmap=cm.plasma)
        
        plt.title('T-SNE plot of all ETF clusters');

        for index, (x_pos, y_pos, label) in enumerate(zip(X_tsne_tick[:, 0], X_tsne_tick[:, 1], 
                                                          clustered_tickers)):

            dx = x_pos - X_tsne_tick[:, 0]
            dx[index] = 0.1
            dy = y_pos - X_tsne_tick[:, 1]
            dy[index] = 0.1
            this_dx = dx[np.argmin(np.abs(dy))]
            this_dy = dy[np.argmin(np.abs(dx))]
            if this_dx > 0:
                horizontalalignment = 'left'
                x_pos = x_pos + 0.05
            else:
                horizontalalignment = 'right'
                x_pos = x_pos - 0.05
            if this_dy > 0:
                verticalalignment = 'bottom'
                y_pos = y_pos + 0.05
            else:
                verticalalignment = 'top'
                y_pos = y_pos - 0.05

            plt.text(x_pos, y_pos, label, size=8,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           alpha=.6))
        plt.show()
        if save_path:    
            plt.savefig(save_path)
            
    def plot_pairs_TSNE(self,
                        tsne_kwargs:dict = {},
                        save_path:str=None):
        """
        Plots the TSNE plot of paired tickers from clusters and saves it
        --------------------
        :param tsne_kwargs: kwargs to pass onto the `sklearn.manifold.TSNE` 
                            instance in dict format. 
        :param save_path: path to output
        """        
        stocks = np.unique(self.pairs_list)
        X_pca_df = pd.DataFrame(index=self.returns.T.index,
                                data=self.X_pca)
        
        
        in_pairs_series = self.clustered_series.loc[stocks]
        stocks = list(np.unique(self.pairs_list))
        X_pairs = X_pca_df.loc[stocks]  
        
        tsne = TSNE(n_components=2, **tsne_kwargs)
        X_tsne = tsne.fit_transform(X_pairs)
        
        plt.figure(figsize=(8, 8), facecolor='white')
        plt.clf()
        plt.axis('off')
        for pair in self.pairs_list:
            ticker1 = pair[0]
            loc1 = X_pairs.index.get_loc(ticker1)
            x1, y1 = X_tsne[loc1, :]

            ticker2 = pair[1]
            loc2 = X_pairs.index.get_loc(ticker2)
            x2, y2 = X_tsne[loc2, :]

            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray')

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, 
                    c=in_pairs_series.values.ravel(), cmap=cm.Paired)
        plt.title('T-SNE Visualization of Validated Pairs')


        # Add the participant names as text labels for each point
        for index, (x_pos, y_pos, label) in enumerate(zip(X_tsne[:, 0], 
                                                          X_tsne[:, 1], 
                                                          self.pairs_list)):

            dx = x_pos - X_tsne[:, 0]
            dx[index] = 0.1
            dy = y_pos - X_tsne[:, 1]
            dy[index] = 0.1
            this_dx = dx[np.argmin(np.abs(dy))]
            this_dy = dy[np.argmin(np.abs(dx))]
            if this_dx > 0:
                horizontalalignment = 'left'
                x_pos = x_pos + 0.0001
            else:
                horizontalalignment = 'right'
                x_pos = x_pos - 0.0001
            if this_dy > 0:
                verticalalignment = 'bottom'
                y_pos = y_pos + 0.0001
            else:
                verticalalignment = 'top'
                y_pos = y_pos - 0.0001

            plt.text(x_pos, y_pos, label, size=8,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           alpha=.6))

        # Show the plot
        plt.show()        
        if save_path:    
            plt.savefig(save_path)        

##Helper Functions:        
    
def _get_hurst_exponent(time_series: np.array, max_lag=20) -> float:
    """
    Returns the Hurst Exponent of the time series
    ------------------
    :param time_series: `np.array` time-series(prices) data
    :param max_lag: maximum lag to search for
    ------------------
    Returns: The Hurst exponent.
    """
    
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(time_series[lag:].to_numpy(), 
                              time_series[:-lag].to_numpy())) for lag in lags]
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]

def _get_mean_reversion_halflife(time_series: np.array) -> float:
    """
    Returns the Half-Life of mean reversion for the time series
    ------------------
    :param time_series: `np.array` time-series(prices) data
    ------------------
    Returns: Half life of mean reversion in units of the original time series.
    """
    series = time_series.to_numpy()
    series_lag = np.roll(series,1)
    series_lag[0] = 0
    series_ret = series - series_lag
    series_ret[0] = 0
    
    series_lag2 = sm.add_constant(series_lag)

    model = sm.OLS(series_ret,series_lag2)
    res = model.fit()

    halflife = -np.log(2) / res.params[1] 
    return halflife

def _get_ticker_list(ticker_name_path: str) -> List[str]:
    """
    Creates a list of ticker names.
    -------------------------------
    :param ticker_name_path: the path to the .csv file with the ticker names
    -------------------------------
    Returns: A list with ticker names to download OHLCV data for
    """
    tickers_df = pd.read_csv(ticker_name_path)
    tickers_list = sorted(list(set(tickers_df['Symbol'].tolist())))
    return tickers_list

def _extract_ticker(ticker: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts individual OHLCV data for a given ticker name
    ------------------------------------------------------
    :param ticker: ticker name, string format 'SPY' for example
    :param master_data: dataframe containing OHLCV data for all tickers
    ------------------------------------------------------
    Returns: `pd.DataFrame` containing OHLCV data for the ticker
    """ 
    data_groupedby_ticker = data.reorder_levels([1,0], axis=1)
    ticker_OHLCV = data_groupedby_ticker[ticker]
    return ticker_OHLCV

def p_value_at_risk(returns: np.array, alpha=0.95) -> float:
    """
    Calculates VaR (Value at Risk) as a percentage from (de-meaned) returns.
    ------
    :param returns: np.array of de-meaned returns
    :param alpha: coverage percentage
    ------
    Returns: VaR as a float, -0.10 -> 10%
    """
    returns = np.nan_to_num(returns, nan=0.0)
    pVaR = np.percentile(returns, 100 * (1-alpha))
    return pVaR

def p_c_value_at_risk(returns: np.array, alpha=0.95) -> float:
    """
    Calculates CVaR (Conditional Value at Risk) as a 
    percentage from (de-meaned) returns.
    ------
    :param returns: np.array of de-meaned returns
    :param alpha: coverage percentage
    ------
    Returns: CVaR as a float, -0.10 -> 10%
    """
    pVaR = p_value_at_risk(returns=returns, alpha=alpha)
    cVaR = np.nanmean(returns[returns <= pVaR])
    return cVaR