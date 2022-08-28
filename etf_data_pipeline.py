import pandas as pd
import numpy as np
import yfinance as yf
import yahoo_fin as yfin
import itertools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm
import backtrader as bt

def create_master_data(ticker_name_path,
                       coint_test_start,
                       coint_test_end,
                       backtest_end,
                       chunk_size=1000,
                       save_path=None):
    """
    Collects and combines OHLCV data of securities into a single `pd.DataFrame`
    ---------------------------------------------------------------
    :param ticker_name_path: a .csv file with 'Symbol' including ticker names
    :param coint_test_start: datetime in 'YYYY-MM-DD' format, 
                             date where OHLCV in-sample data starts
    :param coint_test_end: datetime in 'YYYY-MM-DD' format, 
                           date where OHLCV in-sample data ends
    :param backtest_end: datetime in 'YYYY-MM-DD' format, 
                         date where OHLCV out-sample data ends
    :param chunk_size: optional integer, number of tickers in each chunk, 
                       to help with downloading
    :param save_path: optional, path to save the combined OHLCV dataframe to.
    ---------------------------------------------------------------
    Returns: two dataframes, OHLCV for in sample and OHLCV for out of sample.
    """
    
    tickers_df = pd.read_csv(ticker_name_path)
    tickers_list = sorted(list(set(tickers_df['Symbol'].tolist())))
    ticker_chunks = [tickers_list[i*chunk_size:(i+1)*chunk_size] 
                     for i in range((len(tickers_list)+chunk_size-1)//chunk_size)]
    
    all_tickers_dict = {}
    for i in range(len(ticker_chunks)):
        print('Downloading chunk {0}/{1}...'.format(i+1, len(ticker_chunks)))
        chunk = yf.download(ticker_chunks[i],
                            start=coint_test_start,
                            end=backtest_end,
                            show_errors=False)
        #rename 'Adj Close' to 'Adj_Close'
        print('Re-naming...')
        chunk = chunk.rename(columns={'Adj Close':'Adj_Close'})
        #save to dictionary
        all_tickers_dict['chunk_{0}'.format(i)] = chunk

    #combine into a single dataframe
    tickers_master = pd.concat(list(all_tickers_dict.values()), axis=1)
    #if needed save it for future reference
    if save_path:
        tickers_master.to_csv('save_path')
    #separate into in_sample and out_sample dataframes:
    tickers_master_in_sample = tickers_master[coint_test_start:coint_test_end]
    tickers_master_out_sample = tickers_master[coint_test_end:backtest_end]
    
    return tickers_master_in_sample, tickers_master_out_sample

def get_ticker_list(ticker_name_path):
    """
    -----
    """
    tickers_df = pd.read_csv(ticker_name_path)
    tickers_list = sorted(list(set(tickers_df['Symbol'].tolist())))
    return tickers_list
    
def download_data(tickers_list,
                  start_date,
                  end_date,
                  chunk_size=1000,
                  save_path=None):
    """
    Collects and combines OHLCV data of securities into a single `pd.DataFrame`
    ---------------------------------------------------------------
    :param ticker_name_path: a .csv file with 'Symbol' including ticker names
    :param start_date: datetime in 'YYYY-MM-DD' format, 
                       date where OHLCV in-sample data starts
    :param end_date: datetime in 'YYYY-MM-DD' format, 
                     date where OHLCV in-sample data ends
    :param chunk_size: optional integer, number of tickers in each chunk, 
                       to help with downloading
    :param save_path: optional, path to save the combined OHLCV dataframe to.
    ---------------------------------------------------------------
    Returns: two dataframes, OHLCV for in sample and OHLCV for out of sample.
    """
    ticker_chunks = [tickers_list[i*chunk_size:(i+1)*chunk_size] 
                     for i in range((len(tickers_list)+chunk_size-1)//chunk_size)]
    
    all_tickers_dict = {}
    for i in range(len(ticker_chunks)):
        print('Downloading chunk {0}/{1}...'.format(i+1, len(ticker_chunks)))
        chunk = yf.download(ticker_chunks[i],
                            start=start_date,
                            end=end_date,
                            show_errors=False)
        #rename 'Adj Close' to 'Adj_Close'
        print('Re-naming...')
        chunk = chunk.rename(columns={'Adj Close':'Adj_Close'})
        #save to dictionary
        all_tickers_dict['chunk_{0}'.format(i)] = chunk

    #combine into a single dataframe
    tickers_master = pd.concat(list(all_tickers_dict.values()), axis=1)
    #if needed save it for future reference
    if save_path:
        tickers_master.to_csv('save_path')
    
    return tickers_master

#we want to only take into account ETFs with some minimum dollar volume per day
def usd_vol_threshold(min_dollar_vol, master_data):
    """
    Selects securities with average daily volume in dollars larger than the threshold 
    value
    ---------------------------------------------------------
    :param min_dollar_vol: threshold USD volume, float or int
    :param master_data: securities master dataframe
    ---------------------------------------------------------
    Returns: a `pd.Series` with ticker names as `index` for tickers having average 
             daily dollar volumes larger than the threshold dollar volume value
    """
    closes = master_data['Adj_Close']
    volume = master_data['Volume']
    mean_dollar_volumes = (closes*volume).mean()
    adequate_dollar_volumes = mean_dollar_volumes[mean_dollar_volumes >= min_dollar_vol]
    print('There are {0} tickers with adequate'.format(len(adequate_dollar_volumes)), 
          'average daily USD volume of {0}'.format(min_dollar_vol))
    return adequate_dollar_volumes

def get_pairs(adequate_dollar_volumes):
    """
    Generates all possible pairs of tickers
    ---------------------------------------
    :param adequate_dollar_volumes: `pd.Series` with stock tickers as `index`
    ---------------------------------------
    Returns: List of tuples [('ticker_1', 'ticker_2'), ('ticker_1', 'ticker_3'),...]
    """
    print('Generating ticker pairs...')
    out = list(itertools.combinations(adequate_dollar_volumes.index, 2))
    print('There are {0} pairs'.format(len(out)))
    return out

def coint_pairs(pairs, closes, conf_level, min_data):
    """
    Tests pairs for cointegration and returns a list of pairs that cointegrate 
    above the confidence level
    -----------------------------------------------------------
    :param pairs: list of tuples of pairs
    :param conf_level: either 90, 95 or 99, confidence level
    -----------------------------------------------------------
    Returns: a list of dictionaries with ticker ids
             [{'ticker_1': 'some_ticker', 'ticker_2': 'some_other_ticker'}, 
              {'ticker_1': 'yet_another_ticker', 'ticker_2': 'yet_other_another_ticker'},...]
    """
    if conf_level not in [90, 95, 99]:
        raise ValueError(f"Confidence level parameter `conf_level` must be one of [90, 95, 99]")
    cointegrating_pairs = []
    pbar = tqdm(pairs)
    for i, (ticker_1, ticker_2) in enumerate(pbar):
        pbar.set_description(f"Confidence Level={conf_level}%")
        pair_closes = closes[[ticker_1, ticker_2]].dropna()
        #skip if we have less than `min_data` data points
        if len(pair_closes) < min_data:
            continue
        
        #second and third parameters indicate constant term, with a lag of 1
        result = coint_johansen(pair_closes, 1, 1)
        # the 90%, 95%, and 99% confidence levels for the trace statistic and maximum 
        # eigenvalue statistic are stored in the first, second, and third column of 
        # cvt and cvm, respectively
        confidence_level_cols = {90: 0,
                                 95: 1,
                                 99: 2}
        confidence_level_col = confidence_level_cols[conf_level]
        trace_crit_value = result.cvt[:, confidence_level_col]
        eigen_crit_value = result.cvm[:, confidence_level_col]

        # The trace statistic and maximum eigenvalue statistic are stored in lr1 and lr2;
        # see if they exceeded the confidence threshold
        if np.all(result.lr1 >= trace_crit_value) and np.all(result.lr2 >= eigen_crit_value):
            cointegrating_pairs.append(dict(ticker_1=ticker_1,
                                            ticker_2=ticker_2))
    print('There are {0} cointegrating pairs'.format(len(cointegrating_pairs)), 
          'at {0}% confidence level'.format(conf_level))
                                                                            
    return cointegrating_pairs

def extract_ticker(ticker, master_data):
    """
    Extracts individual OHLCV data for a given ticker name
    ------------------------------------------------------
    :param ticker: ticker name, string format 'SPY' for example
    :param master_data: dataframe containing OHLCV data for all tickers
    ------------------------------------------------------
    Returns: dataframe containing OHLCV data for the ticker
    """ 
    master_data_grouped_ticker = master_data.reorder_levels([1,0], axis=1)
    ticker_OHLCV = master_data_grouped_ticker[ticker]
    return ticker_OHLCV

#define a `bt.feed` class to handle `pd.DataFrame` objects with 'Adj_Close' as one of the fields
class PandasDataAdj(bt.feed.DataBase):
    """"
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    """

    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', None),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('adj_close', -1),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
    )
    
def p_value_at_risk(returns, alpha=0.95):
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

def p_c_value_at_risk(returns, alpha=0.95):
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