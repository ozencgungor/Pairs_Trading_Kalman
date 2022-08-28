from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import argparse
import datetime

import backtrader as bt
import pair_pipeline as psel

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (18, 18)
plt.ioff()

from tqdm import tqdm

import quantstats as qs

from cashmarket import CashMarket

class KalmanPairs(bt.Strategy):
    packages = (('numpy', 'np'), 
                'math', 
                ('pandas', 'pd'),
               )
    params = dict(delta=1e-3,
                  vt=1e-2,
                  quantity=100,
                  burn_in=10,
                  threshold=0.8,
                 )
    
    def __init__(self):
        self.wt = self.p.delta / (1 - self.p.delta) * np.eye(2)
        self.theta = np.zeros(2)
        self.P = np.ones((2, 2))
        self.R = np.ones((2, 2))
        
        self.d0_prev = self.data0(-1) # data0 yesterday's price
        self.d1_prev = self.data1(-1) # data1 yesterday's price        
    
        self.position_type = None
        self.quantity = self.params.quantity
        
        self.out_of_market = 0
    
    def next(self):
        if not self.position_type:
            self.out_of_market = self.out_of_market + 1
        
        F = np.asarray([self.data0[0], 1.0]).reshape((1, 2))
        y = self.data1[0]
        
        self.R = self.P + self.wt
        
        yhat = F.dot(self.theta)
        et = y - yhat
        
        # Q_t is the variance of the prediction of observations and hence
        # \sqrt{Q_t} is the standard deviation of the predictions
        Qt = F.dot(self.R).dot(F.T) + self.p.vt
        sqrt_Qt = np.sqrt(Qt)
        
        Kt = self.R.dot(F.T) / Qt  # Kalman gain
        
        self.theta += Kt.flatten() * et  # State update
        
        self.P = self.R - Kt * F.dot(self.R)
        
        sizer = self.getsizer() # get the sizer 
        perc = sizer.params.percents # get the stake
        cash = self.broker.get_cash()
        if self.out_of_market >= 21: #1 month
            #print('Decreasing trading threshold...')
            self.p.threshold = self.p.threshold/1.4
            #print('New threshold is {0}'.format(self.p.threshold))
        if len(self) >= self.p.burn_in:
            if self.position:
                if (self.position_type == 'long' and et >= -self.p.threshold*sqrt_Qt):
                    self.close(self.data1)
                    self.close(self.data0)
                    self.potision_type = None
                if (self.position_type == 'short' and et <= self.p.threshold*sqrt_Qt):
                    self.close(self.data0)
                    self.close(self.data1)
                    self.position_type = None
            else:
                if et < -self.p.threshold*sqrt_Qt:
                    stake = int(math.floor((cash/self.data1.close[0])*(perc/100)))
                    #stake = self.quantity
                    hedge = int(math.floor(self.theta[0]*stake))
                    self.sell(data=self.data0, size=hedge)
                    self.buy(data=self.data1, size=stake)
                    self.position_type = 'long'
                    self.out_of_market = 0
                if et > self.p.threshold*sqrt_Qt:
                    stake = int(math.floor((cash/self.data1.close[0])*(perc/100)))
                    #stake = self.quantity
                    hedge = int(math.floor(self.theta[0]*stake))
                    self.sell(data=self.data1, size=stake)
                    self.buy(data=self.data0, size=hedge)
                    self.position_type = 'short'
                    self.out_of_market = 0


def run_test(args=None):
    args = parse_args(args)
    
    #parse PCA-cluster-pair-select arguments:
    pca_kwargs = eval( 'dict(' + args.PCAparams + ')')
    cluster_kwargs = eval( 'dict(' + args.clusterparams + ')')
    pair_selection_params = eval( 'dict(' + args.pairselectionparams + ')')
    #check pair_selection_param keys:, if not given, put default values in
    default_sel_params = {"n_pca_components": 0.80,
                          "cluster_alg": 'OPTICS',
                          "max_halflife": 126,
                          "coint_significance": 0.10,
                          "max_hurst_exp": 0.5,
                          "dbscan_eps": 0.5,}
    
    for key in default_sel_params.keys():
        if key not in pair_selection_params.keys():
            pair_selection_params[key] = default_sel_params[key]
    
    ticker_list = psel._get_ticker_list(ticker_name_path='data/etf-list.csv')
    ticker_list.remove('IAU')  #bad yfinance data
    ticker_list.remove('SDOW')  #bad yfinance data
    
    in_sample = psel.PairSelection(tickers=ticker_list,
                                   fromdate=args.insamplestartdate,
                                   todate=args.insampleenddate,
                                   min_usd_vol=int(args.minusdvol),
                                   save_ohlc=args.saveohlc,
                                   data_path=args.datapath,)
    
    in_sample_ohlc = in_sample.ohlc 
    
    (pairs_list,
     pairs_list_dict, 
     cluster_dict,) = in_sample.get_clustered_pairs(
                            n_pca_components=pair_selection_params['n_pca_components'],
                            cluster_alg=pair_selection_params['cluster_alg'],
                            max_halflife=pair_selection_params['max_halflife'],
                            pca_kwargs=pca_kwargs,
                            cluster_kwargs=cluster_kwargs,
                            coint_significance=pair_selection_params['coint_significance'],
                            max_hurst_exp=pair_selection_params['max_hurst_exp'],
                            eps=pair_selection_params['dbscan_eps'],)
    
    
    
    # Data feed kwargs
    kwargs = dict()
    
    # Parse from/to-date
    dtfmt = '%Y-%m-%d'
    for a, d in ((getattr(args, x), x) for x in ['insamplestartdate', 
                                                 'insampleenddate', 
                                                 'outsampleenddate',]):
        if a:
            kwargs[d] = datetime.datetime.strptime(a, dtfmt)
            
    # Parse analysis timeframe:
    if args.analysistimeframe not in ['Daily', 'Weekly', 'Monthly', 'Yearly']:
        raise ValueError("Analysis timeframe '--analysistimeframe' must be one of",
                         "'Daily', 'Weekly', 'Monthly', 'Yearly'")
    if args.analysistimeframe == 'Daily':
        time_frame = bt.TimeFrame.Days
    if args.analysistimeframe == 'Weekly':
        time_frame = bt.TimeFrame.Weeks
    if args.analysistimeframe == 'Monthly':
        time_frame = bt.TimeFrame.Months
    if args.analysistimeframe == 'Yearly':
        time_frame = bt.TimeFrame.Years
    
    
    #stuff to create the analysis dataframe:
    tickers_1 = []
    tickers_2 = []
    sharpes = []
    tot_returns = []
    norm_returns = []
    drawdowns = []
    daily_returns = []
    VaRs = []
    CVaRs = []
    final_vals = []
    
    
    
    pairs_pbar = tqdm(pairs_list_dict)
    for i, tickers in enumerate(pairs_pbar):
        ticker_1 = tickers['ticker_1']
        ticker_2 = tickers['ticker_2']
        tickers_1.append(ticker_1)                                         
        tickers_2.append(ticker_2)
        
        pairs_pbar.set_description('In-Sample Test: {0}-{1}'.format(ticker_1,
                                                                    ticker_2)) 
                                                 
        cerebro = bt.Cerebro()

        ticker_1_df = psel._extract_ticker(ticker=ticker_1, 
                                           data=in_sample_ohlc)

        ticker_2_df = psel._extract_ticker(ticker=ticker_2, 
                                           data=in_sample_ohlc)

        data0 = bt.feeds.PandasData(dataname=ticker_1_df)
        cerebro.adddata(data0, name='{0}'.format(ticker_1))
        
        data1 = bt.feeds.PandasData(dataname=ticker_2_df)
        data1.plotmaster = data0
        cerebro.adddata(data1, name='{0}'.format(ticker_2))

        # Broker
        cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))

        # Sizer
        cerebro.addsizer(bt.sizers.PercentSizer, **eval('dict(' + args.sizer + ')'))

        # Strategy
        cerebro.addstrategy(KalmanPairs, **eval('dict(' + args.strat + ')'))
        
        cerebro.addanalyzer(bt.analyzers.DrawDown)
        
        cerebro.addanalyzer(CashMarket, _name='cashmarket')

        # Execute
        in_sample_tests = cerebro.run(**eval('dict(' + args.cerebro + ')'))
        in_sample_test = in_sample_tests[0]
        
        df_values = pd.DataFrame(in_sample_test.analyzers.getbyname("cashmarket").get_analysis()).T
        df_values = df_values.iloc[:, 1]
        qs.extend_pandas()
        qs_returns = qs.utils.to_returns(df_values)
        qs_returns.index = pd.to_datetime(qs_returns.index)
        #de-mean the returns:
        qs_returns_no_mean = (qs_returns - qs_returns.mean()).to_numpy()
        
        sharpe = qs.stats.sharpe(returns=qs_returns)
        sharpes.append(sharpe)
        
        var = psel.p_value_at_risk(returns=qs_returns_no_mean, alpha=0.95)
        VaRs.append(var)
        
        cvar = psel.p_c_value_at_risk(returns=qs_returns_no_mean, alpha=0.95)
        CVaRs.append(cvar)  
                                
        cagr = qs.stats.cagr(returns=qs_returns)
        norm_returns.append(cagr)
        
        drawdown = in_sample_test.analyzers.drawdown.get_analysis()['max']['drawdown']
        drawdowns.append(drawdown)
        
        final_val = cerebro.broker.getvalue()
        final_vals.append(final_val)
        
    in_sample_dict = {'Ticker_1': tickers_1,
                      'Ticker_2': tickers_2, 
                      'Sharpe_Ratio': sharpes,
                      'CAGR': norm_returns,
                      'Max_Drawdowns': drawdowns,
                      'VaR_(perc.)': VaRs,
                      'CVaR_(perc.)': CVaRs,
                      'Final_Value': final_vals,
                      }
    
    in_sample_results = pd.DataFrame(data=in_sample_dict)
    in_sample_results = in_sample_results.sort_values(by=['Final_Value'], ascending=False)
    print('Top Performers')
    top5_pairs = in_sample_results.head()
    print(top5_pairs)
    
    if args.saveinsample:
        in_sample_results.to_csv('results/insample_results_{0}-{1}'.format(args.insamplestartdate,
                                                                args.insampleenddate))
    
    top5_ticker1 = top5_pairs['Ticker_1']
    top5_ticker1_list = top5_ticker1.tolist()
    top5_ticker2 = top5_pairs['Ticker_2']
    top5_ticker2_list = top5_ticker2.tolist()
    best_pairs = [{'ticker_1': top5_ticker1_list[i], 
                   'ticker_2': top5_ticker2_list[i]} for i in range(len(top5_ticker1_list))]
    
    if args.bestpairs: #print best pairs if requested
        print(best_pairs)
    
    #test the best pairs in the backtest period: cointenddate-backtestenddate
    bt_tickers_1 = []
    bt_tickers_2 = []
    bt_sharpes = []
    bt_norm_returns = []
    bt_drawdowns = []
    bt_VaRs = []
    bt_CVaRs = []
    bt_final_vals = []
    
    best_ticker_list = list(set().union(*[{elem['ticker_1'], 
                                           elem['ticker_2']} for elem in best_pairs]))
    
    out_sample = psel.PairSelection(tickers=best_ticker_list,
                                    fromdate=args.insampleenddate,
                                    todate=args.outsampleenddate)
    
    out_sample_ohlc = out_sample.ohlc
    
    pbar = tqdm(best_pairs)
    for i, tickers in enumerate(pbar):
        ticker_1 = tickers['ticker_1']
        ticker_2 = tickers['ticker_2']
        bt_tickers_1.append(ticker_1)                                         
        bt_tickers_2.append(ticker_2)
        
        pbar.set_description('Out-Sample Test: {0}-{1}'.format(ticker_1,
                                                               ticker_2)) 
                                                 
        bt_cerebro = bt.Cerebro()

        ticker_1_df = psel._extract_ticker(ticker=ticker_1, 
                                           data=out_sample_ohlc)

        ticker_2_df = psel._extract_ticker(ticker=ticker_2, 
                                           data=out_sample_ohlc)

        data0 = bt.feeds.PandasData(dataname=ticker_1_df)
        bt_cerebro.adddata(data0, name='{0}'.format(ticker_1))
        
        data1 = bt.feeds.PandasData(dataname=ticker_2_df)
        data1.plotmaster = data0
        bt_cerebro.adddata(data1, name='{0}'.format(ticker_2))

        # Broker
        bt_cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))

        # Sizer
        bt_cerebro.addsizer(bt.sizers.PercentSizer, **eval('dict(' + args.sizer + ')'))

        # Strategy
        bt_cerebro.addstrategy(KalmanPairs, **eval('dict(' + args.strat + ')'))
        
        bt_cerebro.addanalyzer(bt.analyzers.DrawDown, _name='bt_drawdown')
        
        bt_cerebro.addanalyzer(CashMarket, _name='cashmarket')
        
        # Execute
        out_samples = bt_cerebro.run(**eval('dict(' + args.cerebro + ')'))
        out_sample = out_samples[0]
        
        df_values = pd.DataFrame(out_sample.analyzers.getbyname("cashmarket").get_analysis()).T
        df_values = df_values.iloc[:, 1]
        qs.extend_pandas()
        qs_returns = qs.utils.to_returns(df_values)
        qs_returns.index = pd.to_datetime(qs_returns.index)
        #de-mean the returns:
        qs_returns_no_mean = (qs_returns - qs_returns.mean()).to_numpy()
        
        bt_sharpe = qs.stats.sharpe(returns=qs_returns)
        bt_sharpes.append(bt_sharpe)
        
        bt_var = psel.p_value_at_risk(returns=qs_returns_no_mean, alpha=0.95)
        bt_VaRs.append(bt_var)
        
        bt_cvar = psel.p_c_value_at_risk(returns=qs_returns_no_mean, alpha=0.95)
        bt_CVaRs.append(bt_cvar)  
                                
        bt_cagr = qs.stats.cagr(returns=qs_returns)
        bt_norm_returns.append(bt_cagr)
        
        bt_drawdown = out_sample.analyzers.bt_drawdown.get_analysis()['max']['drawdown']
        bt_drawdowns.append(bt_drawdown)
        
        bt_final_val = bt_cerebro.broker.getvalue()
        bt_final_vals.append(bt_final_val)
        
        if args.plot:  # Plot if requested to
            fig = bt_cerebro.plot(**eval('dict(' + args.plot + ')'))[0][0]
            fig.savefig('results/out_sample_{0}-{1}.pdf'.format(ticker_1,
                                                                ticker_2),)
            
            qs.reports.html(qs_returns,
                            title='Strategy Tearsheet, {0}-{1}'.format(ticker_1,
                                                                       ticker_2),
                            output="qs.html", 
                            download_filename='results/ts_{0}-{1}.html'.format(ticker_1,
                                                                               ticker_2),)
            import imgkit
            imgkit.from_file('results/ts_{0}-{1}.html'.format(ticker_1,
                                                              ticker_2), 
                             'results/tearsheet_{0}-{1}.jpg'.format(ticker_1,
                                                                    ticker_2))
    
    backtest_results_dict = {'Ticker_1': bt_tickers_1,
                             'Ticker_2': bt_tickers_2, 
                             'Sharpe_Ratio': bt_sharpes,
                             'CAGR': bt_norm_returns,
                             'Max_Drawdowns': bt_drawdowns,
                             'VaR_(perc.)': bt_VaRs,
                             'CVaR_(perc.)': bt_CVaRs,
                             'Final_Value': bt_final_vals,
                             }
    
    backtest_results = pd.DataFrame(data=backtest_results_dict)
    backtest_results = backtest_results.sort_values(by=['Final_Value'], ascending=False)
    
    print(backtest_results)
    if args.savetest:
        backtest_results.to_csv('results/outsample_results_{0}-{1}'.format(args.insampleenddate,
                                                                  args.backtestenddate))


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Kalman Pairs Trading Strategy'))
    
    # Defaults for dates
    parser.add_argument('--insamplestartdate', required=False, default='2016-06-01',
                        help='Date[time] in YYYY-MM-DD format',)

    parser.add_argument('--insampleenddate', required=False, default='2020-06-01',
                        help='Date[time] in YYYY-MM-DD format',)

    parser.add_argument('--outsampleenddate', required=False, default='2022-06-01',
                        help='Date[time] in YYYY-MM-DD format',)
                                                 
    parser.add_argument('--minusdvol', required=False, default=60000000,
                        help='Minimum average USD volume, integer',)
                                                 
    parser.add_argument('--confidencelevel', required=False, default=90,
                        help='Cointegration test confidence level, 90, 95, or 99',)
    
    parser.add_argument('--analysistimeframe', required=False, default='Yearly',
                        help='Analysis timeframe for Sharpe Ratio and Returns calculation',)

    parser.add_argument('--cerebro', required=False, default='runonce=False',
                        metavar='kwargs', help='kwargs in key=value format',)

    parser.add_argument('--broker', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format',)

    parser.add_argument('--sizer', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format',)

    parser.add_argument('--strat', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format',)

    parser.add_argument('--plot', required=False, default='',
                        nargs='?', const='{}',
                        metavar='kwargs', help='kwargs in key=value format',)
    
    parser.add_argument('--bestpairs', required=False, default=True,
                        help='True to return best pairs')
    
    parser.add_argument('--clusterparams', 
                        required=False, 
                        default='',
                        nargs='?',
                        const='{}',
                        metavar='kwargs',
                        help='''
                             kwargs to pass onto `sklearn.cluster` 
                             instance in key=value format.
                             example: --clusterparams 'min_samples=10, max_eps=2.3'   
                             ''')
    
    parser.add_argument('--saveinsample', required=False, default=True,
                        help='True to save backtest results into a .csv')
    
    parser.add_argument('--PCAparams', 
                        required=False, 
                        default='',
                        nargs='?',
                        const='{}',
                        metavar='kwargs',
                        help='''
                             kwargs to pass onto `sklearn.decomposition.PCA` 
                             instance in key=value format.
                             example: --PCA 'whiten=True, n_oversamples=10'   
                             ''')
                                        
    parser.add_argument('--saveohlc', required=False, default='',
                        help='Path to save ohlc data')
    
    parser.add_argument('--savetest', required=False, default='',
                        help='Path to save ohlc data')
    
    parser.add_argument('--datapath', required=False, default='',
                        help='Path to securities master')
    
    parser.add_argument('--pairselectionparams',
                        required=False,
                        default="",
                        nargs='?',
                        const='{}',
                        metavar='kwargs',
                        help='''pair selection parameters. 
                             options are: n_pca_components, cluster_alg, max_halflife
                             coint_significance, max_hurst_exp, dbscan_eps
                             example: --pairselectionparams 'n_pca_components=15, max_halflife=21'
                             ''')

    return parser.parse_args(pargs)


if __name__ == '__main__':
    run_test()
