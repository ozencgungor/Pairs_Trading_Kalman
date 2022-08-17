from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import argparse
import datetime

import backtrader as bt

import etf_data_pipeline as pipe

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30, 18)

from tqdm import tqdm

class KalmanMovingAverage(bt.indicators.MovingAverageBase):
    packages = ('pykalman',)
    frompackages = (('pykalman', [('KalmanFilter', 'KF')]),)
    lines = ('kma',)
    alias = ('KMA',)
    params = (
        ('initial_state_covariance', 1.0),
        ('observation_covariance', 1.0),
        ('transition_covariance', 0.05),
    )
    plotinfo = dict(subplot=False, plot=False)

    def __init__(self):
        self.addminperiod(self.p.period)  # when to deliver values
        self._dlast = self.data(-1)  # get previous day value

    def nextstart(self):
        self._k1 = self._dlast[0]
        self._c1 = self.p.initial_state_covariance

        self._kf = pykalman.KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            observation_covariance=self.p.observation_covariance,
            transition_covariance=self.p.transition_covariance,
            initial_state_mean=self._k1,
            initial_state_covariance=self._c1,
        )

        self.next()

    def next(self):
        k1, self._c1 = self._kf.filter_update(self._k1, self._c1, self.data[0])
        self.lines.kma[0] = self._k1 = k1


class NumPy(object):
    packages = (('numpy', 'np'),)


class KalmanFilterInd(bt.Indicator, NumPy):
    _mindatas = 2  # needs at least 2 data feeds

    packages = ('pandas',)
    lines = ('et', 'sqrt_qt', 'theta')

    params = dict(
        delta=7e-5,
        vt=8e-4,
    )
    
    plotinfo = dict(subplot=False, plot=False)

    def __init__(self):
        self.wt = self.p.delta / (1 - self.p.delta) * np.eye(2)
        self.theta = np.zeros(2)
        self.R = None

        self.d1_prev = self.data1(-1)  # data1 yesterday's price

    def next(self):
        F = np.asarray([self.data0[0], 1.0]).reshape((1, 2))
        y = self.d1_prev[0]

        if self.R is not None:  # self.R starts as None, self.C set below
            self.R = self.C + self.wt
        else:
            self.R = np.zeros((2, 2))

        yhat = F.dot(self.theta)
        et = y - yhat

        # Q_t is the variance of the prediction of observations and hence
        # \sqrt{Q_t} is the standard deviation of the predictions
        Qt = F.dot(self.R).dot(F.T) + self.p.vt
        sqrt_Qt = np.sqrt(Qt)

        # The posterior value of the states \theta_t is distributed as a
        # multivariate Gaussian with mean m_t and variance-covariance C_t
        At = self.R.dot(F.T) / Qt
        self.theta = self.theta + At.flatten() * et
        self.C = self.R - At * F.dot(self.R)
        theta = self.theta

        # Fill the lines
        self.lines.et[0] = et
        self.lines.sqrt_qt[0] = sqrt_Qt
        self.lines.theta[0] = theta[0]


class KalmanSignals(bt.Indicator):
    _mindatas = 2  # needs at least 2 data feeds

    lines = ('long', 'short', 'theta', )
    plotinfo = dict(subplot=False, plot=False)

    def __init__(self):
        kf = KalmanFilterInd()
        et, sqrt_qt, theta = kf.lines.et, kf.lines.sqrt_qt, kf.lines.theta
        
        self.lines.theta = theta
        self.lines.long = et < -1.0 * sqrt_qt
        # longexit is et > -1.0 * sqrt_qt ... the opposite of long
        self.lines.short = et > sqrt_qt
        # shortexit is et < sqrt_qt ... the opposite of short


class St(bt.Strategy):
    packages = ('math',)
    
    params = dict(
        ksigs=True,  # attempt trading
        period=30,
    )
    
    def __init__(self):
        if self.p.ksigs:
            self.ksig = KalmanSignals()
            KalmanFilterInd()
            self.ksig.plotlines.long._plotskip = True
            self.ksig.plotlines.short._plotskip = True
            self.ksig.plotlines.theta._plotskip = True

        #KalmanMovingAverage(period=self.p.period)
        #bt.ind.SMA(period=self.p.period)
        if True:
            kf = KalmanFilterInd()
            kf.plotlines.sqrt_qt._plotskip = True
            kf.plotlines.et._plotskip = True
            kf.plotlines.theta._plotskip = True
            

    def next(self):
        if not self.p.ksigs:
            return
        
        sizer = self.getsizer()
        stake = sizer.params.stake
        size = self.position.size
        if not size:
            if self.ksig.long:
                hedge = int(math.floor(self.ksig.theta*stake))
                self.buy(data=self.datas[1], size=stake)
                self.sell(data=self.datas[0], size=hedge)
            elif self.ksig.short:
                hedge = int(math.floor(self.ksig.theta*stake))
                self.sell(data=self.datas[1], size=stake)
                self.buy(data=self.datas[0], size=hedge)

        elif size > 0:
            if not self.ksig.long:
                self.close(data=self.datas[1])
                self.close(data=self.datas[0])
        elif size < 0:
            if not self.ksig.short:
        #elif not self.ksig.short:  # implicit size < 0
                self.close(data=self.datas[1])
                self.close(data=self.datas[0])


def run_coint_test(args=None):
    args = parse_args(args)
    
    ticker_list = pipe.get_ticker_list(ticker_name_path='data/etf-list.csv')
    
    master_data_in_sample = pipe.download_data(tickers_list=ticker_list,
                                               start_date=args.cointstartdate,
                                               end_date=args.cointenddate)
    
    minusdvol = int(args.minusdvol)
    
    adq_usd_vol = pipe.usd_vol_threshold(min_dollar_vol=minusdvol, 
                                         master_data=master_data_in_sample)

    pairs = pipe.get_pairs(adequate_dollar_volumes=adq_usd_vol)
    
    confidencelevel = int(args.confidencelevel)

    conf_pairs = pipe.coint_pairs(pairs=pairs, 
                                  closes=master_data_in_sample['Close'],
                                  conf_level=args.confidencelevel,
                                  min_data=400)
    
    # Data feed kwargs
    kwargs = dict()
    
    # Parse from/to-date
    dtfmt = '%Y-%m-%d'
    for a, d in ((getattr(args, x), x) for x in ['cointstartdate', 
                                                 'cointenddate', 
                                                 'backtestenddate']):
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
    
    conf_pbar = tqdm(conf_pairs)
    for i, tickers in enumerate(conf_pbar):
        ticker_1 = tickers['ticker_1']
        ticker_2 = tickers['ticker_2']
        tickers_1.append(ticker_1)                                         
        tickers_2.append(ticker_2)
        
        conf_pbar.set_description('Testing {0}-{1}'.format(ticker_1,
                                                           ticker_2)) 
                                                 
        cerebro = bt.Cerebro()

        ticker_1_df = pipe.extract_ticker(ticker=ticker_1, 
                                          master_data=master_data_in_sample)

        ticker_2_df = pipe.extract_ticker(ticker=ticker_2, 
                                          master_data=master_data_in_sample)

        data1 = bt.feeds.PandasData(dataname=ticker_1_df)
        cerebro.adddata(data1, name='{0}'.format(ticker_1))
        
        data2 = bt.feeds.PandasData(dataname=ticker_2_df)
        data2.plotmaster = data1
        cerebro.adddata(data2, name='{0}'.format(ticker_2))

        # Broker
        cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))
        cerebro.broker.setcommission(commission=0.005)

        # Sizer
        cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))

        # Strategy
        cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))

        #sharpe_ratio
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                            riskfreerate=0.035, timeframe=time_frame)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', 
                            timeframe=time_frame)
        cerebro.addanalyzer(bt.analyzers.DrawDown)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days,
                            _name='timereturns')

        # Execute
        coint_tests = cerebro.run(**eval('dict(' + args.cerebro + ')'))
        coint_test = coint_tests[0]
        
        sharpe_ratio = coint_test.analyzers.sharpe.get_analysis()['sharperatio']
        sharpes.append(sharpe_ratio)
        
        totreturn = coint_test.analyzers.returns.get_analysis()['rtot']
        tot_returns.append(totreturn)
        
        normreturn = coint_test.analyzers.returns.get_analysis()['rnorm100']
        norm_returns.append(normreturn)
        
        drawdown = coint_test.analyzers.drawdown.get_analysis()['max']['drawdown']
        drawdowns.append(drawdown)
        
        tret_analyzer = coint_test.analyzers.getbyname('timereturns')
        tret_analysis = tret_analyzer.get_analysis()
        returns_df = pd.DataFrame(data = {'Returns': tret_analysis.values()}, 
                                  index = tret_analysis.keys())
        returns_df.index.name = 'Date'
        returns_df['Returns_NoMean'] = returns_df['Returns'] - returns_df['Returns'].mean()
        returns_np = returns_df['Returns_NoMean'].to_numpy()
        pVaR = pipe.p_value_at_risk(returns=returns_np, alpha=0.95)
        VaRs.append(pVaR)
        
        pCVaR = pipe.p_c_value_at_risk(returns=returns_np, alpha=0.95)
        CVaRs.append(pCVaR)
        
    coint_dict = {'Ticker_1': tickers_1,
                  'Ticker_2': tickers_2, 
                  'Sharpe_Ratio': sharpes,
                  'Total_Returns': tot_returns,
                  'Norm_Yearly_Returns': norm_returns,
                  'Max_Drawdowns': drawdowns,
                  'VaR (perc.)': VaRs,
                  'CVaR (perc.)': CVaRs}
    
    coint_results = pd.DataFrame(data=coint_dict)
    coint_results = coint_results.sort_values(by=['Norm_Yearly_Returns'], ascending=False)
    top5_pairs_sharpe = coint_results.head()
    print(top5_pairs_sharpe)
    
    if args.savecoint:
        coint_results.to_csv('results/coint_results_{0}-{1}'.format(args.cointstartdate,
                                                            args.cointenddate))
    
    top5_ticker1 = top5_pairs_sharpe['Ticker_1']
    top5_ticker1_list = top5_ticker1.tolist()
    top5_ticker2 = top5_pairs_sharpe['Ticker_2']
    top5_ticker2_list = top5_ticker2.tolist()
    best_pairs = [{'ticker_1': top5_ticker1_list[i], 
                   'ticker_2': top5_ticker2_list[i]} for i in range(len(top5_ticker1_list))]
    
    if args.bestpairs: #print best pairs if requested
        print(best_pairs)
    
    #test the best pairs in the backtest period: cointenddate-backtestenddate
    bt_tickers_1 = []
    bt_tickers_2 = []
    bt_sharpes = []
    bt_tot_returns = []
    bt_norm_returns = []
    bt_drawdowns = []
    bt_VaRs = []
    bt_CVaRs = []
    
    best_ticker_list = list(set().union(*[{elem['ticker_1'], 
                                           elem['ticker_2']} for elem in best_pairs]))
    
    master_data_out_sample = pipe.download_data(tickers_list=best_ticker_list,
                                                start_date=args.cointenddate,
                                                end_date=args.backtestenddate)
    
    pbar = tqdm(best_pairs)
    for i, tickers in enumerate(pbar):
        ticker_1 = tickers['ticker_1']
        ticker_2 = tickers['ticker_2']
        bt_tickers_1.append(ticker_1)                                         
        bt_tickers_2.append(ticker_2)
        
        pbar.set_description('Backtesting {0}-{1}'.format(ticker_1,
                                                          ticker_2)) 
                                                 
        bt_cerebro = bt.Cerebro()

        ticker_1_df = pipe.extract_ticker(ticker=ticker_1, 
                                          master_data=master_data_out_sample)

        ticker_2_df = pipe.extract_ticker(ticker=ticker_2, 
                                          master_data=master_data_out_sample)

        data1 = bt.feeds.PandasData(dataname=ticker_1_df)
        bt_cerebro.adddata(data1, name='{0}'.format(ticker_1))
        
        data2 = bt.feeds.PandasData(dataname=ticker_2_df)
        data2.plotmaster = data1
        bt_cerebro.adddata(data2, name='{0}'.format(ticker_2))

        # Broker
        bt_cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))
        bt_cerebro.broker.setcommission(commission=0.005)

        # Sizer
        bt_cerebro.addsizer(bt.sizers.FixedSize, **eval('dict(' + args.sizer + ')'))

        # Strategy
        bt_cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))

        #sharpe_ratio
        bt_cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='bt_sharpe',
                               riskfreerate=0.035, timeframe=time_frame)
        bt_cerebro.addanalyzer(bt.analyzers.Returns, _name='bt_returns', 
                               timeframe=time_frame)
        bt_cerebro.addanalyzer(bt.analyzers.DrawDown, _name='bt_drawdown')
        bt_cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days,
                               _name='bt_timereturns')

        # Execute
        backtests = bt_cerebro.run(**eval('dict(' + args.cerebro + ')'))
        backtest = backtests[0]
        
        bt_sharpe_ratio = backtest.analyzers.bt_sharpe.get_analysis()['sharperatio']
        bt_sharpes.append(bt_sharpe_ratio)
        
        bt_totreturn = backtest.analyzers.bt_returns.get_analysis()['rtot']
        bt_tot_returns.append(bt_totreturn)
        
        bt_normreturn = backtest.analyzers.bt_returns.get_analysis()['rnorm100']
        bt_norm_returns.append(bt_normreturn)
        
        bt_drawdown = backtest.analyzers.bt_drawdown.get_analysis()['max']['drawdown']
        bt_drawdowns.append(bt_drawdown)
        
        bt_tret_analyzer = backtest.analyzers.getbyname('bt_timereturns')
        bt_tret_analysis = bt_tret_analyzer.get_analysis()
        bt_returns_df = pd.DataFrame(data = {'Returns': bt_tret_analysis.values()}, 
                                     index = bt_tret_analysis.keys())
        bt_returns_df.index.name = 'Date'
        bt_returns_df['Returns_NoMean'] = bt_returns_df['Returns'] - bt_returns_df['Returns'].mean()
        bt_returns_np = bt_returns_df['Returns_NoMean'].to_numpy()
        bt_pVaR = pipe.p_value_at_risk(returns=bt_returns_np, alpha=0.95)
        bt_VaRs.append(bt_pVaR)
        
        bt_pCVaR = pipe.p_c_value_at_risk(returns=bt_returns_np, alpha=0.95)
        bt_CVaRs.append(bt_pCVaR)
        
        if args.plot:  # Plot if requested to
            fig = bt_cerebro.plot(**eval('dict(' + args.plot + ')'))[0][0]
            fig.savefig('results/{0}-{1}.pdf'.format(ticker_1,
                                             ticker_2))
               
    #print(bt_tickers_1, bt_tickers_2, bt_sharpes, bt_tot_returns, 
    #      bt_norm_returns, bt_drawdowns, bt_VaRs, bt_CVaRs)
    
    backtest_results_dict = {'Ticker_1': bt_tickers_1,
                             'Ticker_2': bt_tickers_2, 
                             'Sharpe_Ratio': bt_sharpes,
                             'Total_Returns': bt_tot_returns,
                             'Norm_Yearly_Returns': bt_norm_returns,
                             'Max_Drawdowns': bt_drawdowns,
                             'VaR (perc.)': bt_VaRs,
                             'CVaR (perc.)': bt_CVaRs}
    
    backtest_results = pd.DataFrame(data=backtest_results_dict)
    backtest_results = backtest_results.sort_values(by=['Sharpe_Ratio'], ascending=False)
    
    print(backtest_results)
    if args.savetest:
        backtest_results.to_csv('results/backtest_results_{0}-{1}'.format(args.cointenddate,
                                                                  args.backtestenddate))


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Kalman Pairs Trading Strategy'))
    
    # Defaults for dates
    parser.add_argument('--cointstartdate', required=False, default='2018-01-01',
                        help='Date[time] in YYYY-MM-DD format')

    parser.add_argument('--cointenddate', required=False, default='2021-01-01',
                        help='Date[time] in YYYY-MM-DD format')

    parser.add_argument('--backtestenddate', required=False, default='2022-05-30',
                        help='Date[time] in YYYY-MM-DD format')
                                                 
    parser.add_argument('--minusdvol', required=False, default=100e6,
                        help='Minimum average USD volume, integer')
                                                 
    parser.add_argument('--confidencelevel', required=False, default=90,
                        help='Cointegration test confidence level, 90, 95, or 99')
    
    parser.add_argument('--analysistimeframe', required=False, default='Yearly',
                        help='Analysis timeframe for Sharpe Ratio and Returns calculation')

    parser.add_argument('--cerebro', required=False, default='runonce=False',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--broker', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--sizer', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--strat', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--plot', required=False, default='',
                        nargs='?', const='{}',
                        metavar='kwargs', help='kwargs in key=value format')
    
    parser.add_argument('--bestpairs', required=False, default=True,
                        help='True to return best pairs')
    
    parser.add_argument('--savecoint', required=False, default=True,
                        help='True to save cointegration test results into a .csv')
    
    parser.add_argument('--savetest', required=False, default=True,
                        help='True to save backtest results into a .csv')

    return parser.parse_args(pargs)


if __name__ == '__main__':
    run_coint_test()
