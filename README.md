Pairs trading strategy implementation based on Kalman filtering to create signals and Johansen co-integration test to identify ETF pairs.
Install the requirements using ```python -m pip install -r requirements.txt``` and to generate the results run 

```Python
python kalman_pairs_testing.py --bestpairs 'True' --cointstartdate '2016-01-01' --cointenddate '2019-01-01' --backtestenddate '2022-01-01' --size stake=60 --minusdvol 80000000 --plot
```
