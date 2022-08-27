Pairs trading strategy implementation based on Kalman filtering to create signals.

Pairs are selected by clustering and checking for co-integration within clusters. Pairs are then further tested for 
their spreads mean reversion behavior by calculating the Hurst exponent and the halflife of mean reversion of the spread. 

Install the requirements using `python -m pip install -r requirements.txt`. The requirements are generated via pipreqs.

To generate the results run: 

```SHELL
python kalman_pairs_testing.py --plot iplot=False --minusdvol 50000000 \ 
--broker 'cash=100000' --sizer percents=40 \ 
--clusterparams "min_cluster_size=2, min_samples=5, xi=0.05, metric='cosine'" \ 
--PCAparams "svd_solver='full'" --pairselectionparams "n_pca_components=0.8"
```

To see available CLI arguements, use `python kalman_pairs_testing.py --help`.
