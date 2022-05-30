# This study focuses on the prediction and
# of GDP growth based on economic indicators
# that are reported/obtainable before quarterly
# GDP growth is released.

# We first used iterative differencing
# techniques on time series data from
# FRED to find Q/Q growth rates for up-trending 
# indicators. 
# We then trained a LASSO model on the
# time series of these growth rates, along
# with 1-period and 2-period lags for each of
# these growth rates, the unemployment rate of
# the quarter, and the average yield curve of
# the quarter. We fitted our LASSO model by
# iterating through OOS data with different
# penalty parameters and minimizing RMSE.

# Then, we used the regression coefficient
# estimates obtained from our LASSO model,
# along with the unscaled covariance of each 
# co-variate, to derive conditional posterior 
# probability distributions for each regression
# coefficient and the marginal posterior
# distribution for coefficient variance

# We then created posterior predictive
# distributions for our OOS Lasso estimates
# and evaluate our model using bayesian
# test statistics. Finally, we found and 
# plotted the predictive distribution of 
# the forecast for the upcoming quarter.

# Will be adding support for live web scrapping
# for FRED data using Quandl in the future
