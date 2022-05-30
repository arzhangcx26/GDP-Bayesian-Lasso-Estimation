##------------------------------------------
## Study Description
##------------------------------------------
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

##------------------------------------------
## Model Assumptions
##------------------------------------------
## Regression Coefficient Normality: ##
# We are assuming a normal distribution for
# all regression coefficients with a mean equal
# to the estimate derived from a LASSO
# regression

## Regression Coefficient Stationarity: ##
# We are assuming that the regression coefficients
# are stationary. In other words, over time we
# assume the underlying probability of each
# regression coefficient does not change

## Penalized Coefficient Variability: ##
# We are assuming that coefficients for variables
# penalized to 0 by LASSO have a probability
# of being relevant in forecasting and thus
# include posterior probability distributions
# for penalized covariates with a mean of 0
# and variable standard deviation

## Non-Informative Prior Distributions: ##
# We are using non-informative prior distributions
# for all regression coefficients, reflecting
# weak/non-existent prior beliefs

##------------------------------------------
## Loading R Libraries
##------------------------------------------
library(plyr) #for lasso regression
library(readr) #for lasso regression
library(dplyr) #for lasso regression
library(caret) #for lasso regression
library(ggplot2) #for lasso regression
library(repr) #for lasso regression
library(glmnet) #for lasso regression
library(xts) #for time series data
library(zoo) #for time series data
library(MASS) #for bayesian regression

##------------------------------------------
## Growth Rate Calculations
##------------------------------------------

data <- read.csv("~/Shit You Care About/Data Science Projects/GDP Bayesian Lasso/DATA.csv")
dat <- data[-1,]
n <- nrow(dat)

# Make Quarterly Dummy Variables
dat$Q1 <- rep(0, n)
dat$Q2 <- rep(0, n)
dat$Q3 <- rep(0, n)
dat$Q4 <- rep(0, n)

for(k in 1:n) {
  if(dat$QUARTER[k]==1){
    dat$Q1[k] <- 1
  }
  if(dat$QUARTER[k]==2){
    dat$Q2[k] <- 1
  }
  if(dat$QUARTER[k]==3){
    dat$Q3[k] <- 1
  }
  if(dat$QUARTER[k]==4){
    dat$Q4[k] <- 1
  }
}

# Creating Growth Rate Variables
dat$gdpG <- rep(NA, n)
dat$pcepilfeG <- rep(NA, n)
dat$pcepdgG <- rep(NA, n)
dat$indproG <- rep(NA, n)
dat$cpiG <- rep(NA, n)
dat$m2G <- rep(NA, n)
dat$wtiG <- rep(NA, n)
dat$ppiG <- rep(NA, n)
dat$dspiG <- rep(NA, n)
dat$payemsG <- rep(NA, n)
dat$houseG <- rep(NA, n)

for(i in 1:n) {
  if(!is.na(data$GDPC1[i+1])) {
    dat$gdpG[i] <- data$GDPC1[i+1]/data$GDPC1[i]-1
  }
  dat$pcepilfeG[i] <- data$PCEPILFE[i+1]/data$PCEPILFE[i]-1
  dat$pcepdgG[i] <- data$PCEPDG[i+1]/data$PCEPDG[i]-1
  dat$indproG[i] <- data$INDPRO[i+1]/data$INDPRO[i]-1
  dat$cpiG[i] <- data$CoreCPI[i+1]/data$CoreCPI[i]-1
  dat$m2G[i] <- data$M2Real[i+1]/data$M2Real[i]-1
  dat$wtiG[i] <- data$WTIOIL[i+1]/data$WTIOIL[i]-1
  dat$ppiG[i] <- data$PPIACO[i+1]/data$PPIACO[i]-1
  dat$dspiG[i] <- data$DSPI[i+1]/data$DSPI[i]-1
  dat$payemsG[i] <- data$PAYEMS[i+1]/data$PAYEMS[i]-1
  dat$houseG[i] <- data$HouseSales[i+1]/data$HouseSales[i]-1
}

# Creating Lag Variables
dat$gdpG_lag1 <- lag(dat$gdpG, 1)
dat$gdpG_lag2 <- lag(dat$gdpG, 2)
dat$pcepilfeG_lag1 <- lag(dat$pcepilfeG, 1)
dat$pcepilfeG_lag2 <- lag(dat$pcepilfeG, 2)
dat$pcepdgG_lag1 <- lag(dat$pcepdgG, 1)
dat$pcepdgG_lag2 <- lag(dat$pcepdgG, 2)
dat$indproG_lag1 <- lag(dat$indproG, 1)
dat$indproG_lag2 <- lag(dat$indproG, 2)
dat$cpiG_lag1 <- lag(dat$cpiG, 1)
dat$cpiG_lag2 <- lag(dat$cpiG, 2)
dat$m2G_lag1 <- lag(dat$m2G, 1)
dat$m2G_lag2 <- lag(dat$m2G, 2)
dat$wtiG_lag1 <- lag(dat$wtiG, 1)
dat$wtiG_lag2 <- lag(dat$wtiG, 2)
dat$ppiG_lag1 <- lag(dat$indproG, 1)
dat$ppiG_lag2 <- lag(dat$indproG, 2)

dat$dspiG_lag1 <- lag(dat$dspiG, 1)
dat$dspiG_lag2 <- lag(dat$dspiG, 2)
dat$payemsG_lag1 <- lag(dat$payemsG, 1)
dat$payemsG_lag2 <- lag(dat$payemsG, 2)
dat$houseG_lag1 <- lag(dat$houseG, 1)
dat$houseG_lag2 <- lag(dat$houseG, 2)
dat$TB3M_lag1 <- lag(dat$TB3M, 1)
dat$TB3M_lag2 <- lag(dat$TB3M, 2)
dat$TB1Y_lag1 <- lag(dat$TB1Y, 1)
dat$TB1Y_lag2 <- lag(dat$TB1Y, 2)
dat$TB2Y_lag1 <- lag(dat$TB2Y, 1)
dat$TB2Y_lag2 <- lag(dat$TB2Y, 2)
dat$TB3Y_lag1 <- lag(dat$TB3Y, 1)
dat$TB3Y_lag2 <- lag(dat$TB3Y, 2)
dat$TB5Y_lag1 <- lag(dat$TB5Y, 1)
dat$TB5Y_lag2 <- lag(dat$TB5Y, 2)
dat$TB10Y_lag1 <- lag(dat$TB10Y, 1)
dat$TB10Y_lag2 <- lag(dat$TB10Y, 2)
dat$TB30Y_lag1 <- lag(dat$TB30Y, 1)
dat$TB30Y_lag2 <- lag(dat$TB30Y, 2)
dat$UNRATE_lag1 <- lag(dat$UNRATE, 1)
dat$UNRATE_lag2 <- lag(dat$UNRATE, 2)
dat <- dat[-(1:2),]
n <- nrow(dat)

date <- as.Date(dat$DATE,tryFormats = c("%m/%d/%Y"))
dat <- xts(dat[,colnames(dat)!="DATE"], order.by=as.Date(dat$DATE,"%m/%d/%Y"))

##------------------------------------------
## Data Analysis and Summary
##------------------------------------------
plot(dat$GDPC1)
plot(dat$gdpG)

hist(dat$gdpG,breaks=50)
abline(v=mean(dat$gdpG),col="red")
mean(dat$gdpG)

##------------------------------------------
## Data Partitioning
##------------------------------------------
set.seed(100)

index <- 85

train <- dat[1:index] # Create the training data
test <- dat[index:n] # Create the test data
forecast <- dat[n] # Create the forecast data
if(is.na(dat$GDPC1[n])) {
  test <- dat[index:(n-1)]
}

# Covariates
x <- data.frame(1,train$gdpG_lag1,train$gdpG_lag2,
                train$pcepilfeG,train$pcepilfeG_lag1,
                train$pcepilfeG_lag2,train$pcepdgG,
                train$pcepdgG_lag1,train$pcepdgG_lag2,
                train$indproG,train$indproG_lag1,
                train$indproG_lag2,train$cpiG,
                train$cpiG_lag1,train$cpiG_lag2,
                train$m2G,train$m2G_lag1,
                train$m2G_lag2,train$wtiG,
                train$wtiG_lag1,train$wtiG_lag2,
                train$dspiG,
                train$dspiG_lag1,train$dspiG_lag2,
                train$payemsG,train$payemsG_lag1,
                train$payemsG_lag2,train$houseG,
                train$houseG_lag1,train$houseG_lag2,
                train$TB3M,train$TB3M_lag1,
                train$TB3M_lag2,train$TB1Y,
                train$TB1Y_lag1,train$TB1Y_lag2,
                train$TB2Y,train$TB2Y_lag1,
                train$TB2Y_lag2,train$TB3Y,
                train$TB3Y_lag1,train$TB3Y_lag2,
                train$TB5Y,train$TB5Y_lag1,
                train$TB5Y_lag2,train$TB10Y,
                train$TB10Y_lag1,train$TB10Y_lag2,
                train$TB30Y,train$TB30Y_lag1,
                train$TB30Y_lag2,train$UNRATE,
                train$UNRATE_lag1,train$UNRATE_lag2,
                train$Q1,train$Q2,train$Q3)
x <- as.matrix(x)

x_test <- data.frame(1,test$gdpG_lag1,test$gdpG_lag2,
                test$pcepilfeG,test$pcepilfeG_lag1,
                test$pcepilfeG_lag2,test$pcepdgG,
                test$pcepdgG_lag1,test$pcepdgG_lag2,
                test$indproG,test$indproG_lag1,
                test$indproG_lag2,test$cpiG,
                test$cpiG_lag1,test$cpiG_lag2,
                test$m2G,test$m2G_lag1,
                test$m2G_lag2,test$wtiG,
                test$wtiG_lag1,test$wtiG_lag2,
                test$dspiG,
                test$dspiG_lag1,test$dspiG_lag2,
                test$payemsG,test$payemsG_lag1,
                test$payemsG_lag2,test$houseG,
                test$houseG_lag1,test$houseG_lag2,
                test$TB3M,test$TB3M_lag1,
                test$TB3M_lag2,test$TB1Y,
                test$TB1Y_lag1,test$TB1Y_lag2,
                test$TB2Y,test$TB2Y_lag1,
                test$TB2Y_lag2,test$TB3Y,
                test$TB3Y_lag1,test$TB3Y_lag2,
                test$TB5Y,test$TB5Y_lag1,
                test$TB5Y_lag2,test$TB10Y,
                test$TB10Y_lag1,test$TB10Y_lag2,
                test$TB30Y,test$TB30Y_lag1,
                test$TB30Y_lag2,test$UNRATE,
                test$UNRATE_lag1,test$UNRATE_lag2,
                test$Q1,test$Q2,test$Q3)
x_test <- as.matrix(x_test)

x_for <- data.frame(1,forecast$gdpG_lag1,forecast$gdpG_lag2,
                     forecast$pcepilfeG,forecast$pcepilfeG_lag1,
                     forecast$pcepilfeG_lag2,forecast$pcepdgG,
                     forecast$pcepdgG_lag1,forecast$pcepdgG_lag2,
                     forecast$indproG,forecast$indproG_lag1,
                     forecast$indproG_lag2,forecast$cpiG,
                     forecast$cpiG_lag1,forecast$cpiG_lag2,
                     forecast$m2G,forecast$m2G_lag1,
                     forecast$m2G_lag2,forecast$wtiG,
                     forecast$wtiG_lag1,forecast$wtiG_lag2,
                     forecast$dspiG,
                     forecast$dspiG_lag1,forecast$dspiG_lag2,
                     forecast$payemsG,forecast$payemsG_lag1,
                     forecast$payemsG_lag2,forecast$houseG,
                     forecast$houseG_lag1,forecast$houseG_lag2,
                     forecast$TB3M,forecast$TB3M_lag1,
                     forecast$TB3M_lag2,forecast$TB1Y,
                     forecast$TB1Y_lag1,forecast$TB1Y_lag2,
                     forecast$TB2Y,forecast$TB2Y_lag1,
                     forecast$TB2Y_lag2,forecast$TB3Y,
                     forecast$TB3Y_lag1,forecast$TB3Y_lag2,
                     forecast$TB5Y,forecast$TB5Y_lag1,
                     forecast$TB5Y_lag2,forecast$TB10Y,
                     forecast$TB10Y_lag1,forecast$TB10Y_lag2,
                     forecast$TB30Y,forecast$TB30Y_lag1,
                     forecast$TB30Y_lag2,forecast$UNRATE,
                     forecast$UNRATE_lag1,forecast$UNRATE_lag2,
                     forecast$Q1,forecast$Q2,forecast$Q3)
x_for <- as.matrix(x_for)

##------------------------------------------
## Lasso Model Training/Testing
##------------------------------------------
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

# Lambdas
lambdas <- seq(0, 0.0005, by = .000001)
y_train <- train$gdpG
y_test <- test$gdpG
nLambdas <- length(lambdas)

# Minimize OOS RMSE
error <- rep(NA, nLambdas)
for(j in 1:nLambdas) {
  lambda_test <- lambdas[j]
  lasso_test <- glmnet(x, y_train, alpha = 1, lambda = lambda_test, standardize = TRUE)
  
  predictions_test <- predict(lasso_test, s = lambda_test, newx = x_test)
  error[j] <- as.numeric(eval_results(y_test, predictions_test, test)[1])
}

count <- 0
for(j in 1:nLambdas) {
  count <- count + 1
  if(error[j]==min(error)) {
    min <- count
  }
}
lambda_best <- lambdas[min]

#Plot Model RMSE vs. Lambda
par(mfrow=c(1,1))
plot(lambdas,error,type='l',lwd=2,pch=15,xlab="Lambda",ylab='RMSE')
abline(v=lambda_best, col="red")
title("OOS RMSE")

# Fitting Optimal Model
lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
eval_results(y_train, predictions_train, train)

predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
eval_results(y_test, predictions_test, test)

# Plot Predicted vs. Historical GDP Growth
compare <- data.frame(date[index:n],y_test,predictions_test)
plot(compare[,1],compare[,2],col="black",type="l",lwd=2,xlab="Date",ylab="GDP Growth")
lines(compare[,1],compare[,3],col="red",type="l",lwd=3)
lines(compare[,1],compare[,2],col="black",type="b",lwd=3,pch=19)
legend(0,-0.035,c("actual","projected"), lwd=c(1,1), col=c("black","red"), pch=c(19,19), y.intersp=0.65)
title("OOS GDP Growth vs. Model Estimates")

#Plot Model Residual
diff <- compare[,2]-compare[,3]
hist(diff,xlab="Residual",main="Distribution of Model Error",breaks=25)
abline(v=mean(diff),col="red")
mean(diff)

##------------------------------------------
## Beta and Sigma2 Posterior Distributions
##------------------------------------------

# Sampling from Posterior Distribution
beta.hat <- as.double(lasso_model$beta)
p <- length(beta.hat)
s2 <- (nrow(x)-p)*(sqrt((sum(diff^2)))/(nrow(x)-p))^2
X <- matrix(c(c(x[,1]),c(x[,2]),c(x[,3]),c(x[,4]),c(x[,5]),
              c(x[,6]),c(x[,7]),c(x[,8]),c(x[,9]),c(x[,10]),
              c(x[,11]),c(x[,12]),c(x[,13]),c(x[,14]),c(x[,15]),
              c(x[,16]),c(x[,17]),c(x[,18]),c(x[,19]),c(x[,20]),
              c(x[,21]),c(x[,22]),c(x[,23]),c(x[,24]),c(x[,25]),
              c(x[,26]),c(x[,27]),c(x[,28]),c(x[,29]),c(x[,30]),
              c(x[,31]),c(x[,32]),c(x[,33]),c(x[,34]),c(x[,35]),
              c(x[,36]),c(x[,37]),c(x[,38]),c(x[,39]),c(x[,40]),
              c(x[,41]),c(x[,42]),c(x[,43]),c(x[,44]),c(x[,45]),
              c(x[,46]),c(x[,47]),c(x[,48]),c(x[,49]),c(x[,50]),
              c(x[,51]),c(x[,52]),c(x[,53]),c(x[,54]),c(x[,55]),
              c(x[,56]),c(x[,57])), ncol = ncol(x))
V.beta <- solve(t(X) %*% X)

numsamp <- 1000
beta.samp <- matrix(NA,nrow=numsamp,ncol=p)
sigsq.samp <- rep(NA,numsamp)
for (i in 1:numsamp){
  temp <- rgamma(1,shape=(nrow(x)-p)/2,rate=s2/2)
  cursigsq <- 1/temp
  curvarbeta <- cursigsq*V.beta
  curbeta <- mvrnorm(1,mu=beta.hat,Sigma=curvarbeta)
  sigsq.samp[i] <- cursigsq
  beta.samp[i,] <- curbeta
}

# posterior means
postmean.beta <- apply(beta.samp,2,mean)
postmean.sigsq <- mean(sigsq.samp)
postmean.beta
postmean.sigsq


# 95% posterior intervals
allsamples <- cbind(beta.samp,sigsq.samp)
allsamples.sort <- apply(allsamples,2,sort)
allsamples.sort[25,]
allsamples.sort[975,]

##------------------------------------------
## OOS Posterior Predictive Distributions
##------------------------------------------
# Creating matrix of OOS Observations
Xstar <- matrix(c(c(x_test[,1]),c(x_test[,2]),c(x_test[,3]),c(x_test[,4]),c(x_test[,5]),
              c(x_test[,6]),c(x_test[,7]),c(x_test[,8]),c(x_test[,9]),c(x_test[,10]),
                c(x_test[,11]),c(x_test[,12]),c(x_test[,13]),c(x_test[,14]),c(x_test[,15]),
                c(x_test[,16]),c(x_test[,17]),c(x_test[,18]),c(x_test[,19]),c(x_test[,20]),
                c(x_test[,21]),c(x_test[,22]),c(x_test[,23]),c(x_test[,24]),c(x_test[,25]),
                c(x_test[,26]),c(x_test[,27]),c(x_test[,28]),c(x_test[,29]),c(x_test[,30]),
                c(x_test[,31]),c(x_test[,32]),c(x_test[,33]),c(x_test[,34]),c(x_test[,35]),
                c(x_test[,36]),c(x_test[,37]),c(x_test[,38]),c(x_test[,39]),c(x_test[,40]),
                c(x_test[,41]),c(x_test[,42]),c(x_test[,43]),c(x_test[,44]),c(x_test[,45]),
                c(x_test[,46]),c(x_test[,47]),c(x_test[,48]),c(x_test[,49]),c(x_test[,50]),
                c(x_test[,51]),c(x_test[,52]),c(x_test[,53]),c(x_test[,54]),c(x_test[,55]),
                c(x_test[,56]),c(x_test[,57])), ncol = ncol(x_test))  
Xstar <- t(Xstar)  # making it a row vector
nCol <- ncol(Xstar)

# use posterior samples and OOS data to get posterior predictive distribution of Y
numsamp <- 1000
ystar.samp <- rep(NA,numsamp)
ystar.ts <- matrix(NA,nrow=numsamp,ncol=nCol)
ystar.postmean <- rep(NA, nCol)
ystar.005 <-rep(NA, nCol)
ystar.025 <-rep(NA, nCol)
ystar.160 <-rep(NA, nCol)
ystar.250 <-rep(NA, nCol)
ystar.750 <-rep(NA, nCol)
ystar.840 <-rep(NA, nCol)
ystar.975 <- rep(NA, nCol)
ystar.995 <- rep(NA, nCol)
for (j in 1:nCol) {
  for (i in 1:numsamp){
    xstarbeta <- Xstar[,j]%*%t(t(beta.samp[i,]))
    ystar.samp[i] <- rnorm(1,mean=xstarbeta,sd=sqrt(sigsq.samp[i]))
  } 
  ystar.ts[,j] <- ystar.samp
  ystar.postmean[j] <- mean(ystar.ts[,j])
  ystar.005[j] <- sort(ystar.samp)[5]
  ystar.025[j] <- sort(ystar.samp)[25]
  ystar.160[j] <- sort(ystar.samp)[160]
  ystar.250[j] <- sort(ystar.samp)[250]
  ystar.750[j] <- sort(ystar.samp)[750]
  ystar.840[j] <- sort(ystar.samp)[840]
  ystar.975[j] <- sort(ystar.samp)[975]
  ystar.995[j] <- sort(ystar.samp)[995]
}
ystar.ts <- t(ystar.ts)

ystar.tsmean <- mean(ystar.ts)
ystar.tsmean

# Plot Predicted vs. Historical GDP Growth w/ Posterior Intervals
par(mfrow=c(1,1))
compare <- data.frame(date[index:n],y_test,ystar.postmean,ystar.025,
                      ystar.160,ystar.840,ystar.975,ystar.005,ystar.995,
                      ystar.250,ystar.750)
plot(compare[,1],compare[,3],col="red",type="l",lwd=2,
     ylim=c(-0.1,0.1),xlab="Date,",ylab="GDP Growth")
lines(compare[,1],compare[,4],col="grey",type="l",lwd=1)
lines(compare[,1],compare[,7],col="grey",type="l",lwd=1)
lines(compare[,1],compare[,8],col="blue",type="l",lwd=1)
lines(compare[,1],compare[,9],col="blue",type="l",lwd=1)
lines(compare[,1],compare[,5],col="grey",type="l",lwd=1)
lines(compare[,1],compare[,6],col="grey",type="l",lwd=1)
lines(compare[,1],compare[,10],col="grey",type="l",lwd=1)
lines(compare[,1],compare[,11],col="grey",type="l",lwd=1)
lines(compare[,1],compare[,2],col="black",type="l",lwd=2)
lines(compare[,1],compare[,2],col="black",type="b",lwd=1,pch=19)
legend(0,-0.035,c("actual","projected"), lwd=c(1,1),
       col=c("black","red"), pch=c(19,19), y.intersp=0.625)
title("OOS GDP Growth vs. Model Estimates")

##------------------------------------------
## Bayesian Model Evaluation
##------------------------------------------
# OOS Data Histogram
hist(y_test,breaks=25)

# 50% Interval Accuracy Check
outOfBounds <- rep(1, nCol)
for(k in 1:nCol) {
  if(y_test[k] > ystar.750[k] || y_test[k] < ystar.250[k]) {
    outOfBounds[k] <- 0
  }
}
accuracyRate0 <- sum(outOfBounds)/nCol
accuracyRate0

# 68% Interval Accuracy Check
outOfBounds <- rep(1, nCol)
for(k in 1:nCol) {
  if(y_test[k] > ystar.840[k] || y_test[k] < ystar.160[k]) {
    outOfBounds[k] <- 0
  }
}
accuracyRate1 <- sum(outOfBounds)/nCol
accuracyRate1

# 95% Interval Accuracy Check
outOfBounds <- rep(1, nCol)
for(k in 1:nCol) {
  if(y_test[k] > ystar.975[k] || y_test[k] < ystar.025[k]) {
    outOfBounds[k] <- 0
  }
}
accuracyRate2 <- sum(outOfBounds)/nCol
accuracyRate2

# 99% Interval Accuracy Check
outOfBounds <- rep(1, nCol)
for(k in 1:nCol) {
  if(y_test[k] > ystar.995[k] || y_test[k] < ystar.005[k]) {
    outOfBounds[k] <- 0
  }
}
accuracyRate3 <- sum(outOfBounds)/nCol
accuracyRate3

# Plot Model Interval Error
intError <- rep(0, nCol)
for(k in 1:nCol) {
  if(y_test[k] > ystar.840[k]) {
    intError[k] <- 1
  }
  if(y_test[k] > ystar.975[k]) {
    intError[k] <- 2
  }
  if(y_test[k] > ystar.995[k]) {
    intError[k] <- 3
  }
  if(y_test[k] < ystar.160[k]) {
    intError[k] <- -1
  }
  if(y_test[k] < ystar.025[k]) {
    intError[k] <- -2
  }
  if(y_test[k] < ystar.005[k]) {
    intError[k] <- -3
  }
}

par(mfrow=c(1,1))
plot(intError,main="Model Interval Error",xlab="OOS Observations",
     ylab="Standard Deviations",ylim=c(-3.5,3.5))
abline(h=mean(intError),col="red")

# Plot Predictive Distribution Error
ystar.diff <- matrix(NA,nrow=nCol,ncol=numsamp)
for(h in 1:nCol){
  d <- rep(NA, numsamp)
  for(p in 1:numsamp) {
    d[p] <- ystar.ts[h,p] - y_test[h]
  }
  ystar.diff[h,] <- d
}

hist(ystar.diff,breaks=250)
abline(v=mean(ystar.diff),col="red")
abline(v=sort(ystar.diff)[length(ystar.diff)*0.025],col="blue")
abline(v=sort(ystar.diff)[length(ystar.diff)*0.975],col="blue")
diffAbs <- mean(abs(ystar.diff))
abline(v=diffAbs,col="green") #average abs difference between forecast and data
abline(v=-diffAbs,col="green") #average abs difference between forecast and data

# Plot Average Difference Interval vs. OOS Data
par(mfrow=c(1,1))
compare <- data.frame(date[index:n],y_test,ystar.postmean,diffAbs)
plot(compare[,1],compare[,3],col="red",type="l",lwd=2,ylim=c(-0.1,0.1),xlab="Time",ylab="GDP Growth")
lines(compare[,1],compare[,2]+compare[,4],col="grey",type="l",lwd=1)
lines(compare[,1],compare[,2]-compare[,4],col="grey",type="l",lwd=1)
lines(compare[,1],compare[,2],col="black",type="l",lwd=2)
lines(compare[,1],compare[,2],col="black",type="b",lwd=1,pch=19)
legend(0,-0.035,c("actual","projected"), lwd=c(1,1), col=c("black","red"), pch=c(19,19), y.intersp=0.625)
title("OOS GDP Growth vs. Model Estimates")

# Test Statistic: Model SD
test1.actual <- sd(y_test)
test1.reps <- rep(NA,1000)
for (i in 1:1000){
  test1.reps[i] <- sd(ystar.ts[,i])
}
test1.p <- sum(test1.reps > test1.actual)/1000

# Test Statistic: Model Max
test2.actual <- max(y_test)
test2.reps <- rep(NA,1000)
for (i in 1:1000){
  test2.reps[i] <- max(ystar.ts[,i])
}
test2.p <- sum(test2.reps > test2.actual)/1000

# Test Statistic: Model Min
test3.actual <- min(y_test)
test3.reps <- rep(NA,1000)
for (i in 1:1000){
  test3.reps[i] <- min(ystar.ts[,i])
}
test3.p <- sum(test3.reps > test3.actual)/1000

test1.p
test2.p
test3.p

# Compare Model and Data SD, Max, and Min
par(mfrow=c(1,3))
xmin <- min(test1.actual, test1.reps)
xmax <- max(test1.actual, test1.reps)
hist(test1.reps,xlim=c(xmin,xmax),breaks=15,
     main="SD of GDP Growth")
abline(v=test1.actual,lwd=2,col=2)

xmin <- min(test2.actual, test2.reps)
xmax <- max(test2.actual, test2.reps)
hist(test2.reps,xlim=c(xmin,xmax),breaks=15,
     main="Max of GDP Growth")
abline(v=test2.actual,lwd=2,col=2)

xmin <- min(test3.actual, test3.reps)
xmax <- max(test3.actual, test3.reps)
hist(test3.reps,xlim=c(xmin,xmax),breaks=15,
     main="Min of GDP Growth")
abline(v=test3.actual,lwd=2,col=2)

##------------------------------------------
## Next Quarter Forecast Distribution
##------------------------------------------
# Creating matrix of forecast observations
XstarF <- matrix(c(x_for[1],x_for[2],x_for[3],x_for[4],x_for[5],
                  x_for[6],x_for[7],x_for[8],x_for[9],x_for[10],
                  x_for[11],x_for[12],x_for[13],x_for[14],x_for[15],
                  x_for[16],x_for[17],x_for[18],x_for[19],x_for[20],
                  x_for[21],x_for[22],x_for[23],x_for[24],x_for[25],
                  x_for[26],x_for[27],x_for[28],x_for[29],x_for[30],
                  x_for[31],x_for[32],x_for[33],x_for[34],x_for[35],
                  x_for[36],x_for[37],x_for[38],x_for[39],x_for[40],
                  x_for[41],x_for[42],x_for[43],x_for[44],x_for[45],
                  x_for[46],x_for[47],x_for[48],x_for[49],x_for[50],
                  x_for[51],x_for[52],x_for[53],x_for[54],x_for[55],
                  x_for[56],x_for[57]),ncol=length(x_for))
XstarF <- t(XstarF)  # making it a row vector

# use posterior samples and forecast data to get posterior predictive distribution of Y
numsamp <- 1000
ystarF.samp <- rep(NA,numsamp)
for (i in 1:numsamp){
  xstarbeta <- XstarF[,1]%*%t(t(beta.samp[i,]))
  ystarF.samp[i] <- rnorm(1,mean=xstarbeta,sd=sqrt(sigsq.samp[i]))
} 
ystarF.samp <- t(ystarF.samp)

ystarF.mean <- mean(ystarF.samp)
ystarF.mean

par(mfrow=c(1,1))
hist(ystarF.samp, breaks=25,xlab="GDP Growth",
     main="Posterior Forecast Distribution")
abline(v=ystarF.mean,col="red")
abline(v=sort(ystarF.samp)[25],col="blue") #95% Interval
abline(v=sort(ystarF.samp)[975],col="blue") #95% Interval
if(!is.na(dat$gdpG[nrow(dat)])) {
  abline(v=dat$gdpG[nrow(dat)],col="green") #actual GDP growth (if applicable)
}
