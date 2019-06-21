---
title: "Practical Machine Learning Project"
author: "Jorge Bretones Santamarina"
date: "June 21st 2019"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---





## Introduction

Nowadays, using sport tracking devices it is possible to measure how well a certain exercise is performed, something known as *qualitative activity recognition*. Hence, in this project, data from 6 participants were collected using accelerometers in the belt, forearm, arm and dumbell. They did barbell lifts 5 different ways (1 correct and the rest incorrect). Therefore, the goal of this project is to build a machine learning algorithm to classify exercise performance into 1 of the 5 categories.

## Algorithm selection

In this project we are faced with a classification problem. The reason is the outcome variable "classe" is a factor with 5 levels, each of them corresponding to one way of performing the exercises. Hence, as we want to predict a discrete outcome, we cannot employ linear regression. Inside classification methods, there are multiple possibilities to employ. However, due to its high accuracy, we will use random forest to train our prediction model.

## Cross-validation scheme

To develop a machine learning algorithm, considering only the training error is not sufficient. Doing so will most likely lead to overfitting the training data set, causing the method to perform poorly in new data sets. Thus, we will use cross-validation while training the model in order to account for the generalisation or out-of-sample error. **It is important to point out that the out-of-sample error is given by the OBB output of the finalModel column of a randomforest model. OBB means out-of-bag, an estimate of the generalisation error.**

Inside the cross-validation framework we have several possible choices. For this project, we have ruled out LOOCV (Leave-one-out cross-validation), as it is computationally expensive given we have a large dataset with nearly 20.000 observations. Additionally, we would like to consider the bias-var tradeoff when building our method, so we have opted for 5-fold cross-validation, computationally more feasible and with a nice compromise between bias and variance.

## Cleaning the data:

First of all, we see that the dataset has 19622 observations of 160 predictor variables. The first thing to do is clean the dataset, eliminating those variables with multiple missing variables, which are not explanatory. I have carried 2 procedures to clean the data:

**A. I have eliminated those columns of the dataframe with a high percentage of NA values (greater than 60%).**


```r
indexes <- apply(training,2,function(x){
  my_sum <- sum(is.na(x))
  if (my_sum > length(x)*0.6){
    index <- 1 #Bad column
  }else{
    index <- 0
  }
})
#Now we select those positions which contain 0s (good columns to keep)
good_indexes <- which(indexes == 0)
#We keep the data we want (not NAs)
new_training <- training[,good_indexes]
testing <- testing[,good_indexes]
```

**B. I have elminated factor variables as "kurtosis","skewness","max","min" or "amplitude", which had multiple missing values other than NAs. At the same time, we ommit the X and user_name variables, which are not explanatory.**


```r
bad_colnames <- grep("kurtosis|skewness|amplitude|max|min|X|user_name",colnames(new_training))
new_training <- new_training[,-bad_colnames]
testing <- testing[,-bad_colnames]
```

**At this point, we are left with 58 possible predictors. In my first analysis I ignored the impact of variables as "raw_timestamp_part_1" or "num_window", so I left them in the model. We will review the consequences of this in the next section**

## Results I: full model fitting and variable selection

We have used the caret package train function to train our model for the output "classe" against all 58 predictors left in the model. We have parallelised the code to make it faster. The goal of fitting a model with all predictors is to perform a variable importance analysis afterwards. The output of this test will give us a ranking of the predictors, depending on their relevance to explain the output variable. The full model is the following:




```r
set.seed(45) #It is important to reproduce, as randomForest has a random component
library(parallel)
library(doParallel)
library(randomForest)
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)
#We define the cross-validation scheme, which will be 5-fold crossvalidation here.
#We do not want to use LOOCV, because it is computationally expensive 
#and it might lead to overfitting
fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)
full_model <- train(classe~., data = new_training, method = "rf",trControl = fitControl)
```


```
## Random Forest 
## 
## 19622 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15697, 15698, 15698, 15697 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9942412  0.9927152
##   38    0.9994904  0.9993554
##   75    0.9986750  0.9983240
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 38.
```

The model output shows that the accuracy of the best model is of 0.999 (equivalently, it has a out-of-sample error of 0.01%). However, we do not want to employ a model like this, as it has many predictors and will not probably generalise well. Hence, to reduce the number of variables we run the varImp test to obtain a ranking of predictors according to their relevance to explain the output variable. We show the 10 most important predictors in what follows:




```
##      overall                          names
## 1  3568.6899           raw_timestamp_part_1
## 23 1580.0136                     num_window
## 24 1463.5276                      roll_belt
## 64  902.6137                  pitch_forearm
## 62  618.6035              magnet_dumbbell_z
## 61  521.1328              magnet_dumbbell_y
## 26  476.9921                       yaw_belt
## 63  437.3832                   roll_forearm
## 25  433.2951                     pitch_belt
## 21  361.9149 cvtd_timestamp30/11/2011 17:12
```

**These results are quite surprising, as we see that raw_timestamp_part_1 and num_window are far more important than the rest of the variables, only closely followed by roll_belt. It can be shown that if we fit a model with only both of them, we get an accuracy of more than 99.9%**




```r
two_predmod <- train(classe~raw_timestamp_part_1+num_window,data = new_training,method= "rf",trControl = fitControl)
```


```
## Random Forest 
## 
## 19622 samples
##     2 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15697, 15697, 15696, 15699, 15699 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.9997451  0.9996776
## 
## Tuning parameter 'mtry' was held constant at a value of 2
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.01%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5580    0    0    0    0 0.0000000000
## B    0 3796    0    0    1 0.0002633658
## C    0    0 3422    0    0 0.0000000000
## D    0    0    0 3216    0 0.0000000000
## E    0    0    0    1 3606 0.0002772387
```

**However, let's do a plot of these variables vs classe to illustrate what is going on:**

![](Project_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

In this plot we see that both variables are highly correlated with the outcome "classe", but this is assuredly an artifact. By coloring the plot with respect to the user_name variable, we see that the exercises were performed sequencially by each of the participants. **Hence, these predictors are not reliable and they must be eliminated from the analysis. This exercise highlights the importance of being critical with the varImp results. Moreover, exploratory data analysis is a good way to understanding what is happening inside the data**

## Results II: best model achieved only with accelerometer data predictor variables

If we discard the temporal variables (first columns of the dataset), we are left with the following predictors:




```
## 'data.frame':	19622 obs. of  53 variables:
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

We can then fit a model of classe vs all of them: 


```r
full_model2 <- train(classe~., data = new_training, method = "rf",trControl = fitControl)
```


```
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15697, 15698, 15698, 15697 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9940882  0.9925217
##   27    0.9940882  0.9925216
##   52    0.9882274  0.9851064
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.41%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5578    1    0    0    1 0.0003584229
## B   12 3783    2    0    0 0.0036871214
## C    0   16 3404    2    0 0.0052600818
## D    0    0   37 3176    3 0.0124378109
## E    0    0    0    6 3601 0.0016634322
```

We see that including 52 predictors of physical parameters we get an accuracy of 0.9948 and an out-of-sample error of 0.45%, which corresponds to a probability of getting 20 correct out of 20 in the testing set of 0.8866. The predictions, as we see here, are correct:


```r
predict(full_model2,testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

However, this model has a high number of predictors, so we might want to check if it is possible to reduce that number while keeping a high accuracy level. We do this by running a varImp analysis:


```r
varImpPlot(full_model2$finalModel, sort = TRUE, n.var = 52)
```

![](Project_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

Given this plot, I have fitted a model with the 20, 25, 30 and 40 most important predictors. The results show that with 30 out of 52 predictors we are still able to get an accuracy of 0.994 and OBB of 0.47%, which is quite good. This way we avoid including the last 22 predictors initially employed and we reduce the complexity of the model.




```r
import_var2 <- varImp(full_model2,scale = FALSE)$importance
imp2 <- data.frame (overall = import_var2$Overall,names = rownames(import_var2))
imp2 <- imp2[order(imp2$overall, decreasing = T), ]
imp2$overall <- round(imp2$overall,4)
imp2
#We try the 30 most important predictors:
cols <- paste(imp2$names[1:30], collapse = "+")
formul <- as.formula(paste("classe ~ ",cols))
mod30 <- train(formul,data = new_training, method = "rf",trControl = fitControl)
```


```
## Random Forest 
## 
## 19622 samples
##    30 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15697, 15697, 15698, 15698 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9937315  0.9920708
##   16    0.9937316  0.9920710
##   30    0.9875142  0.9842056
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 16.
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 16
## 
##         OOB estimate of  error rate: 0.47%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5572    4    3    0    1 0.001433692
## B   13 3769   15    0    0 0.007374243
## C    0    9 3403   10    0 0.005552309
## D    0    1   24 3188    3 0.008706468
## E    0    1    4    5 3597 0.002772387
```

The predictions in the test set are correct:


```r
predict(mod30,testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion

In this project we have tried to predict the quality of different exercises according to the data recorded with accelerometers. We have eliminated the missing values in the dataset, leaving 52 predictors. The following conclusions can be extracted from the analysis:

**1. We have seen the importance of evaluating the output of varImp, as variables like raw_timestamp_par_1 and num_window might seem relevant, while they are just artifacts. Using exploratory data analysis has been helpful to address this issue**

**2. Fitting a model with 52 variables gave us an accuracy of 0.994, which is quite high. We have also seen that cutting the number of predictors down to the 30 most important ones is able to reach a very similar accuracy. We might then choose this model over the big one, as it is simpler and probably less prone to overfit the training dataset**
