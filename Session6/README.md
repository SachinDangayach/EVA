# Submission for EVA5 Assignment 6 (Regularization) by Sachin Dangayach

Implemented 5 different regularization on the top of last best model. Inclusion of regularization reduced the training and test accuracy. 
EVA5_S6_NN_Regularization.ipynb contains the code for implementation and call of all different regularization's. Also, uploaded the loss and accuracy curves (Loss_and_Accuracy_Plots.JPEG).
GBN regularization based model is saved and loaded to find 25 incorrect prediction by model. These are plotted in one graph named GBN_RESULTS.JPEG
  
  

## A. Target

Take the last model (5th code), and run bellow versions for 25 epochs and report findings:

  

* with L1 + BN

* with L2 + BN

* with L1 and L2 with BN

* with with GBN

* with L1 and L2 with GBN

  

## B. Results

* L1 + BN

1. Parameters: 9,912

2. Best Training Accuracy in 25 epochs: 99.28%

3. Best Test Accuracy in 25 epochs: 99.37%

* L2 + BN

1. Parameters: 9,912

2. Best Training Accuracy in 25 epochs: 99.27%

3. Best Test Accuracy in 25 epochs: 99.32%

* L1 + L2 + BN

1. Parameters: 9,912

2. Best Training Accuracy in 25 epochs: 99.31%

3. Best Test Accuracy in 25 epochs: 99.35%

* GBN

1. Parameters: 9,912

2. Best Training Accuracy in 25 epochs: 99.16%

3. Best Test Accuracy in 25 epochs: 99.38%

* L1 + L2 + GBN

1. Parameters: 9,912

2. Best Training Accuracy in 25 epochs: 99.25%

3. Best Test Accuracy in 25 epochs: 99.34%

## C. Analysis

### Loss and Accuract Plots

![alt text](https://github.com/SachinDangayach/eva5_session6/blob/master/Loss_and_Accuracy_Plots.JPEG)

### GBN Results

![alt text](https://github.com/SachinDangayach/eva5_session6/blob/master/GBN_RESULTS.JPEG)
  

With the application of regularization, we try to modify the loss function slightly. Regularization helps to deal with the problem of overfitting.

We tried 5 models here and the summary of the analysis is -

* L1+BN: We added the sum of weights as regularizer with lambda as 1e-6. Due to the effect of regularization the test and train accuracy gap got reduced though model coudn't reach to the earlier levels of above 99.40% accurcay

* L2+BN: We used the weight decay parameter in the optimizer function for L2 Norm which is sum of squares of weights. L2 didn't better than L1 as the value of decay parameter is not appropriate and could be explored further

* L1+L2+BN: L1 + L2 performed better than L2. The training accuracies shows L2 is more hard than L1 and L1+L2 combined

* GBN: We used split of 2 and batch size of 4. GBN gave best training accuracy as splitting the batches acts as a regularizer which leads to better training and resulting into better test accuracy
