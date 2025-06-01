# EVA5 Session 4 Assignment by Sachin Dangayach

In this assignment we have to train a neural network on MNIST dataset of handwritten digits with following constrainits-
1. We have to acheive minimum 99.4% validation accuracy
2. We can use less than 20k Parameters only
3. Less than 20 Epochs should be used
4. No fully connected layer

# We have trained the network and acheived 99.41% accuracy in 17 Epoch. 

## Model Details
### 1. Number of parameters:- Total params: 10,066 (Learnt during back propagation)
Trainable params: 10,066 (Learnt during back propagation)
### 2. Batch Size:-64 
Used for deciding number of images feed to the network to calculate the loss, before back propagation happens to adjust the weights and reduce the loss
### 3. Learning Rate:- 0.1
Multiplied with gardient to decide the value to update the weight
### 4. Kernel Size:- 3*3
Kernel help to extract the features. Acts as both filter and gate.
### 5. Optimizer:- SGD 
One of the aglo. used for updating the weights
### 6. Activation Function:- Relu 
Used to act as a gate to let values passed to the next layer or not

### 7. Model Summary
|  Layer (type) |Output Shape   |  Param #  |  Comments |
|---|---|---|---|
| Conv2d-1  |[-1, 8, 26, 26]   | 80  | 3*3 kernel is convolved or 28 * 28 * 1 to extract features and generate 8 Channels  |
| BatchNorm2d-2  |[-1, 8, 26, 26]   |16   |Batch Normalization to sharpen the extracted features   |
| Dropout-3  |[-1, 8, 26, 26]   |0   | Dropout layer added for regularization. To force all neurons to learn  |
|Conv2d-4 |[-1, 16, 24, 24]   | 1168  | 3*3 kernel is convolved or 26 * 26 * 1 to extract features and generate 16 Channels  |
|BatchNorm2d-5  |[-1, 16, 24, 24]   |32   |Batch Normalization to sharpen the extracted features    |
|Dropout-6  |[-1, 16, 24, 24]   |0   |Dropout layer added for regularization. To force all neurons to learn  |
|Conv2d-7 | [-1, 16, 22, 22]  |2320   |3*3 kernel is convolved or 24 * 24 * 1 to extract features and generate new 16 Channels |
|BatchNorm2d-8  |[-1, 16, 22, 22]   |32   | Batch Normalization to sharpen the extracted features   |
|Dropout-9 |[-1, 16, 22, 22]   |0   | Dropout layer added for regularization. To force all neurons to learn   |
|MaxPool2d-10  | [-1, 16, 11, 11]   |0   |Max Pool layer is used to reduce the number of layers in network (22 * 22 * 16     - 11 * 11 * 16)  |
|BatchNorm2d-11  |  [-1, 16, 11, 11]  |32   |Batch Normalization to sharpen the extracted features    |
|Dropout-12  | [-1, 16, 11, 11]   |0   |  Dropout layer added for regularization. To force all neurons to learn  |
|Conv2d-13 |[-1, 8, 11, 11]   |36   |1*1 kernel is used to mix the channels as DJ to reduce the number of channels from 16 to 8. It acts as a filter also to drop unwated features  (convolved or 11 * 11 * 16 to result 11 * 11 * 8    |
|BatchNorm2d-14   |[-1, 8, 11, 11]   |16   | Batch Normalization to sharpen the extracted features   |
|Dropout-15   |[-1, 8, 11, 11]   |0   | Dropout layer added for regularization. To force all neurons to learn   |
|Conv2d-16  |  [-1, 16, 9, 9]  |1168   |3*3 kernel is convolved or 11 * 11 * 8 to extract features to generate new 16 Channels  with output as 9 * 9 * 16 |
|BatchNorm2d-17  |  [-1, 16, 9, 9]  |32   |Batch Normalization to sharpen the extracted features    |
|Dropout-18 | [-1, 16, 9, 9]   |0   |  Dropout layer added for regularization. To force all neurons to learn  |
|Conv2d-19   |[-1, 32, 7, 7]   | 4640  | 3*3 kernel is convolved or 9 * 9 * 16 to extract features to generate new 32 Channels  with output as 7 * 7 * 32 |
|BatchNorm2d-20   | [-1, 32, 7, 7]  |64   |Batch Normalization to sharpen the extracted features    |
|Dropout-21   |[-1, 32, 7, 7]   |0   | Dropout layer added for regularization. To force all neurons to learn   |
|AvgPool2d-22   | [-1, 32, 1, 1]   |0   | Global Average pooling is used to reduce the number of laters from 7 * 7 to 1 * 1   |
|Conv2d-23   | [-1, 32, 1, 1]   |330   | Ten 1 * 1 * 32 kernels used to mix and generate the flatten output to feed Log_softmax   |

# EVA5_99_41.pynp file with code and logs is placed in repo