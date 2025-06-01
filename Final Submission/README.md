# FINAL ASSIGNMENT SUBMISSION FOR EVA5
## The Assignment
Assignment is to create a network that can perform 3 tasks simultaneously:
- Predict the boots, PPE, hardhat, and mask if there is an image
- Predict the depth map of the image
- Predict the Planar Surfaces in the region
>As always you'll find in the industry, people don't really share their datasets. Moreover, datasets can be really HUGE and very difficult to handle. The strategy we will use here is to:
- use pre-trained networks and use their outputs as the ground truth data
- Use the output or MidasNet for the depth map
- Use the output of PlanerRCNN for the plane-surface data
- Use the dataset EVA5 batch collected for boots/PPE/mask/hardhat.

## APPROACH
The Final assignment required the applications of concepts we learnt throughout the course. As we started from understanding the basics of neuarl networks, convolutions, regualarization, how to play with learning rates, data augmentations, advance convolutions and modern nn architecture like Reset. In last three sessions we learn the object detection network - YOLO. We started with data creation to train a network able to classify four different classes (hard hat, vests, masks and boots). We trained the YOLO network with the collective [dataset 1](https://drive.google.com/file/d/1EqtOpF7cS74C56EaVQoKNkQmpT6_HFL2/view?usp=sharing) we prepared.
In the next session we understood the architect of RCNNs, PlaneRCNN and MaskRCNN. We used the same dataset which we used to train the YOLO while we used the pretrained MidasNet to get the depth images [dataset 2](https://drive.google.com/file/d/1ALKKsABUnbI0FodCvdFRRGwcfFUVirvd/view?usp=sharing) and we used the pretrained network and generated the output of plane-surface [dataset 3](https://drive.google.com/file/d/1ycFC7INzTajJFmEV22ns-v813-wfJBta/view?usp=sharing).

**DATASETS FOR TRAINING**-
dataset1 , dataset2 and dataset3 mentioned above are the ground truth for training the final network.

**MODEL ARCHITECTURE**-
As the final expectations is to generate three outputs (depth map, plane-surface and bounding boxes) we need to go for encoder decoder architecture. As Midas model is trained by intel on huge dataset, I decided to follow the below mentioned approach-
1. Keep MidasNet as the central network with pretrained weights. I will freeze this network. As YOLO and PlaneRCNN has got similar encoder decoder architecture, I will connect the decoder of darknet53 while replacing the encoder with encoder of Midas and similarly I will connect the decoder of resnet with PlaneRCNN while keeping the encoder from Midas.
2. Once I am able to connect with networks, I will work on the loss function and train the models ( Decoder part for YOLO and planeRCNN) as described in last session.

**STEPS Followed**-
1. Get the existing architecture for MidasNet, YOLO( darknet53) and PlaneRCNN: Explore the repo code and get the architecture. Files are attached below-
    - [MidasNet](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Model%20Architectures/midas_network)
    - [YOLO-darknet53](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Model%20Architectures/yolo_network)
    - [PlaneRCNN](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Model%20Architectures/planercnn_network)

2. Midasnet with YOLO integration:
    - **Changes** in MidasNet repo code
      - [midas_net.py](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Yolo_Midas_Integration_Code/MiDaS/midas_net.py)

    - Changes in YOLO repo
      - [models.py](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Yolo_Midas_Integration_Code/YoloV3/models.py)
      - [train.py](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Yolo_Midas_Integration_Code/YoloV3/train.py)
      - [yolov3-custom.cfg](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Yolo_Midas_Integration_Code/YoloV3/yolov3-custom.cfg)

    - [common.py](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Yolo_Midas_Integration_Code/common.py)
      I added the run function from Midas with Train function from YOLO in common.py

    - [YOLO_Midas.ipynb](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Yolo_Midas_Integration_Code/Yolo_Midas.ipynb)
      I followed the steps to get the common encoder mode from midas and get it integrate with decoder of YOLO and load after loading the repos with changed codes. I could manage to run the training successfully but for batch size of one. Need to do code changes to let the batch size varied. Current restriction was due to Midas taking 1 image at time.


3. Midasnet with PlaneRCNN integration:
I started with Midasnet and PlaneRCNN integration. I tried to integrate the encoder on Midas with resnet of planercnn as in the linked [file](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/PlanerCNN_Midas_Integration/PlanerCNN_and_Midas_combined.ipynb) but I couldn't get the success as I faced multiple issues due to version miss-match between libraries.

**Conclusion**

I couldn't achieve the desired result of combining all three networks. I could connect YOLO and MIDasnet and get the train loop run in [ipynb notebook](https://github.com/SachinDangayach/EV5_Final_Submission/blob/master/Yolo_Midas_Integration_Code/Yolo_Midas.ipynb) by loading the weights but couldn't finish the integration. I will work on this assignment and complete it by working on gaps in my understand and skills.
