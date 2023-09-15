Programming Assignment 1

*Dependable Artificial Intelligence*

[Rachit (B20AI032)](mailto:rachit.2@iitj.ac.in)

INTRODUCTION

Exploring various types of attacks and finally a detection model to detect these attacks. Attacks such as FGSM, PGD, masking are used to attack on different deep learning models. Implementing FGSM & PGD attack is done by adding noise to the original image while masking based attack is done by masking a patch of image such that the model misclassifies the image. Detection is done by creating a CNN model and treating it as a classification model to detect whether the image is an attack based image or an untouched (unattacked) image.

Using deepfake/faceswap images and using pre trained detection models and fine tuning on original, altered images combined dataset. Using my own image and using a few celebrity images to faceswap my face and thiers.

METHODOLOGY

**Question 1**

**Part 1**

CNN model was created using three convolution layers and a single pool layer finally connected to two fully connected layers. The model architecture is as follows:

![](media/4b54c85690b0ff9e458c23183162ac46.png)

Then we use the ‘checker’ function for training. Loss criteria was cross entropy loss as it is a classification model while the optimizer used was Adam. Learning rate was 0.1 while the batch size was 500.

**FGSM Attack - Fast Gradient Sign Method**

The FGSM attacks is a single step optimization-based attack that aims to find an adversarial example by perturbing the input data in the direction of the gradient of the loss function with respect to the input data.To attack we can simply understand FGSM adds noise to the original image as follows:

Perturbed image = original image + epsilon (sign of gradient),

where we can choose the value of epsilon which is smaller than 1 and closer to 0. To get gradients we train our model and compute loss to get gradients and then use it to generate perturbed images. These perturbed images look very similar to original images to the naked eye but the model looks at them quite differently as it considers value at each pixel and misclassifies it.

**Part 2**

Using pretrained ResNet and ShufffleNetV2 models and fine tuning them on the SVHN dataset we attack using the following 2 models.

**PGD Attack - Projected Gradient Descent**

Similar to FGSM, PGD attack aims to find an adversarial example by perturbing the input data in a way that causes the model to misclassify it. This attack aims to find an adversarial example by modifying the data in a way that causes the model to misclassify it.

Unlike FGSM which uses a single step optimization, PGD uses an iterative based approach for optimization. The idea remains same as FGSM except we add another hyperparameter alpha, we can use another hyperparameter epsilon in clamping but to keep similarity between FGSM and PGD I have avoided using epsilon. It can be shown as follows:

Perturbed image = original image + alpha(sign of gradient).

**Masked Based Attack**

Masking is done by simply adding a mask on the original image.

**Part 3**

**Detection model**

Using a CNN model of the following architecture we make a detection model.

Using FGSM attack and giving normal images as label 1 and perturbed images label as 0. We train our model to classify attacks on not. Hence converting detection into a simpler classification problem.

**Question 2**

Using a github repository[[3]](https://github.com/cedro3/sber-swap/blob/main/SberSwapInference.ipynb) for generating faceswap images and storing them in folders Celebs (orignal), Targets(faceswaps). Further using 50 from each folder for train and another 50 for test dataset. Train and test folders both were further divided into two folders 0 and 1. ) signifying original image and 1 implying face swap image. The train and test folders were stored in a folder named data.After resizing it to 32x32 running the pretrained detection model for fine tuning. Finally testing the fine tuned model on the test dataset to obtain results.

OBSERVATIONS

**Question 1**

**Part 1**

**CNN Classification**

**![](media/d4183bf9ff0c3ef00fd7849350018931.png)![](media/92a819c70cf7ba9fd89df880e9635d76.png)**

**FGSM attack**

Varying epsilon to see how accuracy changes with epsilon.

Train accuracy = **78.73%**, Test Accuracy = **64.76%** (Before Attack)

| **Epsilon**  | 0.025 | 0.05 | 0.075 | 0.1  | 0.125 | 0.15 | 0.175 | 0.2  |
|--------------|-------|------|-------|------|-------|------|-------|------|
| **Accuracy** | 2.25  | 0.27 | 0.3   | 0.65 | 1.27  | 1.94 | 2.61  | 3.14 |

**![](media/56438be6d909fdbd1e5c6825854e0e35.png)**

The first image is an original image without attack, the second perturbed image is after we attack the image using FGSM while the third image is the difference between the two. To the naked eye the first two images look almost similar but the model misclassified it where the ground truth was a truck and the model predicted it as a car.

**![](media/ea7dac1d9eee651e4bd94edefd3b6f25.png)**

**Part 2**

**Attacks of PGD and Mask**

Before Attack: Resnet :- Train accuracy = 86.54%, Test accuracy = 77.93%

Shufflenetv2 :- Train accuracy = 91.8%, Test accuracy = 75.84%

FOR PGD ATTACK:

ON ResNET

| **Alpha**    | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |
|--------------|------|------|------|------|------|
| **Accuracy** | 1.28 | 0.72 | 1.35 | 2.30 | 3.06 |

![](media/787531b1827d031f75606eecea1af64d.png)

ON ShuffleNet

| **Alpha**    | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |
|--------------|------|------|------|------|------|
| **Accuracy** | 0.19 | 0.31 | 0.9  | 1.81 | 2.71 |

![](media/e6e2d07da87dff8f924908b61be469a8.png)

FOR **MASKED** BASED ATTACK:

Before Attack: Resnet :- Train accuracy = 86.54%, Test accuracy = 77.93%

Shufflenetv2 :- Train accuracy = 91.8%, Test accuracy = 75.84%

ON ResNET

| **Alpha**    | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |
|--------------|------|------|------|------|------|
| **Accuracy** | 9.39 | 9.65 | 9.87 | 9.88 | 9.48 |

![](media/04e7e0e283e212414e7cf9141ac84b11.png)

ON ShuffleNet

| **Alpha**    | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |
|--------------|------|------|------|------|------|
| **Accuracy** | 9.49 | 9.89 | 9.85 | 9.92 | 9.87 |

![](media/8052d6dd48c49ff9d8e0dc8d1c69579c.png)

**Part 3**

Training accuracy = 100%

Testing accuracy = 99.99%

**Question 2**

Example:

My image Celebrity Face swapped

![](media/c65fb94d900809c661100e93d6ff6af1.png) ![](media/fc2c5c2cae93433960cbfffcb0e95b1e.png) ![](media/d18cf9ddcbe65f3b602e90d3d6b6d7ec.png)

The finetuning of detection model was as follows:

Train accuracy = 52%

Test accuracy = 48%

# RESULTS & CONCLUSIONS

**Question 1**

**Part 1**

Small, simple changes in pixels helps us effectively misclassify the images and drops the accuracy from 64.72% to 0.27% (for epsilon 0.05) which is more than 100 times poor result. Although a human would still be able to identify the image correctly, the model classifying using pixel values finds a large difference and fails due to the attack. Various deep learning architectures can be tricked with only minor input changes. To substantially alter the predictions of the networks using carefully constructed noise patterns that are undetectable to humans seems very simple yet is very effective to fool such complex models.

**Part 2**

PGD attack performs well and drops accuracy to 0.25%. PGD takes much more time to train compared to FGSM. Since it is similar to FGSM but has an iterative based approach unlike single step of FGSM. It attacks successfully and drops accuracy from 78% in testing to 0.72% on ResNet-18 and from 76% to 0.19% in ShuffleNetV2.

Mask based attacks, although successful dont give as good results as PGD or FGSM. This is because we only make changes in part of an image and might leave out some essential features of an image hence PGF , FGSM are more effective as they alter every pixel. It reduced accuracy from 78% to 9.4% for ResNet-18 while 76% to % in ShuffleNetV2.

**Part 3**

Detection model runs very well and gives training accuracy of 100% and testing of 99.99%. This can mean two things : either our detection model works really well or our attack is very poor and can easily be identified. Based on the images above of the FGSM attack it can be understood as a combination of both that the FGSM model although gives a lot of drop in accuracy but can be identified as an attacked image.

**Question 2**

The detection models performance was not up to the mark this can be due to two major reasons. Firstly faceswap generated were quite good and can be deceptive to humans as well. These look much better than FGSM attack images. Secondly resizing the image to a small size 32x32 making it harder for the model to look into details of images. Thus model performance isn't good. However one may think around 50% accuracy can be a cause that the model predicts either all 0(orignal) or 1(fakes) but that is not the case as read from predicted outputs. This implies the faceswaps generated are of real good quality and it further helps when we downsize all images to make the attack even more efficient. Since the dataset is small it would be difficult to train a good model without overfitting. Hence accuracy can be improved by using a larger dataset.

REFERENCES

1.  Dependable Artificial Intelligence Course Lecture Slides
2.  Machine Learning Mastery
3.  <https://github.com/cedro3/sber-swap/blob/main/SberSwapInference.ipynb>
4.  <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial10/Adversarial_Attacks.html>
5.  Datasets of celebrity images and my image have been attached in the Zip file.
