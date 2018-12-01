# **Steering Angle Prediction using Behavioral Cloning for Self-Driving Cars** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./pictures/cnn-architecture-624x890.png "Model Architecture"

---
#### My project includes the following files

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My model consists of a convolution neural network based on the architecture suggested by Nvidia here https://devblogs.nvidia.com/deep-learning-self-driving-cars/

![alt text][image1]

The data is normalized in the model using a Keras lambda layer (code line 76). 

#### 2. Attempts to reduce overfitting in the model

Nvidia's model isn't very likely to cause over-fitting, plus the dataset is huge. I also confirmed overfitting wasn't occuring by using a validation dataset that showed loss values coherent with the training dataset.

The model was trained and validated on different data sets with 80% to 20% ratio to ensure that the model was not overfitting (code line 36). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses an adam optimizer instead of classical stochastic gradient descent.

#### 4. Appropriate training data

I've collected and used my own data for the project, which consists of the following:
- Two complete runs on the track controlled using the mouse, in the counter-clockwise direction.
- Two complete runs on the track controlled using the mouse, in the clockwise direction.
- A number of off-track situations to train the network on recovering from such scenarios.
- Extra focus on the parts of the track that has different looking borders or terrain.

The data includes for each steering angle a front-facing photo, a left-facing photo, and a right-facing photo.

### Model Architecture and Training Strategy

#### Solution Design Approach

I trained 14 different models before arriving at the final model that I am submitting. At the beginning I wasn't using a sample generator function, and the data size was still small enough so it didn't cause any issues, however the car was only driving up until the bridge then crashing as soon as it gets onto it.

I started collecting more data, which called for the need of using a generator so I did. One thing I was noticing however, is that the training time was very high, and it was taking forever to train the models, which meant I had to stick to low and practical number of epochs and samples, but it was only resulting in models with rather poor performance, unable to successfully finish the entire track.

Later, I figured out that the reason for the slow training was that I had been using python lists for reading and preprocessing the images instead of numpy arrays. Switching to numpy arrays showed a dramatic decrease in training time, allowing me to use higher values for the hyperparameters for the training process.

While reading the data, for each sample, I would randomly pick the center, right, or left image. For the center image, I would add the corresponsing angle as is. For the left image, I would add 0.2 to the value of the steering angle, and for the right image, I would subtract 0.2 from the angle, this is to help the model avoid driving off the track, and knowing to steer in the opposite direction when it gets too close to the sides of the road.

After reading the data, I performed augmentation on it by randomly choosing some samples and flipping them, then I cropped the top-most and bottom-most portions of the image, as well as the right-most and left-most, in order to get rid of useless data that might serve to confuse the model.

After testing the models and continuously tuning the training hyperparameters according to the results for each model, I started getting better performance. Finally, I added a condition to check on steering angle values for each sample, and for each batch, limit the number of samples with angles representing straight driving, so as to avoid force the model to train on the more challenging turns. At the end of the process, the vehicle was able to drive autonomously around the track without ever leaving the road.

#### Final Hyperparameters
- Number of epochs: 40
- Batch Size: 32
- No. of steps per epoch: 384
- No. of validation samples per epoch: 96

#### Validation Accuracy
98.6%
