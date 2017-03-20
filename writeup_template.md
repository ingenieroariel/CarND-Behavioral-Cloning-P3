#**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

[//]: # (Image References)

[image3]: ./sample_images/sample_center.jpg "Driving on center"
[image4]: ./sample_images/sample_close_left.jpg "Recovering from left"
[image5]: ./sample_images/sample_close_right.jpg "Recovering from right"
[image6]: ./sample_images/normal.png "Normal Image"
[image7]: ./sample_images/mirrored.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py ariel.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a classic LeNet (Two cascaded convolution + maxpooling with relu activation) with a lambda normalization layer at the beginning.  (model.py lines 25-39)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 32).

 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. When seeing the bad performance on the advanced track, I realized some overfitting may have occurred, I tried training using less epochs but still got the same result.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 19).

####4. Appropriate training data

Collecting training data was a very interesting activity, whenever the car went off track and into the lake my solution was to collect more data. There was a lot of trial and error but the resulting dataset makes it possible to get very good results even with a simple network architecture.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model from the LeNet lab. I decided to not invest too much time on the neural network architecture and focus instead on data acquisition.

In earlier iterations the car steered more to one side than the other and got off track (more error on the validation set than the training set) this means the model was overfitting to just turning to one side instead of learning to drive.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I increased the speed from 9 to 30 and the car kept driving fine.

####2. Final Model Architecture

The final model architecture (model.py lines 19-50) consisted of a convolution neural network with the following layers and layer sizes:
```
Lambda(lambda x: x/127.5 - 1, input_shape=(32, 32, 3))
Convolution2D(6, 5, 5)
MaxPooling2D(pool_size=(2, 2))
Activation("relu")

Convolution2D(16, 5, 5)
MaxPooling2D(pool_size=(2, 2))
Activation("relu")
Dropout(0.5)

Convolution2D(120, 1, 1)

Dense(84)
Activation("relu")
Dense(1)
```


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center driving][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover.

![From left][image4]
![From right][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![Normal][image6]
![Mirrored][image7]

After the collection process, I then preprocessed this data by selecting 500 different angles, equally spaced from -25 to 25 degrees and 1200 data points for each angle.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 11 as evidenced by a higher error rate (mse) in the validation dataset when using less. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is a video of the car driving itself:
https://www.youtube.com/watch?v=ItEjCzqNa-Y&feature=youtu.be
