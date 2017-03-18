import cv2
import csv
import numpy as np
import random
import sklearn

from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Activation, Input, Embedding, Lambda, ELU, LSTM, Dense, merge, Convolution2D, MaxPooling2D, Reshape, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras import initializations
from keras.callbacks import ModelCheckpoint


def get_model(filename='./model.h5', optimizer='Adam', loss='mse'):
    """Configures model in keras, returns a model and a list of callbacks.
    """
    # Network architecture: LeNet
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(32, 32, 3)))
    model.add(Convolution2D(6, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))

    model.add(Convolution2D(16, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(120, 1, 1, border_mode='valid'))

    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation("relu"))
    model.add(Dense(1))

    # sgd parameters
    model.compile(optimizer=optimizer, loss=loss)

    model_path = filename

    # checkpoint to save
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def get_data(filename='drive.log', clean_data=True):
    """Loads data and labels from a file.
    """
    # read csv file
    samples = []
    list_log = [filename,]
    for log_file in list_log:
        with open(log_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)

    if clean_data:
        samples = filter_samples(samples)

    # split data train and validation
    training, validation = train_test_split(samples, test_size=0.2)

    return training, validation

def train_model(model, training=None, validation=None,
                 callback=[], batches=16, epochs=11):
    """Trains a model. Accepts a model, returns a model.
    """

    train_generator = generator_batch_images(training, batch_size=batches)
    validation_generator = generator_batch_images(validation, batch_size=batches)

    # train network
    samples_epoch_train = 2*batches*(np.floor(len(training)/batches))
    samples_epoch_val =  2*batches*(np.floor(len(validation)/batches))
    model.fit_generator(train_generator, samples_per_epoch=samples_epoch_train,
                        validation_data=validation_generator,
                        nb_val_samples=samples_epoch_val,
                        callbacks=callback, nb_epoch=epochs)
    return model


def filter_samples(list_im_angles, num_max_per_bin=1200):
    """Make sure there is around the same number of samples per angle.
    """
    filter_list = []

    all_angles = []
    for sample_line in list_im_angles:
        all_angles.append(float(sample_line[3]))
    all_angles = np.array(all_angles)


    bin_angles = np.arange(-25.05,25.1,0.1)
    hist, label = np.histogram(all_angles, bins=bin_angles)
    inds = np.digitize(all_angles, bin_angles)
    for k_bin in range(bin_angles.shape[0]):
        list_filter_bin = [list_im_angles[k] for k in range(len(inds)) if inds[k]==k_bin]
        if len(list_filter_bin) > num_max_per_bin:
            list_filter_bin = random.sample(list_filter_bin, num_max_per_bin)
        filter_list = filter_list + list_filter_bin
    return filter_list


def generator_batch_images(list_images, batch_size=4):
    """Takes an image list and returns images loaded as a batch.
    """
    num_samples = len(list_images)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(list_images)
        offset_angle = 0.23
        for offset in range(0, int(np.floor(num_samples/batch_size)), batch_size):
            batch_samples = list_images[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if len(batch_sample[0].split('/')) > 3:
                    name = batch_sample[0]
                    rname = batch_sample[1]
                    lname = batch_sample[2]
                else:
                    name = './IMG/' + batch_sample[0].split('/')[-1]
                    lname = './IMG/' + batch_sample[1].split('/')[-1]
                    rname = './IMG/' + batch_sample[2].split('/')[-1]
                center_image = cv2.imread(name)
                left_image = cv2.imread(lname)
                right_image = cv2.imread(rname)
                if batch_sample[3] != "steering":
                    center_angle = float(batch_sample[3])
                    left_angle = center_angle

                    # data augmentation
                    center_image = center_image[20:155]
                    center_image = cv2.resize(center_image, (32,32))
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_image = center_image.astype(np.float32)

                    left_image = left_image[20:155]
                    left_image = cv2.resize(left_image, (320,160))
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    left_image = left_image.astype(np.float32)

                    right_image = right_image[20:155]
                    right_image = cv2.resize(right_image, (320,160))
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                    right_image = right_image.astype(np.float32)

                    flipped_center_image = np.copy(center_image)
                    flipped_center_image = np.fliplr(flipped_center_image)

                    images.append(center_image)
                    angles.append(center_angle)

                    images.append(flipped_center_image)
                    angles.append(-1*center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == '__main__':
    model, callback = get_model(filename='ariel.h5')
    training, validation = get_data()
    train_model(model, training=training, validation=validation, callback=callback)
