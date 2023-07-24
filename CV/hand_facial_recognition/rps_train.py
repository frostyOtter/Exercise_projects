import tensorflow as tf

import os
import numpy as np
import pandas as pd

import rps_create_csv

from tqdm import tqdm
import cv2
from sklearn import metrics, model_selection


def create_dataset(training_df, image_dir):
    # create empty list to store image
    images = []
    # create empty list to store targets
    targets = []
    # label dictionary
    label2num = {
        'rock':0,
        'paper':1,
        'scissor':2,
        'nothing':3
    }
    # loop over the dataframe
    for index, row in tqdm(
        training_df.iterrows(),
        total = len(training_df),
        desc = 'processing images'
    ):
        # get img dir base on label
        img_dir = os.path.join(image_dir, row['target'])
        # get image id
        image_id = row['ImageId']
        # create image path
        image_path = os.path.join(img_dir, image_id)
        # open image using PIL
        image = cv2.imread(image_path)
        # resize image to 256, 256, using bilinear resampling
        image = cv2.resize(image,[226, 226])
        # threshold img
        _, thresh_img = cv2.threshold(image, 122, 255, cv2.THRESH_BINARY)
        # convert to array
        image = np.array(thresh_img, dtype = np.float32)
        # convert to range 0-1
        image = image/255.
        # ravel
        image = image.ravel()
        # append to list above
        images.append(image)
        targets.append(int(row[['target']].map(label2num)))
    # convert list to array
    images = np.array(images)
    return images, targets


if __name__ == '__main__':
    # init path to files
    image_path = '/Users/mike/SUMMER2023/DPL302m/RPS_project/rps_img'
    # create csv file
    rps_create_csv.create_csv(image_path)
    #get csv file path
    csv_path = '/Users/mike/SUMMER2023/DPL302m/RPS_project/rps_dataset.csv'
    # init input image shape
    input_image_shape = (226, 226)
    # read csv with imageid and target columns
    df = pd.read_csv(csv_path)
    # split into train and test
    df_train, df_test = model_selection.train_test_split(
        df, test_size=0.1, random_state=42
    )
    
    # create train, test dataset
    X_train, y_train = create_dataset(df_train, image_path)
    X_test, y_test = create_dataset(df_test, image_path)


    # Initialize model
    model = tf.keras.Sequential([
        # first convolution
        tf.keras.layers.Convolution2D(
            input_shape = input_image_shape,
            filters = 64,
            kernel_size = 3,
            activation='relu'),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides = (2, 2)
        ),
        # second convolution
        tf.keras.layers.Convolution2D(
            filters = 128,
            kernel_size = 3,
            activation='relu'),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides = (2, 2)
        ),
        # third convolution
        tf.keras.layers.Convolution2D(
            filters = 256,
            kernel_size = 3,
            activation='relu'),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides = (2, 2)
        ),
        # flatten and dropout
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        # 64 neuron dense layer
        tf.keras.layers.Dense(
            units = 512,
            activation='relu'
        ),
        # output layer
        tf.keras.layers.Dense(
            units = 4,
            activation= 'softmax'
        )
    ])

    # init rmsprop optimizer
    rmsprop_optimizer = tf.keras.optimizers.RMSprop()
    # compile model
    model.compile(
        optimizer =rmsprop_optimizer,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    # fit to model for training
    training_history = model.fit(
        X_train, y_train,
        validation_split= 0.1,
        epochs = 10, verbose = 1
    )

    # predict the test df
    preds = model.predict(X_test)
    f1_score = metrics.f1_score(y_test, preds)
    print(f1_score)
    model.save('/Users/mike/SUMMER2023/DPL302m/RPS_project/rps_model_weights.h5')