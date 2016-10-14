from __future__ import print_function

import argparse
import os
import random
from collections import deque
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

rng_seed = 1337
random.seed(rng_seed)
np.random.seed(seed=rng_seed)


# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Keras CNN Training!')
    parser.add_argument('-d', '--data_augmentation', action='store_true', default=False, help='Use data augmentation.')
    parser.add_argument('-j', '--join_train', action='store_true', default=False, help='Join all the training datasets.')
    parser.add_argument('--model', action='store', type=str, default='simple', help='Local path to where datasets are stored.')
    parser.add_argument('--test_file', action='store', type=str, default='', help='Local path to where datasets are stored.')
    parser.add_argument('--path', action='store', type=str, default=r'c:/ekgdb/datasets/', help='Local path to where datasets are stored.')
    parser.add_argument('--batch_size', action='store', type=int, default=32, help='Number of seconds to use before annotation.')
    parser.add_argument('--epochs', action='store', type=int, default=30, help='Number of seconds to use after annotation.')
    parser.add_argument('--channels', action='store', type=int, default=1, help='Percentage of smallest class to use for test dataset.')
    parser.add_argument('--early_stop_var', action='store', type=str, default='val_loss', help='Valid options: val_loss, val_acc, loss, and acc.')
    parser.add_argument('--early_stop_patience', action='store', type=int, default=2, help='Valid options: val_loss, val_acc, loss, and acc.')
    parser.add_argument('--checkpointer_var', action='store', type=str, default='val_loss', help='Valid options: val_loss, val_acc, loss, and acc.')
    args = parser.parse_args()
    argsdict = vars(args)

    data_augmentation = argsdict['data_augmentation']
    join_train = argsdict['join_train']
    batch_size = argsdict['batch_size']
    nb_epoch = argsdict['epochs']
    img_channels = argsdict['channels']
    test_file = argsdict['test_file']
    model_str = argsdict['model']
    path_dataset = argsdict['path']
    early_stop_var = argsdict['early_stop_var']
    early_stop_patience = argsdict['early_stop_patience']
    checkpointer_var = argsdict['checkpointer_var']

    # if os.path.exists(path_images):
    #     shutil.rmtree(path_images)

    # Get the file listing of the database files (.dat)
    if not os.path.exists(path_dataset):
        print('Dataset path does not exist: {}'.format(path_dataset))
        exit()

    train_files = deque()
    x_test = None
    # Get the test files first.
    for f in os.listdir(path_dataset):
        if f.endswith('.npz') or f.endswith('.npy'):
            if f == test_file:
                print('Opening/uncompressing test dataset...')
                ds = np.load(path_dataset + f)

                if 'x_test' in ds:
                    x_test = ds['x_test']
                    y_test = ds['y_test']
                elif 'x_train' in ds:
                    x_test = ds['x_train']
                    y_test = ds['y_train']
                elif 'images' in ds:  # TODO: Remove this later after regeneration of new datasets!
                    x_test = ds['images']
                    y_test = ds['labels']

                ds.close()
            else:
                train_files.append(f)

    if x_test is None:
        print('Test file not found: {}!'.format(test_file))
        exit()

    img_rows = img_cols = int(sqrt(x_test.shape[1]))

# ------------------------------------------------------------------------------------------------------------------
    # Get Model and Compile
    if model_str == 'simple':
        model = get_simple_cnn(img_rows, img_cols)
    elif model_str == 'vgg19':
        model = get_vgg19_model(img_rows, img_cols)
    elif model_str == 'vgg16':
        model = get_vgg16_model(img_rows, img_cols)
    elif model_str == 'alexnet':
        model = get_alexnet_model(img_rows, img_cols)
    else:
        model = get_simple_cnn(img_rows, img_cols)

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor=early_stop_var, patience=early_stop_patience, verbose=1)
    checkpointer = ModelCheckpoint(monitor=checkpointer_var, filepath='{}{}_weights.hdf5'.format(path_dataset, model_str), verbose=1, save_best_only=True)

    if data_augmentation:
        # This will do pre-processing and real-time data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=True)  # randomly flip images

    print('X_test shape:', x_test.shape)
    print('Y_test shape:', y_test.shape)

    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    
    if join_train:
        print('Opening train dataset...')

        x_train, y_train = join_dataset(path_dataset, train_files)

        print('X_train shape:', x_train.shape)
        print('Y_train shape:', y_train.shape)

        history = train_model(model, x_train, y_train, x_test, y_test, img_channels, img_rows, img_cols, checkpointer, early_stopping, data_augmentation, batch_size, nb_epoch)
    else:
        for i, f in enumerate(train_files):
            if f.endswith('.npz') or f.endswith('.npy'):
                print('Opening/uncompressing train dataset {} of {}...'.format(i + 1, len(train_files)))
                ds = np.load(path_dataset + f)

                if 'x_train' in ds:
                    x_train = ds['x_train']
                    y_train = ds['y_train']
                elif 'images' in ds:
                    x_train = ds['images']
                    y_train = ds['labels']
                else:
                    ds.close()
                    continue

                ds.close()
            else:
                continue

            print('X_train shape:', x_train.shape)
            print('Y_train shape:', y_train.shape)

            history = train_model(model, x_train, y_train, x_test, y_test, img_channels, img_rows, img_cols, checkpointer, early_stopping, data_augmentation, batch_size, nb_epoch)

    # ------------------------------------------------------------------------------------------------------------------
    # Load most accurate weights back into model from file.
    model.load_weights('{}{}_weights.hdf5'.format(path_dataset, model_str))

    # Plot model
    plot(model, to_file='{}{}_plot.png'.format(path_dataset, model_str), show_shapes=True)

    # Model as JSON
    json_string = model.to_json()
    open('{}{}_model.json'.format(path_dataset, model_str), 'w').write(json_string)

    # Save the pre-trained weights
    # model.save_weights(path_dataset + db_name + '_vgg19_weights.h5', overwrite=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate the model    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('loss: ', loss)
    print('accuracy: ', accuracy)
    print()

    # Predict
    classes = model.predict_classes(x_test, verbose=1)
    proba = model.predict_proba(x_test, verbose=1)

    print()

    # ------------------------------------------------------------------------------------------------------------------
    val_map = {0: 0, 1: 0, 2: 0, 3: 0}
    histo = deque()
    with open('{}{}_predictions.csv'.format(path_dataset, model_str), 'wb') as csvfile:
        csvfile.write('ImageId, PredictLabel, TestLabel, Proba\n')
        for i, c in enumerate(classes):
            csvfile.write(
                '{},{},{},{},{},{},{}\n'.format(i, c, y_test[i], proba[i][0], proba[i][1], proba[i][2], proba[i][3]))
            if c == np.argmax(y_test[i]):
                val_map[c] += 1
                histo.append(c)

    with open('{}{}_valid.csv'.format(path_dataset, model_str), 'wb') as csvfile:
        csvfile.write('Classs,Correct,Total,Percent\n')
        csvfile.write('Test\n')

        count_map = {0: 0, 1: 0, 2: 0, 3: 0}
        for c in y_test:
            count_map[int(np.argmax(c))] += 1

        for c in count_map:
            csvfile.write('{},{},{},{}\n'.format(c, val_map[c], count_map[c], float(val_map[c]) / count_map[c]))

        csvfile.write('Train\n')

        count_map = {0: 0, 1: 0, 2: 0, 3: 0}
        for c in y_train:
            count_map[int(np.argmax(c))] += 1

        for c in count_map:
            csvfile.write('{},{}\n'.format(c, count_map[c]))

    counts, bins, patches = plt.hist(histo, bins=[0, 1, 2, 3, 4])

    autolabel(patches, plt.gca())
    plt.xlabel('0=N, 1=S, 2=V, 3=F')
    plt.savefig('{}{}_hist.png'.format(path_dataset, model_str))
    plt.show()
    plt.close()

    x = history.epoch
    legend = ['loss']
    y1 = history.history['loss']

    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, y1, marker='.')
    if 'val_loss' in history.history:
        y2 = history.history['val_loss']
        legend.append('val_loss')
        plt.plot(x, y2, marker='.')

        # for xy in zip(x, y2):
        #     plt.annotate('(%.3f, %.3f)' % xy, xy=xy, textcoords='data')

        xy = zip(x, y2)
        xy = xy[len(xy) - 1]
        plt.annotate('(%.3f, %.3f)' % xy, xy=xy, textcoords='data')

    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.xticks(history.epoch, history.epoch)
    plt.legend(legend, loc='upper right')

    plt.savefig('{}{}_loss.png'.format(path_dataset, model_str))
    plt.show()
    plt.close(fig)

    x = history.epoch
    legend = ['acc']

    plt.figure(figsize=(10, 5))
    plt.plot(x, history.history['acc'], marker='.')
    if 'val_acc' in history.history:
        legend.append('val_acc')
        plt.plot(x, history.history['val_acc'], marker='.')

    plt.title('Acc over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.xticks(history.epoch, history.epoch)
    plt.legend(legend, loc='upper right')

    plt.savefig('{}{}_acc.png'.format(path_dataset, model_str))
    plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
def train_model(model, x_train, y_train, x_test, y_test, img_channels, img_rows, img_cols, checkpointer, early_stopping, data_augmentation, batch_size=32, nb_epoch=30, datagen=None):
    """
    Trains the Model with the given list of train files.
    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param img_channels:
    :param img_rows:
    :param img_cols:
    :param data_augmentation:
    :param datagen:
    :return:
    """

    # x_train, y_train = shuffle(x_train, y_train, random_state=rng_seed)  # TODO: Should probably do an inplace shuffle here.

    # Inplace shuffle. Less memory but takes longer.
    # print('Performing in-place shuffle...')
    # rng_state = np.random.get_state()
    # np.random.shuffle(x_train)
    # np.random.set_state(rng_state)
    # np.random.shuffle(y_train)

    # Reshape for Keras
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)

    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            validation_data=(x_test, y_test),
                            # validation_split=0.2,
                            # shuffle='batch',
                            shuffle=True,
                            verbose=1,
                            callbacks=[checkpointer, early_stopping])
    else:
        print('Using real-time data augmentation.')

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)

        # fit the model on the batches generated by datagen.flow()
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                                   batch_size=batch_size),
                                      samples_per_epoch=x_train.shape[0],
                                      nb_epoch=nb_epoch,
                                      validation_data=(x_test, y_test))

    return history


# ----------------------------------------------------------------------------------------------------------------------
def join_dataset(path_dataset, files):
    """
    Numpy concatenate like with less memory. Very slow!
    :param path_dataset:
    :param files:
    :return:
    """
    ds_len = 0
    for f in files:
        if f.endswith('.npz') or f.endswith('.npy'):
            ds = np.load(path_dataset + f)

            if 'x_train' in ds:
                x = ds['x_train']
                y = ds['y_train'] 
            elif 'images' in ds:  # TODO: Remove this later after regeneration of new datasets!
                x = ds['images']
                y = ds['labels']
            else:
                ds.close()
                continue

            ds_len += len(y)
            x_shape = x.shape
            y_shape = y.shape

    x_train = np.empty((ds_len, x_shape[1]), dtype=np.float32)
    y_train = np.empty((ds_len, y_shape[1]), dtype=np.float32)

    ds_index = 0
    for f in files:
        if f.endswith('.npz') or f.endswith('.npy'):
            ds = np.load(path_dataset + f)

            if 'x_train' in ds:
                x = ds['x_train']
                y = ds['y_train']
            else:
                ds.close()
                continue

            x_train[ds_index:ds_index+y.shape[0], :] = x
            y_train[ds_index:ds_index+y.shape[0], :] = y
            ds_index += len(y)

    return x_train, y_train


# ----------------------------------------------------------------------------------------------------------------------
def get_simple_cnn(img_rows, img_cols, img_channels=1, nb_classes=4):
    """
    Get VGG like model.
    :param img_channels: Number of image channels in image dataset.  (Grayscale = 1)
    :param img_rows: Number of rows in image dataset.
    :param img_cols: Number of cols in image dataset.
    :param nb_classes: Number of different classes in the dataset.
    :return: Keras sequential model.
    """

    print('Using Simple CNN Model!')

    nb_filers = [8, 32, 64, 512, 512, 256, 512, 1024]
    model = Sequential()

    model.add(Convolution2D(nb_filers[0], 3, 3, subsample=(1, 1), border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filers[0], 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filers[1], 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filers[1], 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.25))

    # model.add(Convolution2D(nb_filers[3], 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filers[3], 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    # model.add(Dropout(0.25))

    # model.add(Convolution2D(nb_filers[4], 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filers[4], 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    # model.add(Dropout(0.25))

    # model.add(Convolution2D(nb_filers[5], 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filers[5], 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    # model.add(Dropout(0.25))

    # model.add(Convolution2D(nb_filers[6], 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filers[6], 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    # model.add(Dense(nb_filers[7]))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(nb_filers[7]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


# ----------------------------------------------------------------------------------------------------------------------
def get_alexnet_model(img_rows, img_cols, img_channels=1, nb_classes=4):
    """
    Get Alexnet like model.
    :param img_channels: Number of image channels in image dataset.  (Grayscale = 1)
    :param img_rows: Number of rows in image dataset.
    :param img_cols: Number of cols in image dataset.
    :param nb_classes: Number of different classes in the dataset.
    :return: Keras sequential model.
    """

    print('Using Alex-NET Model!')

    nb_filers = [8, 32, 128, 512, 4096]
    model = Sequential()
    model.add(Convolution2D(nb_filers[0], 11, 11, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    # model.add(BatchNormalization((64,226,226)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(nb_filers[1], 7, 7, border_mode='same'))
    # model.add(BatchNormalization((128,115,115)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(nb_filers[2], 3, 3, border_mode='same'))
    # model.add(BatchNormalization((128,112,112)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(nb_filers[3], 3, 3, border_mode='same'))
    # model.add(BatchNormalization((128,108,108)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Flatten())
    model.add(Dense(nb_filers[4], init='normal'))
    model.add(BatchNormalization(nb_filers[4]))
    model.add(Activation('relu'))
    model.add(Dense(nb_filers[4], init='normal'))
    model.add(BatchNormalization(nb_filers[4]))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes, init='normal'))
    model.add(BatchNormalization(nb_classes))
    model.add(Activation('softmax'))

    return model


# ----------------------------------------------------------------------------------------------------------------------
def get_vgg16_model(img_rows, img_cols, img_channels=1, nb_classes=4):
    """
    Get VGG-16 like model.
    :param img_rows:
    :param img_cols:
    :param img_channels:
    :param nb_classes:
    :return:
    """

    print('Using VGG-16 CNN Model!')

    nb_filers = [8, 32, 64, 512, 1024]
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_channels, img_rows, img_cols)))
    model.add(Convolution2D(nb_filers[0], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[0], 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[1], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[1], 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[2], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[2], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[2], 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(nb_filers[4], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_filers[4], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


# ----------------------------------------------------------------------------------------------------------------------
def get_vgg19_model(img_rows, img_cols, img_channels=1, nb_classes=4):
    """
    Get VGG-19 like model.
    :param img_channels: Number of image channels in image dataset.  (Grayscale = 1)
    :param img_rows: Number of rows in image dataset.
    :param img_cols: Number of cols in image dataset.
    :param nb_classes: Number of different classes in the dataset.
    :return: Keras sequential model.
    """

    print('Using VGG-19 CNN Model!')

    nb_filers = [8, 32, 128, 512, 1024]
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_channels, img_rows, img_cols)))
    model.add(Convolution2D(nb_filers[0], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[0], 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[1], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[1], 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[2], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[2], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[2], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[2], 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(nb_filers[3], 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(nb_filers[4], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_filers[4], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


# ----------------------------------------------------------------------------------------------------------------------
def get_class_indeces(dataset_labels):
    """
    Count the number of beats for each class and bin them into a Dict with associated indeces.
    :param dataset_labels: Dataset to bin labels into dict with associated indices.
    :return: Dict where key is class and value is list of indeces.
    """
    class_indeces = {'N': [], 'S': [], 'V': [], 'F': []}
    for idx, label in enumerate(dataset_labels):
        if np.array_equal(label, [1., 0., 0., 0.]):
            class_indeces['N'].append(idx)
        elif np.array_equal(label, [0., 1., 0., 0.]):
            class_indeces['S'].append(idx)
        elif np.array_equal(label, [0., 0., 1., 0.]):
            class_indeces['V'].append(idx)
        elif np.array_equal(label, [0., 0., 0., 1.]):
            class_indeces['F'].append(idx)
        else:
            continue

    return class_indeces


# ----------------------------------------------------------------------------------------------------------------------
def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.95:  # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width() / 2., label_position, '%d' % int(height), ha='center', va='bottom')


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
