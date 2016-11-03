from __future__ import print_function

import argparse
import os
import random
from collections import deque
from math import sqrt

import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from sklearn.metrics import classification_report, confusion_matrix

rng_seed = 11102017
random.seed(rng_seed)
np.random.seed(rng_seed)


# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Keras CNN Training!')
    parser.add_argument('-d', '--data_augmentation', action='store_true', default=False, help='Use data augmentation.')
    parser.add_argument('-j', '--join_train', action='store_true', default=False, help='Join all the training datasets.')
    parser.add_argument('--model', action='store', type=str, default='simple', help='Options: simple, zero, vgg16, vgg19, alexnet')
    parser.add_argument('--test_file', action='store', type=str, default='', help='File to use for final results.')
    parser.add_argument('--val_file', action='store', type=str, default='', help='If present used for early_stop and checkpointer.')
    parser.add_argument('--path', action='store', type=str, default=r'c:/ekgdb/datasets/', help='Local path to where datasets are stored.')
    parser.add_argument('--batch_size', action='store', type=int, default=32, help='Number of seconds to use before annotation.')
    parser.add_argument('--epochs', action='store', type=int, default=30, help='Number of seconds to use after annotation.')
    parser.add_argument('--channels', action='store', type=int, default=1, help='Percentage of smallest class to use for test dataset.')
    parser.add_argument('--early_stop_var', action='store', type=str, default='', help='Valid options: val_loss, val_acc, loss, and acc.')
    parser.add_argument('--early_stop_patience', action='store', type=int, default=2, help='Valid options: val_loss, val_acc, loss, and acc.')
    parser.add_argument('--checkpointer_var', action='store', type=str, default='', help='Valid options: val_loss, val_acc, loss, and acc.')
    args = parser.parse_args()
    argsdict = vars(args)

    data_augmentation = argsdict['data_augmentation']
    join_train = argsdict['join_train']
    batch_size = argsdict['batch_size']
    nb_epoch = argsdict['epochs']
    img_channels = argsdict['channels']
    test_file = argsdict['test_file']
    val_file = argsdict['val_file']
    model_str = argsdict['model']
    path_dataset = argsdict['path']
    early_stop_var = argsdict['early_stop_var']
    early_stop_patience = argsdict['early_stop_patience']
    checkpointer_var = argsdict['checkpointer_var']

    # if os.path.exists(path_images):
    #     shutil.rmtree(path_images)

    if len(path_dataset.split('/')) > 1 and not path_dataset.endswith('/'):
        path_dataset += '/'

    if len(path_dataset.split('\\')) > 1 and not path_dataset.endswith('\\'):
        path_dataset += '\\'

    # Get the file listing of the database files (.dat)
    if not os.path.exists(path_dataset):
        print('Dataset path does not exist: {}'.format(path_dataset))
        exit()

    train_files = deque()
    is_test_present = False
    x_val = None
    # Get the train files first.
    for f in os.listdir(path_dataset):
        if f.endswith('.npz') or f.endswith('.npy'):
            ds = np.load(path_dataset + f)
            if f == val_file or val_file == test_file:
                print('Opening/uncompressing val dataset...')
                if 'x_test' in ds:
                    x_val = ds['x_test']
                    y_val = ds['y_test']
                elif 'x_train' in ds:
                    x_val = ds['x_train']
                    y_val = ds['y_train']

            elif f == test_file:
                # Check for train dataset to add to list:
                if 'x_train' in ds:
                    train_files.append(f)

                is_test_present = True
                continue

            else:
                train_files.append(f)

            ds.close()

    if test_file == '' and len(train_files) == 1:
        test_file = train_files[0]
        is_test_present = True

    if not is_test_present:
        print('Test file not found: {}!'.format(test_file))
        exit()

    ds = np.load(path_dataset + test_file)
    if 'x_test' in ds:
        img_rows = img_cols = int(sqrt(ds['x_test'].shape[1]))
    elif 'x_train' in ds:
        img_rows = img_cols = int(sqrt(ds['x_train'].shape[1]))
    ds.close()

    if x_val is not None:
        print('X_val shape:', x_val.shape)
        print('Y_val shape:', y_val.shape)

        x_val = x_val.reshape(x_val.shape[0], img_channels, img_rows, img_cols)

    # ------------------------------------------------------------------------------------------------------------------
    # Get Model and Compile
    if model_str == 'simple':
        model = get_simple_cnn(img_rows, img_cols)
    elif model_str == 'max':
        model = get_max_cnn(img_rows, img_cols)
    elif model_str == 'vgg19':
        model = get_vgg19_model(img_rows, img_cols)
    elif model_str == 'vgg16':
        model = get_vgg16_model(img_rows, img_cols)
    elif model_str == 'alexnet':
        model = get_alexnet_model(img_rows, img_cols)
    else:
        model = get_simple_cnn(img_rows, img_cols)

    # Plot model
    plot(model, to_file='{}{}_plot.png'.format(path_dataset, model_str), show_shapes=True)

    # Save model as JSON
    json_string = model.to_json()
    with open('{}{}_model.json'.format(path_dataset, model_str), 'w') as model_file:
        model_file.write(json_string)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    callbacks = []
    if early_stop_var != '':
        early_stopping = EarlyStopping(monitor=early_stop_var, patience=early_stop_patience, verbose=1)
        callbacks.append(early_stopping)

    if checkpointer_var != '':
        checkpointer = ModelCheckpoint(monitor=checkpointer_var, filepath='{}{}_weights.hdf5'.format(path_dataset, model_str), verbose=1, save_best_only=True)
        callbacks.append(checkpointer)

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
    
    if join_train:
        print('Opening/uncompressing and joining train dataset...')

        x_train, y_train = join_dataset(path_dataset, train_files)

        print('X_train shape:', x_train.shape)
        print('Y_train shape:', y_train.shape)

        history = train_model(model, x_train, y_train, x_val, y_val, img_channels, img_rows, img_cols, callbacks, data_augmentation, batch_size, nb_epoch)
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

            history = train_model(model, x_train, y_train, x_val, y_val, img_channels, img_rows, img_cols, callbacks, data_augmentation, batch_size, nb_epoch)

    # ------------------------------------------------------------------------------------------------------------------
    # Load most accurate weights back into model from file.
    model.load_weights('{}{}_weights.hdf5'.format(path_dataset, model_str))

    del x_train
    del y_train

    print('Opening/uncompressing test dataset...')
    if val_file == test_file:
        x_test = x_val
        y_test = y_val

    else:
        ds = np.load(path_dataset + test_file)

        if 'x_test' in ds:
            x_test = ds['x_test']
            y_test = ds['y_test']

        elif 'x_train' in ds:
            x_test = ds['x_train']
            y_test = ds['y_train']

        ds.close()

    print('X_test shape:', x_test.shape)
    print('Y_test shape:', y_test.shape)

    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)

    evaluate_model(path_dataset, model_str, history, model, x_test, y_test)


# ----------------------------------------------------------------------------------------------------------------------
def evaluate_model(path_dataset, model_str, history, model, x_test, y_test):
    """
    Evaluate the model and print results.
    :param path_dataset:
    :param model_str:
    :param history:
    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('loss: ', loss)
    print('accuracy: ', accuracy)
    print()

    # Attain predictions from test input.
    y_prob = model.predict_proba(x_test, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_lbls = np.argmax(y_test, axis=1)

    # Create classification matrix
    labels = ['N', 'S', 'V', 'F']
    class_report = classification_report(y_lbls, y_pred, target_names=labels)
    print(class_report)
    with open('{}{}_classification_report.csv'.format(path_dataset, model_str), 'wb') as reportfile:
        reportfile.write(class_report)

    # Create confusion matrix
    cm = confusion_matrix(y_lbls, y_pred)
    np.set_printoptions(precision=3)

    # Plot confusion matrix
    plot_confusion_matrix(cm, classes=labels, title='Confusion Matrix', save_path='{}{}_confusion_matrix.png'.format(path_dataset, model_str))
    plot_confusion_matrix(cm, classes=labels, normalize=True, title='Normalized Confusion Matrix', save_path='{}{}_normalized_confusion_matrix.png'.format(path_dataset, model_str))

    # Create csv file with prediction so we can hand pick examples later.
    with open('{}{}_predictions.csv'.format(path_dataset, model_str), 'wb') as csvfile:
        csvfile.write('ImageId, PredictLabel, TestLabel, Proba\n')
        for i, c in enumerate(y_prob):
            csvfile.write('{},{},{},{},{},{},{}\n'.format(i, c, y_test[i], y_prob[i][0], y_prob[i][1], y_prob[i][2], y_prob[i][3]))

    # Plot loss and val_loss vs epochs.
    h = history
    x = history.epoch

    fig = plt.figure(figsize=(10, 5))
    legend = ['loss']
    y1 = h.history['loss']
    ax = fig.add_subplot(111)
    plt.plot(x, y1, marker='.')
    # for xy in zip(x, y1):
    #     ax.annotate('(%.3f, %.3f)' % xy, xy=xy, textcoords='data')

    if 'val_loss' in h.history:
        legend.append('val_loss')
        y2 = h.history['val_loss']
        plt.plot(x, y2, marker='.')

        # for xy in zip(x, y2):
        #     plt.annotate('(%.3f, %.3f)' % xy, xy=xy, textcoords='data')

    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.xticks(h.epoch, h.epoch)
    plt.legend(legend, loc='upper right')
    plt.savefig('{}{}_loss.png'.format(path_dataset, model_str))
    # plt.show()
    plt.close(fig)

    # Plot acc and val_acc vs epochs.
    fig = plt.figure(figsize=(10, 5))
    legend = ['acc']
    y1 = h.history['acc']
    ax = fig.add_subplot(111)
    plt.plot(x, y1, marker='.')
    # for xy in zip(x, y1):
    #     ax.annotate('(%.3f, %.3f)' % xy, xy=xy, textcoords='data')

    if 'val_acc' in h.history:
        legend.append('val_acc')
        y2 = h.history['val_acc']
        plt.plot(x, y2, marker='.')

        # for xy in zip(x, y2):
        #     plt.annotate('(%.3f, %.3f)' % xy, xy=xy, textcoords='data')

    plt.title('Acc over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.xticks(h.epoch, h.epoch)
    plt.legend(legend, loc='upper right')
    plt.savefig('{}{}_acc.png'.format(path_dataset, model_str))
    # plt.show()
    plt.close(fig)

    # Write the history to a file.
    with open('{}{}_history.csv'.format(path_dataset, model_str), 'wb') as histfile:
        header_str = 'epoch,loss,acc'
        if 'val_loss' in h.history and 'val_acc' in h.history:
            header_str = '{},val_loss,val_acc'.format(header_str)
        histfile.write('{}\n'.format(header_str))

        for i in x:
            data_str = '{},{},{}'.format(i, h.history['loss'][i], h.history['acc'][i])
            if 'val_loss' in h.history and 'val_acc' in h.history:
                data_str = '{},{},{},'.format(data_str, h.history['val_loss'][i], h.history['val_acc'][i])
            histfile.write('{}\n'.format(data_str))


# ----------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    """
    This function prints and plots the confusion matrix.
    Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    :param cm:
    :param classes:
    :param normalize: Normalization can be applied by setting `normalize=True`.
    :param title:
    :param cmap:
    :return:
    """
    fig = plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if normalize:
        vmax = 1
    else:
        vmax = np.sum(cm, axis=1).max()

    plt.imshow(cm, interpolation='nearest', vmin=0, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, '{0:.3f}'.format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
def train_model(model, x_train, y_train, x_val, y_val, img_channels, img_rows, img_cols, callbacks, data_augmentation, batch_size=32, nb_epoch=30, datagen=None):
    """
    Trains the Model with the given list of train files.
    :param model:
    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param img_channels:
    :param img_rows:
    :param img_cols:
    :param callbacks:
    :param data_augmentation:
    :param batch_size:
    :param nb_epoch:
    :param datagen:
    :return:
    """

    # x_train, y_train = shuffle(x_train, y_train, random_state=rng_seed)  # TODO: Should probably do an inplace shuffle here.

    # Inplace shuffle. Less memory but takes longer.
    print('Performing in-place shuffle...')
    rng_state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)

    # Reshape for Keras
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)

    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            validation_data=(x_val, y_val),
                            # validation_split=0.2,
                            # shuffle='batch',
                            shuffle=True,
                            verbose=1,
                            callbacks=callbacks)
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
                                      validation_data=(x_val, y_val),
                                      shuffle=False,
                                      verbose=1,
                                      callbacks=callbacks)

    return history


# ----------------------------------------------------------------------------------------------------------------------
def join_dataset(path_dataset, files):
    """
    Concatenate all datasets in files list trying to use small memory footprint. Very slow!
    :param path_dataset: Path the the dataset files.
    :param files: Files to join.
    :return: Concatenated files.
    """
    ds_len = 0
    for f in files:
        if f.endswith('.npz') or f.endswith('.npy'):
            ds = np.load(path_dataset + f)

            if 'y_train' in ds:
                y = ds['y_train'] 
            else:
                ds.close()
                continue

            ds.close()
            ds_len += len(y)

    x_train = np.empty((ds_len, x.shape[1]), dtype=np.float32)
    y_train = np.empty((ds_len, y.shape[1]), dtype=np.float32)

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
    Get Simple model.
    :param img_channels: Number of image channels in image dataset.  (Grayscale = 1)
    :param img_rows: Number of rows in image dataset.
    :param img_cols: Number of cols in image dataset.
    :param nb_classes: Number of different classes in the dataset.
    :return: Keras sequential model.
    """

    print('Using Simple CNN Model!')
    model = Sequential()

    nb_conv1 = 8
    model.add(Convolution2D(nb_conv1, 3, 3, subsample=(1, 1), border_mode='same', activation='relu', input_shape=(img_channels, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2), border_mode='same'))
    # model.add(Dropout(0.25))

    nb_conv2 = 32
    model.add(Convolution2D(nb_conv2, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv2, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    # model.add(Dropout(0.25))

    nb_conv3 = 64
    model.add(Convolution2D(nb_conv3, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv3, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv3, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    print('Filters: {}, {}, {}'.format(nb_conv1, nb_conv2, nb_conv3))

    return model


# ----------------------------------------------------------------------------------------------------------------------
def get_max_cnn(img_rows, img_cols, img_channels=1, nb_classes=4):
    """
    Get VGG like model.
    :param img_channels: Number of image channels in image dataset.  (Grayscale = 1)
    :param img_rows: Number of rows in image dataset.
    :param img_cols: Number of cols in image dataset.
    :param nb_classes: Number of different classes in the dataset.
    :return: Keras sequential model.
    """

    print('Using Max Layers CNN Model!')
    model = Sequential()

    nb_conv1 = 8
    model.add(Convolution2D(nb_conv1, 3, 3, subsample=(1, 1), border_mode='same', activation='relu', input_shape=(img_channels, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2), border_mode='same'))

    nb_conv2 = 32
    model.add(Convolution2D(nb_conv2, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv2, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    nb_conv3 = 32
    model.add(Convolution2D(nb_conv3, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv3, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    nb_conv4 = 64
    model.add(Convolution2D(nb_conv4, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv4, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    nb_conv5 = 64
    model.add(Convolution2D(nb_conv5, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv5, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    nb_conv6 = 128
    model.add(Convolution2D(nb_conv6, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv6, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    nb_conv7 = 128
    model.add(Convolution2D(nb_conv7, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv7, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    nb_conv8 = 256
    model.add(Convolution2D(nb_conv8, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(nb_conv8, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Flatten())

    nb_dense1 = nb_conv8
    model.add(Dense(nb_dense1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_dense1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    print('Filters: {}, {}, {}, {}, {}, {}, {}, {}'.format(nb_conv1, nb_conv2, nb_conv3, nb_conv4, nb_conv5, nb_conv6, nb_conv7, nb_conv8))

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

    print('Using Alex-NET Like Model!')
    model = Sequential()

    nb_filters = [96, 256, 384, 256, 4096]
    # nb_filters = [48, 128, 192, 128, 2048]
    # nb_filters = [24, 64, 96, 64, 1024]

    model.add(Convolution2D(nb_filters[0], 11, 11, subsample=(4, 4), activation='relu', border_mode='valid', input_shape=(img_channels, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(BatchNormalization(axis=1))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Convolution2D(nb_filters[1], 5, 5, subsample=(1, 1), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(BatchNormalization(axis=1))
    model.add(ZeroPadding2D((1, 0, 1, 0)))

    model.add(Convolution2D(nb_filters[2], 3, 3, subsample=(1, 1), activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[2], 3, 3, subsample=(1, 1), activation='relu', border_mode='same'))

    model.add(Convolution2D(nb_filters[3], 3, 3, subsample=(1, 1), activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 0, 1, 0)))

    model.add(Flatten())
    model.add(Dense(nb_filters[4], init='normal'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(nb_filters[4], init='normal'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes, init='normal'))
    model.add(Activation('softmax'))

    print('Filters: {}, {}, {}, {}, {}'.format(nb_filters[0], nb_filters[1], nb_filters[2], nb_filters[3], nb_filters[4]))

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

    print('Using VGG-16 Like Model!')
    model = Sequential()

    # nb_filters = [64, 256, 512, 512, 4096]
    nb_filters = [4, 32, 64, 128, 1024]

    model.add(Convolution2D(nb_filters[0], 3, 3, activation='relu', border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    model.add(Convolution2D(nb_filters[0], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(nb_filters[1], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[1], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(nb_filters[2], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[2], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[2], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Flatten())
    model.add(Dense(nb_filters[4], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_filters[4], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    print('Filters: {}, {}, {}, {}, {}'.format(nb_filters[0], nb_filters[1], nb_filters[2], nb_filters[3], nb_filters[4]))

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

    print('Using VGG-19 Like Model!')
    model = Sequential()

    # nb_filters = [64, 256, 512, 512, 4096]
    nb_filters = [8, 16, 32, 64, 1024]

    model.add(Convolution2D(nb_filters[0], 3, 3, activation='relu', border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    model.add(Convolution2D(nb_filters[0], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(nb_filters[1], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[1], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(nb_filters[2], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[2], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[2], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[2], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(nb_filters[3], 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))

    model.add(Flatten())
    model.add(Dense(nb_filters[4], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_filters[4], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    print('Filters: {}, {}, {}, {}, {}'.format(nb_filters[0], nb_filters[1], nb_filters[2], nb_filters[3], nb_filters[4]))

    return model


# ----------------------------------------------------------------------------------------------------------------------
def get_class_indices(dataset_labels):
    """
    Count the number of beats for each class and bin them into a Dict with associated indices.
    :param dataset_labels: Dataset to bin labels into dict with associated indices.
    :return: Dict where key is class and value is list of indices.
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
    """
    Labels histograms with value on top of bar.
    :param rects: The list of bars
    :param ax: The axis object of plot.
    :return:
    """
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
