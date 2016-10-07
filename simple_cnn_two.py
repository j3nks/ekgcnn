'''
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mitdb_cnn.py
'''

from __future__ import print_function

import os
import random

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
from sklearn.utils import shuffle

rng_seed = 1337
random.seed(rng_seed)
np.random.seed(seed=rng_seed)


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

    nb_filers = [8, 16, 64, 512, 1024]
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

    nb_filers = [4, 32, 64, 256, 512, 256, 512, 1024]
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

    model.add(Convolution2D(nb_filers[2], 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filers[2], 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filers[3], 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filers[3], 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.25))

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
    model.add(Dense(nb_filers[7]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


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


def get_balanced_dataset(x, y, img_pixels, class_len_idx=2):
    """
    Over-sample images to even out Classes [N, S, V, F]
    :param x: Training/Testing Images
    :param y: Training/Testing Labels
    :param class_len_idx: Choose the idx 0 through 3, 0 being the smallest number labels in a class, 3 the largest.
    :return:
    """
    class_indeces = get_class_indeces(y)
    class_lengths = [len(class_indeces['N']), len(class_indeces['S']), len(class_indeces['V']), len(class_indeces['F'])]
    class_lengths.sort()

    random.shuffle(class_indeces['N'])
    random.shuffle(class_indeces['S'])
    random.shuffle(class_indeces['V'])
    random.shuffle(class_indeces['F'])

    nb_labels = class_lengths[class_len_idx]

    # TODO: Get rid of magic number.
    X = np.empty((nb_labels * len(class_indeces), img_pixels ** 2), dtype=np.float32)
    Y = np.empty((nb_labels * len(class_indeces), len(class_indeces)), dtype=np.float32)

    idx = 0
    for cnt in xrange(nb_labels):
        for key in ['N', 'S', 'V', 'F']:
            if nb_labels > len(class_indeces[key]):
                lbl_idx = cnt % len(class_indeces[key])
            else:
                lbl_idx = cnt

            X[idx] = x[class_indeces[key][lbl_idx]]
            Y[idx] = y[class_indeces[key][lbl_idx]]
            idx += 1

    return shuffle(X, Y, random_state=rng_seed)


def main():
    # Model Parameters
    batch_size = 32
    nb_epoch = 30
    nb_classes = 4
    data_augmentation = False
    overwrite = True

    # Input image dimensions
    img_rows = img_cols = 256
    # The images are Grayscale (mode='L')
    img_channels = 1

    # ------------------------------------------------------------------------------------------------------------------
    # Loading Database and reformatting to get training and testing datasets.
    path_db = r'c:/mitdb/images/mitdb/two/'
    db_name = 'mlii'
    ds = ['ds0', 'ds1']
    if not os.path.exists(path_db + db_name + '_oversampled.npz') or overwrite:
        ds0 = np.load(path_db + 'ds0_' + db_name + '.npz')
        ds1 = np.load(path_db + 'ds1_' + db_name + '.npz')

        X_train = ds0['images']
        Y_train = ds0['labels']
        X_test = ds1['images']
        Y_test = ds1['labels']

        ds0.close()
        ds1.close()

        print('x_train shape:', X_train.shape)
        print('y_train shape:', Y_train.shape)
        print('x_test shape:', X_test.shape)
        print('y_test shape:', Y_test.shape)

        X_train, Y_train = get_balanced_dataset(X_train, Y_train, img_rows)
        X_test, Y_test = get_balanced_dataset(X_test, Y_test, img_rows, 0)

        # Save as numpy array 1 image per row
        np.savez_compressed(path_db + db_name + '_oversampled', X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

    else:
        ds = np.load(path_db + db_name + '_oversampled.npz')

        X_train = ds['X_train']
        Y_train = ds['Y_train']
        X_test = ds['X_test']
        Y_test = ds['Y_test']
        ds.close()

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

    # ------------------------------------------------------------------------------------------------------------------
    # Get Model and Compile
    #model = get_alexnet_model(img_rows, img_cols)
    model = get_simple_cnn(img_rows, img_cols)

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # ------------------------------------------------------------------------------------------------------------------
    # Train the Model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    checkpointer = ModelCheckpoint(filepath=path_db + db_name + '_weights.h5', verbose=1, save_best_only=True)
    history = None

    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test),
                            shuffle=True,
                            verbose=1,
                            callbacks=[checkpointer, early_stopping])
    else:
        print('Using real-time data augmentation.')

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
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        history = model.fit_generator(datagen.flow(X_train, Y_train,
                                                   batch_size=batch_size),
                                      samples_per_epoch=X_train.shape[0],
                                      nb_epoch=nb_epoch,
                                      validation_data=(X_test, Y_test))

    # ------------------------------------------------------------------------------------------------------------------
    # Save Model to file

    # Plot model
    plot(model, to_file=(path_db + db_name + '_plot.png'), show_shapes=True)

    # Model as JSON
    json_string = model.to_json()
    open(path_db + db_name + '_model.json', 'w').write(json_string)

    # Save the pre-trained weights
    # model.save_weights(path_db + db_name + '_vgg19_weights.h5', overwrite=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print('loss: ', loss)
    print('accuracy: ', accuracy)
    print()

    # Predict
    classes = model.predict_classes(X_test, verbose=1)
    proba = model.predict_proba(X_test, verbose=1)

    print()

    with open(path_db + db_name + '_keras-cnn.csv', 'wb') as csvfile:
        csvfile.write('ImageId, Classes, Predict\n')
        for i, c in enumerate(classes):
            csvfile.write(str.format('{},{},="{}"\n', i, c, proba[i]))

    # ------------------------------------------------------------------------------------------------------------------
    histo_map = {0: 0, 1: 0, 2: 0, 3: 0}
    histo = []
    with open(path_db + db_name + '_keras-cnn.csv', 'wb') as csvfile:
        csvfile.write('ImageId, Classes, Predict\n')
        for i, c in enumerate(classes):
            csvfile.write(
                str.format('{},{},{},{},{},{},{}\n', i, c, np.argmax(proba[i]), proba[i][0], proba[i][1], proba[i][2],
                           proba[i][3]))
            if c == np.argmax(Y_test[i]):
                histo_map[c] += 1
                histo.append(c)

    counts, bins, patches = plt.hist(histo, bins=[0, 1, 2, 3, 4])

    autolabel(patches, plt.gca())
    plt.xlabel('0=N, 1=S, 2=V, 3=F')
    plt.savefig(path_db + db_name + '_hist.png')
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

    plt.savefig(path_db + db_name + '_loss.png')
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

    plt.savefig(path_db + db_name + '_acc.png')
    plt.show()
    plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
