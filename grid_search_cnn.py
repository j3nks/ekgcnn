'''
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mitdb_cnn.py
'''

from __future__ import print_function

import os
import random

import matplotlib.pyplot as plt

random.seed(1337)
import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import cross_validation
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten


# ----------------------------------------------------------------------------------------------------------------------
# Model Parameters
batch_size = 32
nb_epoch = 50
nb_classes = 4
data_augmentation = False

# Input image dimensions
img_rows = img_cols = 128
# The images are Grayscale (mode='L')
img_channels = 1

# ----------------------------------------------------------------------------------------------------------------------

path_db = r'c:/mitdb/images/'
db_name = 'mlii'

def data():
    """Load the data-set and reformat the data to fit to Keras Convolution2D input."""
    if not os.path.exists(path_db + db_name + '_oversampled.npz'):
        ds = np.load(path_db + db_name + '.npz')

        images = ds['images']
        labels = ds['labels']

        ds.close()

        # --------------------------------------------------------------------------------------------------------------
        # Over-sample images to even out Classes
        # [N, S, V, F]

        used_idx = {}
        x_train = np.empty((7009 * 4, 128 ** 2), dtype=np.float32)
        y_train = np.empty((7009 * 4, 4), dtype=np.float32)
        train_idx = 0
        for i in xrange(0, 4):
            class_cnt = 0
            while class_cnt < 7009:
                if i == 0:
                    rnd_idx = random.randint(0, len(images))
                    if not used_idx.has_key(rnd_idx) and np.array_equal(labels[rnd_idx], [1., 0., 0., 0.]):
                        x_train[train_idx] = images[rnd_idx]
                        y_train[train_idx] = labels[rnd_idx]
                        train_idx += 1
                        class_cnt += 1
                        used_idx[rnd_idx] = None
                else:
                    for j in xrange(0, len(images)):
                        if i == 1 and class_cnt < 7009 and np.array_equal(labels[j], [0., 1., 0., 0.]):
                            x_train[train_idx] = images[j]
                            y_train[train_idx] = labels[j]
                            train_idx += 1
                            class_cnt += 1
                        elif i == 2 and class_cnt < 7009 and np.array_equal(labels[j], [0., 0., 1., 0.]):
                            x_train[train_idx] = images[j]
                            y_train[train_idx] = labels[j]
                            train_idx += 1
                            class_cnt += 1
                        elif i == 3 and class_cnt < 7009 and np.array_equal(labels[j], [0., 0., 0., 1.]):
                            x_train[train_idx] = images[j]
                            y_train[train_idx] = labels[j]
                            train_idx += 1
                            class_cnt += 1
                        elif class_cnt >= 7009:
                            break

        # Save as numpy array 1 image per row
        np.savez_compressed(path_db + db_name + '_oversampled', X_train=x_train, Y_train=y_train)

        del images
        del labels
    else:
        ds = np.load(path_db + db_name + '_oversampled.npz')

        x_train = ds['X_train']
        y_train = ds['Y_train']

        ds.close()

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_train, y_train, test_size=0.3, random_state=1337)

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    print('X_train shape:', x_train.shape)
    print('Y_train shape:', y_train.shape)
    print('X_test shape:', x_test.shape)
    print('Y_test shape:', y_test.shape)

    return x_train, y_train, x_test, y_test


def model(X_train, Y_train, X_test, Y_test):
    """The function returns a model using a VGG like structure:"""
    model = Sequential()

    model.add(Convolution2D({{choice([2, 4, 8, 16])}}, {{choice([3, 5, 7, 11, 15])}}, {{choice([3, 5, 7, 11, 15])}}, border_mode={{choice(['same', 'valid'])}}, input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D({{choice([2, 4, 8, 16])}}, {{choice([3, 5, 7, 11, 15])}}, {{choice([3, 5, 7, 11, 15])}}, border_mode={{choice(['same', 'valid'])}}))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size={{choice([(4,4),  (8,8)])}}, strides=(2, 2), border_mode={{choice(['same', 'valid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Convolution2D({{choice([8, 16, 32])}}, 3, 3, border_mode={{choice(['same', 'valid'])}}, input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D({{choice([8, 16, 32])}}, 3, 3, border_mode={{choice(['same', 'valid'])}}))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2), border_mode={{choice(['same', 'valid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    if conditional({{choice(['two', 'three'])}}) == 'three':
        model.add(Convolution2D({{choice([16, 32, 64])}}, 3, 3, border_mode={{choice(['same', 'valid'])}}, input_shape=(img_channels, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D({{choice([16, 32, 64])}}, 3, 3, border_mode={{choice(['same', 'valid'])}}))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode={{choice(['same', 'valid'])}}))
        model.add(Dropout({{uniform(0, 1)}}))

    model.add(Flatten())
    model.add(Dense({{choice([128, 256, 512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'adadelta', 'adamax', sgd])}},
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    checkpointer = ModelCheckpoint(filepath=path_db + db_name + '_vgg19_weights.h5', verbose=1, save_best_only=True)

    history = model.fit(X_train, Y_train,
                            batch_size={{choice([16, 32])}},
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test),
                            shuffle=True,
                            verbose=1,
                            callbacks=[checkpointer, early_stopping])

    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Score: ', score)
    print('Test Accuracy: ', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model, 'history': history}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    X_train, Y_train, X_test, Y_test = data()

    # ----------------------------------------------------------------------------------------------------------------------
    # Save Model to file

    # Plot model
    plot(best_model, to_file=(path_db + db_name + '_vgg19_plot.png'), show_shapes=True)

    # Model as JSON
    json_string = best_model.to_json()
    open(path_db + db_name + '_vgg19_model.json', 'w').write(json_string)

    # Save the pre-trained weights
    # best_model.save_weights(path_db + db_name + '_vgg19_weights.h5', overwrite=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # Evaluate the model
    loss, accuracy = best_model.evaluate(X_test, Y_test, verbose=1)
    print('loss: ', loss)
    print('accuracy: ', accuracy)
    print()

    # Predict
    classes = best_model.predict_classes(X_test, verbose=1)
    proba = best_model.predict_proba(X_test, verbose=1)

    print()

    with open(path_db + db_name + '_keras-cnn.csv', 'wb') as csvfile:
        csvfile.write('ImageId, Classes, Predict\n')
        for i, c in enumerate(classes):
            csvfile.write(str.format('{},{},="{}"\n', i, c, proba[i]))

    # ----------------------------------------------------------------------------------------------------------------------
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
