from __future__ import division, print_function

import argparse
import copy
import io
import os
import random
import sys
from collections import deque
from math import ceil, floor

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywfdb
from PIL import Image
from tqdm import *

rng_seed = 2017
random.seed(rng_seed)
np.random.seed(seed=rng_seed)


# ----------------------------------------------------------------------------------------------------------------------
class Beat:
    """Class that holds all info for a specific type of signal."""

    def __init__(self, rec_name='', sig_name='', ann_idx=0, start=0, end=0, lbl_char='', lbl_onehot=[0., 0., 0., 0.], signal=[], pad=0):
        self.rec_name = rec_name
        self.sig_name = sig_name
        self.ann_idx = ann_idx
        self.start = start
        self.end = end
        self.lbl_char = lbl_char
        self.lbl_onehot = lbl_onehot
        self.signal = signal
        self.pad = pad


# ----------------------------------------------------------------------------------------------------------------------
class MaxSample:
    """Class that holds all info for a specific type of signal."""

    def __init__(self, rec_name='', sig_name='', lbl_char='', ann_idx=0, prev_ann_idx=0, next_ann_idx=0, pre_samples=0, post_samples=0, length=0, beat_idx=0):
        self.rec_name = rec_name
        self.sig_name = sig_name
        self.lbl_char = lbl_char
        self.ann_idx = ann_idx
        self.prev_ann_idx = prev_ann_idx
        self.next_ann_idx = next_ann_idx
        self.pre_samples = pre_samples
        self.post_samples = post_samples
        self.length = length
        self.beat_idx = beat_idx


# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate an image/signal dataset to be used with Keras for CNN Training!')
    parser.add_argument('-s', '--gen_signals', action='store_true', default=False, help='Generate signal dataset.')
    parser.add_argument('-i', '--gen_images', action='store_true', default=False, help='Generate image dataset.')
    parser.add_argument('-t', '--gen_test', action='store_true', default=False, help='Generate test images/signals.')
    parser.add_argument('-p', '--gen_png', action='store_true', default=False, help='Generate png image files.')
    parser.add_argument('-b', '--is_balanced', action='store_true', default=False, help='Generate a balanced dataset.')
    parser.add_argument('-r', '--gen_records', action='store_true', default=False, help='Generate separate images/signals datasets from records.')
    parser.add_argument('--pad', action='store_true', default=False, help='Pad randomly in the front and back when oversampling.')
    parser.add_argument('--pre_ann', action='store', type=float, default=1.1, help='Number of seconds to use before annotation.')
    parser.add_argument('--post_ann', action='store', type=float, default=1.1, help='Number of seconds to use after annotation.')
    parser.add_argument('--pixels', action='store', type=int, default=128, help='Number of pixels for rows and cols of image.')
    parser.add_argument('--percent_test', action='store', type=float, default=0.2, help='Percentage of smallest class to use for test dataset.')
    parser.add_argument('--path', action='store', type=str, default=r'c:/ekgdb/', help='Local path to where PhysioNet databses are stored.')
    parser.add_argument('--db', action='store', type=str, default=None, required=True, help='PhysioNet databases to process.')
    parser.add_argument('--dataset_len', action='store', type=int, default=1000, help='Maximum length of the image/signal dataset that will fit into memory.')
    parser.add_argument('--signals', nargs='+', default=[], help='Signal to use for image/signal generation.')
    args = parser.parse_args()
    argsdict = vars(args)

    gen_png = argsdict['gen_png']
    pad_random = argsdict['pad']
    pre_ann = argsdict['pre_ann']
    post_ann = argsdict['post_ann']
    is_balanced = argsdict['is_balanced']

    sig_list = set(argsdict['signals'])
    signals_str = '_'.join(sig_list).lower()
    signals_str = '{}{}'.format('balanced_' if is_balanced else '', 'all' if signals_str == '' else signals_str)

    db_name = argsdict['db']
    path_db = '{}db/{}/'.format(argsdict['path'], db_name)
    path_ds = '{}datasets/{}/{}/'.format(argsdict['path'], db_name, signals_str)

    # Update args dict with new entries.
    argsdict['sig_list'] = sig_list
    argsdict['signals_str'] = signals_str
    argsdict['path_db'] = path_db
    argsdict['path_ds'] = path_ds

    if gen_png:
        if gen_test:
            path = path_ds + 'test/'
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = path_ds + 'train/'
            if not os.path.exists(path):
                os.makedirs(path)

    # if os.path.exists(path_ds):
    #     shutil.rmtree(path_ds)

    # Get the file listing of the database files (.dat)
    if not os.path.exists(path_db):
        print('Database path does not exist: {}'.format(path_db))
        exit()

    files = []
    for f in os.listdir(path_db):
        if f.endswith('.dat'):
            files.append(f.split('.')[0])

    sig_map, lbl_map, max_samples, beats = get_db_info(path_db, files, sig_list, pre_ann, post_ann, pad_random)

    if len(sig_map) == 0 or len(lbl_map) == 0:
        print('No signals found!')
        exit()

    db_info_to_csv(path_ds, db_name, signals_str, sig_map, lbl_map, max_samples, beats)

    del sig_map

    # Convert to numpy array for easy manipulation.
    beats = np.array(beats)

    if argsdict['dataset_len'] == 0:
        argsdict['dataset_len'] = len(beats)

    if is_balanced:
        gen_balanced_dataset(lbl_map, beats, max_samples, argsdict)
    else:
        # TODO: Complete this for unbalanced datasets i.e. MLII Test
        gen_dataset(lbl_map, beats, max_samples, argsdict)


# ----------------------------------------------------------------------------------------------------------------------
def gen_dataset(lbl_map, beats, max_samples, argsdict):
    gen_signals = argsdict['gen_signals']
    gen_images = argsdict['gen_images']
    gen_png = argsdict['gen_png']
    pad_random = argsdict['pad']
    img_pixels = argsdict['pixels']
    dataset_len = argsdict['dataset_len']
    signals_str = argsdict['signals_str']
    db_name = argsdict['db']
    path_ds = argsdict['path_ds']

    dataset_indices = {}
    ratio = dataset_len / len(beats)
    for lbl_char in lbl_map:
        if ratio < 1.0:
            # We preserve class ratio.
            nb_samples = int(round(ratio * len(lbl_map[lbl_char]['beats'])))
            samples = random.sample(lbl_map[lbl_char]['beats'], nb_samples)
            dataset_indices[lbl_char] = samples
            # Remove samples from pool.
            lbl_map[lbl_char]['beats'] = lbl_map[lbl_char]['beats'].difference(samples)
        else:
            dataset_indices[lbl_char] = lbl_map[lbl_char]['beats']
            # Remove samples from pool.
            lbl_map[lbl_char]['beats'] = set()

    # Convert indices to beats.
    dataset_beats = deque()
    delete_indices = set()
    for lbl_char in dataset_indices:
        dataset_beats.extend(beats[list(dataset_indices[lbl_char])])
        # Remove samples from pool.
        delete_indices.update(dataset_indices[lbl_char])

    # Finally delete beats not in pad_list
    # beats = np.delete(beats, list(delete_indices))

    # Randomize beats so they are in no particular order
    dataset_beats = np.array(dataset_beats)
    np.random.shuffle(dataset_beats)

    # Deal with padding.
    dataset_beats = randomize_beats(dataset_beats, pad_random, [])

    # To save memory we do signals first then images.
    if gen_signals:
        # Get max signal length.
        signal_len = sorted(max_samples, key=lambda b: b.length, reverse=True)[0].length  # TODO: Can probably just iterate through the beats to find the max signal length.

        sig_test, y_test = get_dataset(path_ds, dataset_beats, signal_len, gen_signals, False, False, None, False)

        print('\nCompressing signals to file...')
        np.savez_compressed('{}signals_{}_{}_test'.format(path_ds, db_name, signals_str), x_test=sig_test, y_test=y_test)

        # Free Memory
        del y_test
        del sig_test

    if gen_images:
        x_test, y_test = get_dataset(path_ds + 'test/', dataset_beats, None, False, gen_images, gen_png, img_pixels, False)

        print('\nCompressing datasets...')
        np.savez_compressed('{}images_{}_{}_test'.format(path_ds, db_name, signals_str), x_test=x_test, y_test=y_test)


# ----------------------------------------------------------------------------------------------------------------------
def gen_balanced_dataset(lbl_map, beats, max_samples, argsdict):
    gen_signals = argsdict['gen_signals']
    gen_images = argsdict['gen_images']
    gen_test = argsdict['gen_test']
    gen_png = argsdict['gen_png']
    pad_random = argsdict['pad']
    img_pixels = argsdict['pixels']
    percent_test = argsdict['percent_test']
    dataset_len = argsdict['dataset_len']
    signals_str = argsdict['signals_str']
    db_name = argsdict['db']
    path_ds = argsdict['path_ds']

    nb_beats = [0] * len(lbl_map)
    for idx, lbl_char in enumerate(lbl_map):
        nb_beats[idx] = len(lbl_map[lbl_char]['beats'])

    nb_beats.sort()

    max_rows = nb_beats[len(nb_beats) - 1]

    # Get a list of labels that are less than the number of desired images/signals.
    pad_list = set()
    if pad_random:
        for lbl_char in lbl_map:
            if max_rows > len(lbl_map[lbl_char]['beats']):
                pad_list.add(lbl_char)

    # Get max signal length.
    signal_len = sorted(max_samples, key=lambda b: b.length, reverse=True)[0].length  # TODO: Can probably just iterate through the beats to find the max signal length.

    # Loop to break up large datasets into memory manageable chunks.
    total_beats = max_rows * len(lbl_map)
    if dataset_len > total_beats:
        dataset_len = total_beats

    nb_datasets = int(ceil(total_beats / dataset_len))
    nb_rows = int(round(dataset_len / len(lbl_map)))

    for i in trange(nb_datasets, desc='Generating datasets'):
        if total_beats > dataset_len:
            nb_rows = min(total_beats - ((i + 1) * nb_rows), nb_rows)

        nb_test_rows = 0
        test_beats = []
        if gen_test:
            # Generate test dataset beats.
            nb_test_rows = int(nb_beats[0] * percent_test)

            lbl_map, test_beats, test_indices = get_balanced_beats(lbl_map, beats, nb_test_rows, pad_list)
            test_beats = randomize_beats(test_beats, pad_random, pad_list)

        # Generate train dataset beats.
        nb_train_rows = nb_rows - nb_test_rows
        lbl_map, train_beats, train_indices = get_balanced_beats(lbl_map, beats, nb_train_rows, pad_list)
        train_beats = randomize_beats(train_beats, pad_random, pad_list)

        # To save memory we do signals first then images.
        if gen_signals:
            if gen_test:
                sig_test, y_test = get_dataset(path_ds, test_beats, signal_len, gen_signals, False, False, None, True)

            sig_train, y_train = get_dataset(path_ds, test_beats, signal_len, gen_signals, False, False, None, False)

            # print('Compressing signals to file...')
            if gen_test:
                np.savez_compressed('{}signals_{}_{}_train_test_{}'.format(path_ds, db_name, signals_str, i), x_train=sig_train, y_train=y_train, x_test=sig_test, y_test=y_test)

                # Free memory
                del y_test
                del sig_test

            np.savez_compressed('{}signals_{}_{}_train_{}'.format(path_ds, db_name, signals_str, i), x_train=sig_train, y_train=y_train)

            # Free Memory
            del y_train
            del sig_train

        if gen_images:
            if gen_test:
                x_test, y_test = get_dataset(path_ds + 'test/', test_beats, None, False, gen_images, gen_png, img_pixels, True)

            x_train, y_train = get_dataset(path_ds + 'train/', train_beats, None, False, gen_images, gen_png, img_pixels, False)

            # print('\nCompressing datasets...')
            if gen_test:
                np.savez_compressed('{}images_{}_{}_train_test_{}'.format(path_ds, db_name, signals_str, i), x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

            else:
                np.savez_compressed('{}images_{}_{}_train_{}'.format(path_ds, db_name, signals_str, i), x_train=x_train, y_train=y_train)


# ----------------------------------------------------------------------------------------------------------------------
def randomize_beats(beats, pad_random, pad_list):
    if pad_random:
        for idx, beat in enumerate(tqdm(beats, mininterval=2.0, desc='Randomizing beats')):
            pad_beat = copy.deepcopy(beat)
            sig_len = len(pad_beat.signal)
            pad_len = pad_beat.pad

            if pad_beat.lbl_char in pad_list:  # Create a random shift in signal
                shift = random.randint(0, pad_len)
                start = pad_len - shift
                end = sig_len - shift

                # start = pad_len - random.randint(0, pad_len)
                # end = (sig_len - pad_len) + random.randint(0, pad_len)

            else:  # Strip pad from class that doesn't need it.
                start = pad_len
                end = sig_len - pad_len

            pad_beat.signal = pad_beat.signal[start:end]

    return beats


# ----------------------------------------------------------------------------------------------------------------------
def get_balanced_beats(lbl_map, beats, nb_rows, pad_list):
    # Get the dataset beats
    balanced_indices = {}
    for lbl_char in tqdm(lbl_map, desc='Creating balanced dataset'):  # Get balanced data for all classes.
        balanced_indices[lbl_char] = deque(maxlen=nb_rows)

        while len(balanced_indices[lbl_char]) < nb_rows:
            nb_sample = min(len(lbl_map[lbl_char]['beats']), nb_rows)
            samples = random.sample(lbl_map[lbl_char]['beats'], nb_sample)
            balanced_indices[lbl_char].extend(samples)

            # Remove samples from pool.
            if lbl_char not in pad_list:
                lbl_map[lbl_char]['beats'] = lbl_map[lbl_char]['beats'].difference(samples)

    # Convert indices to beats and remove used beats.
    dataset_beats = deque()
    dataset_indices = set()  # TODO: What do we do with the dataset_indices?
    for lbl_char in balanced_indices:
        dataset_beats.extend(beats[balanced_indices[lbl_char]])

        # Samples to remove from pool.
        dataset_indices.update(balanced_indices[lbl_char])

        # Remove samples from pool.
        #if lbl_char in pad_list:
            #lbl_map[lbl_char]['beats'] = lbl_map[lbl_char]['beats'].difference(balanced_indices[lbl_char])

    # Randomize beats so they are in no particular order
    dataset_beats = np.array(dataset_beats)
    np.random.shuffle(dataset_beats)

    return lbl_map, dataset_beats, dataset_indices


# ----------------------------------------------------------------------------------------------------------------------
# TODO: Split this function into two seperate funstion one for image and one for signals.
def get_dataset(path_ds, beats, signal_len, gen_signals, gen_images, gen_png, img_pixels, is_test):
    """
    Will generate either a signal (priority) or image dataset from given beats.
    :param path_ds:
    :param beats:
    :param signal_len:
    :param gen_signals:
    :param gen_images:
    :param gen_png:
    :param img_pixels:
    :param is_test:
    :return:
    """

    if gen_signals:
        # Preallocate memory
        x = np.empty((len(beats), signal_len), dtype=np.float32)
        y = np.empty((len(beats), 4), dtype=np.float32)  # TODO: Fix this magic number

    elif gen_images:
        # csv = open(path_ds + 'db_images.csv', 'wb')
        # csv.write('ImageNumber, Record, SignalName, Label, Onehot\n')

        # Preallocate memory
        x = np.empty((len(beats), int(img_pixels ** 2)), dtype=np.float32)
        y = np.empty((len(beats), 4), dtype=np.float32)  # TODO: Fix this magic number
        buf = io.BytesIO()  # Memory buffer so that image doesn't have to save to disk.

    for idx, beat in enumerate(tqdm(beats, mininterval=2.0, desc='Generating images/signals')):
        if gen_signals:
            x[idx] = pad_signal(beat.signal, signal_len)

        elif gen_images:
            # Convert to image row
            im_row = signal_to_image(path_ds, beat, buf, img_pixels, gen_png, is_test)

            # Assign the images to dataset
            x[idx] = im_row / 255

            # Print to CSV file
            # csv.write('{},{},{},{},[{}{}{}{}]\n'.format(beat.ann_idx, beat.rec_name, beat.sig_name, beat.lbl_char, beat.lbl_onehot[0], beat.lbl_onehot[1], beat.lbl_onehot[2], beat.lbl_onehot[3]))

        # Labels are the same for both images and signals
        y[idx] = beat.lbl_onehot

    if not gen_signals and gen_images:
        buf.close()
        # csv.close()

    return x, y


#  ----------------------------------------------------------------------------------------------------------------------
def signal_to_image(path_ds, beat, image_buffer, img_pixels, gen_png, is_test):
    image_buffer.seek(0)

    # Create image from matplotlib
    fig = plt.figure(figsize=(1, 1), dpi=img_pixels, frameon=False)
    # fig.set_size_inches(1, 1)
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.axis('off')
    ax.set_xlim(0, len(beat.signal))
    # ax.set_ylim(-1.0, 1.5) # TODO: Should we set the y-axis limits?
    ax.plot(beat.signal, color='k', linewidth=.01)

    # Saving image to buffer in memory
    plt.savefig(image_buffer, dpi=img_pixels)

    # Change image to to black and white with PIL
    im = Image.open(image_buffer)
    im = im.convert('L')
    # im = im.point(lambda x: 0 if x < 254 else 255)  # , '1')  #TODO: Investigate if we really need this?
    # im.show()

    # Convert image to 1d vector
    im_arr = np.asarray(im, dtype=np.float32)
    im_row = im_arr.ravel()

    if gen_png:
        if is_test:
            im.save('{}/{}_{}_{}_{}{}'.format(path_ds, beat.ann_idx, beat.rec_name, beat.sig_name, beat.lbl_char, '.png'))
        else:
            im.save('{}/{}_{}_{}_{}{}'.format(path_ds, beat.ann_idx, beat.rec_name, beat.sig_name, beat.lbl_char, '.png'))

    # Free memory
    plt.close(fig)
    im.close()

    return im_row


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
def get_db_info(path_db, rec_list, sig_list, pre_ann, post_ann, pad_random):
    """
    Gets stats on each record. i.e. Beat count.
    :param path_db:
    :param rec_list:
    :param sig_list:
    :param pre_ann:
    :param post_ann:
    :param pad_random:
    :return:
    """
    beats = deque()
    record_map = {}
    sig_map = {}
    lbl_map = {}
    max_sample = 600
    max_samples = deque()
    max_beat = MaxSample()

    for rec_name in tqdm(rec_list, desc='Gathering database info'):
        rec = pywfdb.Record(path_db + rec_name)
        annotations = rec.annotation().read()

        if len(sig_list) != 0 and sig_list.isdisjoint(set(x.strip() for x in rec.signal_names)):
            continue

        for ann_idx, ann in enumerate(annotations):
            if not (1 <= ann.type <= 11 or ann.type == 34):
                continue

            prev_ann_index = ann_idx - 1
            while prev_ann_index >= 0 and not annotations[prev_ann_index].isqrs:
                prev_ann_index -= 1

            next_ann_index = ann_idx + 1
            while next_ann_index < len(annotations) and not annotations[next_ann_index].isqrs:
                next_ann_index += 1

            if prev_ann_index == -1:
                # Then we are on the the first annotation in the record
                prev_r_r = 0
            else:
                prev_ann = annotations[prev_ann_index]
                prev_r_r = ann.time - prev_ann.time

            if next_ann_index == len(annotations):
                # Then we are on the the last annotation in the record
                next_r_r = 0
            else:
                next_ann = annotations[next_ann_index]
                next_r_r = next_ann.time - ann.time

            pre_samples = int(round(pre_ann * prev_r_r))  # Number samples to capture before annotation
            post_samples = int(round(post_ann * next_r_r))  # Number samples to capture after annotation

            # if pre_samples > max_sample:
            #     pre_samples = max_sample
            #
            # if post_samples > max_sample:
            #     post_samples = max_sample

            lbl_char = get_annotation_char(ann)

            # Index where annotation occurs
            ann_time = ann.time

            # Calculate start index of image
            start = ann_time - pre_samples
            if prev_r_r == 0:
                start = 0

            end = 0

            for sig_idx, sig_name in enumerate(rec.signal_names):
                if len(sig_list) != 0 and sig_name.strip() not in sig_list:
                    continue

                if rec_name not in record_map:
                    # Read in the entire patients EKG
                    rec_len = len(rec.read(sig_name))
                    record_map[rec_name] = {sig_name: rec_len}
                elif sig_name not in record_map[rec_name]:
                    # Read in the entire patients EKG
                    rec_len = len(rec.read(sig_name))
                    record_map[rec_name][sig_name] = rec_len
                else:
                    rec_len = record_map[rec_name][sig_name]

                if end == 0:  # Calculate end index of image
                    end = ann_time + post_samples

                # Pad signal with %10 the length in front and back for random shifting later.
                pad = 0
                pad_start = start
                pad_end = end
                if pad_random:
                    pad = int(round((end - start) * 0.1))
                    pad_start -= pad
                    pad_end += pad

                pad_start = max(0, pad_start)
                pad_end = min(pad_end, rec_len)

                # Read in the signal from the record.
                signal = rec.read(sig_name, pad_start, pad_end - pad_start)

                beat = Beat(rec_name, sig_name, ann_idx, start, end, lbl_char, get_annotation_onehot(ann), signal, pad)
                beats.append(beat)
                beat_idx = len(beats) - 1

                if max_beat.length < end - start:
                    max_beat = MaxSample(rec_name, sig_name, lbl_char, ann_idx, prev_ann_index, next_ann_index, pre_samples, post_samples, end - start, beat_idx)

                if pre_samples >= max_sample or post_samples >= max_sample:
                    max_samples.append(MaxSample(rec_name, sig_name, lbl_char, ann_idx, prev_ann_index, next_ann_index, pre_samples, post_samples, end - start, beat_idx))

                if lbl_char not in lbl_map:
                    lbl_map[lbl_char] = {'beats': set(), 'records': {}, 'signals': {}}

                if rec_name not in lbl_map[lbl_char]['records']:
                    lbl_map[lbl_char]['records'][rec_name] = {}

                if sig_name not in lbl_map[lbl_char]['records'][rec_name]:
                    #lbl_map[lbl_char]['records'][rec_name][sig_name] = set()
                    lbl_map[lbl_char]['records'][rec_name][sig_name] = 0

                if sig_name not in lbl_map[lbl_char]['signals']:
                    # lbl_map[lbl_char]['signals'][sig_name] = set()
                    lbl_map[lbl_char]['signals'][sig_name] = 0

                # lbl_map[lbl_char]['records'][rec_name][sig_name].add(beat_idx)
                # lbl_map[lbl_char]['signals'][sig_name].add(beat_idx)
                lbl_map[lbl_char]['records'][rec_name][sig_name] += 1
                lbl_map[lbl_char]['signals'][sig_name] += 1
                lbl_map[lbl_char]['beats'].add(beat_idx)

                if sig_name not in sig_map:
                    sig_map[sig_name] = {'records': set(), 'labels': {lbl_char: 1}}

                if rec_name not in sig_map[sig_name]['records']:
                    sig_map[sig_name]['records'].add(rec_name)

                if lbl_char not in sig_map[sig_name]['labels']:
                    sig_map[sig_name]['labels'][lbl_char] = 1
                else:
                    sig_map[sig_name]['labels'][lbl_char] += 1

        rec.close()

    max_samples.append(max_beat)

    return sig_map, lbl_map, max_samples, beats


# ----------------------------------------------------------------------------------------------------------------------
def db_info_to_csv(path_ds, db_name, sig_str, sig_map, lbl_map, max_samples, beats):
    """

    :param path_ds:
    :param db_name:
    :param sig_str:
    :param sig_map:
    :param lbl_map:
    :param max_samples:
    :param beats:
    :return:
    """

    print('Writing beat info to CSV file...')

    # Create directory to store dataset.
    if not os.path.exists(path_ds):
        os.makedirs(path_ds)

    csv = open('{}{}_{}_db_breakdown.csv'.format(path_ds, db_name, sig_str), 'wb')
    csv.write('Total Beats\n')
    csv.write(',{}\n'.format(len(beats)))

    csv.write('Signals\n')
    for sig in sig_map:
        total = 0
        csv.write(',{}\n'.format(sig))
        for lbl in sig_map[sig]['labels']:
            csv.write(',,{},{}\n'.format(lbl, sig_map[sig]['labels'][lbl]))
            total += sig_map[sig]['labels'][lbl]

        csv.write(',{},,{}\n\n'.format('Total', total))

    csv.write('Classes\n')
    for lbl in lbl_map:
        total = 0
        csv.write(',{}\n'.format(lbl))
        for sig in lbl_map[lbl]['signals']:
            # csv.write(',,{},{}\n'.format(sig, len
            # total += len(lbl_map[lbl]['signals'][sig])
            csv.write(',,{},{}\n'.format(sig, lbl_map[lbl]['signals'][sig]))
            total += lbl_map[lbl]['signals'][sig]

        csv.write(',{},,{}\n\n'.format('Total', total))

    csv.write('Records\n')
    for sig in sig_map:
        csv.write(',{}\n'.format(sig))
        for rec in sig_map[sig]['records']:
            csv.write(',,{}\n'.format(rec))

    csv.write('\n')

    for lbl in lbl_map:
        csv.write(',{}\n'.format(lbl))
        for rec in lbl_map[lbl]['records']:
            csv.write(',,{}\n'.format(rec))

    csv.write('Max Sample\n')
    csv.write(',rec_name,sig_name,lbl_char,ann_idx,prev_ann_idx,next_ann_idx,pre_samples,post_samples,length,beat_idx\n')
    for entry in max_samples:
        csv.write(',{},{},{},{},{},{},{},{},{},{}\n'.format(entry.rec_name, entry.sig_name, entry.lbl_char, entry.ann_idx, entry.prev_ann_idx, entry.next_ann_idx, entry.pre_samples, entry.post_samples, entry.length, entry.beat_idx))

    csv.close()


# ----------------------------------------------------------------------------------------------------------------------
# Get annotation (label) as a char
def get_annotation_char(ann):
    if ann.type == 1 or ann.type == 2 or ann.type == 3 or ann.type == 34 or ann.type == 11:
        return 'N'
    elif ann.type == 8 or ann.type == 4 or ann.type == 7 or ann.type == 9:
        return 'S'
    elif ann.type == 5 or ann.type == 10:
        return 'V'
    elif ann.type == 6:
        return 'F'
    else:
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Get annotation (label) as one-hot
def get_annotation_onehot(ann):
    if ann.type == 1 or ann.type == 2 or ann.type == 3 or ann.type == 34 or ann.type == 11:
        return [1., 0., 0., 0.]
    elif ann.type == 8 or ann.type == 4 or ann.type == 7 or ann.type == 9:
        return [0., 1., 0., 0.]
    elif ann.type == 5 or ann.type == 10:
        return [0., 0., 1., 0.]
    elif ann.type == 6:
        return [0., 0., 0., 1.]
    else:
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Inserts padding into signal so that it is uniform length
def pad_signal(sig, sig_len):
    front_pad = 0
    back_pad = 0

    if len(sig) < sig_len:
        pad = sig_len - len(sig)
        front_pad = int(floor(pad / 2.0))
        back_pad = int(ceil(pad / 2.0))

    return np.concatenate(np.zeros(front_pad, dtype=np.float32), np.array(sig, dtype=np.float32), np.zeros(back_pad, dtype=np.float32))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------------------------------------------------
def get_balanced_indices(lbl_map, nb_rows, percent_test, is_balanced):

    if is_balanced:
        balanced_indices = {}
        for lbl_char in tqdm(lbl_map, desc='Creating balanced datasets'):  # Get balanced data for all classes.
            balanced_indices[lbl_char] = deque(maxlen=nb_rows)
            if len(lbl_map[lbl_char]['beats']) < nb_rows:  # We must not delete from pool until oversampling is finished.
                percent_sample = 1.0
                oversampled = deque()
                while len(balanced_indices[lbl_char]) < nb_rows:  # Ensure we fill the class with all samples.
                    if len(lbl_map[lbl_char]['beats']) > nb_rows - len(balanced_indices[lbl_char]):
                        percent_sample = 1.0 - (len(balanced_indices[lbl_char]) / float(nb_rows))

                    for rec_name in lbl_map[lbl_char]['records']:  # We want an even sample across all records.
                        for sig_name in lbl_map[lbl_char]['records'][rec_name]:  # We want an even sample across all signals.

                            samples = random.sample(lbl_map[lbl_char]['records'][rec_name][sig_name], int(ceil(len(lbl_map[lbl_char]['records'][rec_name][sig_name]) * percent_sample)))
                            balanced_indices[lbl_char].extend(samples)
                            oversampled.extend(samples)

                # It's ok to remove used samples from pool after finishing oversampling.
                oversampled = set(oversampled)
                for rec_name in lbl_map[lbl_char]['records']:  # We want an even sample across all records.
                    for sig_name in lbl_map[lbl_char]['records'][rec_name]:  # We want an even sample across all signals.
                        lbl_map[lbl_char]['records'][rec_name][sig_name] = lbl_map[lbl_char]['records'][rec_name][sig_name].difference(oversampled)
                        lbl_map[lbl_char]['signals'][sig_name] = lbl_map[lbl_char]['signals'][sig_name].difference(oversampled)

                lbl_map[lbl_char]['beats'] = lbl_map[lbl_char]['beats'].difference(oversampled)

            else:  # Don't have to do any oversampling.
                percent_sample = float(nb_rows) / len(lbl_map[lbl_char]['beats'])
                if percent_test < 1:  # Generating a test set
                    percent_sample *= percent_test

                for rec_name in lbl_map[lbl_char]['records']:  # We want an even sample across all records.
                    for sig_name in lbl_map[lbl_char]['records'][rec_name]:  # We want an even sample across all signals.
                        samples = random.sample(lbl_map[lbl_char]['records'][rec_name][sig_name], int(ceil(len(lbl_map[lbl_char]['records'][rec_name][sig_name]) * percent_test)))

                        balanced_indices[lbl_char].extend(samples)

                        # We must remove samples from pool so they are not reused.
                        lbl_map[lbl_char]['records'][rec_name][sig_name] = lbl_map[lbl_char]['records'][rec_name][sig_name].difference(samples)
                        lbl_map[lbl_char]['signals'][sig_name] = lbl_map[lbl_char]['signals'][sig_name].difference(samples)
                        lbl_map[lbl_char]['beats'] = lbl_map[lbl_char]['beats'].difference(samples)

    return lbl_map, balanced_indices