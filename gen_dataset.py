import io
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywfdb
from collections import deque
from math import ceil, floor
from PIL import Image
from tqdm import *
from sklearn.utils import shuffle

rng_seed = 1337
random.seed(rng_seed)
np.random.seed(seed=rng_seed)
# ----------------------------------------------------------------------------------------------------------------------
def main():
    gen_signals = False
    gen_images = False
    gen_test = False
    pre_ann = 1.1  # Time to capture before annotation in seconds
    post_ann = 1.1  # Time to capture after annotation in seconds
    img_pixels = 256
    percent_test = 0.3

    db_name = sys.argv[1]
    path_db = r'C:/mitdb/db/{}/'.format(db_name)
    path_images = r'C:/mitdb/images/{}/'.format(db_name)

    # Create directory to store images
    if not os.path.exists(path_images):
        os.makedirs(path_images)

    # Get the file listing of the database files (.dat)
    files = []
    for f in os.listdir(path_db):
        if f.endswith(".dat"):
            files.append(f.split('.')[0])

    sig_map, lbl_map, max_samples, beats = get_db_info(path_db, files, pre_ann, post_ann)

    db_info_to_csv(path_images, sig_map, lbl_map, max_samples, beats)

    x_train, y_train, sig_train, x_test, y_test, sig_test = get_dataset(path_db, path_images, lbl_map, beats, max_samples, percent_test, img_pixels, gen_images, gen_signals, gen_test)

    print('Compressing datasets...')
    if gen_test:
        if gen_images:
            np.savez_compressed('{}images_{}_train_test'.format(path_images, db_name), x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

        if gen_signals:
            np.savez_compressed('{}signals_{}_train_test'.format(path_images, db_name), x_train=sig_train, y_train=y_train, x_test=sig_test, y_test=y_test)
    else:
        if gen_images:
            np.savez_compressed('{}images_{}_train'.format(path_images, db_name), x_train=x_train, y_train=y_train)

        if gen_signals:
            np.savez_compressed('{}signals_{}_train'.format(path_images, db_name), x_train=sig_train, y_train=y_train)


# ----------------------------------------------------------------------------------------------------------------------
def signal_to_image(image_buffer, img_pixels, signal):
    """

    :param image_buffer:
    :param img_pixels:
    :param signal:
    :return:
    """

    # Create image from matplotlib
    fig = plt.figure(figsize=(1, 1), dpi=img_pixels, frameon=False)
    # fig.set_size_inches(1, 1)
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.axis('off')
    ax.set_xlim(0, len(signal))
    # ax.set_ylim(-1.0, 1.5) # TODO: Should we set the y-axis limits?
    ax.plot(signal, color='k', linewidth=.01)

    # Saving image to buffer in memory
    plt.savefig(image_buffer, dpi=img_pixels)

    # Change image to to black and white with PIL
    im = Image.open(image_buffer)
    im = im.convert('L')
    im = im.point(lambda x: 0 if x < 254 else 255)  # , '1')
    # im.show()

    # Convert image to 1d vector
    im_arr = np.asarray(im)  # , dtype=np.uint8)
    im_row = im_arr.ravel()

    # Free memory
    plt.close(fig)
    image_buffer.seek(0)

    return im, im_row


# ----------------------------------------------------------------------------------------------------------------------
class Beat:
    """Class that holds all info for a specific type of signal."""

    def __init__(self, rec_name='', sig_name='', ann_idx=0, start=0, end=0, lbl_char='', lbl_onehot=[0., 0., 0., 0.]):
        self.rec_name = rec_name
        self.sig_name = sig_name
        self.ann_idx = ann_idx
        self.start = start
        self.end = end
        self.lbl_char = lbl_char
        self.lbl_onehot = lbl_onehot

    def to_string(self):
        return '{},{},{},{},{},{},{}'.format(self.rec_name, self.sig_name, self.ann_idx, self.start, self.end, self.lbl_char, self.lbl_onehot)


# ----------------------------------------------------------------------------------------------------------------------
def get_dataset(path_db, path_images, lbl_map, beats, max_samples, percent_test, img_pixels, gen_images, gen_signals, gen_test):
    """

    :param path_db:
    :param path_images:
    :param lbl_map:
    :param beats:
    :param max_samples:
    :param percent_test:
    :param img_pixels:
    :param gen_images:
    :param gen_signals:
    :param gen_test:
    :return:
    """

    nb_beats = [0] * len(lbl_map);
    for idx, lbl_char in enumerate(lbl_map):
        nb_beats[idx] = len(lbl_map[lbl_char]['beats'])

    nb_beats.sort()

    nb_rows = nb_beats[2]

    # 50000 total beats will fit into 16GB of memory ok.
    if nb_rows > 50000.0 / len(lbl_map):
        nb_rows = 50000.0 / len(lbl_map)

    nb_test_rows = 0
    x_test = None
    y_test = None
    sig_test = None
    if gen_test:
        # Generate Test Dataset
        nb_test_rows = int(nb_beats[0] * percent_test)  # TODO: Probably should move magic number to global so can be easily adjustable.
        x_test, y_test, sig_test, lbl_map, beats = get_balanced_dataset(path_db, path_images, lbl_map, beats, img_pixels, nb_test_rows, percent_test, max_samples, gen_images, gen_signals)

    # Generate Training Dataset
    nb_train_rows = nb_rows - nb_test_rows
    x_train, y_train, sig_train, lbl_map, beats = get_balanced_dataset(path_db, path_images, lbl_map, beats, img_pixels, nb_train_rows, 1.0, max_samples, gen_images, gen_signals)

    return x_train, y_train, sig_train, x_test, y_test, sig_test


# ----------------------------------------------------------------------------------------------------------------------
def get_balanced_dataset(path_db, path_images, lbl_map, beats, img_pixels, nb_rows, percent_test, max_samples, gen_images, gen_signals):
    """
    Create a balanced set of beats with even distribution across all records.
    :param path_db:
    :param path_images:
    :param lbl_map:
    :param beats:
    :param img_pixels:
    :param nb_rows:
    :param percent_test:
    :param max_samples:
    :param gen_images:
    :param gen_signals:
    :return:
    """

    print('Creating balanced datasets...')
    balanced_indices = {}
    for lbl_char in lbl_map:  # Get balanced data for all classes.
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

    # Convert indices to beats and remove all used beats.
    np_beats = np.array(beats)
    balanced_beats = deque()
    for lbl in balanced_indices:
        balanced_beats.extend(np_beats[balanced_indices[lbl]])
        np.delete(np_beats, balanced_indices[lbl])

    # Sort the beats by record name.
    balanced_beats = list(balanced_beats)
    balanced_beats.sort(key=lambda b: b.rec_name)

    buf = io.BytesIO()  # Memory buffer so that image doesn't have to save to disk.

    csv = open(path_images + 'db_images.csv', 'wb')
    csv.write('ImageNumber, Record, SignalName, Label, Onehot\n')

    # Preallocate memory
    x = np.empty((nb_rows * len(lbl_map), img_pixels ** 2), dtype=np.float32)
    y = np.empty((nb_rows * len(lbl_map), len(lbl_map)), dtype=np.float32)
    signals = None

    if gen_signals:
        signals = np.empty((nb_rows * len(lbl_map), 1000), dtype=np.float32)

    print('Generating images/signals...')
    if percent_test < 1:
        # Create directory to store images
        if not os.path.exists(path_images + 'test/'):
            os.makedirs(path_images + 'test/')
    else:
        # Create directory to store images
        if not os.path.exists(path_images + 'train/'):
            os.makedirs(path_images + 'train/')

    record = ''
    for idx, beat in enumerate(tqdm(balanced_beats)):
        if beat.rec_name != record:
            record = beat.rec_name
            path_record = path_db + record
            rec = pywfdb.Record(path_record)

        # Read in the signal from the record.
        signal = rec.read(beat.sig_name, beat.start, beat.end - beat.start)

        # Convert to image
        im, im_row = signal_to_image(buf, img_pixels, signal)

        # Assign the signal to dataset
        if gen_images:
            x[idx] = im_row / 255
            if percent_test < 1:
                im.save('{}{}/{}_{}_{}_{}{}'.format(path_images, 'test', idx, beat.rec_name, beat.sig_name, beat.lbl_char, '.png'))
            else:
                im.save('{}{}/{}_{}_{}_{}{}'.format(path_images, 'train', idx, beat.rec_name, beat.sig_name, beat.lbl_char, '.png'))

        if gen_signals:
            signals[idx] = pad_signal(signal, 1000)

        y[idx] = beat.lbl_onehot

        # Print to CSV file
        csv.write('{},{},{},{},[{}{}{}{}]\n'.format(idx, beat.rec_name, beat.sig_name, beat.lbl_char, beat.lbl_onehot[0], beat.lbl_onehot[1], beat.lbl_onehot[2], beat.lbl_onehot[3]))
        im.close()

    x, y = shuffle(x, y, random_state=rng_seed)

    buf.close()

    return x, y, signals, lbl_map, beats


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
def get_db_info(path_db, rec_list, pre_ann, post_ann):
    """
    Gets stats on each record. i.e. Beat count.
    :param path_db:
    :param rec_list:
    :param pre_ann:
    :param post_ann:
    :return:
    """
    beats = deque()
    record_map = {}
    sig_map = {}
    lbl_map = {}
    max_sample = 1000
    max_samples = deque()

    print('Compiling beat info into databases...')
    for rec_name in tqdm(rec_list):
        rec = pywfdb.Record(path_db + rec_name)
        annotations = rec.annotation().read()

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
            start = 0
            if pre_samples != 0 and ann_time - pre_samples >= 0:
                start = ann_time - pre_samples

            for sig_idx, sig_name in enumerate(rec.signal_names):

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

                if sig_idx < 1:
                    # Calculate end index of image
                    end = rec_len - 1
                    if post_samples != 0 and ann_time + post_samples < rec_len:
                        end = ann_time + post_samples

                beat = Beat(rec_name, sig_name, ann_idx, start, end, lbl_char, get_annotation_onehot(ann))
                beats.append(beat)
                beat_idx  = len(beats) - 1

                if pre_samples >= max_sample or post_samples >= max_sample:
                    max_samples.append({'ann_idx': ann_idx, 'pre': pre_samples, 'post': post_samples, 'rec': rec_name, 'lbl': lbl_char, 'beat_idx': beat_idx})

                if lbl_char not in lbl_map:
                    lbl_map[lbl_char] = {'beats': set(), 'records': {}, 'signals': {}}

                if rec_name not in lbl_map[lbl_char]['records']:
                    lbl_map[lbl_char]['records'][rec_name] = {}

                if sig_name not in lbl_map[lbl_char]['records'][rec_name]:
                    lbl_map[lbl_char]['records'][rec_name][sig_name] = set()

                if sig_name not in lbl_map[lbl_char]['signals']:
                    lbl_map[lbl_char]['signals'][sig_name] = set()

                lbl_map[lbl_char]['records'][rec_name][sig_name].add(beat_idx)
                lbl_map[lbl_char]['signals'][sig_name].add(beat_idx)
                lbl_map[lbl_char]['beats'].add(beat_idx)

                if sig_name not in sig_map:
                    sig_map[sig_name] = {'beats': set(), 'records': {}, 'labels': {}}

                if rec_name not in sig_map[sig_name]['records']:
                    sig_map[sig_name]['records'][rec_name] = set()

                # if lbl_char not in sig_map[sig_name]['records'][rec_name]:
                #     lbl_map[lbl_char]['records'][rec_name][sig_name] = set()

                if lbl_char not in sig_map[sig_name]['labels']:
                    sig_map[sig_name]['labels'][lbl_char] = set()

                sig_map[sig_name]['records'][rec_name].add(beat_idx)
                sig_map[sig_name]['labels'][lbl_char].add(beat_idx)
                sig_map[sig_name]['beats'].add(beat_idx)

    return sig_map, lbl_map, max_samples, beats


# ----------------------------------------------------------------------------------------------------------------------
def db_info_to_csv(path_images, sig_map, lbl_map, max_samples, beats):
    """

    :param path_images:
    :param sig_map:
    :param lbl_map:
    :param max_samples:
    :param beats:
    :return:
    """

    print('Writing beat info to CSV file...')

    csv = open(path_images + 'db_breakdown.csv', 'wb')
    csv.write('Total Beats\n')
    csv.write(',{}\n'.format(len(beats)))

    csv.write('Signals\n')
    for sig in sig_map:
        total = 0
        csv.write(',{}\n'.format(sig))
        for lbl in sig_map[sig]['labels']:
            csv.write(',,{},{}\n'.format(lbl, len(sig_map[sig]['labels'][lbl])))
            total += len(sig_map[sig]['labels'][lbl])

        csv.write(',{},,{}\n\n'.format('Total', total))

    csv.write('Classes\n')
    for lbl in lbl_map:
        total = 0
        csv.write(',{}\n'.format(lbl))
        for sig in lbl_map[lbl]['signals']:
            csv.write(',,{},{}\n'.format(sig, len(lbl_map[lbl]['signals'][sig])))
            total += len(lbl_map[lbl]['signals'][sig])

        csv.write(',{},,{}\n\n'.format('Total', total))

    csv.write('Max Sample\n')
    csv.write(',ann_idx,pre_samples,post_samples,rec_name,lbl_char,beat_idx\n')
    for entry in max_samples:
        csv.write(',{},{},{},{},{},{}\n'.format(entry['ann_idx'], entry['pre'], entry['post'], entry['rec'], entry['lbl'], entry['beat_idx']))

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
        return 'X'


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
        return [0., 0., 0., 0.]


# ----------------------------------------------------------------------------------------------------------------------
# Inserts padding into signal so that it is uniform length
def pad_signal(sig, sig_len):
    front_pad = 0
    back_pad = 0

    if len(sig) < sig_len:
        pad = sig_len - len(sig)
        front_pad = int(floor(pad / 2.0))
        back_pad = int(ceil(pad / 2.0))

    return np.concatenate(
        (np.zeros(front_pad, dtype=np.float32), np.array(sig, dtype=np.float32), np.zeros(back_pad, dtype=np.float32)))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

