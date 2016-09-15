import io
import os
import random
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywfdb
from math import ceil, floor
from PIL import Image
from tqdm import *

gen_signals = True
gen_images = True
gen_test = False
pre_ann = 1.1  # Time to capture before annotation in seconds
post_ann = 1.1  # Time to capture after annotation in seconds
img_pixels = 256
nb_classes = 4
nb_rows = 30000


# ----------------------------------------------------------------------------------------------------------------------
def main():

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

    print('Compiling beat info into databases...\n\n')
    num_beats, max_sample, sig_per_rec, sig_map, class_map, beat_map = get_db_info(path_db, files)

    print(' Writing beat info to CSV file...\n\n')
    db_info_to_csv(path_images, num_beats, max_sample, sig_map, class_map)

    print('Creating balanced datasets...\n\n')
    x_train, y_train, sig_train, x_test, y_test, sig_test = get_balanced_dataset(path_db, path_images, beat_map, max_sample, img_pixels, gen_test)

    print('Compressing datasets...\n\n')
    if gen_test:
        if gen_images:
            np.savez_compressed('{}images_{}'.format(path_images, db_name), x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

        if gen_signals:
            np.savez_compressed('{}signals_{}'.format(path_images, db_name), x_train=sig_train, y_train=sig_train, x_test=sig_test, y_test=sig_test)
    else:
        if gen_images:
            np.savez_compressed('{}images_{}'.format(path_images, db_name), x_train=x_train, y_train=y_train)

        if gen_signals:
            np.savez_compressed('{}signals_{}'.format(path_images, db_name), x_train=sig_train, y_train=sig_train)


# ----------------------------------------------------------------------------------------------------------------------
def db_info_to_csv(path_images, num_beats, max_sample, sig_map, class_map):
    """
    Writes database info to a CSV file.
    :param path_images:
    :param num_beats:
    :param max_sample:
    :param sig_map:
    :param class_map:
    :return:
    """
    csv = open(path_images + 'db_breakdown.csv', 'wb')
    csv.write('Total Beats\n')
    csv.write(',{}\n'.format(num_beats))

    csv.write('Max Sample\n')
    csv.write(',Index,{}\n'.format(max_sample['idx']))
    csv.write(',Length,{}\n'.format(max_sample['len']))
    csv.write(',Record,{}\n'.format(max_sample['rec']))
    csv.write(',Label,{}\n'.format(max_sample['lbl']))

    csv.write('Signals\n')
    for sig in sig_map:
        csv.write(',{}\n'.format(sig))
        for lbl in sig_map[sig]:
            csv.write(',,{},{}\n'.format(lbl, len(sig_map[sig][lbl])))

    csv.write('Classes\n')
    for lbl in class_map:
        csv.write(',{}\n'.format(lbl))
        for sig in class_map[lbl]:
            csv.write(',,{},{}\n'.format(sig, len(class_map[lbl][sig])))

    csv.close()


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
def get_balanced_dataset(path_db, path_images, beat_map, max_sample, img_pixels, gen_test=False):
    """

    :param path_db:
    :param beat_map:
    :param max_sample:
    :param img_pixels:
    :param gen_test:
    :return:
    """
    sample_len = [len(beat_map['N']), len(beat_map['S']), len(beat_map['V']), len(beat_map['F'])]
    sample_len.sort()

    nb_labels = sample_len[2]  #TODO: Might have to adjust this if smallest class is very small so oversampling is more realistic.

    random_beats = {}
    for key in beat_map:
        if len(beat_map[key]) > nb_labels:
            random_beats[key] = set(random.sample(list(beat_map[key]), nb_labels))  #TODO: Change this to garantee beats from each patient.
        else:
            random_beats[key] = beat_map[key]

    nb_test_labels = 0
    x_test = None
    y_test = None
    sig_test = None
    if gen_test:
        # Generate Test Dataset
        nb_test_labels = int(sample_len[0] * 0.3)  # Reserve 30% TODO: Probably should move magic number to global so can be easily adjustable.
        x_test, y_test, sig_test, random_beats = get_balanced_dataset_helper(path_db, path_images, random_beats, img_pixels, nb_test_labels, max_sample)

    # Generate Training Dataset
    nb_train_labels = nb_labels - nb_test_labels
    x_train, y_train, sig_train, random_beats = get_balanced_dataset_helper(path_db, path_images, random_beats, img_pixels, nb_train_labels, max_sample)

    return x_train, y_train, sig_train, x_test, y_test, sig_test, beat_map


# ----------------------------------------------------------------------------------------------------------------------
def get_balanced_dataset_helper(path_db, path_images, beat_map, img_pixels, nb_labels, max_sample):
    """

    :param path_db:
    :param path_images:
    :param beat_map:
    :param img_pixels:
    :param nb_labels:
    :param max_sample:
    :return:
    """
    buf = io.BytesIO()  # Memory buffer so that image doesn't have to save to disk.

    csv = open(path_images + 'db_images.csv', 'wb')
    csv.write('ImageNumber, Record, SignalName, Label, Onehot\n')

    # Preallocate memory
    beats = [None] * (nb_labels * len(beat_map))
    x = np.empty((nb_labels * len(beat_map), img_pixels ** 2), dtype=np.float32)
    y = np.empty((nb_labels * len(beat_map), len(beat_map)), dtype=np.float32)
    signals = np.empty((nb_labels * len(beat_map), max_sample['len']), dtype=np.float32)

    for cnt in xrange(nb_labels):
        for idx, key in enumerate(beat_map): # TODO: This may need to be randomized.
            beats[cnt * 4 + idx] = beat_map[key].pop()

    beats.sort(key=lambda b: b.rec_name)

    record = ''
    for idx, beat in tqdm(beats):
        if beat.rec_name != record:
            record = beat.rec_name
            path_record = path_db + record
            rec = pywfdb.Record(path_record)

        # Read in the signal from the record.
        signal = rec.read(beat.sig_name, beat.start, beat.stop)

        # Convert to image
        im, im_row = signal_to_image(buf, img_pixels, signal)

        # Assign the signal to dataset
        if gen_images:
            x[idx] = im_row / 255
            im.save('{}{}_{}_{}_{}{}'.format(path_images, idx, beat.rec_name, beat.sig_name, beat.lbl_char, '.png'))

        if gen_signals:
            signals[idx] = pad_signal(y, max_sample['len'])

        y[idx] = beat.lbl_onehot

        # Print to CSV file
        csv.write('{},{},{},{},[{}{}{}{}]\n'.format(idx, beat.rec_name, beat.sig_name, beat.lbl_char, beat.lbl_onehot[0], beat.lbl_onehot[1], beat.lbl_onehot[2], beat.lbl_onehot[3]))
        im.close()

    x, y = random.shuffle(x, y, random_state=rng_seed)

    buf.close()

    return x, y, signals, beat_map


# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
def get_db_info(path_db, rec_list):
    """
    Gets stats on each record. i.e. Beat count.
    :param rec_list:
    :return:
    """
    beat_cnt = 0
    num_signals = 0
    sig_map = {}
    class_map = {}
    record_map = {}
    beat_map = {}
    max_sample = {'idx': 0, 'len': 0, 'rec': '', 'lbl': ''}

    for rec_name in tqdm(rec_list):
        rec = pywfdb.Record(path_db + rec_name)
        annotations = rec.annotation().read()

        if num_signals < len(rec.signal_names):
            num_signals = len(rec.signal_names)

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

            total_samples = pre_samples + post_samples

            if total_samples > max_sample['len']:
                max_sample['idx'] = ann_idx
                max_sample['len'] = total_samples
                max_sample['rec'] = rec_name
                max_sample['lbl'] = get_annotation_char(ann)

            lbl = get_annotation_char(ann)

            # Index where annotation occurs
            ann_time = ann.time

            # Calculate start index of image
            start = 0
            if pre_samples != 0 and ann_time - pre_samples >= 0:
                start = ann_time - pre_samples

            for sig_idx, sig_name in enumerate(rec.signal_names):

                beat_cnt += 1

                if rec_name not in record_map:
                    # Read in the entire patients EKG
                    signal_len = len(rec.read(sig_name))
                    record_map[rec_name] = {sig_name: signal_len}
                elif sig_name not in record_map[rec_name]:
                    # Read in the entire patients EKG
                    signal_len = len(rec.read(sig_name))
                    record_map[rec_name][sig_name] = signal_len
                else:
                    signal_len = record_map[rec_name][sig_name]

                if sig_idx < 1:
                    # Calculate end index of image
                    end = signal_len - 1
                    if post_samples != 0 and ann_time + post_samples < signal_len:
                        end = ann_time + post_samples

                beat = Beat(rec_name, sig_name, ann_idx, start, end, lbl, get_annotation_onehot(ann))

                if lbl not in beat_map:
                    beat_map[lbl] = set()
                    beat_map[lbl].add(beat)
                else:
                    beat_map[lbl].add(beat)

                if sig_name not in sig_map:
                    sig_map[sig_name] = {lbl: set()}
                    sig_map[sig_name][lbl].add(beat)
                elif lbl not in sig_map[sig_name]:
                    sig_map[sig_name][lbl] = set()
                    sig_map[sig_name][lbl].add(beat)
                else:
                    sig_map[sig_name][lbl].add(beat)

                if lbl not in class_map:
                    class_map[lbl] = {sig_name: set()}
                    class_map[lbl][sig_name].add(beat)
                elif sig_name not in class_map[lbl]:
                    class_map[lbl][sig_name] = set()
                    class_map[lbl][sig_name].add(beat)
                else:
                    class_map[lbl][sig_name].add(beat)

    return beat_cnt, max_sample, num_signals, sig_map, class_map, beat_map


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

