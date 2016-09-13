import io
import os
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywfdb
from math import ceil, floor
from PIL import Image

gen_signals = True
gen_images = True
pre_ann = 1.1  # Time to capture before annotation in seconds
post_ann = 1.1  # Time to capture after annotation in seconds
img_pixels = 256
nb_classes = 4
nb_rows = 30000
signal_list = []
# ----------------------------------------------------------------------------------------------------------------------
def main():
    buf = io.BytesIO()  # Memory buffer so that image doesn't have to save to disk.
    file_num = 0

    signal_list = sys.argv[2:]

    path_db = r'C:/mitdb/db/{}/'.format(sys.argv[1])
    path_images = r'C:/mitdb/images/{}/'.format(sys.argv[1])

    # Create directory to store images
    if not os.path.exists(path_images):
        os.makedirs(path_images)

    # Get the file listing of the database files (.dat)
    files = []
    for f in os.listdir(path_db):
        if f.endswith(".dat"):
            files.append(f.split('.')[0])

    num_beats, max_sample, num_signals, sig_map, class_map, record_map = get_record_info(path_db, files)

    # Write results from get_record_info
    record_info_to_csv(path_images, num_beats, max_sample, num_signals, sig_map, class_map, record_map)

    prog_bar = ProgressBar(total_beats=num_beats, num_recs=len(files))

    csv = open(path_images + 'db_images.csv', 'wb')
    csv.write('ImageNumber, Record, SignalName, Label, Onehot\n')

    img_cnt = 0
    # ------------------------------------------------------------------------------------------------------------------
    for key, value in sig_map.iteritems():

        if len(signal_list) > 0:
            if key == 'rec_list' or key not in signal_list:
                continue
        else:
            if key == 'rec_list':
                continue

        if img_cnt == 0 or img_cnt > nb_rows:

            if img_cnt > nb_rows:
                img_cnt = 0
                for sig_name in rec.signal_names:
                    if gen_images:
                        np.savez_compressed('{}images_{}_{}'.format(path_images, sig_name, img_cnt / nb_rows),
                                            images=images, labels=labels)

                    if gen_signals:
                        np.savez_compressed('{}signals_{}_{}'.format(path_images, sig_name, img_cnt / nb_rows),
                                            images=signals, labels=labels)

            # Create directory to store images
            path_signal = '{}{}{}/'.format(path_images, key, img_cnt / nb_rows)

            if not os.path.exists(path_signal):
                os.makedirs(path_signal)

            # Preallocate the memory for all images
            images = np.empty((nb_rows, img_pixels ** 2), dtype=np.float32)
            labels = np.empty((nb_rows, nb_classes), dtype=np.float32)
            signals = np.empty((nb_rows, max_sample['len']), dtype=np.float32)

        for rec_idx, record in enumerate(value['rec_list']):
            path_record = path_db + record
            rec = pywfdb.Record(path_record)

            #Read in the entire patients EKG
            signal = rec.read(key)

            for beat_idx, beat in enumerate(record_map[record][key]['beats']):
                img_cnt += 1

                # Create the data points to plot
                # t = xrange(start, end)
                y = signal[beat.start:beat.end]

                # Print CSV file
                csv.write('{},{},{},{},[{}{}{}{}]\n'.format(img_cnt, record, key, beat.lbl_char, beat.lbl_onehot[0], beat.lbl_onehot[1], beat.lbl_onehot[2], beat.lbl_onehot[3]))

                im, im_row = signal_to_image(buf, img_pixels, y)

                if gen_images:
                    images[img_cnt] = im_row / 255
                    im.save('{}{}_{}_{}_{}{}'.format(path_signal, img_cnt, record, key, beat.lbl_char, '.png'))

                if gen_signals:
                    signals[img_cnt] = pad_signal(y, max_sample['len'])

                labels[img_cnt] = beat.lbl_onehot

                im.close()

                #prog_bar.print_progress(rec_idx, beat_idx)

        # Save as numpy array 1 image/signal per row
        for sig_name in rec.signal_names:
            if gen_images:
                np.savez_compressed('{}images_{}'.format(path_images, sig_name), images=sig_map[sig_name]['images'], labels=sig_map[sig_name]['labels'])

            if gen_signals:
                np.savez_compressed('{}signals_{}'.format(path_images, sig_name), images=sig_map[sig_name]['signals'], labels=sig_map[sig_name]['labels'])

    # Free memory
    del images
    del labels
    del signals

    print '\nData set ' + str(rec_idx + 1) + ' complete!'

    buf.close()


def record_info_to_csv(path_images, num_beats, max_sample, num_signals, sig_map, class_map, record_map):
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


def signal_to_image(image_buffer, img_pixels, signal):

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


class ProgressBar:

    def __init__(self, rec_idx=0, num_recs=0, ann_idx=0, num_anns=0, total_beats=0, start_time=time.time(), ann_count=0):
        self.rec_idx = rec_idx
        self.num_recs = num_recs
        self.ann_idx = ann_idx
        self.num_anns = num_anns
        self.start_time = start_time
        self.ann_count = ann_count

        if total_beats > 0:
            self.total_beats = total_beats
        else:
            self.total_beats = self.num_anns * self.num_recs

    def print_progress(self, rec_idx, ann_idx):
        """
        Prints progress bar.
        :param rec_idx: The index of the current record being processed.
        :param ann_idx: The index of the current annotation being processed.
        :return:
        """
        self.ann_cnt += 1

        # Only printing progress bar for the current record.
        if ann_idx % (self.num_anns / 26) == 0:
            completed_tot = float(self.ann_cnt) / self.total_beats
            completed_rec = float(self.ann_cnt) / (self.num_anns * self.num_recs)
            elapsed_time = time.time() - self.start_time
            hashes = '#' * int(completed_rec * 50)

        elif self.ann_cnt == 1:
            completed_tot = float(self.ann_cnt) / self.total_beats
            elapsed_time = time.time() - self.start_time
            hashes = ' ' * 49

        print '\rRecord[{0:02}/{1}][{2:49s}] Total {3:.1f}% | Est. Time Remaining: {4:.2f}(Min)'.format(
            rec_idx + 1,
            self.num_recs,
            hashes,
            completed_tot * 100,
            ((elapsed_time / completed_tot) - elapsed_time) / 60,
        ),


# ----------------------------------------------------------------------------------------------------------------------
# Helper Class
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
        return '{},{},{},{}'.format(self.rec_name, self.sig_name, self.ann_idx, self.start, self.end, self.lbl_char, self.lbl_onehot)

# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
def get_record_info(path_db, rec_list):
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
    max_sample = {'idx': 0, 'len': 0, 'rec': '', 'lbl': ''}

    for rec_str in rec_list:
        rec = pywfdb.Record(path_db + rec_str)
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
                max_sample['rec'] = rec_str
                max_sample['lbl'] = get_annotation_char(ann)

            img_class = get_annotation_char(ann)

            # Index where annotation occurs
            ann_sig_index = ann.time

            # Calculate start index of image
            start = 0
            if pre_samples != 0 and ann_sig_index - pre_samples >= 0:
                start = ann_sig_index - pre_samples

            for sig_idx, sig_name in enumerate(rec.signal_names):

                beat_cnt += 1

                if rec_str not in record_map:
                    # Read in the entire patients EKG
                    signal_len = len(rec.read(sig_name))
                    record_map[rec_str] = {sig_name: {'len': signal_len, 'beats': set()}}
                elif sig_name not in record_map[rec_str]:
                    signal_len = len(rec.read(sig_name))
                    record_map[rec_str][sig_name] = {'len': signal_len, 'beats': set()}
                    record_map[rec_str][sig_name]['len'] = signal_len
                else:
                    signal_len = record_map[rec_str][sig_name]['len']

                if sig_idx < 1:
                    # Calculate end index of image
                    end = signal_len - 1
                    if post_samples != 0 and ann_sig_index + post_samples < signal_len:
                        end = ann_sig_index + post_samples

                beat = Beat(rec_str, sig_name, ann_idx, start, end, img_class, get_annotation_onehot(ann))

                record_map[rec_str][sig_name]['beats'].add(beat)

                if sig_name not in sig_map:
                    sig_map[sig_name] = {img_class: set(), 'rec_list': set()}
                    sig_map[sig_name][img_class].add(beat)
                    sig_map[sig_name]['rec_list'].add(rec_str)
                else:
                    if img_class not in sig_map[sig_name]:
                        sig_map[sig_name][img_class] = set()
                        sig_map[sig_name][img_class].add(beat)
                    else:
                        sig_map[sig_name][img_class].add(beat)

                    if rec_str not in sig_map[sig_name]['rec_list']:
                        sig_map[sig_name]['rec_list'].add(rec_str)

                if img_class not in class_map:
                    class_map[img_class] = {sig_name: set()}
                    class_map[img_class][sig_name].add(beat)
                elif sig_name not in class_map[img_class]:
                    class_map[img_class][sig_name] = set()
                    class_map[img_class][sig_name].add(beat)
                else:
                    class_map[img_class][sig_name].add(beat)

    return beat_cnt, max_sample, num_signals, sig_map, class_map, record_map


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
