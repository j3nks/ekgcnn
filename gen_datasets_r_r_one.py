import io
import os
import time
from math import ceil, floor

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pywfdb

from PIL import Image

records = [
    '100',
    '101',
    '103',
    '105',
    '106',
    '108',
    '109',
    '111',
    '112',
    '113',
    '114',
    '115',
    '116',
    '117',
    '118',
    '119',
    '121',
    '122',
    '123',
    '124',
    '200',
    '201',
    '202',
    '203',
    '205',
    '207',
    '208',
    '209',
    '210',
    '212',
    '213',
    '214',
    '215',
    '219',
    '220',
    '221',
    '222',
    '223',
    '228',
    '230',
    '231',
    '232',
    '233',
    '234'
]

path_mitdb = r'c:/mitdb/db/mitdb/'
path_images = r'c:/mitdb/images/mitdb/'

# Create directory to store images
if not os.path.exists(path_images):
    os.makedirs(path_images)


# Count how many beats there are in the data set
def get_record_info(rec_list):
    beats = 0
    max_samples = 0

    for rec_str in rec_list:
        rec = pywfdb.Record(path_mitdb + rec_str)
        annotations = rec.annotation().read()

        for ann_idx, ann in enumerate(annotations):
            if 1 <= ann.type <= 11 or ann.type == 34:
                beats += 1

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
                if total_samples > max_samples:
                    max_samples = total_samples

    return beats, max_samples


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


# Inserts padding into signal so that it is uniform length
def pad_signal(sig, sig_len):
    front_pad = 0
    back_pad = 0

    if len(sig) < sig_len:
        pad = sig_len - len(sig)
        front_pad = int(floor(pad / 2.0))
        back_pad = int(ceil(pad / 2.0))

    return np.concatenate((np.zeros(front_pad, dtype=np.float32), np.array(sig, dtype=np.float32), np.zeros(back_pad, dtype=np.float32)))


buf = io.BytesIO()  # Memory buffer so that image doesn't have to save to disk.
pre_ann = .99  # Time to capture before annotation in seconds
post_ann = .99  # Time to capture after annotation in seconds
img_count = 0  # Image counter to keep track of indices
img_pixels = 256
num_signals = 2
nb_classes = 4

start_time = time.time()
num_beats, max_samples = get_record_info(records)
total_beats = num_beats * num_signals
mlii_count = 0
v_count = 0

# Preallocate the memory for all images
images_mlii = np.empty((num_beats, img_pixels ** 2), dtype=np.float32)
images_v = np.empty((num_beats, img_pixels ** 2), dtype=np.float32)
labels = np.empty((num_beats, nb_classes), dtype=np.float32)
signals_mlii = np.empty((num_beats, max_samples), dtype=np.float32)
signals_v = np.empty((num_beats, max_samples), dtype=np.float32)

csv = open(path_images + 'images.csv', 'wb')
csv.write('ImageNumber, Record, SignalIndex, SignalName, Label, Onehot\n')

gen_signals = True
gen_images = True
# ----------------------------------------------------------------------------------------------------------------------
for rec_idx, record in enumerate(records):
    path_record = path_mitdb + record
    rec = pywfdb.Record(path_record)
    ann_cnt = 0

    for sig_name in rec.signal_names:
        sig_index = 0
        if sig_name != 'MLII':
            sig_index = 1

        # Read in the entire patients EKG
        signal = rec.read(sig_name)
        signal_len = len(signal)

        path_image = '{}ds{}/'.format(path_images, sig_index)
        # Create directory to store images
        if not os.path.exists(path_image):
            os.makedirs(path_image)

        # Read all annotations from associated annotation file with extension (.atr)
        annotations = rec.annotation().read()

        for ann_idx, ann in enumerate(annotations):
            # Check if it is an annotation we are interested in
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

            # Index where annotation occurs
            ann_sig_index = ann.time

            # Calculate start index of image
            start = 0
            if pre_samples != 0 and ann_sig_index - pre_samples >= 0:
                start = ann_sig_index - pre_samples

            # Calculate end index of image
            end = signal_len - 1
            if post_samples != 0 and ann_sig_index + post_samples < signal_len:
                end = ann_sig_index + post_samples

            # Create the data points to plot
            # t = xrange(start, end)
            y = signal[start:end]

            # Create image from matplotlib
            fig = plt.figure(figsize=(1, 1), dpi=img_pixels, frameon=False)
            # fig.set_size_inches(1, 1)
            ax = fig.add_axes([0., 0., 1., 1.])
            ax.axis('off')
            ax.set_xlim(0, end - start)
            # ax.set_ylim(-1.0, 1.5) # TODO: Should we set the y-axis limits?
            ax.plot(y, color='k', linewidth=.01)

            # Saving image to buffer in memory
            plt.savefig(buf, dpi=img_pixels)

            # Change image to to black and white with PIL
            im = Image.open(buf)
            im = im.convert('L')
            im = im.point(lambda x: 0 if x < 254 else 255)  # , '1')
            # im.show()

            # Convert image to 1d vector
            im_arr = np.asarray(im)  # , dtype=np.uint8)
            im_row = im_arr.ravel()

            # Print CSV file
            csv.write('{},{},{},{},{},{}\n'.format(mlii_count, entry, sig_index, sig_name, get_annotation_char(ann), get_annotation_onehot(ann)))

            if not sig_index:
                if gen_images:
                    images_mlii[mlii_count] = im_row / 255
                    im.save('{}{}_{}_{}_{}{}'.format(path_image, mlii_count, record, sig_name, get_annotation_char(ann), '.png'))

                if gen_signals:
                    signals_mlii[mlii_count] = pad_signal(y, max_samples)

                labels[mlii_count] = get_annotation_onehot(ann)
                mlii_count += 1

            else:
                if gen_images:
                    images_v[v_count] = im_row / 255
                    im.save('{}{}_{}_{}_{}{}'.format(path_image, v_count, record, sig_name, get_annotation_char(ann), '.png'))

                if gen_signals:
                    signals_v[v_count] = pad_signal(y, max_samples)

                v_count += 1

            # Free memory
            plt.close(fig)
            im.close()
            buf.seek(0)

            # Print progress bar
            ann_cnt += 1
            curr_beat = mlii_count + v_count
            if ann_idx % (len(annotations) / 26) == 0:
                completed_tot = float(curr_beat) / total_beats
                completed_rec = float(ann_cnt) / (len(annotations) * 2)
                elapsed_time = time.time() - start_time
                print '\rRecord[{3:02}/{4}][{0:49s}] Total {1:.1f}% | Est. Time Remaining: {2:.2f}(Min)'.format(
                    '#' * int(completed_rec * 50), completed_tot * 100,
                    ((elapsed_time / completed_tot) - elapsed_time) / 60, rec_idx + 1, len(records)),
            elif ann_cnt == 1:
                completed_tot = float(curr_beat) / total_beats
                elapsed_time = time.time() - start_time
                print '\rRecord[{3:02}/{4}][{0:49s}] Total {1:.1f}% | Est. Time Remaining: {2:.2f}(Min)'.format(
                    ' ' * 49, completed_tot * 100, ((elapsed_time / completed_tot) - elapsed_time) / 60, rec_idx + 1,
                    len(records)),

# Save as numpy array 1 image/signal per row
if gen_images:
    np.savez_compressed('{}images_mlii'.format(path_images), images=images_mlii, labels=labels)
    np.savez_compressed('{}images_v'.format(path_images), images=images_v, labels=labels)

if gen_signals:
    np.savez_compressed('{}signals_mlii'.format(path_images), images=signals_mlii, labels=labels)
    np.savez_compressed('{}signals_v'.format(path_images), images=images_mlii, labels=labels)

# Free memory
del images_mlii
del images_v
del labels
del signals_mlii
del signals_v

print '\nData set ' + str(rec_idx + 1) + ' complete!'

buf.close()
csv.close()
