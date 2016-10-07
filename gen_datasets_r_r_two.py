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

ds1 = [
    '101',
    '106',
    '108',
    '109',
    '112',
    '114',
    '115',
    '116',
    '118',
    '119',
    '122',
    '124',
    '201',
    '203',
    '205',
    '207',
    '208',
    '209',
    '215',
    '220',
    '223',
    '230'
]

ds2 = [
    '100',
    '103',
    '105',
    '111',
    '113',
    '117',
    '121',
    '123',
    '200',
    '202',
    '210',
    '212',
    '213',
    '214',
    '219',
    '221',
    '222',
    '228',
    '231',
    '232',
    '233',
    '234'
]

data_sets = [ds1, ds2]

path_db = r'c:/ekgdb/db/mitdb/'
path_images = r'c:/ekgdb/images/mitdb/two/'

# Create directory to store images
if not os.path.exists(path_images):
    os.makedirs(path_images)


# Count how many beats there are in the data set
def get_record_info(rec_list):
    beats = 0
    max_samples = 0
    labels = []

    for rec_str in rec_list:
        rec = pywfdb.Record(path_db + rec_str)
        annotations = rec.annotation().read()

        for ann_idx, ann in enumerate(annotations):
            if 1 <= ann.type <= 11 or ann.type == 34:

                labels.append(get_annotation_onehot(ann))
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

    return beats, max_samples, np.array(labels)


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
pre_ann = 1.1  # Time to capture before annotation in seconds
post_ann = 1.1  # Time to capture after annotation in seconds
img_count = 0  # Image counter to keep track of indices
img_pixels = 256
num_signals = 2
nb_classes = 4

csv = open(path_images + 'images.csv', 'wb')
csv.write('ImageNumber, Record, SignalIndex, SignalName, Label, Onehot\n')
# ----------------------------------------------------------------------------------------------------------------------
for ds_idx, data_set in enumerate(data_sets):
    start_time = time.time()
    num_beats, max_samples, labels = get_record_info(data_set)
    total_beats = num_beats * num_signals
    mlii_count = 0
    v_count = 0

    # Preallocate the memory for all images
    images_mlii = np.empty((num_beats, img_pixels ** 2), dtype=np.uint8)
    images_v = np.empty((num_beats, img_pixels ** 2), dtype=np.uint8)
    signals_mlii = np.empty((num_beats, max_samples), dtype=np.float32)
    signals_v = np.empty((num_beats, max_samples), dtype=np.float32)

    for rec_idx, entry in enumerate(data_set):
        path_record = path_db + entry
        record = pywfdb.Record(path_record)
        ann_cnt = 0
        
        for sig_name in record.signal_names:
            sig_index = 0
            if sig_name != 'MLII':
                sig_index = 1

            # Read in the entire patients EKG
            signal = record.read(sig_name)
            signal_len = len(signal)

            path_image = '{}ds{}/signal{}/'.format(path_images, ds_idx, sig_index)
            # Create directory to store images
            if not os.path.exists(path_image):
                os.makedirs(path_image)

            # Read all annotations from associated annotation file with extension (.atr)
            annotations = record.annotation().read()

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
                im = im.point(lambda x: 0 if x < 255 else 255)  # , '1')
                #im.show()

                # Convert image to 1d vector
                im_arr = np.asarray(im)  # , dtype=np.uint8)
                im_row = im_arr.ravel()

                # Print CSV file
                csv.write('{},{},{},{},{},{}\n'.format(mlii_count, entry, sig_index, sig_name, get_annotation_char(ann), get_annotation_onehot(ann)))

                if not sig_index:
                    images_mlii[mlii_count] = im_row / 255
                    signals_mlii[mlii_count] = pad_signal(y, max_samples)
                    im.save('{}{}_{}_{}_{}{}'.format(path_image, mlii_count, entry, sig_name, get_annotation_char(ann), '.png'))
                    mlii_count += 1

                else:
                    images_v[v_count] = im_row / 255
                    signals_v[v_count] = pad_signal(y, max_samples)
                    im.save('{}{}_{}_{}_{}{}'.format(path_image, v_count, entry, sig_name, get_annotation_char(ann), '.png'))
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
                        '#' * int(completed_rec * 50), completed_tot * 100, ((elapsed_time / completed_tot) - elapsed_time) / 60, rec_idx + 1, len(data_sets[ds_idx])),
                elif ann_cnt == 1:
                    completed_tot = float(curr_beat) / total_beats
                    elapsed_time = time.time() - start_time
                    print '\rRecord[{3:02}/{4}][{0:49s}] Total {1:.1f}% | Est. Time Remaining: {2:.2f}(Min)'.format(
                        ' ' * 49, completed_tot * 100, ((elapsed_time / completed_tot) - elapsed_time) / 60, rec_idx + 1, len(data_sets[ds_idx])),

    # Save as numpy array 1 image per row
    np.savez_compressed('{}ds{}_mlii'.format(path_images, ds_idx), images=images_mlii, labels=labels)
    np.savez_compressed('{}ds{}_v'.format(path_images, ds_idx), images=images_v, labels=labels)
    np.savez_compressed('{}ds{}_signals_mlii'.format(path_images, ds_idx), images=signals_mlii, labels=labels)
    np.savez_compressed('{}ds{}_signals_v'.format(path_images, ds_idx), images=signals_v, labels=labels)

    # Free memory
    del images_mlii
    del images_v
    del signals_mlii
    del signals_v

    print '\nData set ' + str(ds_idx) + ' complete!'

buf.close()
csv.close()
