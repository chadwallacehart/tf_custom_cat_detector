'''
Source: https://gist.github.com/saghiralfasly/ee642af0616461145a9a82d7317fb1d6
'''
import os
import io
import glob
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import random
from models.research.object_detection.utils import dataset_util

from PIL import Image


from shutil import move

'''
this script automatically divides dataset into training and evaluation (10% for evaluation)
this scripts also shuffles the dataset before converting it into tfrecords
if u have different structure of dataset (rather than pascal VOC ) u need to change
the paths and names input directories(images and annotation) and output tfrecords names.
(note: this script can be enhanced to use flags instead of changing parameters on code).
default expected directories tree:
dataset- 
   -JPEGImages
   -Annotations
    dataset_to_tfrecord.py   
to run this script:
$ python dataset_to_tfrecord.py 
'''

def create_tf_example_orig(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def class_text_to_int(row_label):
    if row_label == 'cat':
        return 1
    elif row_label == 'person':
        return 2
    else:
        return None

def create_example(xml_file, img_file):
    # process the xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_name = root.find('filename').text
    file_name = image_name.encode('utf8')
    size = root.find('size')
    width = int(size[0].text)
    height = int(size[1].text)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for member in root.findall('object'):
        obj_class = member[0].text
        classes_text.append(obj_class.encode('utf8'))
        xmin.append(float(member[4][0].text) / width)
        ymin.append(float(member[4][1].text) / height)
        xmax.append(float(member[4][2].text) / width)
        ymax.append(float(member[4][3].text) / height)
        difficult_obj.append(0)

        classes.append(class_text_to_int(obj_class))  # i wrote 1 because i have only one class(person)
        truncated.append(0)
        poses.append('Unspecified'.encode('utf8'))

    # read corresponding image
    # full_path = os.path.join('./images', '{}'.format(image_name))  # provide the path of images directory
    with tf.gfile.GFile(img_file, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # create TFRecord Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    writer_train = tf.python_io.TFRecordWriter('gcp-data/train.record')
    writer_test = tf.python_io.TFRecordWriter('gcp-data/test.record')
    # provide the path to annotation xml files directory
    filename_list = tf.train.match_filenames_once("images/*.xml")
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    list = sess.run(filename_list)
    random.shuffle(list)  # shuffle files list
    i = 1
    tst = 0  # to count number of images for evaluation
    trn = 0  # to count number of images for training
    for xml_file in list:
        jpg_file = os.path.splitext(xml_file.decode('ascii'))[0]+'.jpg'
        if (i % 10) == 0:  # each 10th file (xml and image) write it for evaluation
            new_xml_file = os.path.join('images/test', os.path.basename(xml_file.decode('ascii')))
            move(xml_file, new_xml_file)

            new_jpg_file = os.path.join('images/test', os.path.basename(jpg_file))
            move(jpg_file, new_jpg_file)

            example = create_example(new_xml_file, new_jpg_file)
            writer_test.write(example.SerializeToString())
            tst = tst + 1
        else:  # the rest for training
            new_xml_file = os.path.join('images/train', os.path.basename(xml_file.decode('ascii')))
            move(xml_file, new_xml_file)

            new_jpg_file = os.path.join('images/train', os.path.basename(jpg_file))
            move(jpg_file, new_jpg_file)

            example = create_example(new_xml_file, new_jpg_file)
            writer_train.write(example.SerializeToString())
            trn = trn + 1
        # copy the files
        # new_jpg_file = os.path.join(os.getcwd(), 'images/test', os.path.basename(jpg_file))
        i = i + 1
        print(new_xml_file)
    writer_test.close()
    writer_train.close()
    print('Successfully converted dataset to TFRecord.')
    print('training dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)


if __name__ == '__main__':
    tf.app.run()