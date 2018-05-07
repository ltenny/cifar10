from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf 

# although the CIFAR-10 are 32x32 we take just the center 24x24
IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# reads a single example from CIFAR-10 files
def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # dimensions of the images in the CIFAR-10 dataset
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * rsult.depth

    # every record has a label followed by the image
    record_bytes = label_bytes + image_bytes

    # read a record getting the filenames from the input queue of 
    # filenames, note that using FixedLengthRecords() (i.e. the data API)
    # is more efficient than the feed approach via placeholders
    reader = tf.FixedLengthRecords(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # convert from a string to a vector of uint8
    record_bytes = tf.decode_raw(valuem, tf.uint8)
    
    # the first bytes represent the label, so we convert it to int32
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    # the remaining bytes represent the image that we need to reshape into 32x32x3
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],[labels_bytes+image_bytes]),
                            [result.depth, result.height, result.width])
    
    result.uint8image = tf.transpose(depth_major, [1,2,0])

    return result
    

