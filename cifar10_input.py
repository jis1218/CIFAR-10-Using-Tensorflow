# coding: utf-8
'''
Created on 2018. 4. 10.

@author: Insup Jung
'''

# sys.path 상의 가장 상위 모듈을 import하는 것을 보장해 줌
from __future__ import absolute_import
# 뭔지 잘 모르겠다
from __future__ import division

# print 함수에 ()를 사용할 수 있게 함... 왜???? 원래 되어있는 것인데?
from __future__ import print_function

import os



from six.moves import xrange
import tensorflow as tf

#32x32 사이즈의 이미지를 랜덤하게 24x24 사이즈로 자름으로써 전체 데이터 수를 늘릴 수 있다.
IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# 파일 이름 목록을 받아와 CIFAR-10의 바이너리 데이터를 읽고 파싱하여 단일 오브젝트 형태로 반환한다.
def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  
  Args:
    filename_queue: A queue of strings with the filenames to read from.
    
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    
    #모든 레코드는 이미지 바이트 다음에 라벨 바이트가 따라온다.
    record_bytes = label_bytes + image_bytes
    
    # 레코드를 읽고 filename_queue에서 파일이름을 가져온다.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes) #고정된 크기의 record를 가져온다.
    result.key, value = reader.read(filename_queue)
    
    # string에서 uint8로 바꿔준다. 왜냐면 위에서 받은 value가 string 값이기 때문에
    record_bytes = tf.decode_raw(value, tf.uint8)
    
    #record_bytes는 라벨과 그림의 바이트인데 라벨만 떼어내가 위해 아래와 같은 함수를 쓴다. 그리고 int32로 cast 해준다.
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    
    #라벨을 떼고 남은 바이트는 이미지로 바꿔준다. 이미지는 지금 [depth * height * width] 형태로 되어 있는데 이를 [depth, height, width]로 바꿔준다.
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]), [result.depth, result.height, result.width])
    # [depth, height, width] 형태로 되어있는 list의 형태를 [height, width, depth]의 형태로 바꿔준다.
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    
    # image를 셔플하여 queue를 만든다. 
    # batch size의 image와 label을 queue로부터 읽어들인다.
    
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label], batch_size = batch_size, num_threads=num_preprocess_threads, 
                                                    capacity=min_queue_examples + 3*batch_size, min_after_dequeue=min_queue_examples)
        
        # training image를 visualizer로 보여준다.
        tf.summary.image('images', images)
        
        return images, tf.reshape(label_batch, [batch_size])

# 데이터셋 확대를 위한 이미지 왜곡 작업을 진행
def distorted_inputs(data_dir, batch_size):
    # os.path.join 함수는 전달받은 파라미터를 이어 새로운 경로를 만드는 함수
    # 아래 코드는 이 함수에서 파라미터로 받은 data_dir 경로와 그 경로 아래에 있는
    # CIFAR-10의 이미지 파일이 담긴 data_batch_1.bin ~ data_batch_5.bin의
    # 5개 파일에 대한 전체 경로를 요소로 하는 텐서를 만드는 것이다.
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    
    # 배열 내에 파일 경로가 없으면 에러 발생
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    with tf.name_scope('data_augumentation'):
        read_input=read_cifar10(filename_queue)
        
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        
        height = IMAGE_SIZE
        width = IMAGE_SIZE
        
        # 32*32 그리고 3채널의 이미지를 24*24 그리고 3채널의 이미지로 랜덤하게 crop한다.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
        
        # 랜덤하게 수평적으로 이미지를 뒤집는다.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        
        # 마치 배치 정규화를 한 것처럼 이미지도 그렇게 하는 것 같다.
        float_image = tf.image.per_image_standardization(distorted_image)
        
        # float_image의 모양을 set 해준다. 이게 무슨 의미지?
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])
        
        # random shuffling이 잘 되었는지 확인한다.
        # 전체 테스트용 이미지의 40%, 즉, 총 50000개의 테스트 이미지 중 20000개를 사용
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_fraction_of_examples_in_queue)
        print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
        
        # Generate a batch of images and labels by building up a queue of examples.
        # 배치 작업에 사용할 128개의 이미지를 shuffle하여 리턴
        return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)
    
    def inputs(eval_data, data_dir, batch_size):
        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
            
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)
        
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(filenames)
            
            
            
        read_input=read_cifar10(filename_queue)
        
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        
        height = IMAGE_SIZE
        width = IMAGE_SIZE
        
        #evaluation하기 위한 Image Processing
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
        
        # 마치 배치 정규화를 한 것처럼 이미지도 그렇게 하는 것 같다.
        float_image = tf.image.per_image_standardization(distorted_image)
        
        # float_image의 모양을 set 해준다. 이게 무슨 의미지?
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])
        
        # random shuffling이 잘 되었는지 확인한다.
        # 전체 테스트용 이미지의 40%, 즉, 총 50000개의 테스트 이미지 중 20000개를 사용
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_fraction_of_examples_in_queue)
        print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
        
        # Generate a batch of images and labels by building up a queue of examples.
        # 배치 작업에 사용할 128개의 이미지를 shuffle하여 리턴
        return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)

        