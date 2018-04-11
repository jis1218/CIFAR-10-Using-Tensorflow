# coding: utf-8
'''
Created on 2018. 4. 11.

@author: Insup Jung
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
from CNN import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train', """Directory where to write event logs""" """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement""")
tf.app.flags.DEFINE_integer('log_frequency', 10, """How often to log results to the console""")

def train():
    
    # 지금까지 만들었던 모든 그래프 구성요소를 하나의 전역 Graph 안에서 사용하겠다는 의미
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        
        # CIFAR-10으로부터 이미지와 라벨을 가지고 온다.
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()
            
        # interference 모델로부터 logits 예측값을 계산할 수 있는 그래프를 그린다.
        logits = cifar10.inference(images)
        
        # Loss 값을 계산한다.
        loss = cifar10.loss(logits, labels)
        
        train_op = cifar10.train(loss, global_step)
        
        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()
            
            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)
            
            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
                    
                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
                    
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir, hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps), tf.train.NanTensorHook(loss), _LoggerHook()], 
                                               config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
                
        