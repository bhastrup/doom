# Import packages
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import threading
import multiprocessing

import gym
from skimage import transform

from estimator import Model
from worker import Worker
from gdoom_env import *

# For cropping and further image resizing, grayscaling done gdoom env
def preprocess(state):
    #s = state[30:-10,30:-30] # cropping, remove roof
    s = np.asarray(state)
    s = s / 255
    s = transform.resize(s, [84, 84])

# For setting up the gdoom environment
def make_env():
    env = gym.make("doom_scenario0_96-v0")
    frame = env.reset()

    return env

NUM_WORKERS = 3
T_MAX = 5
VALID_ACTIONS = [0, 1, 2]

name_of_run = 'a3c'
summary_dir = 'logs/'+name_of_run
if not os.path.exists(summary_dir): os.makedirs(summary_dir)

summary_writer = tf.summary.FileWriter(summary_dir)

with tf.device("/cpu:0"):

  with tf.variable_scope("global") as vs:
    model_net = Model(num_outputs=len(VALID_ACTIONS))

  global_counter = itertools.count()

  workers = []
  for worker_id in range(NUM_WORKERS):
    worker = Worker(
      name="worker_{}".format(worker_id),
      env=make_env(),
      model_net=model_net,
      discount_factor = 0.99,
      t_max=T_MAX,
      global_counter=global_counter,
      summary_writer=summary_writer)
    workers.append(worker)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()

  worker_threads = []
  for worker in workers:
    worker_fn = lambda: worker.run(sess, coord)
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)

  coord.join(worker_threads)