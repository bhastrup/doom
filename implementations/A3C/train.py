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

from network import Model
from worker import Worker
from gdoom_env import *

# For setting up the gdoom environment
def make_env():
    # 0  - Basic
    # 1 - Corridor
    # 2 - DefendCenter
    # 3 - DefendLine
    # 4 - HealthGathering
    # 5 - MyWayHome
    # 6 - PredictPosition
    # 7 - TakeCover
    # 8 - Deathmatch

    env = gym.make("doom_scenario3_96-v0")
    frame = env.reset()

    return env

num_workers = multiprocessing.cpu_count() # set the number as workers = available CPU threads
t_max = 5 # update parameters in each thread after this many steps in that thread
action_space = [0, 1, 2] # the possible actions

name_of_run = 'a3c'
summary_dir = 'logs/'+name_of_run
if not os.path.exists(summary_dir): os.makedirs(summary_dir)

summary_writer = tf.summary.FileWriter(summary_dir)

with tf.device("/cpu:0"):

  with tf.variable_scope("global") as vs:
    model_net = Model(num_outputs=len(action_space))

  global_counter = itertools.count()

  # create process for each worker
  workers = []
  for worker_id in range(num_workers):
    worker = Worker(
      name="worker_{}".format(worker_id),
      env=make_env(),
      model_net=model_net,
      discount_factor = 0.99,
      t_max=t_max,
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
