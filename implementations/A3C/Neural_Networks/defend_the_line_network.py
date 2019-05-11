import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model():
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        self._build_model()

    def _build_model(self):
        self.states = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name='X')
        self.targets_pi = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_pi")
        self.targets_v = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_v")
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.states)[0]

        # First convolutional layer
        self.conv_1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.states, num_outputs=32,
                                  kernel_size=[8, 8], stride=4, padding='SAME')
        # Second convolutional layer
        self.conv_2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_1, num_outputs=64,
                                  kernel_size=[4, 4], stride=2, padding='SAME')
        # Third convolutional layer
        self.conv_3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv_2, num_outputs=64,
                                  kernel_size=[3, 3], stride=1, padding='SAME')

        # Flatten the network
        flattened = slim.fully_connected(slim.flatten(self.conv_3), 1024, activation_fn=tf.nn.elu)

        # The policy output
        fc1_pi = slim.fully_connected(flattened, 3,
                                      activation_fn=tf.nn.softmax,
                                      biases_initializer=None)

        self.logits_pi = tf.contrib.layers.fully_connected(fc1_pi, self.num_outputs, activation_fn=None)
        self.probs_pi = tf.nn.softmax(self.logits_pi) + 1e-8

        # Entropy, to encourage exploration
        self.entropy_pi = -tf.reduce_sum(self.probs_pi * tf.log(self.probs_pi), 1, name="entropy")

        # Predictions from chosen actions
        gather_indices_pi = tf.range(batch_size) * tf.shape(self.probs_pi)[1] + self.actions
        self.picked_action_probs_pi = tf.gather(tf.reshape(self.probs_pi, [-1]), gather_indices_pi)

        self.losses_pi = - (tf.log(self.picked_action_probs_pi) * self.targets_pi + 0.008 * self.entropy_pi)
        self.loss_pi = tf.reduce_sum(self.losses_pi, name="loss_pi")

        # The value output
        fc1_v = slim.fully_connected(flattened,
                                     1,
                                     activation_fn=None,
                                     biases_initializer=None)

        self.logits_v = tf.contrib.layers.fully_connected(inputs=fc1_v, num_outputs=1, activation_fn=None)
        self.logits_v = tf.squeeze(self.logits_v, squeeze_dims=[1])

        self.losses_v = 0.5 * tf.squared_difference(self.logits_v, self.targets_v)
        self.loss_v = tf.reduce_sum(self.losses_v, name="loss_v")

        # Combine loss
        self.loss = self.loss_pi + 0.5 * self.loss_v

        # Using the RMSPropOptimizer, https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
        # Inputs to the optimizer:
        # learning_rate: A Tensor or a floating point value. The learning rate
        # decay: Discounting factor for the history/coming gradient
        # momentum: A scalar tensor
        # epsilon: Small value to avoid zero denominator.

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]

        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                       global_step=tf.contrib.framework.get_global_step())
