import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import pandas as pd          # Handle dataframes
from vizdoom import *        # Doom Environment
from vizdoom import GameVariable

import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

"""
Here we create our environment
REWARDS:

+101 for killing the monster

-5 for missing
Episode ends after killing the monster or on timeout.
living reward = -1
"""
def create_environment():
    game = DoomGame()

    # Load the correct configuration
    game.load_config("basic.cfg")

    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("basic.wad")

    game.init()

    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions


"""
Here we performing random action to test the environment
"""
def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()

game, possible_actions = create_environment()

"""
Here we define the preprocessing functions
Our steps:
    Grayscale each of our frames (because color does not add important information ). But this is already done by the config file.
    Crop the screen (in our case we remove the roof because it contains no information)
    We normalize pixel values
    Finally we resize the preprocessed frame
"""
def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)

    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    return preprocessed_frame

# stack frames
stack_size = 4  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

"""
Here we set up our hyperparameters
    First, you begin by defining the neural networks hyperparameters when you implement the model.
    Then, you'll add the training hyperparameters when you implement the training algorithm
"""
### MODEL HYPERPARAMETERS
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 100        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0005            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

"""
Here we create our Deep Q-learning Neural Network model
    We take a stack of 4 frames as input
    It passes through 3 convnets
    Then it is flatened
    Finally it passes through 2 FC layers
    It outputs a Q value for each actions
"""


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]

            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")

            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3,
                                          activation=None)

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

"""
Experience Replay
Here we'll create the Memory object that creates a deque. A deque (double ended queue) is a data type that removes the oldest element each time that you add a new element.
"""
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)

        return [self.buffer[i] for i in index]

# Here we'll deal with the empty memory problem: we pre-populate our memory by taking random actions and storing the experience (state, action, reward, new_state).

# Instantiate memory
memory = Memory(max_size=memory_size)

# Render the environment
game.new_episode()

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Random action
    action = random.choice(possible_actions)

    # Get the rewards
    reward = game.make_action(action)

    # Look if the episode is finished
    done = game.is_episode_finished()

    # If we're dead
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        game.new_episode()

        # First we need a state
        state = game.get_state().screen_buffer

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Get the next state
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our state is now the next_state
        state = next_state

"""
Set up Tensorboard
"""

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("./tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

# To launch tensorboard: tensorboard --logdir=/home/wamberg/ViZDoom-master/tensorboard/dqn/1
# and go to: http://localhost:6006/


"""
Train our Agent

 The algorithm:
 - Initialize the weights
 - Init the environment
 - Initialize the decay rate (that will use to reduce epsilon) 

 For episode to max_episode do
    Make new episode
    Set step to 0
    Observe the first state s_0
    
    While step < max_steps do:
        Increase decay_rate
        With epsilon select a random action a_t, otherwise select a_t = argmax{a} [Q(s_t,a)]
        Execute action a_t in simulator and observe reward r_{t+1} and new state s_{t+1}
        Store transition
        Sample random mini-batch from D 
        Set Q_hat = r if the episode ends at +1, otherwise set Q_hat = r + gamma max_{a'} [Q(s', a')]
        Make a gradient descent step with loss Q_hat - Q(s, a))^2
    endfor
 endfor 
"""

"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


# Saver will help us to save our model
saver = tf.train.Saver()


# initialize variable arrays
step_data = []
episode_data = []
reward_data = []
loss_data = []
explore_data = []
killcount_data = []
itemcount_data = []
secretcount_data = []
fragcount_data = []
deathcount_data = []
hitcount_data = []
hits_taken_data = []
damagecount_data = []
damage_taken_data = []
health_data = []
armor_data = []
dead_data = []
on_ground_data = []
attack_ready_data = []
alattack_ready_data = []
selected_weapon_data = []
selected_weapon_ammo_data = []
ammo_data = []
x_pos_data = []
y_pos_data = []
z_pos_data = []
angle_data = []
pitch_data = []
roll_data = []
x_vel_data = []
y_vel_data = []
y_vel_data = []
z_vel_data = []
cam_pos_x_data = []
cam_pos_y_data = []
cam_pos_z_data = []
cam_angle_data = []
cam_pitch_data = []
cam_roll_data = []
cam_fov_data = []


all_data = {'step':step_data,'episode':episode_data,'reward':reward_data, 'explore':explore_data,
            'killcount':killcount_data,'itemcount':itemcount_data, 'secretcount':secretcount_data, 'fragcount':fragcount_data,
            'deathcount':deathcount_data,'hitcount':hitcount_data, 'hitstaken':hits_taken_data, 'damagecount':damagecount_data,
            'damage_taken':damage_taken_data,'health':health_data, 'armor':armor_data, 'dead':dead_data,
            'on_ground':on_ground_data,'attack_ready':attack_ready_data, 'alattack_ready':alattack_ready_data, 'selected_weapon':selected_weapon_data,
            'ammo':ammo_data,'x_pos':x_pos_data, 'y_pos':y_pos_data, 'z_pos':z_pos_data,
            'angle':angle_data,'pitch':pitch_data, 'roll':roll_data, 'x_vel':x_vel_data,
            'y_vel':y_vel_data,'z_vel':z_vel_data, 'cam_pos_x':cam_pos_x_data, 'cam_pos_y':cam_pos_y_data,
            'cam_pos_y':cam_pos_y_data,'cam_pos_z':cam_pos_z_data, 'cam_angle':cam_angle_data, 'cam_pitch':cam_pitch_data,
            'cam_roll':cam_roll_data,'cam_fov':cam_fov_data
            }

df = pd.DataFrame(all_data)

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Init the game
        game.init()

        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            game.new_episode()
            state = game.get_state().screen_buffer

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1

                # Increase decay_step
                decay_step += 1

                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)

                # Do the action
                reward = game.make_action(action)

                # Look if the episode is finished
                done = game.is_episode_finished()

                # Add the reward to total reward
                episode_rewards.append(reward)
                
               

                # If the game is finished
                if done:
                    # Get game variables
                    killcount = game.get_game_variable(GameVariable.KILLCOUNT)
                    itemcount = game.get_game_variable(GameVariable.ITEMCOUNT)
                    secretcount = game.get_game_variable(GameVariable.SECRETCOUNT)
                    fragcount = game.get_game_variable(GameVariable.FRAGCOUNT)
                    deathcount = game.get_game_variable(GameVariable.DEATHCOUNT)
                    hitcount = game.get_game_variable(GameVariable.HITCOUNT)
                    hits_taken = game.get_game_variable(GameVariable.HITS_TAKEN)
                    damagecount = game.get_game_variable(GameVariable.DAMAGECOUNT)
                    damage_taken = game.get_game_variable(GameVariable.DAMAGE_TAKEN)
                    health = game.get_game_variable(GameVariable.HEALTH)
                    armor = game.get_game_variable(GameVariable.ARMOR)
                    dead = game.get_game_variable(GameVariable.DEAD)
                    on_ground = game.get_game_variable(GameVariable.ON_GROUND)
                    attack_ready = game.get_game_variable(GameVariable.ATTACK_READY)
                    alattack_ready = game.get_game_variable(GameVariable.ALTATTACK_READY)
                    selected_weapon = game.get_game_variable(GameVariable.SELECTED_WEAPON)
                    selected_weapon_ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
                    ammo = game.get_game_variable(GameVariable.AMMO2)
                    x_pos = game.get_game_variable(GameVariable.POSITION_X)
                    y_pos = game.get_game_variable(GameVariable.POSITION_Y)
                    z_pos = game.get_game_variable(GameVariable.POSITION_Z)
                    angle = game.get_game_variable(GameVariable.ANGLE)
                    pitch = game.get_game_variable(GameVariable.PITCH)
                    roll = game.get_game_variable(GameVariable.ROLL)
                    x_vel = game.get_game_variable(GameVariable.VELOCITY_X)
                    y_vel = game.get_game_variable(GameVariable.VELOCITY_Y)
                    z_vel = game.get_game_variable(GameVariable.VELOCITY_Z)
                    cam_pos_x = game.get_game_variable(GameVariable.CAMERA_POSITION_X)
                    cam_pos_y = game.get_game_variable(GameVariable.CAMERA_POSITION_Y)
                    cam_pos_z = game.get_game_variable(GameVariable.CAMERA_POSITION_Z)
                    cam_angle = game.get_game_variable(GameVariable.CAMERA_ANGLE)
                    cam_pitch = game.get_game_variable(GameVariable.CAMERA_PITCH)
                    cam_roll = game.get_game_variable(GameVariable.CAMERA_ROLL)
                    cam_fov = game.get_game_variable(GameVariable.CAMERA_FOV)
                    
                    
                    # add to variable data arrays
                    step_data.append(step)
                    episode_data.append(episode)
                    reward_data.append(reward)
                    explore_data.append(explore_probability)
                    killcount_data.append(killcount)
                    itemcount_data.append(itemcount)
                    secretcount_data.append(secretcount)
                    fragcount_data.append(fragcount)
                    deathcount_data.append(deathcount)
                    hitcount_data.append(hitcount)
                    hits_taken_data.append(hits_taken)
                    damagecount_data.append(damagecount)
                    damage_taken_data.append(damage_taken)
                    health_data.append(health)
                    armor_data.append(armor)
                    dead_data.append(dead)
                    on_ground_data.append(on_ground)
                    attack_ready_data.append(attack_ready)
                    alattack_ready_data.append(alattack_ready)
                    selected_weapon_data.append(selected_weapon)
                    selected_weapon_ammo_data.append(selected_weapon_ammo)
                    ammo_data.append(ammo)
                    x_pos_data.append(x_pos)
                    y_pos_data.append(y_pos)
                    z_pos_data.append(z_pos)
                    angle_data.append(angle)
                    pitch_data.append(pitch)
                    roll_data.append(roll)
                    x_vel_data.append(x_vel)
                    y_vel_data.append(y_vel)
                    z_vel_data.append(z_vel)
                    cam_pos_x_data.append(cam_pos_x)
                    cam_pos_y_data.append(cam_pos_y)
                    cam_pos_z_data.append(cam_pos_z)
                    cam_angle_data.append(cam_angle)
                    cam_pitch_data.append(cam_pitch)
                    cam_roll_data.append(cam_roll)
                    cam_fov_data.append(cam_fov)

                    # the episode ends so no next state
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    # Get game variables
                    ammo = game.get_game_variable(GameVariable.AMMO2)
                    health = game.get_game_variable(GameVariable.HEALTH)
                    y_pos = game.get_game_variable(GameVariable.POSITION_Y)
                    x_pos = game.get_game_variable(GameVariable.POSITION_X)
                    x_vel = game.get_game_variable(GameVariable.VELOCITY_X)
                    y_vel = game.get_game_variable(GameVariable.VELOCITY_Y)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))

                    print("Ammo:", ammo,
                          "health:", health,
                          "y_pos:", y_pos,
                          "x_pos:", x_pos,
                          "step:", step)

                    memory.add((state, action, reward, next_state, done))

                else:
                    # Get the next state
                    next_state = game.get_state().screen_buffer

                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                   feed_dict={DQNetwork.inputs_: states_mb,
                                              DQNetwork.target_Q: targets_mb,
                                              DQNetwork.actions_: actions_mb})

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

"""
Watch our Agent play
"""
with tf.Session() as sess:
    game, possible_actions = create_environment()

    totalScore = 0

    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(1):

        done = False

        game.new_episode()

        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not game.is_episode_finished():
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]

            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()

            if done:
                break

            else:
                print("else")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state

        score = game.get_total_reward()
        print("Score: ", score)
    game.close()
