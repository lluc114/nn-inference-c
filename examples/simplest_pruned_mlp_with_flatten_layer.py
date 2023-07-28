# %%
INPUT_HEIGHT = 28
INPUT_WIDTH = 28
NUM_HIDDEN_UNITS = 1568
NUM_OUTPUTS = 10
PRUNING = 0.95 # e.g. 0.95 is 95% pruning in the hidden layer

import os
import sys
import numpy as np
import tensorflow as tf
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

# such a long import might seem an inconvenient but this way it will be easier to
# extend the library in futer to include more NN code generators
from nn_inference.simplest_pruned_mlp_inference.builder.NetBuilder import PrunedMLPBuilder

tf.random.set_seed(2)

# %% define an example model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(NUM_HIDDEN_UNITS, activation='relu')) # currently, only relu activation is supported
model.add(tf.keras.layers.Dense(10))

# %% evaluate model on random data so that the model knows the input dimensions
x = tf.random.normal(shape=[1, INPUT_HEIGHT, INPUT_WIDTH])
out = model(x)
model.summary()

# %% put some weights to zero to test if generated code is taking it into account
parameters = model.get_weights()
example_pruning_mask = (np.random.permutation(parameters[0].size) / parameters[0].size) > (1 - PRUNING) # 0.05 corresponds to 95% weight pruning in the hidden layer
example_pruning_mask = example_pruning_mask.reshape(parameters[0].shape)
parameters[0][example_pruning_mask] = 0
parameters[1] = 10 * tf.random.normal(shape=parameters[1].shape)
parameters[3] = 10 * tf.random.normal(shape=parameters[3].shape)
model.set_weights(parameters)
model.compile()

# %% inference using the example data
expected_output = model(x)

# %% save the model
model_path = os.path.join(parent_dir, '../out', 'model')
#tf.saved_model.save(model, model_path)
model.save(model_path)

# %% initialize the model builder
nb = PrunedMLPBuilder(model_path, out_directory='../out')

# %% check and modify model if needed
nb.check_model()

# %% generate the code
nb.generate(x.numpy())

# %% compile the example code
nb.compile()

# %% run the example code
nb._execute('../out/main')
print('TF output:')
print(expected_output.numpy())

# %%
