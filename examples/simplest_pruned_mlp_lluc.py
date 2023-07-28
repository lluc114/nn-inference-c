# %%
MODEL_PATH = '../models/MLP_FASHION_1568_pruned'

import os
import sys
import numpy as np
import tensorflow as tf
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

# such a long import might seem an inconvenient but this way it will be easier to
# extend the library in futer to include more NN code generators
from nn_inference.simplest_pruned_mlp_inference.builder.NetBuilder import PrunedMLPBuilder

# %% initialize the model builder
nb = PrunedMLPBuilder(MODEL_PATH, out_directory='../out')

# %% check and modify model if needed
nb.check_model()

# %% evaluate model on MNIST data sample
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
x = test_images[:1] / 255 # shape is 1, height, width

# %% inference using the example data, this is the golden reference for the generated code
expected_output = nb.model(x)

# %% generate the code
assert type(x) == type(np.array([]))
nb.generate(x)

# %% compile the example code
nb.compile()

# %% run the example code
nb._execute('../out/main')
print('TF output:')
print(expected_output.numpy())

# %%
