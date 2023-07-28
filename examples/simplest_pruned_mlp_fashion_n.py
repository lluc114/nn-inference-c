# %%
#MODEL_PATH = '../../tensor_flow_2/models/MLP_FASHION_300_100_pruned'
#MODEL_PATH = '../models/MLP_FashionMNIST_256_256_256_256_THP'
MODEL_PATH = '../models/MLP_FASHION_1568_pruned'

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

# such a long import might seem an inconvenient but this way it will be easier to
# extend the library in futer to include more NN code generators
from nn_inference.simplest_pruned_mlp_inference_n.builder.NetBuilder import PrunedMLPBuilder

# %% initialize the model builder
nb = PrunedMLPBuilder(MODEL_PATH, out_directory='../out')

model = tf.keras.models.load_model(MODEL_PATH)

# %% check and modify model if needed
nb.check_model()

# %% evaluate model on MNIST data sample
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
x = test_images[:1] / 255 # shape is 1, height, width
#x = x.reshape((1,-1))

# input_samples = (np.random.random_sample(10)*10000).astype(np.int16)
# samples = test_images[input_samples]
# samples = samples.reshape((10,1,784))
# samples = samples/255
# np.save('./fashion_10samples.npy', samples)

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
# expected_output = nb.model(x)
# print('TF output:')
# print(expected_output.numpy())
# %%
