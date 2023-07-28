import os
import shutil
import subprocess
#import tempfile

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
THIRD_PARTY_DIR = os.path.join(BASE_DIR, 'third_party')
#OUTPUT_DIR = os.path.join(BASE_DIR, 'out')

class NetBuilder:
    """Base network builder class including basic methods
        
    """
    def __init__(self, model_path, 
            model_format='saved_model', 
            template_file='mlp_hidden_layer_pruned.cpp',
            out_directory='out',
            compiler='gcc'):
        """Initialize basic settings
            
        Args:
            model_path (str): path to NN model directory/file(s)
            model_format (str, optional): Format of the model stored in 'model_path'. Defaults to 'saved_model'
            template_file (str, optional): C/C++ template file name. Defaults to 'mlp_single_hidden_layer_pruned.cpp'
            out_directory (str, optional): Output directory
            compiler (str, optional): Currently, only 'gcc' is supported and tested. Defaults to 'gcc'
        """
        
        self.model_path = model_path
        self.model_format = model_format
        self.template_path = os.path.join(TEMPLATE_DIR, template_file)
        self.out_directory = out_directory
        self.compiler = compiler

        print('Loading model...', end='')
        self._load_model()
        print(' OK')

        print('Loading code template...', end='')
        self._load_template()
        print(' OK')

    def _load_model(self, model_path=None):
        # override self.model_path when an different model_path is required
        if model_path is None:
            model_path = self.model_path

        # load the model
        if self.model_format == 'saved_model':
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            raise NotImplementedError(f'The specified model format ({self.model_format}) is not supported')
        
    def _load_template(self):
        self.template = open(self.template_path, 'r').read()

    def _copy_dependencies(self, dependencies):
        for dep in dependencies:
            src = os.path.join(THIRD_PARTY_DIR, dep)
            shutil.copy(src, self.out_directory)
        
    def _dump_code(self):
        f = open(os.path.join(self.out_directory, 'main.cpp'), 'w')
        f.write(self.template)
        f.close()

    def _execute(self, command):
        """Executes a command as a subprocess

        Args:
            command (list): command represented by a list of strings
        """
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while p.poll() is None:
            l = p.stdout.readline() # blocks until new line is received
            print(l.decode('ascii'))
        last_line = p.stdout.read()
        if len(last_line) > 0:
            print(last_line.decode('ascii'))


class PrunedMLPBuilder(NetBuilder):
    """Extends the NetBuilder base class with additional methods for building code and compiling software

    Args:
        NetBuilder: Parent class including basic methods
    """
    def __init__(self, model_path, model_format='saved_model', out_directory='out', compiler='gcc'):
        """Initialize basic settings

        Args:
            model_path (str): path to NN model directory/file(s)
            model_format (str, optional): Format of the model stored in 'model_path'. Defaults to 'saved_model'. Defaults to 'saved_model'
            out_directory (str, optional): Output directory
        """
        super().__init__(model_path, model_format, template_file='mlp_hidden_layer_pruned.cpp', out_directory=out_directory, compiler=compiler)

    def _extract_model_parameters(self):
        #parameters = [param.T for param in self.model.get_weights()]

        parameters = [None]*6
        n = 0
        for layer in self.model.layers:
            layer_params = layer.get_weights()
            if len(layer_params) > 1: # use bias
                parameters[n] = layer_params[0].T
                parameters[n + 1] = layer_params[1].T
            else: # no bias
                parameters[n] = layer_params[0].T
                parameters[n + 1] = np.zeros(1)
            n = n + 2

        return parameters
    
    def _skip_zeros_and_flatten(self, w):
        cond = np.abs(w) > 1e-6
        neuron_inputs = cond.sum(axis=1)
        wi = np.zeros(shape=w.shape, dtype=int)
        wi = wi + np.arange(wi.shape[1])
        w_pruned = w[cond]
        wi_pruned = wi[cond]
        return neuron_inputs, w_pruned, wi_pruned
        
    def _array2str(self, array, delimiter=','):
        np.set_printoptions(threshold=np.inf)

        # define supported types
        integer_types = [np.uint8, np.uint16, np.uint32, np.uint64,
                         np.int8,  np.int16,  np.int32,  np.int64]
        float_types = [np.float16, np.float32, np.float64]

        # vectorize hex function
        hexv = np.vectorize(lambda t: t.hex())

        # modify the array as needed
        array_mod = array.reshape(-1)
        dtype = type(array_mod[0])
        if dtype in integer_types:
            pass
        elif dtype in float_types:
            array_mod = hexv(array_mod.astype(float))
        else:
            raise NotImplementedError
        
        # convert to 1d C array format, not including {}
        array_str = str(array_mod)[1:-1]
        array_str = ' '.join(array_str.split())
        array_str = array_str.replace(' ', delimiter)
        array_str = array_str.replace('\'', '')

        # remove unnecessary zeros depending on the data format
        if dtype == np.float64:
            pass
        elif dtype == np.float32:
            array_str = array_str.replace('0000000p', 'p')
        elif dtype == np.float16:
            array_str = array_str.replace('0000000000p', 'p') 


        np.set_printoptions(threshold=None)
        return array_str
    
    def _replace(self, a, b):
        self.template = self.template.replace(a, b)
        
    def check_model(self):
        readout_activation = self.model.layers[-1].activation
        something_changed = False

        if len(self.model.layers) < 3:
            print ("The model does not have 2 layers, excluding output layer!")
            raise NotImplementedError

        if readout_activation is not None:
            print(f'WARNING: Found unsupported readout activation function: {readout_activation}.')
            print("         This activation function has been changed to 'linear'")
            self.model.layers[-1].activation = None
            something_changed = True
            
        if something_changed:
            print('Applying changes to the model...', end='')
            self.model.compile(loss='categorical_crossentropy')
            print(' OK')
            
        
    def generate(self, x, half_precision=True, disable_prints=False):

        print('Copying third party dependencies...', end='')
        if half_precision:
            self._copy_dependencies(['half-2.2.0/include/half.hpp'])
        print(' OK')
        
        # print('Extracting model parameters...', end='')
        # parameters = self._extract_model_parameters()
        # print(' OK')

        # convert to float16 if needed
        # if half_precision:
        #     print('Converting model to float16...', end='')
        #     for i, p in enumerate(parameters):
        #         parameters[i] = p.astype(np.float16)
        #     print(' OK')
        #     print('Converting example input to float16...', end='')
        #     x = x.astype(np.float16)
        #     print(' OK')

        print('Formatting parameters...', end='')
        # n_inputs = parameters[0].shape[1]
        # n_activs_h1 = parameters[0].shape[0]
        # n_activs_h2 = parameters[2].shape[0]
        # n_outputs = parameters[4].shape[0]
        # neuron_inputs, parameters[0], wi0_pruned = self._skip_zeros_and_flatten(parameters[0])
        # n_weights_h1 = neuron_inputs.sum()
        # _neuron_inputs, parameters[2], wi1_pruned = self._skip_zeros_and_flatten(parameters[2])
        # n_weights_h2 = _neuron_inputs.sum()
        # neuron_inputs = np.concatenate([neuron_inputs, _neuron_inputs])
        # #wi1_pruned = wi1_pruned + n_inputs
        # wi_pruned = np.concatenate([wi0_pruned, wi1_pruned])
        # wi_pruned = self._array2str(wi_pruned)
        # w0_pruned, b0, w1_pruned, b1, w2, b2 = [self._array2str(array) for array in parameters]
        # #n_weights_h1 = neuron_inputs.sum()

        # n_inputs = parameters[0].shape[1]
        # n_outputs = parameters[-1].shape[0]

        n_layers = len(self.model.layers)
        
        if self.model.get_config()['layers'][1]['class_name'] == 'Flatten': # If flatten layer
           n_inputs = self.model.layers[1].get_weights()[0].shape[0]
           n_hiddens = n_layers - 1
        else:
            n_inputs = self.model.layers[0].get_weights()[0].shape[0]
            n_hiddens = n_layers
        
        n_outputs = self.model.layers[-1].get_weights()[0].shape[1]
        n_activs_h = np.zeros(n_hiddens, dtype=np.uint16)
        use_bias_h = np.zeros(n_hiddens, dtype=np.uint8)
        n_weights_h = np.zeros(n_hiddens, dtype=np.uint16)

        b = np.array([])
        n_activs_h_total = int(0)

        firstLayer = False
        n = 0
        for i, layer in enumerate(self.model.layers):
            if self.model.get_config()['layers'][i+1]['class_name'] != 'Dense': #Only dense layers are allowed
                firstLayer = True
                if i > 0:
                    raise NotImplementedError
                continue

            use_bias_h[n] = layer.get_config()['use_bias']
            n_activs_h[n] = layer.get_config()['units']
            n_activs_h_total += int(n_activs_h[n])

            layer_w = layer.get_weights()
            
            if (n == n_layers - 1): #Output layer
                _w = layer_w[0].T
            else:
                _neuron_inputs, _w, _wi_pruned = self._skip_zeros_and_flatten(layer_w[0].T)
                n_weights_h[n] = _neuron_inputs.sum()

            if use_bias_h[n]:
                _b = (layer_w[1].T).reshape(-1)
                b = np.concatenate([b, _b])

            if firstLayer:
                neuron_inputs = _neuron_inputs
                wi_pruned = _wi_pruned
                w_pruned = _w.reshape(-1)
                firstLayer = False
            else:
                neuron_inputs = np.concatenate([neuron_inputs, _neuron_inputs])
                wi_pruned = np.concatenate([wi_pruned, _wi_pruned.reshape(-1)])
                w_pruned = np.concatenate([w_pruned, _w.reshape(-1)])
            
            n = n + 1


        # neuron_inputs, parameters[0], wi_pruned = self._skip_zeros_and_flatten(parameters[0])
        # w_pruned = parameters[0]
        # b = parameters[1]
        # n_weights_h[0] = neuron_inputs.sum()

        # for i in range(len(parameters)):
        #     if i < 2:
        #         continue
        #     if i % 2 == 0:
        #         _neuron_inputs, parameters[i], _wi_pruned = self._skip_zeros_and_flatten(parameters[i])
        #         neuron_inputs = np.concatenate([neuron_inputs, _neuron_inputs])
        #         wi_pruned = np.concatenate([wi_pruned, _wi_pruned])
        #         w_pruned = np.concatenate([w_pruned, parameters[i]])
        #         n_weights_h
        #     else:
        #         b = np.concatenate([b, parameters[i]])

        print(' OK')

        # TODO: use header file for weights
        print('Generating code...', end='')
        self._replace('<N_INPUTS>', str(n_inputs))
        self._replace('<N_ACTIVS_H_TOTAL>', str(n_activs_h_total))
        self._replace('<N_HIDDENS>', str(n_hiddens))
        # self._replace('<N_ACTIVS_H1>', str(n_activs_h1))
        # self._replace('<N_ACTIVS_H2>', str(n_activs_h2))
        self._replace('<N_OUTPUTS>', str(n_outputs))
        # self._replace('<N_WEIGHTS_H1>', str(n_weights_h1))
        # self._replace('<N_WEIGHTS_H2>', str(n_weights_h2))
        self._replace('<NEURON_INPUTS>', self._array2str(neuron_inputs))
        self._replace('<INPUT_INDICES>', self._array2str(wi_pruned))
        # self._replace('<WEIGHTS_H1>', w0_pruned)
        # self._replace('<WEIGHTS_H2>', w1_pruned)
        # self._replace('<BIAS_H1>', b0)
        # self._replace('<BIAS_H2>', b1)
        # self._replace('<WEIGHTS_OUT>', w2)
        # self._replace('<BIAS_OUT>', b2)
        self._replace('<EXAMPLE_INPUT_VALUES>', self._array2str(x))

        self._replace('<N_WEIGHTS_H>', self._array2str(n_weights_h))
        self._replace('<USE_BIAS_H>', self._array2str(use_bias_h))
        self._replace('<N_ACTIVS_H>', self._array2str(n_activs_h))
        self._replace('<WEIGHTS_H>', self._array2str(w_pruned))
        self._replace('<BIAS_H>', self._array2str(b))


        # if self.model.layers[0].get_config()['use_bias']:
        #     self._replace('<USE_BIAS_H1>', '1')
        # else:
        #     self._replace('<USE_BIAS_H1>', '0')
        
        # if self.model.layers[1].get_config()['use_bias']:
        #     self._replace('<USE_BIAS_H2>', '1')
        # else:
        #     self._replace('<USE_BIAS_H2>', '0')
        
        # if self.model.layers[2].get_config()['use_bias']:
        #     self._replace('<USE_BIAS_OUT>', '1')
        # else:
        #     self._replace('<USE_BIAS_OUT>', '0')

        if half_precision:
            weights_ctype = 'float'
            bias_ctype = 'float'
            activations_ctype = 'float'
            additional_includes = '#include "half.hpp"'
            additional_namespaces = 'using half_float::half;'
        else:
            weights_ctype = 'float'
            bias_ctype = 'float'
            activations_ctype = 'float'
            additional_includes = ''
            additional_namespaces = ''

        self._replace('<WEIGHTS_CTYPE>', weights_ctype)
        self._replace('<BIAS_CTYPE>', bias_ctype)
        self._replace('<ACTIVATIONS_CTYPE>', activations_ctype)
        self._replace('<ADDITIONAL_INCLUDES>', additional_includes)
        self._replace('<ADDITIONAL_NAMESPACES>', additional_namespaces)

        if disable_prints:
            self._replace('printf(', '//printf(')
        self._dump_code()
        print(' OK')

    def compile(self, optimization_level=3):
        print('Compiling...')
        command = f"{self.compiler} -O{optimization_level} {os.path.join(self.out_directory, 'main.cpp')} -o {os.path.join(self.out_directory, 'main')}"
        print(command)
        self._execute(command)
        print('')

        
        