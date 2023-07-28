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
            template_file='mlp_single_hidden_layer_pruned.cpp',
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
        super().__init__(model_path, model_format, template_file='mlp_single_hidden_layer_pruned.cpp', out_directory=out_directory, compiler=compiler)

    def _extract_model_parameters(self):
        parameters = [param.T for param in self.model.get_weights()]
        return parameters
    
    def _skip_zeros_and_flatten(self, w):
        cond = np.abs(w) > 1e-6
        neuron_inputs = cond.sum(1)
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
        
        print('Extracting model parameters...', end='')
        parameters = self._extract_model_parameters()
        print(' OK')

        # convert to float16 if needed
        if half_precision:
            print('Converting model to float16...', end='')
            for i, p in enumerate(parameters):
                parameters[i] = p.astype(np.float16)
            print(' OK')
            print('Converting example input to float16...', end='')
            x = x.astype(np.float16)
            print(' OK')

        print('Formatting parameters...', end='')
        n_inputs = parameters[0].shape[1]
        n_activs_h1 = parameters[0].shape[0]
        n_outputs = parameters[2].shape[0]
        neuron_inputs, parameters[0], wi0_pruned = self._skip_zeros_and_flatten(parameters[0])
        wi0_pruned = self._array2str(wi0_pruned)
        w0_pruned, b0, w1, b1 = [self._array2str(array) for array in parameters]
        n_weights_h1 = neuron_inputs.sum()
        print(' OK')

        # TODO: use header file for weights
        print('Generating code...', end='')
        self._replace('<N_INPUTS>', str(n_inputs))
        self._replace('<N_ACTIVS_H1>', str(n_activs_h1))
        self._replace('<N_OUTPUTS>', str(n_outputs))
        self._replace('<N_WEIGHTS_H1>', str(n_weights_h1))
        self._replace('<NEURON_INPUTS>', self._array2str(neuron_inputs))
        self._replace('<INPUT_INDICES>', wi0_pruned)
        self._replace('<WEIGHTS_H1>', w0_pruned)
        self._replace('<BIAS_H1>', b0)
        self._replace('<WEIGHTS_OUT>', w1)
        self._replace('<BIAS_OUT>', b1)
        self._replace('<EXAMPLE_INPUT_VALUES>', self._array2str(x))

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

        
        