#include <stdio.h>
#include <stdint.h> 
<ADDITIONAL_INCLUDES>

<ADDITIONAL_NAMESPACES>

#define N_INPUTS           <N_INPUTS> // number of inputs
#define N_ACTIVS_H1        <N_ACTIVS_H1> // number of neurons in hidden layer
#define N_OUTPUTS          <N_OUTPUTS> // number of outputs

#define N_WEIGHTS_H1       <N_WEIGHTS_H1>
#define N_PARAMETERS_H1    N_WEIGHTS_H1 + N_ACTIVS_H1 // maximum is (N_INPTS + 1) * N_ACTIVS_H1
#define N_PARAMETERS_OUT   (N_ACTIVS_H1 + 1) * N_OUTPUTS // set the maximum since readout layer is not pruned

#define N_PARAMETERS       N_PARAMETERS_H1 + N_PARAMETERS_OUT // total number of parameters

//#define DMEM_BASE          0
#define WAIT               {}

const uint16_t neuron_inputs[N_ACTIVS_H1] = {<NEURON_INPUTS>};
const uint16_t input_indices[] = {<INPUT_INDICES>};

// memory reserved for weights
const <WEIGHTS_CTYPE> w[] = {
    // hidden layer
    <WEIGHTS_H1>,
    // output layer
    <WEIGHTS_OUT>
};

// memory reserved for biases
const <BIAS_CTYPE> b[] = {
    // hidden layer
    <BIAS_H1>,
    // output layer
    <BIAS_OUT>
};

// memory reserved for activations
<ACTIVATIONS_CTYPE> a[N_INPUTS+N_ACTIVS_H1+N_OUTPUTS];

// NN inference code
void mlp_single_hidden_layer_pruned(
    float a[], 
    const uint16_t neuron_inputs[], 
    const uint16_t input_indices[], 
    const float w[], 
    const float b[]) {

    float a_tmp = 0; // holds current activation value

    // ------------------------------------------------------------------------
    // inference code starts HERE, DO NOT MODIFY
    // ------------------------------------------------------------------------
    /*
        HIDDEN LAYER COMPUTATIONS
    */
    // initialize array offsets
    size_t in_activation_ofst = 0;
    size_t out_activation_ofst = N_INPUTS;
    size_t weight_ofst = 0;
    size_t bias_ofst = 0;

    // compute hidden layer activations
    for (size_t i = 0; i < N_ACTIVS_H1; i++) // i is the hidden neuron index --> iterate for each hidden neuron
    {
        // dot products
        a_tmp = b[i+bias_ofst]; // initialize using the bias value
        for (size_t j = 0; j < neuron_inputs[i]; j++)
        {
            a_tmp += w[j+weight_ofst] * a[input_indices[j+weight_ofst]+in_activation_ofst];
        }
        // relu activations
        if (a_tmp > 0) {
            a[i+out_activation_ofst] = a_tmp;
        } else {
            a[i+out_activation_ofst] = 0;
        }
        // update weight offset
        weight_ofst += neuron_inputs[i];
    }

    /*
        OUTPUT LAYER COMPUTATIONS
    */
    // modify memory offsets
    in_activation_ofst = N_INPUTS;
    out_activation_ofst = N_INPUTS + N_ACTIVS_H1;
    weight_ofst = N_WEIGHTS_H1;
    bias_ofst = N_ACTIVS_H1;

    // compute outputs
    printf("C output:\n");
    for (size_t i = 0; i < N_OUTPUTS; i++)
    {
        // dot products
        a_tmp = b[i+bias_ofst]; // initialize using the bias value
        for (size_t j = 0; j < N_ACTIVS_H1; j++)
        {
            a_tmp += w[j+weight_ofst] * a[j+in_activation_ofst];
            //printf("%f ", w[j+weight_ofst]);
        }
        
        // no activation is applied to the output
        a[i+out_activation_ofst] = a_tmp;
        // update weight offset
        weight_ofst += N_ACTIVS_H1;

        printf("%f ", a[i+out_activation_ofst]);
        //printf("\n");
    }
    printf("\n");

    // ------------------------------------------------------------------------
    // end of inference code
    // ------------------------------------------------------------------------


}

int main() {

    printf("\nRunning test program for pruned MLP...\n\n");

    // some example input
    <ACTIVATIONS_CTYPE> x[] = {<EXAMPLE_INPUT_VALUES>};

    // replace this loop with your preprocessing code
    for (size_t i = 0; i < N_INPUTS; i++)
    {
        a[i] = x[i];
    }

    // NN inference
    mlp_single_hidden_layer_pruned(a, neuron_inputs, input_indices, w, b);
}