#include <stdio.h>
#include <stdint.h> 
<ADDITIONAL_INCLUDES>

<ADDITIONAL_NAMESPACES>

#define N_INPUTS           <N_INPUTS> // number of inputs
#define N_ACTIVS_H1        <N_ACTIVS_H1> // number of neurons in hidden layer 1
#define N_ACTIVS_H2        <N_ACTIVS_H2> // number of neurons in hidden layer 2
#define N_OUTPUTS          <N_OUTPUTS> // number of outputs

#define N_WEIGHTS_H1       <N_WEIGHTS_H1>
#define N_WEIGHTS_H2       <N_WEIGHTS_H2>
#define N_PARAMETERS_H1    N_WEIGHTS_H1 + N_ACTIVS_H1 // maximum is (N_INPTS + 1) * N_ACTIVS_H1
#define N_PARAMETERS_H2    N_WEIGHTS_H2 + N_ACTIVS_H2 
#define N_PARAMETERS_OUT   (N_ACTIVS_H2 + 1) * N_OUTPUTS // set the maximum since readout layer is not pruned

#define N_PARAMETERS       N_PARAMETERS_H1 + N_PARAMETERS_H2 + N_PARAMETERS_OUT // total number of parameters

#define USE_BIAS_H1        <USE_BIAS_H1>
#define USE_BIAS_H2        <USE_BIAS_H2>
#define USE_BIAS_OUT       <USE_BIAS_OUT>

//#define DMEM_BASE          0
#define WAIT               {}

const uint16_t neuron_inputs[N_ACTIVS_H1 + N_ACTIVS_H2] = {<NEURON_INPUTS>};
const uint16_t input_indices[] = {<INPUT_INDICES>};

// memory reserved for weights
const <WEIGHTS_CTYPE> w[] = {
    // hidden layer 1
    <WEIGHTS_H1>,
    //hidden layer 2
    <WEIGHTS_H2>,
    // output layer
    <WEIGHTS_OUT>
};

// memory reserved for biases
const <BIAS_CTYPE> b[] = {
    // hidden layer 1
    #if USE_BIAS_H1 == 1
        <BIAS_H1>,
    #endif
    // hidden layer 2
    #if USE_BIAS_H2 == 1
        <BIAS_H2>,
    #endif
    // output layer
    #if USE_BIAS_OUT == 1
        <BIAS_OUT>
    #endif
};

// memory reserved for activations
<ACTIVATIONS_CTYPE> a[N_INPUTS+N_ACTIVS_H1+N_ACTIVS_H2+N_OUTPUTS];

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
    size_t activation_ofst = 0;
    size_t out_activation_ofst = N_INPUTS;
    size_t weight_ofst = 0;
    size_t bias_ofst = 0;
    size_t n_inputs_ofst = 0;

    // compute hidden layer 1 activations
    for (size_t i = 0; i < N_ACTIVS_H1; i++) // i is the hidden neuron index --> iterate for each hidden neuron
    {
        // dot products
        #if USE_BIAS_H1 == 1
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        #else
            a_tmp = 0;
        #endif

        for (size_t j = 0; j < neuron_inputs[i+n_inputs_ofst]; j++)
        {
            a_tmp += w[j+weight_ofst] * a[input_indices[j+weight_ofst]+activation_ofst];
        }
        // relu activations
        if (a_tmp > 0) {
            a[i+out_activation_ofst] = a_tmp;
        } else {
            a[i+out_activation_ofst] = 0;
        }
        // update weight offset
        weight_ofst += neuron_inputs[i+n_inputs_ofst];
    }

    // compute hidden layer 2 activations
    activation_ofst = N_INPUTS;
    bias_ofst = N_ACTIVS_H1*USE_BIAS_H1;
    out_activation_ofst = N_INPUTS + N_ACTIVS_H1;
    weight_ofst = N_WEIGHTS_H1;
    n_inputs_ofst = N_ACTIVS_H1;

    for (size_t i = 0; i < N_ACTIVS_H2; i++) // i is the hidden neuron index --> iterate for each hidden neuron
    {
        // dot products
        #if USE_BIAS_H2 == 1
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        #else
            a_tmp = 0;
        #endif

        for (size_t j = 0; j < neuron_inputs[i+n_inputs_ofst]; j++)
        {
            a_tmp += w[j+weight_ofst] * a[input_indices[j+weight_ofst]+activation_ofst];
        }
        // relu activations
        if (a_tmp > 0) {
            a[i+out_activation_ofst] = a_tmp;
        } else {
            a[i+out_activation_ofst] = 0;
        }
        // update weight offset
        weight_ofst += neuron_inputs[i+n_inputs_ofst];
    }


    /*
        OUTPUT LAYER COMPUTATIONS
    */
    // modify memory offsets
    activation_ofst = N_INPUTS + N_ACTIVS_H1;
    out_activation_ofst = N_INPUTS + N_ACTIVS_H1 + N_ACTIVS_H2;
    weight_ofst = N_WEIGHTS_H1 + N_WEIGHTS_H2;
    bias_ofst = N_ACTIVS_H1*USE_BIAS_H1 + N_ACTIVS_H2*USE_BIAS_H2;

    // compute outputs
    printf("C output:\n");
    for (size_t i = 0; i < N_OUTPUTS; i++)
    {
        // dot products
        #if USE_BIAS_OUT == 1
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        #else
            a_tmp = 0;
        #endif

        for (size_t j = 0; j < N_ACTIVS_H2; j++)
        {
            a_tmp += w[j+weight_ofst] * a[j+activation_ofst];
            //printf("%f ", w[j+weight_ofst]);
        }
        
        // no activation is applied to the output
        a[i+out_activation_ofst] = a_tmp;
        // update weight offset
        weight_ofst += N_ACTIVS_H2;

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