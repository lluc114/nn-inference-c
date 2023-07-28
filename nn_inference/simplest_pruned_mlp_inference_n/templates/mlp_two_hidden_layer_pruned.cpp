#include <stdio.h>
#include <stdint.h> 
<ADDITIONAL_INCLUDES>

//<ADDITIONAL_NAMESPACES>

#define N_INPUTS           <N_INPUTS> // number of inputs
// #define N_ACTIVS_H1        <N_ACTIVS_H1> // number of neurons in hidden layer 1
// #define N_ACTIVS_H2        <N_ACTIVS_H2> // number of neurons in hidden layer 2
#define N_ACTIVS_H_TOTAL   <N_ACTIVS_H_TOTAL>
#define N_OUTPUTS          <N_OUTPUTS> // number of outputs

// #define N_WEIGHTS_H1       <N_WEIGHTS_H1>
// #define N_WEIGHTS_H2       <N_WEIGHTS_H2>
// #define N_PARAMETERS_H1    N_WEIGHTS_H1 + N_ACTIVS_H1 // maximum is (N_INPTS + 1) * N_ACTIVS_H1
// #define N_PARAMETERS_H2    N_WEIGHTS_H2 + N_ACTIVS_H2 
// #define N_PARAMETERS_OUT   (N_ACTIVS_H2 + 1) * N_OUTPUTS // set the maximum since readout layer is not pruned

// #define N_PARAMETERS       N_PARAMETERS_H1 + N_PARAMETERS_H2 + N_PARAMETERS_OUT // total number of parameters

// #define USE_BIAS_H1        <USE_BIAS_H1>
// #define USE_BIAS_H2        <USE_BIAS_H2>
// #define USE_BIAS_OUT       <USE_BIAS_OUT>

#define N_HIDDENS          <N_HIDDENS>

//#define DMEM_BASE          0
#define WAIT               {}

const uint16_t neuron_inputs[] = {<NEURON_INPUTS>};
const uint16_t input_indices[] = {<INPUT_INDICES>};

const uint16_t n_weights_H[] {<N_WEIGHTS_H>};
const bool use_bias_H[] {<USE_BIAS_H>};
const uint16_t n_activs_H[] {<N_ACTIVS_H>};

// memory reserved for weights
const <WEIGHTS_CTYPE> w[] = {
    <WEIGHTS_H>
};

// memory reserved for biases
const <BIAS_CTYPE> b[] = {
    <BIAS_H>
};

// memory reserved for activations
<ACTIVATIONS_CTYPE> a[N_INPUTS+N_ACTIVS_H_TOTAL+N_OUTPUTS];

void compute_hidden(
    size_t activation_ofst,
    size_t bias_ofst,
    size_t out_activation_ofst,
    size_t weight_ofst,
    size_t n_inputs_ofst,
    uint8_t n_H
){
    float a_tmp = 0; // holds current activation value
    for (size_t i = 0; i < n_activs_H[n_H]; i++) // i is the hidden neuron index --> iterate for each hidden neuron
    {
        // dot products
        if (use_bias_H[n_H]){
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        }else{
            a_tmp = 0;
        }

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
}

// NN inference code
void mlp_single_hidden_layer_pruned() {

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
    for (size_t i = 0; i < n_activs_H[0]; i++) // i is the hidden neuron index --> iterate for each hidden neuron
    {
        // dot products
        if(use_bias_H[0]){
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        }else{
            a_tmp = 0;
        }
            
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
     
    // Hidden layers
    //out_activation_ofst = N_INPUTS;
    activation_ofst = N_INPUTS;
    out_activation_ofst = N_INPUTS + n_activs_H[0];
    bias_ofst = n_activs_H[0]*use_bias_H[0];
    n_inputs_ofst  = n_activs_H[0];
    for (uint8_t n_H = 1; n_H < N_HIDDENS - 1; n_H++)
    {
        // compute hidden layer activations
        // activation_ofst = N_INPUTS;
        // bias_ofst = N_ACTIVS_H1*USE_BIAS_H1;
        // out_activation_ofst = N_INPUTS + N_ACTIVS_H1;
        // weight_ofst = N_WEIGHTS_H1;
        // n_inputs_ofst = N_ACTIVS_H1;

        compute_hidden(activation_ofst, bias_ofst, out_activation_ofst, weight_ofst, n_inputs_ofst, n_H);
                
        activation_ofst += n_activs_H[n_H - 1];
        out_activation_ofst += n_activs_H[n_H];
        bias_ofst += n_activs_H[n_H]*use_bias_H[n_H];
        n_inputs_ofst  += n_activs_H[n_H];
        weight_ofst += n_weights_H[n_H];
    }

    /*
        OUTPUT LAYER COMPUTATIONS
    */
    // modify memory offsets
    // activation_ofst = N_INPUTS + N_ACTIVS_H1;
    // out_activation_ofst = N_INPUTS + N_ACTIVS_H1 + N_ACTIVS_H2;
    // weight_ofst = N_WEIGHTS_H1 + N_WEIGHTS_H2;
    // bias_ofst = N_ACTIVS_H1*USE_BIAS_H1 + N_ACTIVS_H2*USE_BIAS_H2;

    // compute outputs
    printf("C output:\n");
    for (size_t i = 0; i < N_OUTPUTS; i++)
    {
        // dot products
        if(use_bias_H[N_HIDDENS - 1] == 1){
            a_tmp = b[i+bias_ofst]; // initialize using the bias value
        }else{
            a_tmp = 0;
        }

        for (size_t j = 0; j < n_activs_H[N_HIDDENS - 2]; j++)
        {
            a_tmp += w[j+weight_ofst] * a[j+activation_ofst];
        }
        
        // no activation is applied to the output
        a[i+out_activation_ofst] = a_tmp;
        // update weight offset
        weight_ofst += n_activs_H[N_HIDDENS - 2];

        printf("%f ", a[i+out_activation_ofst]);
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
    mlp_single_hidden_layer_pruned();
}