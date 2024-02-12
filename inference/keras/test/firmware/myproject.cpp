#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &fc1_input,
    hls::stream<result_t> &layer2_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=fc1_input,layer2_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<fc1_default_t, 3080>(w2, "w2.txt");
        nnet::load_weights_from_txt<fc1_default_t, 2>(b2, "b2.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    nnet::dense<input_t, result_t, config2>(fc1_input, layer2_out, w2, b2); // fc1

}
