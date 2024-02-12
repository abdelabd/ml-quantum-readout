#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 1540
#define N_LAYER_2 2

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<12,12>, 1540*1> input_t;
typedef ap_fixed<22,14> fc1_default_t;
typedef nnet::array<ap_fixed<22,14>, 2*1> result_t;
typedef ap_uint<1> layer2_index;

#endif
