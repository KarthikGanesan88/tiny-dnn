/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {

inline void
fully_connected_op_custom(const tensor_t&     in_data,
                          const vec_t&        W,
                          const vec_t&        bias,
                          tensor_t&           out_data,
                          const fully_params& params,
                          const bool          layer_parallelize) {

#ifdef TIMING
    std::chrono::high_resolution_clock::time_point t1,t2;
    t1 = std::chrono::high_resolution_clock::now();
#endif

#ifdef PRINT_DEBUG
    std::cout << "Starting FC layer: (in size =  "<<params.in_size_<<", out size= "<<params.out_size_<<") : \n";
#endif

    //for_i(layer_parallelize, in_data.size(), [&](int sample) {
    for (auto sample = 0; sample < in_data.size(); sample++) {
        const vec_t &in = in_data[sample];
        vec_t &out = out_data[sample];

#ifdef PRINT_DEBUG
        std::cout << "Inputs :\n";
        for (cnn_size_t c = 0; c < params.in_size_; c++) {
            std::cout << in[c] <<",";
        }
        std::cout << std::endl;

        std::cout << "Outputs :\n";
#endif

        for (cnn_size_t i = 0; i < params.out_size_; i++) {
            out[i] = float_t(0);
            if ((i%params.skip_nodes_) == 0) {
                for (cnn_size_t c = 0; c < params.in_size_; c++) {
                    out[i] += W[c * params.out_size_ + i] * in[c];
                }
                if (params.has_bias_) {
                    out[i] += bias[i];
                }
            }
#ifdef PRINT_DEBUG
            std::cout << out[i] << ",";
#endif
        }
#ifdef PRINT_DEBUG
        std::cout << std::endl;
#endif
    }
    //});

#ifdef TIMING
    t2 = std::chrono::high_resolution_clock::now();
    auto timeElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    std::cout << std::setw(30) << "Time in FC layer : " << timeElapsed << std::endl;
#endif
}

inline void
fully_connected_op_custom(const tensor_t& prev_out,
                          const vec_t&    W,
                          tensor_t&       dW,
                          tensor_t&       db,
                          tensor_t&       curr_delta,
                          tensor_t&       prev_delta,
                          const fully_params& params,
                          const bool      layer_parallelize) {
    for (cnn_size_t sample = 0; sample < prev_out.size(); sample++) {
        for (cnn_size_t c = 0; c < params.in_size_; c++) {
            // propagate delta to previous layer
            // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
            prev_delta[sample][c] += vectorize::dot(&curr_delta[sample][0],
                &W[c * params.out_size_],
                params.out_size_);
        }

        for_(layer_parallelize, 0, size_t(params.out_size_), [&](const blocked_range& r) {
            // accumulate weight-step using delta
            // dW[c * out_size + i] += current_delta[i] * prev_out[c]
            for (cnn_size_t c = 0; c < params.in_size_; c++) {
                vectorize::muladd(&curr_delta[sample][r.begin()],
                    prev_out[sample][c], r.end() - r.begin(),
                    &dW[sample][c * params.out_size_ + r.begin()]);
            }

            if (params.has_bias_) {
                // vec_t& db = *in_grad[2];
                for (int i = r.begin(); i < r.end(); i++) {
                    db[sample][i] += curr_delta[sample][i];
                }
            }
        });
    }
}

}  // namespace kernels
}  // namespace tiny_dnn
