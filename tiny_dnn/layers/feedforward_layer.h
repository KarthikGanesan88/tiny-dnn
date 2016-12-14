/*
    Copyright (c) 2016, Taiga Nomi
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
#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/activations/activation_function.h"

namespace tiny_dnn {

/**
 * single-input, single-output network with activation function
 **/
template<typename Activation>
class feedforward_layer : public layer {
public:
    explicit feedforward_layer(const std::vector<vector_type>& in_data_type)
        : layer(in_data_type, std_output_order(true)) {}
    activation::function& activation_function() { return h_; }
    std::pair<float_t, float_t> out_value_range() const override { return h_.scale(); }

public:
    void forward_activation(tensor_t& a_tensor, tensor_t& out_tensor) {

        cnn_size_t out_dim = out_shape()[0].size();

#ifdef PRINT_DEBUG
        std::cout<<"\n\nActivation function"<<std::endl;
#endif

#ifdef TIMING
        std::chrono::high_resolution_clock::time_point t1, t2;
        t1 = std::chrono::high_resolution_clock::now();
#endif

        //for_i(a_tensor.size(), [&](int sample) {
        for (auto sample = 0; sample < a_tensor.size(); sample++) {
            vec_t &out = a_tensor[sample];
            vec_t &a = out_tensor[sample];

            out.resize(out_dim);
            a.resize(out_dim);

#ifdef PRINT_DEBUG
            std::cout<<"Input :\n";
            for (cnn_size_t i = 0; i < out_dim; i++) {
                std::cout<< a[i] <<",";
            }
            std::cout<< "\nOutput :\n";
#endif
            /*std::cout<<std::endl<<"vec_t out (initial) :";

            for (cnn_size_t i = 0; i < out_dim; i++) {
                std::cout<< out[i] <<",";
            }*/

            for (cnn_size_t i = 0; i < out_dim; i+=anytime_param_) {
                out[i] = this->h_.f(a, i);
#ifdef PRINT_DEBUG
                std::cout<<out[i]<<",";
#endif
            }
#ifdef PRINT_DEBUG
            std::cout<<std::endl;
#endif
        }
        //});

#ifdef TIMING
        t2 = std::chrono::high_resolution_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        std::cout << std::setw(30) << "Time in activation function : " << timeElapsed << std::endl;
#endif
    }

    void backward_activation(const tensor_t& prev_delta, const tensor_t& this_out, tensor_t& curr_delta) {

        // @todo consider parallelism
        for_i(this_out.size(), [&](cnn_size_t sample) {
            const vec_t& out_vec = this_out[sample];
            const vec_t& prev_delta_vec = prev_delta[sample];
            vec_t& curr_delta_vec = curr_delta[sample];

            const cnn_size_t len = prev_delta_vec.size();
            
            if (h_.one_hot()) {
                for (cnn_size_t c = 0; c < len; c++) {
                    curr_delta_vec[c] = prev_delta_vec[c] * h_.df(out_vec[c]);
                }
            }
            else {
                for (cnn_size_t c = 0; c < len; c++) {
                    vec_t df = h_.df(out_vec, c);
                    curr_delta_vec[c] = vectorize::dot(&prev_delta_vec[0], &df[0], len);
                }
            }
        });
    }

    virtual void on_set_anytime_param() {}

    void set_anytime_param(int anytime_param) override{
        anytime_param_ = anytime_param;
        on_set_anytime_param();
    }

    int get_anytime_param() {
        return anytime_param_;
    }

    Activation h_;
    int anytime_param_ = 1;
};

} // namespace tiny_dnn
