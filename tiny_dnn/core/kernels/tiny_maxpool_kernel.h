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

#include <chrono>
#include <thread>

namespace tiny_dnn {
namespace core {
namespace kernels {

inline void tiny_maxpool_kernel(const tensor_t& in_data,
                                tensor_t&       out_data,
                                std::vector<std::vector<cnn_size_t>>& max_idx,
                                const std::vector<std::vector<cnn_size_t>>& out2in,
                                const bool layer_parallelize,
								cnn_size_t stride_adjust = 1
 			      				) {
    
    //std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
    //std::chrono::high_resolution_clock::time_point t1,t2;
  
    for_i(layer_parallelize, in_data.size(), [&](int sample) {
         
		const vec_t& in = in_data[sample];
			vec_t& a = out_data[sample];
			std::vector<cnn_size_t>& max = max_idx[sample];

		//std::cout << "out2in:" << out2in.size() << std::endl;

		cnn_size_t inc = stride_adjust;

		float_t max_float_value = std::numeric_limits<float_t>::lowest();

		int loop_count;

		//t1 = std::chrono::high_resolution_clock::now();

		//for (loop_count=0; loop_count<10; loop_count++){
		  for (cnn_size_t i = 0; i < out2in.size(); i+=inc) {

			  const auto& in_index = out2in[i];
			  float_t max_value = max_float_value;

			  //std::cout << i << ",";
			  // in_index is always pooling_size^2 in size.

			  for (auto j : in_index) {
			  if (in[j] > max_value) {
				  max_value = in[j];
				  max[i] = j;
			  }
			  }
			  a[i] = max_value;

		  }
		  //std::cout << std::endl;
		//}
        
        //t2 = std::chrono::high_resolution_clock::now();
	
    });    
    
    //auto timeElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    //std::cout << "Time in Max pool Layer : " << timeElapsed << std::endl;
}

inline void tiny_maxpool_back_kernel(tensor_t& prev_delta,
                                     const tensor_t&  curr_delta,
                                     std::vector<std::vector<cnn_size_t>>& max_idx,
                                     const std::vector<cnn_size_t>& in2out,
                                     const bool layer_parallelize) {

    for_i(layer_parallelize, prev_delta.size(), [&](int sample) {
        vec_t& prev       = prev_delta[sample];
        const vec_t& curr = curr_delta[sample];
        const std::vector<cnn_size_t>& max = max_idx[sample];

        for (cnn_size_t i = 0; i < in2out.size(); i++) {
            cnn_size_t outi = in2out[i];
            prev[i] = (max[outi] == static_cast<cnn_size_t>(i)) ?
                       curr[outi] : float_t(0);
        }
    });
    
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
