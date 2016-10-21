/*
    Copyright (c) 2013, Taiga Nomi
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
#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

static void test_lenet(const std::string& dictionary, const std::string& data_dir_path) {

    network<sequential> nn;

    std::cout << "load models..." << std::endl;
    
    // Load the saved network model & trained weights
    nn.load(dictionary);
    
    // load nets
    std::ifstream ifs(dictionary.c_str());
    ifs >> nn;

    // load MNIST dataset
    std::vector<label_t> test_labels;
    std::vector<vec_t> test_images;

    parse_mnist_labels(data_dir_path+"/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images(data_dir_path+"/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    std::cout << "Start testing" << std::endl;

    /* construct nets
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6,  	// C1, 1@32x32-in, 6@28x28-out
            padding::valid, true, 1, 1, backend_type)    
       << max_pooling_layer<tan_h>(28, 28, 6, 2)   	    // S2, 6@28x28-in, 6@14x14-out    
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16, 	// C3, 6@14x14-in, 16@10x10-in
            padding::valid, true, 1, 1, backend_type)       
       << max_pooling_layer<tan_h>(10, 10, 16, 2)  	    // S4, 16@10x10-in, 16@5x5-out       
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120, 	// C5, 16@5x5-in, 120@1x1-out
            padding::valid, true, 1, 1, backend_type)       
       << fully_connected_layer<tan_h>(120, 84,        	// F6, 120-in, 84-out
            true, backend_type)           
       << fully_connected_layer<softmax>(84, 10,       	// F7, 120-in, 10-out
            true, backend_type) 
    ;*/

    // Anytime params must be initialised to 1 to perform the 'full calculations'
    std::vector<int> anytime_params;
    anytime_params.push_back(1); // [0] - C1 Layer
    anytime_params.push_back(2); // [1] - S2 Layer
    anytime_params.push_back(1); // [2] - C3 Layer
    anytime_params.push_back(1); // [3] - S4 Layer
    anytime_params.push_back(1); // [4] - C5 Layer
    anytime_params.push_back(1); // [5] - F6 Layer
    anytime_params.push_back(1); // [6] - F7 Layer

    //nn.test1(test_images, test_labels).print_summary(std::cout);
    
    for (int ia = 1; ia<=4; ia*=2 ){    		// [0] - C1 Layer
        anytime_params[0] = ia;	
	for (int ib = 1; ib<=4; ib*=2 ){  		// [1] - S2 Layer  
	  anytime_params[1] = ib;	  
	  for (int ic = 1; ic<=4; ic*=2 ){    		// [2] - C3 Layer
	    anytime_params[2] = ic;
	    for (int id = 1; id<=4; id*=2 ){    	// [2] - C3 Layer
	      anytime_params[3] = id;
	      for (int ie = 1; ie<=4; ie*=2 ){    	// [2] - C3 Layer
		anytime_params[4] = ie;
		  for (int ig = 1; ig<=4; ig*=2 ){    	// [2] - C3 Layer
		    anytime_params[5] = ig;
		    	    
		    nn.set_anytime_params(anytime_params);  
				
		    for (auto iter: anytime_params) { std::cout<< iter << ","; }
		  
		    nn.test(test_images, test_labels).print_summary(std::cout);
		  }
	      }
	    }	    
	  }
	}	
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage : " << argv[0] << " path_to_data (example:../data)" << std::endl;
        return -1;
    }
    test_lenet("LeNet-model-2FC", argv[1]);
    return 0;
}
