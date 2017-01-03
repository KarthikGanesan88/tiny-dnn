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
#include <chrono>
#include "tiny_dnn/tiny_dnn.h"

#define NN_SIZE 7

using namespace tiny_dnn;
using namespace tiny_dnn::activation;


void run_test(network<sequential>& nn, std::vector<label_t>& test_labels, std::vector<vec_t>& test_images, const std::vector<int> ap) 
{
    //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    nn.set_anytime_params(ap); for (auto iter: ap) { std::cout<< iter << ","; }	std::cout<< ":"<<std::endl;
    nn.test(test_images, test_labels).print_summary(std::cout);
    
    /*std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    
    auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Time elapsed: " << timeElapsed << std::endl;    */
}
  

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

    // Initial run to warm the cache?
    nn.testN(test_images, test_labels, (cnn_size_t)1);

    anytime_params ap[NN_SIZE] = {0};

    /* Anytime params must be initialised to 1 to perform the 'full calculations'
    for (int i=0; i<NN_SIZE; i++){ ap[0].cs=1; ap[0].ls=1; }
        
    std::cout << "-------------------------Full run-------------------------" << std::endl;
    nn.testN(test_images, test_labels, (cnn_size_t)1).print_summary(std::cout);
    //nn.test(test_images, test_labels).print_summary(std::cout);
    
    std::cout << "-------------------------Partial run-------------------------" << std::endl;
    ap[0].cs=1; ap[0].ls=1; for (int i=1; i<(NN_SIZE-1); i++){ ap[0].cs=4; ap[0].ls=4; }
    nn.set_anytime_params(ap);

    //for (int i=0; i<NN_SIZE; i++){ std::cout<< ap[i].ls << ","ap[0].cs=1; ap[0].ls=1; } std::cout<< ":"<<std::endl;

    nn.testN(test_images, test_labels, (cnn_size_t)1).print_summary(std::cout);
    //nn.test(test_images, test_labels).print_summary(std::cout);
    
    /*ap[0]=8;ap[1]=8;ap[2]=8;ap[3]=8;ap[4]=8;ap[5]=8; run_test(nn,test_labels, test_images,ap);
    ap[0]=4;ap[1]=4;ap[2]=8;ap[3]=8;ap[4]=8;ap[5]=8; run_test(nn,test_labels, test_images,ap);        
    ap[0]=4;ap[1]=4;ap[2]=4;ap[3]=4;ap[4]=8;ap[5]=8; run_test(nn,test_labels, test_images,ap);
    ap[0]=4;ap[1]=4;ap[2]=4;ap[3]=4;ap[4]=4;ap[5]=8; run_test(nn,test_labels, test_images,ap);
        
    ap[0]=4;ap[1]=4;ap[2]=4;ap[3]=4;ap[4]=4;ap[5]=4; run_test(nn,test_labels, test_images,ap);    
    ap[0]=2;ap[1]=2;ap[2]=4;ap[3]=4;ap[4]=4;ap[5]=4; run_test(nn,test_labels, test_images,ap);    
    ap[0]=2;ap[1]=2;ap[2]=2;ap[3]=2;ap[4]=4;ap[5]=4; run_test(nn,test_labels, test_images,ap); 
    ap[0]=2;ap[1]=2;ap[2]=2;ap[3]=2;ap[4]=2;ap[5]=4; run_test(nn,test_labels, test_images,ap);    
    
    ap[0]=2;ap[1]=2;ap[2]=2;ap[3]=2;ap[4]=2;ap[5]=2; run_test(nn,test_labels, test_images,ap);     
    ap[0]=1;ap[1]=1;ap[2]=2;ap[3]=2;ap[4]=2;ap[5]=2; run_test(nn,test_labels, test_images,ap); 
    ap[0]=1;ap[1]=1;ap[2]=1;ap[3]=1;ap[4]=2;ap[5]=2; run_test(nn,test_labels, test_images,ap); 
    ap[0]=1;ap[1]=1;ap[2]=1;ap[3]=1;ap[4]=1;ap[5]=2; run_test(nn,test_labels, test_images,ap); 
    ap[0]=1;ap[1]=1;ap[2]=1;ap[3]=1;ap[4]=1;ap[5]=1; run_test(nn,test_labels, test_images,ap); */
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage : " << argv[0] << " path_to_model path_to_data" << std::endl;
        std::cerr << "Example: ./test_lenet lenet_model ../data/" << std::endl;
        return -1;
    }
    test_lenet(argv[1],argv[2]);
    return 0;
}
















/*for (int ia = 1; ia<=4; ia*=2 ){    		// [0] - C1 Layer
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
		    	    
		    nn.set_anytime_params(ap);  
				
		    for (auto iter: ap) { std::cout<< iter << ","; }
		  
		    nn.test(test_images, test_labels).print_summary(std::cout);
		  }
	      }
	    }	    
	  }
	}	
    }*/



