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
#include <chrono>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;


void run_test(network<sequential>& nn, std::vector<label_t>& test_labels, std::vector<vec_t>& test_images, const std::vector<int> ap)
{
    nn.set_anytime_params(ap); for (auto iter: ap) { std::cout<< iter << ","; }	std::cout<< ":"<<std::endl;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //nn.testN(test_images, test_labels,10).print_summary(std::cout);
    nn.test(test_images, test_labels).print_summary(std::cout);

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Time elapsed: " << timeElapsed << std::endl;
}

static void test_lenet(const std::string& dictionary, const std::string& data_dir_path) {

    network<sequential> nn;

    std::cout << "load models..." << std::endl;
    
    // Load the saved network model & trained weights
    nn.load(dictionary);
    
    // load nets
    std::ifstream ifs(dictionary.c_str());
    ifs >> nn;

    nn.output_model_details();

    //load MNIST dataset
    std::vector<label_t> test_labels;
    std::vector<vec_t> test_images;

    parse_mnist_labels(data_dir_path+"/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images(data_dir_path+"/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    //Dry runs to fix the stupid activation function timing.
    nn.testN(test_images, test_labels, (cnn_size_t)1);

    std::cout << "*******************************\n" \
                 "          Full model           \n" \
                 "*******************************\n";

    nn.testN(test_images, test_labels, (cnn_size_t)1).print_summary(std::cout);
    //nn.test(test_images, test_labels).print_summary(std::cout);

    std::cout << "*******************************\n" \
                 "          Partial model        \n" \
                 "*******************************\n";

    std::vector<int> ap;
    ap.push_back(1); ap.push_back(1);

    ap[0]=4; ap[1]=1;
    nn.set_anytime_params(ap); for (auto iter: ap) { std::cout<< iter << ","; }	std::cout<< ":"<<std::endl;

    nn.testN(test_images, test_labels, (cnn_size_t)1).print_summary(std::cout);
    /*nn.test(test_images, test_labels).print_summary(std::cout);

    ap[0]=2; ap[1]=1;
    nn.set_anytime_params(ap); for (auto iter: ap) { std::cout<< iter << ","; }	std::cout<< ":"<<std::endl;

    nn.testN(test_images, test_labels, (cnn_size_t)1).print_summary(std::cout);
    //nn.test(test_images, test_labels).print_summary(std::cout);

    ap[0]=4; ap[1]=1;
    nn.set_anytime_params(ap); for (auto iter: ap) { std::cout<< iter << ","; }	std::cout<< ":"<<std::endl;

    nn.testN(test_images, test_labels, (cnn_size_t)1).print_summary(std::cout);
    //nn.test(test_images, test_labels).print_summary(std::cout);*/
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage : " << argv[0] << " path_to_model path_to_data" << std::endl;
        std::cerr << "Example: ./test_JP lenet_model.dnn ../data/" << std::endl;
        return -1;
    }
    test_lenet(argv[1],argv[2]);

    return 0;
}










/*for (int i = 8; i>0; i/=2 ){

    anytime_params[0] = i;

    for (int j = 8; j>0; j/=2 ){

        anytime_params[1] = j;

        for (int k = 8; k>0; k/=2 ){

            anytime_params[2] = k;

            nn.set_anytime_params(anytime_params);

            std::cout<< i << " " << j << " " << k << ":";
            nn.test(test_images, test_labels).print_summary(std::cout);
        }
    }
}*/