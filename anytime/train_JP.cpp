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

static void construct_net(network<sequential>& nn) {

    core::backend_t backend_type = core::backend_t::tiny_dnn;
   
    // construct nets
    nn << fully_connected_layer<tan_h>(32*32, 32,    // C1, 32*32-in, 32-out
            true, backend_type)
       << fully_connected_layer<softmax>(32, 10,     // C3, 32-in, 10-out
            true, backend_type)           
    ;
}

static void train_lenet(const std::string& data_dir_path) {
    // specify loss-function and learning strategy
    network<sequential> nn;
    adagrad optimizer;

    construct_net(nn);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels(data_dir_path+"/train-labels.idx1-ubyte",
                       &train_labels);
    parse_mnist_images(data_dir_path+"/train-images.idx3-ubyte",
                       &train_images, -1.0, 1.0, 2, 2);
    parse_mnist_labels(data_dir_path+"/t10k-labels.idx1-ubyte",
                       &test_labels);
    parse_mnist_images(data_dir_path+"/t10k-images.idx3-ubyte",
                       &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start training" << std::endl;

    progress_display disp((unsigned long)train_images.size());
    timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    // Original Alpha from example    
    //optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));
    
    //Experimental smaller learning rate
    optimizer.alpha *= static_cast<tiny_dnn::float_t>(0.1);   

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart((unsigned long)train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save network model & trained weights
    nn.save("LeNet-JP-2layers-32-10");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage : " << argv[0]
                  << " path_to_data (example:../data)" << std::endl;
        return -1;
    }
    train_lenet(argv[1]);
    return 0;
}
