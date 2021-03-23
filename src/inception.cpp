#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include <thread>
#include <unistd.h>
#include "inception.h"

/*

event_idx : branch_num in inception (for recording event)
input_idx : the index of the input from the current layer
skip : Number of layer modules in one branch (How many more signals do thread have to send)
branch_idx : The last layer index of the branch to determine if the operation is complete(exe_success)

*/

namespace F = torch::nn::functional;
using namespace std;

void get_submodule_inception(torch::jit::script::Module module, Net &net){
    Layer t_layer;    
    Dummy temp;
    for(auto children : module.named_children()){
        if(children.name == "Mixed_5b" || children.name == "Mixed_5c" || children.name == "Mixed_5d"){ //InceptionA
            int event_idx = -1;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.input_idx = -7;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "avg_pool2d";
                    t_layer.skip = 2;
                    net.layers.push_back(t_layer);    
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                    t_layer.branch_idx = {-7, -5, -2, 0};
                }
                if(branch.name == "branch1x1"){
                    t_layer.input_idx = 0;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = 1;
                    t_layer.branch_idx = {2, 5, 7};
                }
                else if(branch.name == "branch5x5_1"){
                    t_layer.input_idx = -2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = 2;
                }
                else if(branch.name == "branch5x5_2"){
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                    t_layer.branch_idx = {-2, 3, 5};
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = -4;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = 3;
                }
                else if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                    t_layer.branch_idx = {-5, -3, 2};
                }
                else{
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                }
                t_layer.name = "A_" + branch.name;
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                net.layers.push_back(t_layer);
            }
            t_layer.event_idx = -1;
            t_layer.input_idx = 0;
            t_layer.from_idx = {-8,-6,-3, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = 0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_6a"){   //InceptionB
            int event_idx = -1;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch3x3"){
                    t_layer.input_idx = 0;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = 1;
                    t_layer.branch_idx = {3, 4};
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = -2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = 3;
                }
                else if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                    t_layer.branch_idx = {-3, 1};
                }
                else{
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "B_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = -5;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "max_pool2d";
                    t_layer.skip = 1;
                    t_layer.branch_idx = {-4, -1, 0};
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.event_idx = -1;
            t_layer.input_idx = 0;
            t_layer.from_idx = {-5,-2, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = 0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_6b" || children.name == "Mixed_6c" || children.name == "Mixed_6d" || children.name == "Mixed_6e" ){ //InceptionC
            int event_idx = -1;
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.input_idx = -10;
                    t_layer.layer = temp;
                    t_layer.event_idx = ++event_idx;
                    t_layer.exe_success = false;
                    t_layer.name = "avg_pool2d";
                    t_layer.skip = 2;
                    net.layers.push_back(t_layer);
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                    t_layer.branch_idx = {-10,-7,-2, 0};
                }
                else if(branch.name == "branch1x1"){
                    t_layer.input_idx = 0;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = 1;
                    t_layer.branch_idx = {3,8,10};
                }
                else if(branch.name == "branch7x7_1"){
                    t_layer.input_idx = -2;
                    t_layer.event_idx = ++event_idx;
                    t_layer.skip = 3;
                }
                else if(branch.name == "branch7x7_3"){
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                    t_layer.branch_idx = {-3,5,7};
                }
                else if(branch.name == "branch7x7dbl_1"){
                    t_layer.event_idx = ++event_idx;
                    t_layer.input_idx = -5;
                    t_layer.skip = 5;
                }
                else if(branch.name == "branch7x7dbl_3"){
                    t_layer.input_idx = 0;
                    t_layer.skip = 0;
                    t_layer.branch_idx = {-8,-5,2};
                }
                else{
                    t_layer.skip = 0;
                    t_layer.input_idx = 0;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "C_" + branch.name;
                net.layers.push_back(t_layer);
            }
            t_layer.event_idx = -1;
            t_layer.from_idx = {-11,-8,-3, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "concat";
            t_layer.skip = 0;
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_7a"){   //InceptionD
            int event_idx = -1;
            for(auto branch : children.value.named_children()){
                t_layer.skip = 0;
                if(branch.name == "branch7x7x3_1"){
                    t_layer.event_idx = ++event_idx;
                    t_layer.input_idx = -3;
                    t_layer.skip = 4;
                }
                else {
                    t_layer.input_idx = 0;
                    if(branch.name == "branch3x3_1"){
                        t_layer.skip = 2;
                        t_layer.event_idx = ++event_idx;
                    }
                    else if(branch.name == "branch7x7x3_4"){
                        t_layer.branch_idx = {-4, 1};
                    }
                    else if(branch.name == "branch3x3_2"){
                        t_layer.branch_idx = {4, 5};
                    }
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "D_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch7x7x3_4"){
                    t_layer.input_idx = -7;
                    t_layer.layer = temp;
                    t_layer.skip = 1;
                    t_layer.event_idx = ++event_idx;
                    t_layer.exe_success = false;
                    t_layer.name = "max_pool2d";
                    t_layer.branch_idx = {-5, -1, 0};
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.event_idx = -1;
            t_layer.from_idx = {-6,-2, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.skip = 0;
            t_layer.name = "concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_7b" || children.name == "Mixed_7c"){    //InceptionE
            int event_idx = -1;
            for(auto branch : children.value.named_children()){
                t_layer.skip = 0;
                if(branch.name == "branch_pool"){
                    t_layer.input_idx = -11;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.event_idx = ++event_idx;
                    t_layer.name = "avg_pool2d";
	                t_layer.skip = 2;
                    net.layers.push_back(t_layer);
                    t_layer.branch_idx = {-11, -7, -2, 0}; 
                    t_layer.input_idx = 0;
                }
                else if(branch.name == "branch3x3_1" || branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
                    t_layer.input_idx = -2;
                    if(branch.name == "branch3x3_1"){
	                    t_layer.skip = 4;
                        t_layer.event_idx = ++event_idx;
                    }
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.event_idx = ++event_idx;
	                t_layer.skip = 5;
                    t_layer.input_idx = -6;
                }
                else{
                    t_layer.input_idx = 0;
                    if(branch.name == "branch1x1"){
                        t_layer.skip = 1;
                        t_layer.event_idx = ++event_idx;
                        t_layer.branch_idx = {4, 9, 11};
                    }
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "E_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
                    if(branch.name == "branch3x3dbl_3b") t_layer.branch_idx = {-9, -5, 2}; 
                    else t_layer.branch_idx = {-4, 5, 7}; 
                    t_layer.input_idx = 0;
                    t_layer.from_idx = {-2, -1};
                    t_layer.layer = temp;
                    t_layer.skip = 0;
                    t_layer.exe_success = false;
                    t_layer.name = "concat";
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.skip = 0;
            t_layer.input_idx = 0;
            t_layer.from_idx = {-12,-8,-3, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.event_idx =-1;
            t_layer.name = "concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name != "AuxLogits")
        {   
            t_layer.input_idx = 0;
            t_layer.event_idx = -1;
            t_layer.layer = children.value;
            t_layer.skip = 0;
            t_layer.name = "";
            t_layer.exe_success = false;
            net.layers.push_back(t_layer);   
        }
    }
}


void *predict_inception(Net *inception){
	
	int i;
    
    auto x_ch0 = torch::unsqueeze(inception->input[0].toTensor().index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
    auto x_ch1 = torch::unsqueeze(inception->input[0].toTensor().index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
    auto x_ch2 = torch::unsqueeze(inception->input[0].toTensor().index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
    
    x_ch0.to(at::kCUDA);
    x_ch1.to(at::kCUDA);
    x_ch2.to(at::kCUDA);

    auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(at::kCUDA);

    inception->input[0] = x_cat;

	for(i=0;i<inception->layers.size();i++){
		pthread_mutex_lock(&mutex_t[inception->index_n]);
		cond_i[inception->index_n] = 1;
		inception->layers[i].exe_success = false;

		netlayer nl;
		nl.net = inception;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		thpool_add_work(thpool,(void(*)(void *))forward_inception,&th);
        std::cout << "After thpool add work inception "<< i << "\n";
		
        while (cond_i[inception->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[inception->index_n], &mutex_t[inception->index_n]);
    	}
        i = nl.net->index;
		inception->input.clear();
		inception->input.push_back(inception->layers[i].output);
		pthread_mutex_unlock(&mutex_t[inception->index_n]);
	}
	std::cout << "\n*****"<<inception->name<<"*****" << "\n";
	std::cout << (inception->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}

void forward_inception(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	int k = nl->net->index;
    int n_all = nl->net->n_all;
    std::vector<torch::jit::IValue> inputs;
    std::vector<int> stream_id = {nl->net->index_n, n_streamPerPool-1, n_streamPerPool-2,n_streamPerPool-3};
    //at::cuda::setCurrentCUDAStream(streams[stream_id[0]]);
    if(nl->net->layers[k].input_idx != 0){
        inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
    }
    else {
        inputs = nl->net->input;
    }
    if(nl->net->layers[k + nl->net->layers[k].skip].skip > 0){ // +1 why? predict for loop 
        nl->net->index = k + nl->net->layers[k].skip - 1;
        cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]); 
	at::Tensor out;
    {
        at::cuda::CUDAStreamGuard guard(streams[stream_id[0]]);
        if(k == nl->net->flatten){
            out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
            inputs.clear();
            inputs.push_back(out);
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
        }
        else if(nl->net->layers[k].skip > 0){   //branch
            //at::cuda::setCurrentCUDAStream(streams[stream_id[(nl->net->layers[k].event_idx)%4]]);
            {
                at::cuda::CUDAStreamGuard guard(streams[stream_id[(nl->net->layers[k].event_idx)%4]]); //event_idx == branch_num
                out = inputs[0].toTensor();
                int T = nl->net->layers[k].skip;
                for(int t=0;t<T;k++,t++){
                    if(nl->net->layers[k].input_idx != 0){
                        inputs.clear();
                        inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
                    }
                    else {
                        inputs.clear();
                        inputs.push_back(out);
                    } 
                    
                    if(nl->net->layers[k].name == "concat"){
                        std::vector<at::Tensor> cat_input;
                        for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
                            cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
                        }
                        out = torch::cat(cat_input, 1);
                    }
                    else if(nl->net->layers[k].name == "avg_pool2d"){
                        out = F::avg_pool2d(out,F::AvgPool2dFuncOptions(3).stride(1).padding(1));
                    }
                    else if(nl->net->layers[k].name == "max_pool2d"){
                        out = F::max_pool2d(out,F::MaxPool2dFuncOptions(3).stride(2));
                    }
                    else{
                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
                    }
                    nl->net->layers[k].output = out;
                }
            }
            k--;
            //int record = nl->net->layers[--k].event_idx;
            //cudaEventRecord(nl->net->record[record], 0);
        }
        else if(nl->net->layers[k].name == "concat"){  //brach out
            std::vector<at::Tensor> cat_input;
            for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
                cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
            }
            out = torch::cat(cat_input, 1);
        }
        else{
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
        }
    }
    if(nl->net->layers[k].event_idx >= 0){
		//cudaEventSynchronize(nl->net->record[nl->net->layers[k].event_idx]);
		nl->net->layers[k].output = out;
		nl->net->layers[k].exe_success = true;
	}
    nl->net->layers[k].output = out;

    pthread_mutex_lock(&mutex_t[nl->net->index_n]);

    if(nl->net->layers[k].exe_success == false){
        cond_i[nl->net->index_n]=0;
        pthread_cond_signal(&cond_t[nl->net->index_n]);
    }
    else{
       for(int i=0;i<nl->net->layers[k].branch_idx.size();i++){
           if(nl->net->layers[k + nl->net->layers[k].branch_idx[i]].exe_success == false){
               pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
               return;
           }
       }
       nl->net->index = k + nl->net->layers[k].branch_idx.back(); // last layer index of branch
       cond_i[nl->net->index_n]=0;
       pthread_cond_signal(&cond_t[nl->net->index_n]);
    }
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}

