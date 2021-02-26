#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "inception.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_inception(torch::jit::script::Module module, Net &net){
    Layer t_layer;    
    Dummy temp;
    for(auto children : module.named_children()){
        if(children.name == "Mixed_5b" || children.name == "Mixed_5c" || children.name == "Mixed_5d"){
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.input_idx = -7;
                    t_layer.name = "avg_pool2d";
                    net.layers.push_back(t_layer);    
                }
                if(branch.name == "branch5x5_1"){
                    t_layer.input_idx = -2;
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = -4;
                }else{
                    t_layer.input_idx = 0;
                }
                t_layer.name = "A_" + branch.name;
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                net.layers.push_back(t_layer);
            }
            t_layer.input_idx = 0;
            t_layer.from_idx = {-8,-6,-3, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "Concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_6a"){
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = -2;
                }
                else{
                    t_layer.input_idx = 0;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "B_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch3x3dbl_3"){
                    t_layer.input_idx = -5;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.name = "max_pool2d";
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.input_idx = 0;
            t_layer.from_idx = {-5,-2, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "Concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_6b" || children.name == "Mixed_6c" || children.name == "Mixed_6d" || children.name == "Mixed_6e" ){
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.input_idx = -10;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.name = "avg_pool2d";
                    net.layers.push_back(t_layer);
                    t_layer.input_idx = 0;
                }
                else if(branch.name == "branch7x7_1"){
                    t_layer.input_idx = -2;
                }
                else if(branch.name == "branch7x7dbl_1"){
                    t_layer.input_idx = -5;
                }
                else{
                    t_layer.input_idx = 0;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "C_" + branch.name;
                net.layers.push_back(t_layer);
            }
            t_layer.input_idx = 0;
            t_layer.from_idx = {-11,-8,-3, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "Concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_7a"){
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch7x7x3_1"){
                    t_layer.input_idx = -3;
                }
                else {
                    t_layer.input_idx = 0;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "D_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch7x7x3_4"){
                    t_layer.input_idx = -7;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.name = "max_pool2d";
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.input_idx = 0;
            t_layer.from_idx = {-6,-2, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "Concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name == "Mixed_7b" || children.name == "Mixed_7c"){
            for(auto branch : children.value.named_children()){
                if(branch.name == "branch_pool"){
                    t_layer.input_idx = -11;
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.name = "avg_pool2d";
                    net.layers.push_back(t_layer);
                    t_layer.input_idx = 0;
                }
                else if(branch.name == "branch3x3_1" || branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
                    t_layer.input_idx = -2;
                }
                else if(branch.name == "branch3x3dbl_1"){
                    t_layer.input_idx = -6;
                }
                else{
                    t_layer.input_idx = 0;
                }
                t_layer.layer = branch.value;
                t_layer.exe_success = false;
                t_layer.name = "E_" + branch.name;
                net.layers.push_back(t_layer);
                if(branch.name == "branch3x3_2b" || branch.name == "branch3x3dbl_3b"){
                    t_layer.input_idx = 0;
                    t_layer.from_idx = {-2, -1};
                    t_layer.layer = temp;
                    t_layer.exe_success = false;
                    t_layer.name = "Concat";
                    net.layers.push_back(t_layer);
                }
            }
            t_layer.input_idx = 0;
            t_layer.from_idx = {-12,-8,-3, -1};
            t_layer.layer = temp;
            t_layer.exe_success = false;
            t_layer.name = "Concat";
            net.layers.push_back(t_layer);
            continue;
        }
        else if(children.name != "AuxLogits")
        {   
            t_layer.input_idx = 0;
            t_layer.layer = children.value;
            t_layer.name = "";
            t_layer.exe_success = false;
            net.layers.push_back(t_layer);   
        }
    }
}


void *predict_inception(Net *input){
	
	int i;
	std::cout<<"num layer"<<input->layers.size()<<"\n"; 
    
    auto x_ch0 = torch::unsqueeze(input->input[0].toTensor().index({torch::indexing::Slice(), 0}), 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5;
    auto x_ch1 = torch::unsqueeze(input->input[0].toTensor().index({torch::indexing::Slice(), 1}), 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5;
    auto x_ch2 = torch::unsqueeze(input->input[0].toTensor().index({torch::indexing::Slice(), 2}), 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5;
    
    x_ch0.to(at::kCUDA);
    x_ch1.to(at::kCUDA);
    x_ch2.to(at::kCUDA);

    auto x_cat = torch::cat({x_ch0,x_ch1,x_ch2},1).to(at::kCUDA);

    input->input[0] = x_cat;

	for(i=0;i<input->layers.size();i++){
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1;
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		//std::cout<<"index = "<<nl.index<<'\n';
		//std::cout << "Before thpool add work inception "<< i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_inception,&th);
		//std::cout << "After thpool add work inception "<< i << "\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->input.clear();
		input->input.push_back(input->layers[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout << "\n*****inception result*****" << "\n";
	std::cout << (input->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}

void forward_inception(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	int k = nl->net->index;
    std::vector<torch::jit::IValue> inputs;
    if(nl->net->layers[k].input_idx != 0){
        inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
    }
    else {
        inputs = nl->net->input;
    }
	at::Tensor out;
	if(k == nl->net->layers.size()-2){
		out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
		inputs.clear();
		inputs.push_back(out);
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
	}
    else if(nl->net->layers[k].name == "Concat"){
        std::vector<at::Tensor> cat_input;
        for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
            cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
        }
        out = torch::cat(cat_input, 1);
    }
    else if(nl->net->layers[k].name == "avg_pool2d"){
        out = F::avg_pool2d(inputs[0].toTensor(),F::AvgPool2dFuncOptions(3).stride(1).padding(1));
    }
    else if(nl->net->layers[k].name == "max_pool2d"){
        out = F::max_pool2d(inputs[0].toTensor(),F::MaxPool2dFuncOptions(3).stride(2));
    }
	else{
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
	}
	nl->net->layers[k].output = out;
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}

