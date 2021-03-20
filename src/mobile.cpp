#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "mobile.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_mobilenet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
    if(module.children().size() == 0){
        t_layer.layer = module;
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		if(children.name == "features"){
			for(auto inverted : children.value.named_children()){
				int idx = net.layers.size();
				get_submodule_mobilenet(inverted.value, net);
				if(inverted.name == "3" || inverted.name == "5" || inverted.name == "6" || inverted.name == "8" || inverted.name == "9" || inverted.name == "10" || inverted.name == "12" || inverted.name == "13" || inverted.name == "15" || inverted.name == "16"){
					net.layers.back().name = "last_use_res_connect";
					net.layers[idx].name = "first_use_res_connect";
				}
			}
			continue;
		}
		get_submodule_mobilenet(children.value, net);
	}
}

void *predict_mobilenet(Net *input){
	std::vector<torch::jit::IValue> inputs = input->input;
	int i;
	//std::cout<<input->layers.size()<<"\n";
	for(i=0;i<input->layers.size();i++){
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1;
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		//std::cout << "Before thpool add work Mobilenet "<< i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_mobilenet,&th);
		//std::cout << "After thpool add work Mobilenet "<< i << "\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->input.clear();
		input->input.push_back(input->layers[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout << "\n*****Mobilenet result*****" << "\n";
	std::cout << (input->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}

void forward_mobilenet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;
	//std::cout<<"k = "<<k<<"\n";
	at::cuda::setCurrentCUDAStream(streams[(nl->net->index_n)]);
	if(nl->net->layers[k].name == "first_use_res_connect"){
		nl->net->identity = inputs[0].toTensor();
	}

	if(k == nl->net->layers.size()-2){
		out = torch::nn::functional::adaptive_avg_pool2d(inputs[0].toTensor(), F::AdaptiveAvgPool2dFuncOptions(1)).reshape({inputs[0].toTensor().size(0), -1});
	}
	else if(nl->net->layers[k].name == "last_use_res_connect"){
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
		out = nl->net->identity + out;
	}
	else{
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
	}
	//std::cout<<"before out\n";
	nl->net->layers[k].output = out;
	//std::cout<<"after out\n";
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}

