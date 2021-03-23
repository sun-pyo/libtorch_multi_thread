#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "alex.h"

namespace F = torch::nn::functional;


void get_submodule_alexnet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
    if(module.children().size() == 0){ 
        t_layer.layer = module;
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		get_submodule_alexnet(children.value, net);
	}
}

void *predict_alexnet(Net *alex){
	std::vector<torch::jit::IValue> inputs = alex->input;
	int i;
	std::cout<<alex->layers.size()<<"\n";
	for(i=0;i<alex->layers.size();i++){
		pthread_mutex_lock(&mutex_t[alex->index_n]);
		cond_i[alex->index_n] = 1;
		
		netlayer nl;
		nl.net = alex;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		thpool_add_work(thpool,(void(*)(void *))forward_alexnet,&th);

		while (cond_i[alex->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[alex->index_n], &mutex_t[alex->index_n]);
    	}
		alex->input.clear();
		alex->input.push_back(alex->layers[i].output);
		pthread_mutex_unlock(&mutex_t[alex->index_n]);
	}
	std::cout << "\n*****"<<alex->name<<" result*****" << "\n";
	std::cout << (alex->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}

void forward_alexnet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;
	{
		at::cuda::CUDAStreamGuard guard(streams[(nl->net->index_n)]);
		if(k == nl->net->flatten){  //flatten
			out = inputs[0].toTensor().view({nl->net->layers[k-1].output.size(0), -1});
			inputs.clear();
			inputs.push_back(out);
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}
		else{
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}
	}
	nl->net->layers[k].output = out;
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}

