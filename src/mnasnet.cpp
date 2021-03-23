#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>

#include "mnasnet.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_MNASNet(torch::jit::script::Module module, Net &net){
	Layer t_layer;
	Dummy dummy;
    if(module.children().size() == 0){
        t_layer.layer = module;
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		if(children.name == "layers"){
			for(auto ch2 : children.value.named_children()){
				if(ch2.value.children().size() != 0){
					for(auto ch3 : ch2.value.named_children()){
						if(ch3.value.children().size() != 0){
							for(auto ch4 : ch3.value.named_children()){
								get_submodule_MNASNet(ch4.value, net);
							}
						}
						else{
							t_layer.layer = ch3.value;
							t_layer.name = "";
        					net.layers.push_back(t_layer);
						}
						/*
						class _InvertedResidual

						def forward(self, input):
							if self.apply_residual:    // ch3.name != "0"
								return self.layers(input) + input
							else:
								return self.layers(input)    // ch3.name == "0"
						*/
						if(ch3.name != "0"){
							t_layer.layer = dummy;
							t_layer.name = "Residual";
							t_layer.from_idx = {-1, -9};
							net.layers.push_back(t_layer);
						}
					}
				}
				else{
					t_layer.layer = ch2.value;
					t_layer.name = "";
					net.layers.push_back(t_layer);
				}
			}
		}
		else{
			get_submodule_MNASNet(children.value, net);
		}
	}
}

void *predict_MNASNet(Net *mnasnet){
	int i;
	for(i=0;i<mnasnet->layers.size();i++){
		pthread_mutex_lock(&mutex_t[mnasnet->index_n]);
		cond_i[mnasnet->index_n] = 1;
		
		netlayer nl;
		nl.net = mnasnet;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		thpool_add_work(thpool,(void(*)(void *))forward_MNASNet,&th);

		while (cond_i[mnasnet->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[mnasnet->index_n], &mutex_t[mnasnet->index_n]);
    	}
		mnasnet->input.clear();
		mnasnet->input.push_back(mnasnet->layers[i].output);
		pthread_mutex_unlock(&mutex_t[mnasnet->index_n]);
	}
	std::cout << "\n*****"<<mnasnet->name<<"*****" << "\n";
	std::cout << (mnasnet->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}

void forward_MNASNet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;
	//at::cuda::setCurrentCUDAStream(streams[(nl->net->index_n)]);
	{
		at::cuda::CUDAStreamGuard guard(streams[(nl->net->index_n)]);
		if(k == nl->net->flatten){
			out = inputs[0].toTensor().mean({2,3});
		}
		else if(nl->net->layers[k].name == "Residual"){
			int add_index = k + nl->net->layers[k].from_idx[0];
			out = nl->net->layers[add_index].output;
			for(int i=1;i<nl->net->layers[k].from_idx.size();i++){
				int add_index = k + nl->net->layers[k].from_idx[i];
				out += nl->net->layers[add_index].output;
			}
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

