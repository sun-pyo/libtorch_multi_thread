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

void *predict_MNASNet(Net *input){
	std::vector<torch::jit::IValue> inputs = input->input;
	int i;
	std::cout<<input->layers.size()<<"\n";
	for(i=0;i<input->layers.size();i++){
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1;
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		//std::cout<<"index = "<<nl.index<<'\n';
		std::cout << "Before thpool add work MNASNet "<< i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_MNASNet,&th);
		std::cout << "After thpool add work MNASNet "<< i << "\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->input.clear();
		input->input.push_back(input->layers[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout << "\n*****MNASNet result*****" << "\n";
	std::cout << (input->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
	}

void forward_MNASNet(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;
	std::cout<<"k = "<<k<<" "<<nl->net->layers[k].name<<"\n";
	if(k == nl->net->layers.size()-2){
		out = inputs[0].toTensor().mean({2,3});
	}
	else if(nl->net->layers[k].name == "Residual"){
		out = nl->net->layers[k + nl->net->layers[k].from_idx[0]].output;
		std::cout<<"size = "<<nl->net->layers[k].from_idx.size()<<"\n";
		for(int i=1;i<nl->net->layers[k].from_idx.size();i++){
			std::cout<<"\n\nidx and = "<<k<<" "<<nl->net->layers[k].from_idx[i]<<"\n";
			out += nl->net->layers[k + nl->net->layers[k].from_idx[i]].output;
		}
	}
	else{
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
	}
	std::cout<<"before out\n";
	nl->net->layers[k].output = out;
	std::cout<<"after out\n";
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}

