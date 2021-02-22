
  #include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "vgg.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_vgg(torch::jit::script::Module module, Net &net){
	Layer t_layer;
    if(module.children().size() == 0){
        t_layer.layer = module;
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		get_submodule_vgg(children.value, net);
	}
}

void *predict_vgg(Net *input){
	std::vector<torch::jit::IValue> inputs = input->input;
	int i;
	for(i=0;i<input->layers.size();i++){
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1;
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;
		std::cout << "Before thpool add work VGG " << i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_vgg,&th);
		std::cout << "After thpool add work VGG " << i <<"\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->input.clear();
		input->input.push_back(input->layers[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout << "\n*****VGG result*****" << "\n";
	std::cout << (input->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";	
}

void forward_vgg(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;

	if(k == 32){
		//out = out.view({out.size(0), -1});
		out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
		inputs.clear();
		inputs.push_back(out);
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
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
