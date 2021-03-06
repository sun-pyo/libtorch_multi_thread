
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
		t_layer.name = "Normal";
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		get_submodule_vgg(children.value, net);
	}
}

void *predict_vgg(Net *vgg){
	std::vector<torch::jit::IValue> inputs = vgg->input;
	int i;
	for(i=0;i<vgg->layers.size();i++){
		pthread_mutex_lock(&mutex_t[vgg->index_n]);
		cond_i[vgg->index_n] = 1;
		
		netlayer nl;
		nl.net = vgg;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		thpool_add_work(thpool,(void(*)(void *))forward_vgg,&th);

		while (cond_i[vgg->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[vgg->index_n], &mutex_t[vgg->index_n]);
    	}
		vgg->input.clear();
		vgg->input.push_back(vgg->layers[i].output);
		pthread_mutex_unlock(&mutex_t[vgg->index_n]);
	}
  

	std::cout << "\n*****"<<vgg->name<<" result*****" << "\n";
	std::cout << (vgg->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";	
}

void forward_vgg(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	int k = nl->net->index;
	at::Tensor out;
	
	//at::cuda::setCurrentCUDAStream(streams[(nl->net->index_n)]);
	{
		at::cuda::CUDAStreamGuard guard(streams[(nl->net->index_n)]);
		if(k == nl->net->flatten){
			//out = out.view({out.size(0), -1});
			out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
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
