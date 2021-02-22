#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "densenet.h"

using namespace std;
namespace F = torch::nn::functional;

struct Concat : torch::jit::Module{
   /* at::Tensor forward(Layer *layer){
        at::Tensor out = layer->output[layer->index];
        out = torch::cat({out[layer->index-1], out[layer->index-7]}, 1);
        return out;
    }*/
};
void get_submodule_densenet(torch::jit::script::Module module,Net &net){
	Concat concat;
    Layer t_layer;
    if(module.children().size() == 0){
        t_layer.layer = module;
        net.layers.push_back(t_layer);
        return;
    }
    for(auto children : module.named_children()){
        if(children.name.find("denseblock") != std::string::npos){
            int size = net.layers.size();
            for(auto layer : children.value.named_children()){
                if(layer.name.find("denselayer") != std::string::npos){
                    int i=0;
                    for(auto in_denselayer : layer.value.children()){
                        if(i++ == 0){
                            t_layer.name = "layer_start";
                        }
                        t_layer.layer = in_denselayer;
                        net.layers.push_back(t_layer);
                    }
                }
                else
                    get_submodule_densenet(layer.value, net);
                t_layer.layer = concat;
                t_layer.name = "Concat";
                net.layers.push_back(t_layer);
            }
            continue;
        }
        get_submodule_densenet(children.value, net);
    }
}

void *predict_densenet(Net *input){
    std::cout<< "dense" <<"\n";
	std::vector<torch::jit::IValue> inputs = input->input;
    int i;

    for(i=0;i<input->layers.size();i++){
        std::cout<< "start dense layer" << i <<"\n";
        pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1; //right?
        netlayer nl;
		nl.net = input;
        nl.net->index = i;

		th_arg th;
		th.arg = &nl;
        std::cout << "Before thpool add work DENSE " << i << "\n";
        thpool_add_work(thpool,(void(*)(void *))forward_densenet,&th);
        std::cout << "After thpool add work DENSE " << i << "\n";
        while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
        i = nl.net->index;
        cout<<nl.net->index<<"\n";
		input->input.clear();
		input->input.push_back(input->layers[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
    }
    std::cout << "\n*****Dense result*****" << "\n";
	std::cout << (input->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
}

void forward_densenet(th_arg *th){ 
    pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
    int k =nl->net->index;
    at::Tensor out;
    
    if(k == nl->net->layers.size()-1){
        out = F::relu(inputs[0].toTensor(), F::ReLUFuncOptions().inplace(true));
        out = F::adaptive_avg_pool2d(out, F::AdaptiveAvgPool2dFuncOptions(1));
        out = out.view({out.size(0), -1});
	    inputs.clear();
	    inputs.push_back(out);
        out = nl->net->layers[k].layer.forward(inputs).toTensor();
    }
    else if(nl->net->layers[k].name == "Concat"){
        out = torch::cat({nl->net->layers[k-7].output, nl->net->layers[k-1].output}, 1);
    }
    else if(nl->net->layers[k].name == "layer_start"){
        out = torch::cat({nl->net->layers[k-1].output},1);
        inputs.clear();
        inputs.push_back(out);
        out = nl->net->layers[k].layer.forward(inputs).toTensor();
    }
    else{
        out = nl->net->layers[k].layer.forward(inputs).toTensor();
    }
    at::Tensor ten = out;
    nl->net->layers[k].output = out;
    nl->net->index = k;
	cond_i[nl->net->index_n]=0;
    std::cout<< "dense forward end\n";
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
}
