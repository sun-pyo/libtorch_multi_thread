#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "densenet.h"

using namespace std;
namespace F = torch::nn::functional;

void get_submodule_densenet(torch::jit::script::Module module,Net &net){
	Dummy concat;
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
                    t_layer.from_idx = {-1};
                    t_layer.layer = concat;
                    t_layer.name = "Concat";
                    net.layers.push_back(t_layer);
                    for(auto in_denselayer : layer.value.named_children()){
                        t_layer.from_idx.clear();
                        t_layer.layer = in_denselayer.value;
                        t_layer.name = "None";
                        net.layers.push_back(t_layer);
                        
                    }
                    t_layer.from_idx = {-7, -1};
                    t_layer.layer = concat;
                    t_layer.name = "Concat";
                    net.layers.push_back(t_layer);
                }
                else
                    get_submodule_densenet(layer.value, net);
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
 //       std::cout<< "start dense layer" << i <<"\n";
        pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1; //right?
        netlayer nl;
		nl.net = input;
        nl.net->index = i;

		th_arg th;
		th.arg = &nl;
//        std::cout << "Before thpool add work DENSE " << i << "\n";
        thpool_add_work(thpool,(void(*)(void *))forward_densenet,&th);
 //       std::cout << "After thpool add work DENSE " << i << "\n";
        while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
        i = nl.net->index;
       // cout<<nl.net->index<<"\n";
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
        if(k==18) cout<<"1111111111111\n";
        out = F::relu(inputs[0].toTensor(), F::ReLUFuncOptions().inplace(true));
        out = F::adaptive_avg_pool2d(out, F::AdaptiveAvgPool2dFuncOptions(1));
        out = out.view({out.size(0), -1});
	    inputs.clear();
	    inputs.push_back(out);
        out = nl->net->layers[k].layer.forward(inputs).toTensor();
    }
    else if(nl->net->layers[k].name == "Concat"){
        if(k==18) cout<<"2222222222222\n";
        std::vector<at::Tensor> cat_input;
        for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
            cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
        }
        out = torch::cat(cat_input, 1);
    }
    else{
        if(k==18) {
            out = torch::cat({nl->net->layers[k-1].output},1);
            inputs.clear();
            inputs.push_back(out);
        }
        out = nl->net->layers[k].layer.forward(inputs).toTensor();
    }
    
    //cout<<"output size = "<<out.sizes()<<"\n\n\n";
    nl->net->layers[k].output = out;
    nl->net->index = k;
	cond_i[nl->net->index_n]=0;
    //std::cout<< "dense forward end\n";
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
}
