#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "resnet18.h"

using namespace std;
namespace F = torch::nn::functional;

void get_submodule_resnet18(torch::jit::script::Module module, Net &net){
    Layer t_layer;
    string name;
    if(module.children().size() == 0){
        t_layer.layer = module;
        t_layer.name = "";
        net.layers.push_back(t_layer);
        return;
    }
    for(auto children : module.named_children()){
		if(children.name.find("layer") != std::string::npos){
			for(auto in_layer : children.value.named_children()){
				int count = 0;
                for(auto in_block_count : in_layer.value.children()) count++;
                int i = 0;
                name = "";
                if(count > 6) {
                    for(auto in_block : in_layer.value.named_children()){
                        if(i==6) {
                            i++;
                            continue;
                        }
                        t_layer.layer = in_block.value;
                        t_layer.name = name;
                        if(i==1 || i==3){
                            t_layer.name += "bn_relu";
                        }
                        
                        if(i == 0){
                            t_layer.name += "_first";
                        }
                        else if(count-1 == i || (count == 7 && i==5)){
                            t_layer.name += "_last";
                            if(in_block.name == "downsample")
                                t_layer.name += "_downsample";
                        }
                        net.layers.push_back(t_layer);
                        i++;
                    }
                }
                else{
                    for(auto in_block : in_layer.value.named_children()){
                        if(i==2) {
                            i++;
                            continue;
                        }
                        t_layer.layer = in_block.value;
                        t_layer.name = name;
                        if(i==1){
                            t_layer.name += "bn_relu";
                        }
                        if(i == 0){
                            t_layer.name += "_first";
                        }
                        else if(count-1 == i){
                            t_layer.name += "_last";
                            if(in_block.name == "downsample")
                                t_layer.name += "_downsample";
                        }
                        net.layers.push_back(t_layer);
                        i++;
                    }
                }
			}
			continue;
		}
		get_submodule_resnet18(children.value, net);

    }
}

void *predict_resnet18(Net *input){
	std::vector<torch::jit::IValue> inputs = input->input;
	int i;
    std::cout<<"resnet start\n";
	for(i = 0;i<input->layers.size();i++) {
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1; //right?
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;
		//std::cout << "Before thpool add work RES " << i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_resnet18,&th);
		//std::cout << "After thpool add work RES " << i <<"\n";
		//cout<<"id = " <<(nl.net->identity)<<"\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		i = nl.net->index;
		input->input.clear();
		input->input.push_back(input->layers[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout << "\n*****Res result*****" << "\n";
	std::cout << (input->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
}

void forward_resnet18(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs = nl->net->input;
	at::Tensor identity = nl->net->identity;
	vector<torch::jit::IValue> inputs_cpy;
	int k =nl->net->index;
	at::Tensor out;

    if(nl->net->layers[k].name.find("_first") != std::string::npos){
       identity = inputs[0].toTensor(); 
    }

    if(k == nl->net->layers.size()-1) //flatten
    {	
        
		out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
       	inputs.clear();
       	inputs.push_back(out);
       	out = nl->net->layers[k].layer.forward(inputs).toTensor();

    }
    else if(nl->net->layers[k].name.find("_last") != std::string::npos){
		//BasicBlock and downsample 
        if(nl->net->layers[k].name.find("_downsample") != std::string::npos){
			inputs_cpy.clear();
			inputs_cpy.push_back(identity);
			identity = nl->net->layers[k].layer.forward(inputs_cpy).toTensor();
		    out = nl->net->layers[k-1].output;
		} //BasicBlock
		else{
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
        }
		out += identity;
		out = torch::relu(out);
    }
	else{
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
        if(nl->net->layers[k].name.find("bn_relu") != std::string::npos ){
            out = torch::relu(out);
        }
	}

    nl->net->layers[k].output = out;
	nl->net->identity = identity;
	nl->net->index = k; //check
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
    //cout<<out.sizes()<<"\n\n";
}
