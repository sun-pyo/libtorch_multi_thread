#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "resnet.h"

using namespace std;
namespace F = torch::nn::functional;

void get_submodule_resnet(torch::jit::script::Module module, Net &net){ 
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
                //Bottleneck
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
                        else if(count-1 == i || (count == 7 && i==5)){  // if count=8 downsample 
                            t_layer.name += "_last";
                            if(in_block.name == "downsample")
                                t_layer.name += "_downsample";
                        }
                        net.layers.push_back(t_layer);
                        i++;
                    }
                }
                //Basicblock
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
		get_submodule_resnet(children.value, net);

    }
}

void *predict_resnet(Net *res){
	std::vector<torch::jit::IValue> inputs = res->input; 
	int i;
	for(i = 0;i<res->layers.size();i++) {
		pthread_mutex_lock(&mutex_t[res->index_n]);
		cond_i[res->index_n] = 1; 
		
		netlayer nl;
		nl.net = res;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;
		thpool_add_work(thpool,(void(*)(void *))forward_resnet,&th);

		while (cond_i[res->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[res->index_n], &mutex_t[res->index_n]);
    	}
		i = nl.net->index;
		res->input.clear();
		res->input.push_back(res->layers[i].output);
		pthread_mutex_unlock(&mutex_t[res->index_n]);
	}
    std::cout << "\n*****"<<res->name<<" result*****" << "\n";
	std::cout << (res->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
}

void forward_resnet(th_arg *th){
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
    {
        at::cuda::CUDAStreamGuard guard(streams[(nl->net->index_n)]);
        if(k == nl->net->flatten) //flatten
        {	 
            out = inputs[0].toTensor().view({inputs[0].toTensor().size(0), -1});
            inputs.clear();
            inputs.push_back(out);
            out = nl->net->layers[k].layer.forward(inputs).toTensor();
        } 
        else if(nl->net->layers[k].name.find("_last") != std::string::npos){ 
            if(nl->net->layers[k].name.find("_downsample") != std::string::npos){   // downsample
                inputs_cpy.clear();
                inputs_cpy.push_back(identity); 
                identity = nl->net->layers[k].layer.forward(inputs_cpy).toTensor();
                out = nl->net->layers[k-1].output;
            } 
            else{
                out = nl->net->layers[k].layer.forward(inputs).toTensor();
            }
            out += identity;
            out = torch::relu(out); // downsample self.relu(out)
        }
        else{   //Basicblock(Bottleneck)
            out = nl->net->layers[k].layer.forward(inputs).toTensor(); 
            if(nl->net->layers[k].name.find("bn_relu") != std::string::npos ){   
                out = torch::relu(out); // Basicblock(Botteleneck) self.relu(out)
            }
        }
    }

    nl->net->layers[k].output = out;
	nl->net->identity = identity;
	nl->net->index = k;
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
}
