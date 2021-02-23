#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include "cuda_runtime.h"

#include "squeeze.h"


namespace F = torch::nn::functional;
using namespace std;

void get_submodule_squeeze(torch::jit::script::Module module, Net &net){
	Layer t_layer;
    if(module.children().size() == 0){
        t_layer.layer = module;
		t_layer.flag = false;
		t_layer.name = "none";
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		if(children.name == "squeeze"){
			int i = 0;
			for( auto fire_sub : module.named_children()){
				if(i == 1 || i == 3 || i ==5){
					i++;
					continue;
				}
				t_layer.name = fire_sub.name; // expand1x1 , expand3x3 , squeeze
				t_layer.layer = fire_sub.value;
				t_layer.flag = false;
				net.layers.push_back(t_layer);
				if(fire_sub.name == "expand3x3"){
					t_layer.name = "concat";
					net.layers.push_back(t_layer);
				}
				i++;
			}
			break;
		}
		get_submodule_squeeze(children.value, net);
	}
}

void *predict_squeeze(Net *input){
	std::vector<torch::jit::IValue> inputs = input->input;
	int i;
	std::cout<<"input layers size = "<<input->layers.size()<<"\n";
	for(i=0;i<input->layers.size();i++){
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1;
		
		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		std::cout << "Before thpool add work Squeeze "<< i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_squeeze,&th);
		std::cout << "After thpool add work Squeeze "<< i << "\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->input.clear();
		input->input.push_back(input->layers[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout<<"\n*****Squeeze result*****" << "\n";
	std::cout<<(input->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) <<"\n";
}

void forward_squeeze(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs;
	int k = nl->net->index;
	if(nl->net->layers[k].name == "expand1x1"){
		inputs.push_back(nl->net->layers[k-1].output); //check 
	}
	else if(nl->net->layers[k].name == "expand3x3"){
		inputs.push_back(nl->net->layers[k-2].output);
	}
	else{ 
		inputs = nl->net->input;
	}
	if(nl->net->layers[k].name == "expand1x1"){
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);  
	at::Tensor out;
	std::cout<<"k = "<<k<<"\n";

	if(k == nl->net->layers.size()-1){
		cout<<"11111111111111\n";
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
		out = out.view({out.size(0), -1});
	}
	else if(nl->net->layers[k].name == "concat"){
		cout<<"\n cat \n";
		out = torch::cat({nl->net->layers[k-2].output, nl->net->layers[k-1].output}, 1);
	}
	else if(nl->net->layers[k].name != "none"){
		if(nl->net->layers[k].name == "expand1x1"){
			cout<<"expand1x1  out\n";
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			out = torch::relu(out);
			cudaEventRecord(nl->net->record[0],0);
		}
		else if(nl->net->layers[k].name == "expand3x3"){
			cout<<"expand3x3 out\n";
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			out = torch::relu(out);
			cudaEventRecord(nl->net->record[1],0);
			cout<<"expand3x3 record\n";
		}
		else{
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			out = torch::relu(out);
		}
	}
	else{
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
	}


	std::cout<<"before out\n";

	if(nl->net->layers[k].name == "expand1x1"){
		cudaEventSynchronize(nl->net->record[0]);
		nl->net->layers[k].output = out;
		nl->net->layers[k].flag = true;
	}
	else if(nl->net->layers[k].name == "expand3x3"){
		cudaEventSynchronize(nl->net->record[1]);
		nl->net->layers[k].output = out;
		nl->net->layers[k].flag = true;
	}
	else
		nl->net->layers[k].output = out;


	
	std::cout<<"after out\n";
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	cond_i[nl->net->index_n]=0;
	if(nl->net->layers[k].name != "expand1x1" && nl->net->layers[k].name != "expand3x3"){
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}else if(nl->net->layers[k].name == "expand1x1" && nl->net->layers[k+1].flag){
		nl->net->layers[k+1].flag = false;
		nl->net->layers[k].flag = false;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}
	else if(nl->net->layers[k].name == "expand3x3" && nl->net->layers[k-1].flag){
		nl->net->layers[k-1].flag = false;
		nl->net->layers[k].flag = false;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}

	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}




