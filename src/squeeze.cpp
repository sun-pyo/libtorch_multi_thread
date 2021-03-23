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

bool is_ReLu(int idx){
	if(idx == 1 || idx == 3 || idx == 5)  // ReLu index
		return true;
	return false;
}

void get_submodule_squeeze(torch::jit::script::Module module, Net &net){
	Dummy dummy;
	Layer t_layer;
    if(module.children().size() == 0){
        t_layer.layer = module;
		t_layer.exe_success = false;
		t_layer.input_idx = 0;
		t_layer.name = "normal";
        net.layers.push_back(t_layer);
        return;
    }
	for(auto children : module.named_children()){
		if(children.name == "squeeze"){   // module is Fire Module
			int index = 0;
			for( auto fire_sub : module.named_children()){
				if(is_ReLu(index)){  
					index++;
					continue;
				}
				else if(fire_sub.name == "expand3x3"){
					t_layer.input_idx = -2;
				}
				else{
					t_layer.input_idx = 0;
				}
				t_layer.name = fire_sub.name; // expand1x1 , expand3x3 , squeeze
				t_layer.layer = fire_sub.value;
				t_layer.exe_success = false;
				net.layers.push_back(t_layer);
				if(fire_sub.name == "expand3x3"){
					t_layer.input_idx = 0;
					t_layer.layer = dummy;
					t_layer.from_idx = {-2, -1};
					t_layer.name = "concat";
					net.layers.push_back(t_layer);
				}
				index++;
			}
			break;
		}
		get_submodule_squeeze(children.value, net);
	}
}

void *predict_squeeze(Net *squeeze){
	int i;
	for(i=0;i<squeeze->layers.size();i++){
		pthread_mutex_lock(&mutex_t[squeeze->index_n]);
		cond_i[squeeze->index_n] = 1;
		squeeze->layers[i].exe_success = false;

		netlayer nl;
		nl.net = squeeze;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		thpool_add_work(thpool,(void(*)(void *))forward_squeeze,&th);

		while (cond_i[squeeze->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[squeeze->index_n], &mutex_t[squeeze->index_n]);
    	}
		squeeze->input.clear();
		squeeze->input.push_back(squeeze->layers[i].output);
		pthread_mutex_unlock(&mutex_t[squeeze->index_n]);
	}
	std::cout<<"\n*****"<<squeeze->name<<"*****" << "\n";
	std::cout<<(squeeze->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) <<"\n";
}
void forward_squeeze(th_arg *th){
	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
	std::vector<torch::jit::IValue> inputs;
	int k = nl->net->index;
	int n_all = nl->net->n_all;
	int success_check_idx; // expand1x1 = k+1, expand3x3 = k-1
	std::vector<int> stream_id = {nl->net->index_n, n_streamPerPool-2};

	//at::cuda::setCurrentCUDAStream(streams[(nl->net->index_n)]);
	
	if(nl->net->layers[k].name == "expand3x3"){
		int input_idx = k + nl->net->layers[k].input_idx;
		inputs.push_back(nl->net->layers[input_idx].output);
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
	{
		at::cuda::CUDAStreamGuard guard(streams[stream_id[0]]);
		
		if(k == nl->net->flatten){
			out = nl->net->layers[k].layer.forward(inputs).toTensor();
			out = out.view({out.size(0), -1});
		}
		else if(nl->net->layers[k].name == "concat"){
			std::vector<at::Tensor> cat_input;
			for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
				int concat_idx = k + nl->net->layers[k].from_idx[i];
				cat_input.push_back(nl->net->layers[concat_idx].output);
			}
			out = torch::cat(cat_input, 1);
		}
		else if(nl->net->layers[k].name != "normal"){ // expand1x1, expand3x3, squeeze
			if(nl->net->layers[k].name == "expand1x1"){
				//at::cuda::setCurrentCUDAStream(streams[stream_id[0]]);
				{
					at::cuda::CUDAStreamGuard guard(streams[stream_id[0]]);
					success_check_idx=k+1;
					out = nl->net->layers[k].layer.forward(inputs).toTensor();
					out = torch::relu(out);
					//cudaEventRecord(nl->net->record[0],0);
				}
			}
			else if(nl->net->layers[k].name == "expand3x3"){
				//at::cuda::setCurrentCUDAStream(streams[stream_id[1]]);
				{
					at::cuda::CUDAStreamGuard guard(streams[stream_id[1]]);
					success_check_idx=k-1;
					out = nl->net->layers[k].layer.forward(inputs).toTensor();
					out = torch::relu(out);
					//cudaEventRecord(nl->net->record[1],0);
				}
			}
			else{
				out = nl->net->layers[k].layer.forward(inputs).toTensor();
				out = torch::relu(out);
			}
		}
		else{
		out = nl->net->layers[k].layer.forward(inputs).toTensor();
		}
	}

	if(nl->net->layers[k].name == "expand1x1"){
		//cudaEventSynchronize(nl->net->record[0]);
		nl->net->layers[k].exe_success = true;
	}
	else if(nl->net->layers[k].name == "expand3x3"){
		//cudaEventSynchronize(nl->net->record[1]);
		nl->net->layers[k].exe_success = true;
	}
	
	nl->net->layers[k].output = out;

	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	if(nl->net->layers[k].name != "expand1x1" && nl->net->layers[k].name != "expand3x3"){
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}else if(nl->net->layers[success_check_idx].exe_success){
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}




