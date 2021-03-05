#include <torch/script.h> // 필요한 단 하나의 헤더파일.
#include <torch/torch.h>
#include <typeinfo>
#include <inttypes.h>
#include <iostream>
#include <string>
#include <memory>
#include "cuda_runtime.h"

#include "shuffle.h"

namespace F = torch::nn::functional;
using namespace std;


void get_submodule_shuffle(torch::jit::script::Module module, Net &net){
    Layer t_layer;
    Dummy concat;
    if(module.children().size() == 0){
        t_layer.input_idx = -1;
        t_layer.layer = module;
        t_layer.exe_success = false;
        net.layers.push_back(t_layer);
        return;
    }
    for(auto children : module.named_children()){
        if(children.name.find("stage") != std::string::npos){ //stage
            for(auto seq : children.value.named_children()){
                    bool exist_branch1 = false;
                    for(auto branch : seq.value.named_children()){
                        if(branch.name == "branch1"){
                            if(branch.value.children().size() != 0){
                                exist_branch1 = true;
                                t_layer.input_idx = -1;
                                t_layer.name = branch.name;
                                t_layer.layer = branch.value;
                                t_layer.exe_success = false;
                                net.layers.push_back(t_layer);
                            }
                        }
                        else if(branch.name == "branch2"){
                                if(exist_branch1){
                                        t_layer.input_idx = -2;
                                        t_layer.name = branch.name;
                                }
                                else{
                                        t_layer.input_idx = -1;
                                        t_layer.name = "chunk_and_branch2";
                                }
                                t_layer.layer = branch.value;
                                t_layer.exe_success = false;
                                net.layers.push_back(t_layer);
                        }
                    }
                    if(exist_branch1){
                        t_layer.from_idx = {-2, -1};

                    }
                    else{
                        t_layer.from_idx = {0, -1}; 
                    }
                    t_layer.input_idx = -1;
                    t_layer.name = "concat";
                    t_layer.layer = concat;
                    t_layer.exe_success = false;
                    net.layers.push_back(t_layer);
                }
        }else if(children.name.find("conv") != std::string::npos){
                t_layer.input_idx = -1;
                t_layer.name = children.name;
                t_layer.layer = children.value;
                t_layer.exe_success = false;
                net.layers.push_back(t_layer);
        }else{
                get_submodule_shuffle(children.value,net); //fc
        }
    }
}

at::Tensor channel_shuffle(at::Tensor x, int groups){
        int batchsize = x.sizes()[0];
        int num_channels = x.sizes()[1];
        int height = x.sizes()[2];
        int width = x.sizes()[3];
        int channels_per_group = num_channels / groups;
        x = x.view({batchsize,groups, channels_per_group,height, width});
        x = x.transpose(1,2).contiguous();
        x = x.view({batchsize, -1, height, width});

        return x;
}

void *predict_shuffle(Net *input){
        std::vector<torch::jit::IValue> inputs = input->input;
	int i;
	std::cout<<"input layers size = "<<input->layers.size()<<"\n";
	for(i=0;i<input->layers.size();i++){
		pthread_mutex_lock(&mutex_t[input->index_n]);
		cond_i[input->index_n] = 1;
                input->layers[i].exe_success = false;

		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
		nl.net = input;
		nl.net->index = i;

		th_arg th;
		th.arg = &nl;

		//std::cout << "Before thpool add work Shuffle "<< i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_shuffle,&th);
		//std::cout << "After thpool add work Shuffle "<< i << "\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		input->input.clear();
		input->input.push_back(input->layers[i].output);
		pthread_mutex_unlock(&mutex_t[input->index_n]);
	}
	std::cout<<"\n*****Shuffle result*****" << "\n";
	std::cout<<(input->layers[i-1].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) <<"\n";
}

void forward_shuffle(th_arg *th){
        pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
	netlayer *nl = th->arg;
        std::vector<torch::jit::IValue> inputs;
        int k = nl->net->index;
        int j;
        if(k==0) 
                inputs = nl->net->input;
        else 
                inputs.push_back(nl->net->layers[k + nl->net->layers[k].input_idx].output);
        
        if(nl->net->layers[k].name == "branch1"){
		cond_i[nl->net->index_n]=0;
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}
        cout<<"k = "<<k<<"   "<<nl->net->layers[k].name<<"\n";
        pthread_mutex_unlock(&mutex_t[nl->net->index_n]);

        at::Tensor out;
        if(k == nl->net->layers.size()-1){  //mean
                out = inputs[0].toTensor().mean({2,3});
                inputs.clear();
                inputs.push_back(out);
                out = nl->net->layers[k].layer.forward(inputs).toTensor();
	}
	else if(nl->net->layers[k].name == "concat"){
		std::vector<at::Tensor> cat_input;
                for(int i=0;i<nl->net->layers[k].from_idx.size();i++){
                        if(nl->net->layers[k].from_idx[i]>=0)
                                cat_input.push_back(nl->net->chunk[nl->net->layers[k].from_idx[i]]);
                        else
                                cat_input.push_back(nl->net->layers[k + nl->net->layers[k].from_idx[i]].output);
                }
                out = torch::cat(cat_input, 1);
                out = channel_shuffle(out, 2);
	}else{
                //chunk_and_branch , branch1, branch2 , conv, maxpool
                cout<<"else\n";
                if(nl->net->layers[k].name == "branch1"){
                        j=1;
                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
                        cudaEventRecord(nl->net->record[0],0);
                }else if(nl->net->layers[k].name == "branch2"){
                        j=-1;
                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
                        cudaEventRecord(nl->net->record[1],0);
                }else if(nl->net->layers[k].name == "chunk_and_branch2"){
                        nl->net->chunk = inputs[0].toTensor().chunk(2,1); //check
                        inputs.clear();
                        inputs.push_back(nl->net->chunk[1]);
                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
                }else{
                        out = nl->net->layers[k].layer.forward(inputs).toTensor();
                }
        }
        if(nl->net->layers[k].name == "branch1"){
		cudaEventSynchronize(nl->net->record[0]);
		nl->net->layers[k].output = out;
		nl->net->layers[k].exe_success = true;
	}
	else if(nl->net->layers[k].name == "branch2"){
		cudaEventSynchronize(nl->net->record[1]);
		nl->net->layers[k].output = out;
		nl->net->layers[k].exe_success = true;
	}
	else
		nl->net->layers[k].output = out;

        pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
        cond_i[nl->net->index_n]=0;
        if(nl->net->layers[k].name != "branch1" && nl->net->layers[k].name != "branch2"){
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}else if(nl->net->layers[k].exe_success && nl->net->layers[k+j].exe_success){
		pthread_cond_signal(&cond_t[nl->net->index_n]);
	}
        nl->net->index = k;
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
}
