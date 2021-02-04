#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>

#include "vgg.h"

namespace F = torch::nn::functional;
//using namespace std;

void get_submodule_vgg(torch::jit::script::Module module, std::vector<torch::jit::Module> &child){
	if(module.children().size() == 0){
		child.push_back(module);
		return;
	}
	for(auto children : module.named_children()){
		get_submodule_vgg(children.value, child);
	}
}

void *predict_vgg(Net *input){
	std::vector<torch::jit::Module> child = input->child;
	std::vector<torch::jit::IValue> inputs = input->inputs;
	
	for(int i=0;i<child.size();i++){
		cond_i[input->index_n] = 1;
		
		netlayer *nl = (netlayer *)malloc(sizeof(netlayer));
		nl->net = input;
		nl->index = i;
		th_arg th_input;
		th_input.arg = nl;
		std::cout << "Before thpool add work VGG " << i << "\n";
		thpool_add_work(thpool,(void(*)(void *))forward_vgg,&th_input);
		std::cout << "After thpool add work VGG " << i <<"\n";
		while (cond_i[input->index_n] == 1)
    	{
           	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
    	}
		inputs.clear();
		inputs.push_back(input->output);
	}
	std::cout << "\n*****VGG result*****" << "\n";
	std::cout << (input->output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";	

}

void forward_vgg(th_arg *th_input){
	netlayer *nl = th_input->arg;
	pthread_mutex_lock(&mutex_t[nl->net->index_n]);
	std::cout<<"forward_vgg start"<<"\n";
	at::Tensor out;
	std::vector<torch::jit::Module> child = nl->net->child;
	std::vector<torch::jit::IValue> inputs = nl->net->inputs;
	int k = nl->index;
	if(k == 32){
		std::cout<<"vgg flatten"<<"\n";
		out = nl->net->output.view({nl->net->output.size(0), -1});
		std::cout<<"????\n";
		inputs.clear();
		inputs.push_back(out);
		out = child[k].forward(inputs).toTensor();
	}
	else{
		std::cout<<"vgg else "<< k <<"\n";
		out = child[k].forward(inputs).toTensor();
	}
	std::cout<<"out sizes = "<<out.sizes()<<"\n";
	nl->net->inputs.clear();
	nl->net->inputs.push_back(out);
	nl->net->output = out;
	cond_i[nl->net->index_n]=0;
	pthread_cond_signal(&cond_t[nl->net->index_n]);
	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);		
}
