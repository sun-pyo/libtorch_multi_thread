// #include <torch/script.h>
// #include <torch/torch.h>
// #include <typeinfo>
// #include <iostream>
// #include <inttypes.h>
// #include <memory>

// #include "resnet18.h"

// using namespace std;
// namespace F = torch::nn::functional;

// void get_submodule_resnet18(torch::jit::script::Module module,std::vector<torch::jit::Module> &child, vector<pair<int, int>> &block){
	
//     if(module.children().size() == 0){
// 	    child.push_back(module);
//             return;
//     }
//     for(auto children : module.named_children()){
// 		if(children.name.find("stage") != std::string::npos){
// 			for(auto layer : children.value.named_children()){
// 				int idx = child.size();
// 				int count = 0;
// 				for(auto c : layer.value.children()){
// 					count++;
// 				}
// 				get_submodule_resnet18(layer.value, child, block);
// 				block.push_back(make_pair(idx, count));
// 			}
// 			continue;
// 		}

// 		if(children.name != "downsample") //conv1x1
// 			get_submodule_resnet18(children.value, child, block);
// 		else
// 			child.push_back(children.value);

//     }
// }

// void *predict_resnet18(Net *input){
//     std::vector<torch::jit::Module> child = input->child;
// 	std::vector<torch::jit::IValue> inputs = input->input;
// 	//std::cout<<"child = "<<child.size()<<'\n';
// 	//std::cout<<"basic = "<<input->block.size()<<'\n';
// 	int i;
// 	for(i = 0;i<child.size();i++) {
// 		pthread_mutex_lock(&mutex_t[input->index_n]);
// 		cond_i[input->index_n] = 1; //right?
		
// 		netlayer nl;// = (netlayer *)malloc(sizeof(netlayer));
// 		nl.net = input;
// 		nl.net->index = i;

// 		th_arg th;
// 		th.arg = &nl;
// 		//std::cout << "Before thpool add work RES " << i << "\n";
// 		thpool_add_work(thpool,(void(*)(void *))forward_resnet18,&th);
// 		//std::cout << "After thpool add work RES " << i <<"\n";
// 		//cout<<"id = " <<(nl.net->identity)<<"\n";
// 		while (cond_i[input->index_n] == 1)
//     	{
//            	pthread_cond_wait(&cond_t[input->index_n], &mutex_t[input->index_n]);
//     	}
// 		i = nl.net->index;
// 		input->input.clear();
// 		input->input.push_back(input->layer[i].output);
// 		pthread_mutex_unlock(&mutex_t[input->index_n]);
// 	}
// 	std::cout << "\n*****Res result*****" << "\n";
// 	std::cout << (input->layer[i].output).slice(/*dim=*/1, /*start=*/0, /*end=*/15) << "\n";
// }

// void forward_resnet18(th_arg *th){
// 	pthread_mutex_lock(&mutex_t[th->arg->net->index_n]);
// 	netlayer *nl = th->arg;
// 	std::vector<torch::jit::Module> child = nl->net->child;
// 	std::vector<torch::jit::IValue> inputs = nl->net->input;
// 	std::vector<pair<int,int>> basicblock = nl->net->block;
// 	at::Tensor identity = nl->net->identity;
// 	//vector<torch::jit::IValue> input; // ==inputs?
// 	vector<torch::jit::IValue> inputs_cpy;
// 	int k =nl->net->index;
// 	at::Tensor out;
// 	int j;

// 	if(k==0)
//         j=0;
//     else
//         j = nl->net->j;
	

//     //std::cout<<"res layer index = "<<k<<"\n";
//     //output.clear();
//     if(j < basicblock.size() && k == basicblock[j].first){
// 	   //std::cout<<"basicblock\n";
// 	   identity = inputs[0].toTensor(); 
//     }

//     if(k == child.size()) //flatten
//     {	
// 		out = out.view({inputs[0].toTensor().size(0), -1});
// 		//out = out.view({out.size(0), -1});
//        	inputs.clear();
//        	//out =  out.to(at::kCPU);
//        	inputs.push_back(out);
//        	//child[k].to(at::kCPU);
//        	out = child[k].forward(inputs).toTensor();
//     }
//     else if(j < basicblock.size() && basicblock[j].second <= 6 && k == basicblock[j].first + basicblock[j].second - 1){
// 		//BasicBlock and downsample 
// 		if(basicblock[j].second == 6){
// 			inputs_cpy.clear();
// 			inputs_cpy.push_back(identity);
// 			identity = child[k].forward(inputs_cpy).toTensor();
// 		} //BasicBlock
// 		else{
// 			out = child[k].forward(inputs).toTensor();
// 		}
// 		out += identity;
// 		inputs.clear();
// 		inputs.push_back(out);
// 		out = child[basicblock[j].first + 2].forward(inputs).toTensor();
// 		j += 1;
//     }

//     else if(j < basicblock.size() && basicblock[j].second > 6 && k == basicblock[j].first + 6){
// 		// Bottleneck and downsample
// 		if(basicblock[j].second == 8){
// 			inputs_cpy.clear();
// 			inputs_cpy.push_back(identity);
// 			identity = child[k+1].forward(inputs_cpy).toTensor();
// 			k++;
// 		} // Bottleneck 
// 		out += identity;
// 		inputs.clear();
// 		inputs.push_back(out);
// 		// ReLu
// 		out = child[basicblock[j].first + 6].forward(inputs).toTensor();
// 		j += 1;
//     }
// 	else if(j < basicblock.size() && basicblock[j].second > 6 && (k == basicblock[j].first + 1 || k == basicblock[j].first + 3) )
// 	{
// 		// Bottleneck -> ReLu
// 		out = child[k].forward(inputs).toTensor();
// 		inputs.clear();
// 		inputs.push_back(out);
// 		out = child[basicblock[j].first + 6].forward(inputs).toTensor();
// 	}
// 	else{
// 		out = child[k].forward(inputs).toTensor();
// 	}

//     nl->net->layer[k].output = out;
// 	nl->net->identity = identity;
// 	nl->net->j = j;
// 	nl->net->index = k; //check
// 	cond_i[nl->net->index_n]=0;
// 	pthread_cond_signal(&cond_t[nl->net->index_n]);
// 	pthread_mutex_unlock(&mutex_t[nl->net->index_n]);
//     //cout<<out.sizes()<<"\n\n";
// }
