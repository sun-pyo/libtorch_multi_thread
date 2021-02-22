#include <torch/script.h> // 필요한 단 하나의 헤더파일.
#include <torch/torch.h>
#include <typeinfo>

#include <iostream>
#include <inttypes.h>

#include <iostream>
#include <memory>


namespace F = torch::nn::functional;
using namespace std;

// https://blog.kerrycho.com/Image-Segmentation-Libtorch/

at::Tensor Fire(vector<torch::jit::Module> module_list, vector<torch::jit::IValue> inputs){
	at::Tensor out = module_list[0].forward(inputs).toTensor();
	inputs.clear();
	inputs.push_back(out);
	out = module_list[1].forward(inputs).toTensor();
	inputs.clear();
	inputs.push_back(out);


	at::Tensor out1 = module_list[2].forward(inputs).toTensor();
	vector<torch::jit::IValue> input;
	input.push_back(out1);
	out1 = module_list[3].forward(input).toTensor();

	at::Tensor out2 = module_list[4].forward(inputs).toTensor();
	input.clear();
	input.push_back(out2);
	out2 = module_list[5].forward(input).toTensor();

	out = torch::cat({out1, out2}, 1);
	return out;
}



void get_submodule_squeeze(torch::jit::script::Module module, std::vector<vector<torch::jit::Module>> &child, vector<int> &fire){
	if(module.children().size() == 0){
		vector<torch::jit::Module> temp;
		temp.push_back(module);
		child.push_back(temp);
		return;
	}
	for(auto children : module.named_children()){
    	  	if(children.name == "squeeze"){
			fire.push_back(child.size());
			vector<torch::jit::Module> squeeze;
			for( auto ch : module.children()){
				squeeze.push_back(ch);	
			}
			child.push_back(squeeze);
			break;
		}
		get_submodule_squeeze(children.value, child, fire);
	}

}

at::Tensor forward_squeeze(vector<vector<torch::jit::Module>> &child, vector<torch::jit::IValue> inputs, vector<int> fire){
	at::Tensor out;
	cout<<"\n";
	int j = 0;
	for(int i=0;i<child.size();i++){
		cout<<i<<"\n";
		if(i==16){
			out = child[i][0].forward(inputs).toTensor();
			out = out.view({out.size(0), -1});
		}
		else if(j<fire.size() && i==fire[j]){
			out = Fire(child[i], inputs);
			j++;
		}	
		else{
			out = child[i][0].forward(inputs).toTensor();
		}

		inputs.clear();
		inputs.push_back(out);
	}
	return out;
}



