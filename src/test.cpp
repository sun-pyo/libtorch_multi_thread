#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <memory>
#include <stdlib.h>
#include <pthread.h>

#include "test.h"
#include "alex.h"
#include "vgg.h"

#define n_alex 1
#define n_vgg 1
#define n_threads 4

extern void *predict_alexnet(Net *input);
extern void *predict_vgg(Net *input);

namespace F = torch::nn::functional;
using namespace std;

void print_script_module(const torch::jit::script::Module& module, size_t spaces) {
    for (const auto& sub_module : module.named_children()) {
        if (!sub_module.name.empty()) {
            std::cout << std::string(spaces, ' ') << sub_module.value.type()->name().value().name()
                << " " << sub_module.name << "\n";    
        }
        print_script_module(sub_module.value, spaces + 2);
    }
}

void print_vector(vector<int> v){
	for(int i=0;i<v.size();i++){
		cout<<v[i]<<" ";
	}
	cout<<"\n";
}

threadpool thpool;
pthread_cond_t* cond_t;
pthread_mutex_t* mutex_t;
int* cond_i;

int main(int argc, const char* argv[]) {

  int n_all = n_alex +n_vgg;

  thpool = thpool_init(n_threads);

  torch::jit::script::Module alexModule[n_alex];
  torch::jit::script::Module vggModule[n_vgg];

  try {
    for(int i=0;i<n_alex;i++){
    	alexModule[i] = torch::jit::load("../alex_model.pt");
    }
    for (int i=0;i<n_vgg;i++){    
    	vggModule[i] = torch::jit::load("../vgg_model.pt");
    }
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return -1;
  }
  cout<<"Model Load compelete"<<"\n";

  cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
  mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
  cond_i = (int *)malloc(sizeof(int) * n_all);

  for (int i = 0; i < n_all; i++)
  {
      pthread_cond_init(&cond_t[i], NULL);
      pthread_mutex_init(&mutex_t[i], NULL);
      cond_i[i] = 0;
  }


  vector<torch::jit::IValue> inputs;
  //module.to(at::kCPU);
   
  torch::Tensor x = torch::ones({1, 3, 224, 224});
  inputs.push_back(x);
 
  Net net_input_alex[n_alex];
  Net net_input_vgg[n_vgg];

  pthread_t networkArray_alex[n_alex];
  pthread_t networkArray_vgg[n_vgg];
  

  std::vector<torch::jit::Module> alexchild[n_alex];
  std::vector<torch::jit::Module> vggchild[n_vgg]; 
 
  for(int i=0;i<n_alex;i++){
	  get_submodule_alexnet(alexModule[i], alexchild[i]);
    //std::cout<<alexchild[i].size()<<'\n';
    std::cout << "End get submodule_alex" << "\n";
	  //net_input_alex[i] = (Net *)malloc(sizeof(Net));
    std::cout<<"11111"<<"\n";
	  net_input_alex[i].child = alexchild[i];
    std::cout<<"22222"<<"\n";
	  net_input_alex[i].inputs = inputs;
    std::cout<<"33333"<<"\n";
    net_input_alex[i].index_n = i;
    std::cout<<"4444"<<"\n";
  }

  for(int i=0;i<n_vgg;i++){
	  get_submodule_vgg(vggModule[i], vggchild[i]);
    std::cout << "End get submodule_vgg" << "\n";
	  //net_input_vgg[i] = (Net *)malloc(sizeof(Net));
	  net_input_vgg[i].child = vggchild[i];
	  net_input_vgg[i].inputs = inputs;
    net_input_vgg[i].index_n = i + n_alex;
  }

  for(int i=0;i<n_alex;i++){
    if (pthread_create(&networkArray_alex[i], NULL, (void *(*)(void*))predict_alexnet, &net_input_alex[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_vgg;i++){
	  if (pthread_create(&networkArray_vgg[i], NULL, (void *(*)(void*))predict_vgg, &net_input_vgg[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for (int i = 0; i < n_alex; i++){
    pthread_join(networkArray_alex[i], NULL);
  }
  for (int i = 0; i < n_vgg; i++){
    pthread_join(networkArray_vgg[i], NULL);
  }

  free(cond_t);
  free(mutex_t);
  free(cond_i);
}
