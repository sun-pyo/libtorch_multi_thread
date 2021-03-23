#include <torch/script.h>
#include <torch/torch.h>
#include <typeinfo>
#include <iostream>
#include <inttypes.h>
#include <functional>
#include <memory>
#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>

#include "test.h"
#include "alex.h"
#include "vgg.h"
#include "resnet.h"
#include "densenet.h"
#include "squeeze.h"
#include "mobile.h"
#include "mnasnet.h"
#include "inception.h"
#include "shuffle.h"

#define n_dense 2
#define n_res 2
#define n_alex 2
#define n_vgg 2
#define n_wide 2
#define n_squeeze 2
#define n_mobile 2
#define n_mnasnet 2
#define n_inception 2
#define n_shuffle 2
#define n_resX 2

#define n_threads 10


extern void *predict_alexnet(Net *input);
extern void *predict_vgg(Net *input);
extern void *predict_resnet(Net *input);
extern void *predict_densenet(Net *input);
extern void *predict_squeeze(Net *input);
extern void *predict_mobilenet(Net *input);
extern void *predict_MNASNet(Net *input);
extern void *predict_inception(Net *input);
extern void *predict_shuffle(Net *input);

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
std::vector<at::cuda::CUDAStream> streams;

int main(int argc, const char* argv[]) {
  
  int n_all = n_alex + n_vgg + n_res + n_dense + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle + n_resX;

  thpool = thpool_init(n_threads);

  for(int i=0; i<n_streamPerPool; i++){
    //cout<< "Make stream\n";
    streams.push_back(at::cuda::getStreamFromPool(false,0));
    //at::cuda::CUDAMultiStreamGuard multi_guard(streams);
    //cout<<"stream id = "<<streams[0].unwrap().id()<<"\n";
  }
  torch::jit::script::Module denseModule[n_dense];
  torch::jit::script::Module resModule[n_res];
  torch::jit::script::Module alexModule[n_alex];
  torch::jit::script::Module vggModule[n_vgg];
  torch::jit::script::Module wideModule[n_wide];
  torch::jit::script::Module squeezeModule[n_squeeze];
  torch::jit::script::Module mobileModule[n_mobile];
  torch::jit::script::Module mnasModule[n_mnasnet];
  torch::jit::script::Module inceptionModule[n_inception];
  torch::jit::script::Module shuffleModule[n_shuffle];
  torch::jit::script::Module resXModule[n_resX];
  try {
    for (int i=0;i<n_dense;i++){
      cout<<"Dens model loading ";
    	denseModule[i] = torch::jit::load("../densenet_model.pt");
      denseModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_res;i++){
      cout<<"Res model loading ";  
    	resModule[i] = torch::jit::load("../resnet_model.pt");
      resModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for(int i=0;i<n_alex;i++){
      cout<<"Alex model loading "; 
    	alexModule[i] = torch::jit::load("../alexnet_model.pt");
      alexModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_vgg;i++){
      cout<<"VGG model loading "; 
    	vggModule[i] = torch::jit::load("../vgg_model.pt");
      vggModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_wide;i++){
      cout<<"Wide Res model loading "; 
    	wideModule[i] = torch::jit::load("../wideresnet_model.pt");
      wideModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_squeeze;i++){
      cout<<"Squeeze model loading "; 
    	squeezeModule[i] = torch::jit::load("../squeeze_model.pt");
      squeezeModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_mobile;i++){
      cout<<"Mobile model loading "; 
    	mobileModule[i] = torch::jit::load("../mobilenet_model.pt");
      mobileModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_mnasnet;i++){
      cout<<"MNAS model loading "; 
    	mnasModule[i] = torch::jit::load("../mnasnet_model.pt");
      mnasModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_inception;i++){
      cout<<"Inception model loading "; 
    	inceptionModule[i] = torch::jit::load("../inception_model.pt");
      inceptionModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_shuffle;i++){
      cout<<"Shuffle model loading "; 
    	shuffleModule[i] = torch::jit::load("../shuffle_model.pt");
      shuffleModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
    for (int i=0;i<n_resX;i++){
      cout<<"ResNext model loading "; 
    	resXModule[i] = torch::jit::load("../resnext_model.pt");
      resXModule[i].to(at::kCUDA);
      cout<<"end\n";
    }
  }
  catch (const c10::Error& e) {
    cerr << "error loading the model\n";
    return -1;
  }
  cout<<"***** Model Load compelete *****"<<"\n";

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
  vector<torch::jit::IValue> inputs2;
  //module.to(at::kCPU);
   
  torch::Tensor x = torch::ones({1, 3, 224, 224}).to(at::kCUDA);
  torch::Tensor x2 = torch::ones({1, 3, 299, 299}).to(at::kCUDA);
  inputs.push_back(x);
  inputs2.push_back(x2);
  
  
  Net net_input_dense[n_dense];
  Net net_input_res[n_res];
  Net net_input_alex[n_alex];
  Net net_input_vgg[n_vgg];
  Net net_input_wide[n_wide];
  Net net_input_squeeze[n_squeeze];
  Net net_input_mobile[n_mobile];
  Net net_input_mnasnet[n_mnasnet];
  Net net_input_inception[n_inception];
  Net net_input_shuffle[n_shuffle];
   Net net_input_resX[n_resX];

  pthread_t networkArray_dense[n_dense];
  pthread_t networkArray_res[n_res];
  pthread_t networkArray_alex[n_alex];
  pthread_t networkArray_vgg[n_vgg];
  pthread_t networkArray_wide[n_wide];
  pthread_t networkArray_squeeze[n_squeeze];
  pthread_t networkArray_mobile[n_mobile];
  pthread_t networkArray_mnasnet[n_mnasnet];
  pthread_t networkArray_inception[n_inception];
  pthread_t networkArray_shuffle[n_shuffle];
  pthread_t networkArray_resX[n_resX];

  for(int i=0;i<n_dense;i++){
	  get_submodule_densenet(denseModule[i], net_input_dense[i]);
    std::cout << "End get submodule_densenet "<< i << "\n";
    net_input_dense[i].input = inputs;
    net_input_dense[i].name = "DenseNet";
    net_input_dense[i].flatten = 641;
    net_input_dense[i].index_n = i;
  }

  for(int i=0;i<n_res;i++){
	  get_submodule_resnet(resModule[i], net_input_res[i]);
    std::cout << "End get submodule_resnet "<< i << "\n";
    net_input_res[i].name = "ResNet";
    net_input_res[i].flatten = net_input_res[i].layers.size()-1;
	  net_input_res[i].input = inputs;
    net_input_res[i].index_n = i+n_dense;
  }

  for(int i=0;i<n_alex;i++){
	  get_submodule_alexnet(alexModule[i], net_input_alex[i]);
    std::cout << "End get submodule_alexnet " << i <<"\n";
	  net_input_alex[i].input = inputs;
    net_input_alex[i].name = "AlexNet";
    net_input_alex[i].flatten = net_input_alex[i].layers.size()-7;
    net_input_alex[i].index_n = i+ n_res + n_dense;
  }

  for(int i=0;i<n_vgg;i++){
	  get_submodule_vgg(vggModule[i], net_input_vgg[i]);
    std::cout << "End get submodule_vgg " << i << "\n";
	  net_input_vgg[i].input = inputs;
    net_input_vgg[i].name = "VGG";
    net_input_vgg[i].flatten = 32;
    net_input_vgg[i].index_n = i + n_alex + n_res + n_dense;
  }

  for(int i=0;i<n_wide;i++){
	  get_submodule_resnet(wideModule[i], net_input_wide[i]);
    std::cout << "End get submodule_widenet "<< i << "\n";
	  net_input_wide[i].input = inputs;
    net_input_wide[i].name = "WideResNet";
    net_input_wide[i].flatten = net_input_wide[i].layers.size()-1;
    net_input_wide[i].index_n = i+n_alex + n_res + n_dense + n_vgg;
  }

  for(int i=0;i<n_squeeze;i++){
	  get_submodule_squeeze(squeezeModule[i], net_input_squeeze[i]);
    std::cout << "End get submodule_squeezenet "<< i << "\n";
    for(int j=0;j<2;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_squeeze[i].record.push_back(event_temp);
    }
    net_input_squeeze[i].name = "SqueezeNet";
    net_input_squeeze[i].flatten = net_input_squeeze[i].layers.size()-1;
    net_input_squeeze[i].n_all = n_all;
	  net_input_squeeze[i].input = inputs;
    net_input_squeeze[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide;
  }

  for(int i=0;i<n_mobile;i++){
	  get_submodule_mobilenet(mobileModule[i], net_input_mobile[i]);
    std::cout << "End get submodule_mobilenet "<< i << "\n";
	  net_input_mobile[i].input = inputs;
    net_input_mobile[i].name = "Mobile";
    net_input_mobile[i].flatten = net_input_mobile[i].layers.size()-2;
    net_input_mobile[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze;
  }

  for(int i=0;i<n_mnasnet;i++){
	  get_submodule_MNASNet(mnasModule[i], net_input_mnasnet[i]);
    std::cout << "End get submodule_mnasnet "<< i << "\n";
	  net_input_mnasnet[i].input = inputs;
    net_input_mnasnet[i].name = "MNASNet";
    net_input_mnasnet[i].flatten = net_input_mnasnet[i].layers.size()-2;
    net_input_mnasnet[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile;
  }
  for(int i=0;i<n_inception;i++){
	  get_submodule_inception(inceptionModule[i], net_input_inception[i]);
    std::cout << "End get submodule_inception "<< i << "\n";
    for(int j=0;j<4;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_inception[i].record.push_back(event_temp);
    }
    net_input_inception[i].n_all = n_all;
	  net_input_inception[i].input = inputs2;
    net_input_inception[i].name = "Inception_v3";
    net_input_inception[i].flatten = 123;
    net_input_inception[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet;
  }
  for(int i=0;i<n_shuffle;i++){
	  get_submodule_shuffle(shuffleModule[i], net_input_shuffle[i]);
    std::cout << "End get submodule_shuffle "<< i << "\n";
    for(int j=0;j<2;j++){
      cudaEvent_t event_temp;
      cudaEventCreate(&event_temp);
      net_input_shuffle[i].record.push_back(event_temp);
    }
    net_input_shuffle[i].n_all = n_all;
	  net_input_shuffle[i].input = inputs;
    net_input_shuffle[i].name = "ShuffleNet";
    net_input_shuffle[i].flatten = 38;
    net_input_shuffle[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception;
  }
  for(int i=0;i<n_resX;i++){
	  get_submodule_resnet(resXModule[i], net_input_resX[i]);
    std::cout << "End get submodule_resnext "<< i << "\n";
	  net_input_resX[i].input = inputs;
    net_input_resX[i].name = "ResNext";
    net_input_resX[i].flatten = net_input_resX[i].layers.size()-1;
    net_input_resX[i].index_n = i + n_alex + n_res + n_dense + n_vgg + n_wide + n_squeeze + n_mobile + n_mnasnet + n_inception + n_shuffle;
  }

for(int i=0;i<n_dense;i++){
    if (pthread_create(&networkArray_dense[i], NULL, (void *(*)(void*))predict_densenet, &net_input_dense[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_res;i++){
    if (pthread_create(&networkArray_res[i], NULL, (void *(*)(void*))predict_resnet, &net_input_res[i]) < 0){
      perror("thread error");
      exit(0);
    }
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
  for(int i=0;i<n_wide;i++){
    if (pthread_create(&networkArray_wide[i], NULL, (void *(*)(void*))predict_resnet, &net_input_wide[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_squeeze;i++){
    if (pthread_create(&networkArray_squeeze[i], NULL, (void *(*)(void*))predict_squeeze, &net_input_squeeze[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_mobile;i++){
    if (pthread_create(&networkArray_mobile[i], NULL, (void *(*)(void*))predict_mobilenet, &net_input_mobile[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_mnasnet;i++){
    if (pthread_create(&networkArray_mnasnet[i], NULL, (void *(*)(void*))predict_MNASNet, &net_input_mnasnet[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for(int i=0;i<n_inception;i++){
    if (pthread_create(&networkArray_inception[i], NULL, (void *(*)(void*))predict_inception, &net_input_inception[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_shuffle;i++){
    if (pthread_create(&networkArray_shuffle[i], NULL, (void *(*)(void*))predict_shuffle, &net_input_shuffle[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }
  for(int i=0;i<n_resX;i++){
    if (pthread_create(&networkArray_resX[i], NULL, (void *(*)(void*))predict_resnet, &net_input_resX[i]) < 0){
      perror("thread error");
      exit(0);
    }
  }

  for (int i = 0; i < n_dense; i++){
    pthread_join(networkArray_dense[i], NULL);
  }
  for (int i = 0; i < n_res; i++){
    pthread_join(networkArray_res[i], NULL);
  }
  for (int i = 0; i < n_alex; i++){
    pthread_join(networkArray_alex[i], NULL);
  }
  for (int i = 0; i < n_vgg; i++){
    pthread_join(networkArray_vgg[i], NULL);
  }
  for (int i = 0; i < n_wide; i++){
    pthread_join(networkArray_wide[i], NULL);
  }
  for (int i = 0; i < n_squeeze; i++){
    pthread_join(networkArray_squeeze[i], NULL);
  }
  for (int i = 0; i < n_mobile; i++){
    pthread_join(networkArray_mobile[i], NULL);
  }
  for (int i = 0; i < n_mnasnet; i++){
    pthread_join(networkArray_mnasnet[i], NULL);
  }
  for (int i = 0; i < n_inception; i++){
    pthread_join(networkArray_inception[i], NULL);
  }
  for (int i = 0; i < n_shuffle; i++){
    pthread_join(networkArray_shuffle[i], NULL);
  }
  for (int i = 0; i < n_resX; i++){
    pthread_join(networkArray_resX[i], NULL);
  }
  //at::cuda::device_synchronize();
  free(cond_t);
  free(mutex_t);
  free(cond_i);
}
