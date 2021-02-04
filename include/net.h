#ifndef NET_H
#define NET_H
#include<vector>
#include <torch/torch.h>

//extern "C"{

typedef struct _net
{
	std::vector<torch::jit::Module> child;
	std::vector<torch::jit::IValue> inputs;
	at::Tensor output;
	int index_n;
	int index;
}Net;
typedef struct _netlayer
{
	Net *net;
	int index; //layer index
}netlayer;

//}
#endif
