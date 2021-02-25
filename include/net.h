#ifndef NET_H
#define NET_H

#include <vector>
#include <torch/torch.h>
#include <functional>
#include "cuda_runtime.h"

struct Concat : torch::jit::Module{};

typedef struct _layer
{
	at::Tensor output;
	std::string name;
	torch::jit::Module layer;
	bool exe_success;
	std::vector<int> concat_idx;
}Layer;

typedef struct _net
{
	std::vector<Layer> layers;
	std::vector<torch::jit::IValue> input;
	at::Tensor identity;
	std::vector<cudaEvent_t> record;
	int index; //layer index
	int index_n; //network
}Net;

typedef struct _netlayer
{
	Net *net;
}netlayer;

#endif
