#ifndef NET_H
#define NET_H

#include <vector>
#include <torch/torch.h>
#include <string>
#include <functional>
#include "cuda_runtime.h"

struct Dummy : torch::jit::Module{};

typedef struct _layer
{
	at::Tensor output;
	std::string name;
	torch::jit::Module layer;
	bool exe_success;
	std::vector<int> from_idx; //concat
	std::vector<int> branch_idx; // last layer idx of branch for eventrecord
	int input_idx;
	int event_idx;
	int skip; //inception wait skip num
}Layer;

typedef struct _net
{
	std::vector<Layer> layers;
	std::vector<torch::jit::IValue> input;
	at::Tensor identity;
	std::vector<cudaEvent_t> record;
	std::vector<at::Tensor> chunk; //shuffle
	string name;
	int index; //layer index
	int index_n; //network
	int n_all; // all network num
	int flatten;
}Net;

typedef struct _netlayer
{
	Net *net;
}netlayer;

#endif
