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
	std::string name;	//layer name
	torch::jit::Module layer;
	bool exe_success;	//layer operation complete or not
	std::vector<int> from_idx;	//concat
	std::vector<int> branch_idx;	// last layer idx of branch for eventrecord
	int input_idx; 	//network with branch
	int event_idx;	//network with branch
	int skip;	//inception skip num in a branch
	int stream_idx;	//stream index of current layers
}Layer;

typedef struct _net
{
	std::vector<Layer> layers;
	std::vector<torch::jit::IValue> input;
	at::Tensor identity;	//resnet
	std::vector<cudaEvent_t> record;
	std::vector<at::Tensor> chunk; //shuffle
	std::string name;	//network name
	int index; //layer index
	int index_n; //network index
	int n_all; // all network num
	int flatten; //flatten layer index
}Net;

typedef struct _netlayer
{
	Net *net;
}netlayer;

#endif
