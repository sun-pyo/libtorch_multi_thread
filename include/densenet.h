#ifndef DENSENET_H
#define DENSENET_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_densenet(torch::jit::script::Module module, Net& net);
at::Tensor vector_cat(std::vector<torch::jit::IValue> inputs);
at::Tensor denselayer_forward(std::vector<torch::jit::Module> module_list, std::vector<torch::jit::IValue> inputs, int idx);
at::Tensor denseblock_forward(std::vector<torch::jit::Module> module_list, std::vector<torch::jit::IValue> inputs, int idx, int num_layer);
void *predict_densenet(Net *input);
void forward_densenet(th_arg *th);


#endif