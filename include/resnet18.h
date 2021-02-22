#ifndef RESNET18_H
#define RESNET18_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_resnet18(torch::jit::script::Module module, Net &net);
void *predict_resnet18(Net *input);
void forward_resnet18(th_arg *th);

#endif
