#ifndef SQUEEZE_H
#define SQUEEZE_H

#include "net.h"
#include "test.h"
#include "thpool.h"

void get_submodule_squeeze(torch::jit::script::Module module, Net &net);
void *predict_squeeze(Net *input);
void forward_squeeze(th_arg *th);

#endif
