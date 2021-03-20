#ifndef TEST_H
#define TEST_H

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <pthread.h>
#include "thpool.h"

//#define n_streams 2

extern threadpool thpool; 
extern pthread_cond_t *cond_t;
extern pthread_mutex_t *mutex_t;
extern int *cond_i;
extern std::vector<at::cuda::CUDAStream> streams;

#endif
