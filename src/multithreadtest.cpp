// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/impl/CUDAGuardImpl.h>
// #include <c10/core/impl/InlineEvent.h>
// #include <c10/core/Event.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAMultiStreamGuard.h>
// #include <ATen/cuda/CUDAEvent.h>

// #include <torch/script.h>
// #include <torch/torch.h>
// #include <typeinfo>
// #include <iostream>
// #include <inttypes.h>
// #include <functional>
// #include <memory>

// #include <cuda_runtime.h>
// #include <thread>

// void *pre_Hong(at::cuda::CUDAStream *stream){
//     torch::Tensor tensor0 = torch::ones({2, 2}, torch::device(torch::kCUDA));
//     torch::Tensor tensor1 = torch::ones({2, 2}, torch::device(torch::kCUDA));
//     for(int i=0;i<1000;i++){
//         {
//             at::cuda::CUDAStreamGuard stream_guard(*stream);
//             //at::cuda::setCurrentCUDAStream(*stream);
//             tensor0.sum();
//         }
//     }
//     //tensor0.sub(tensor1);
// }

// void *pre_Cho(at::cuda::CUDAStream *stream){
//     torch::Tensor tensor1 = torch::ones({2, 2}, torch::device({torch::kCUDA,0}));
//     torch::Tensor tensor2 = torch::ones({2, 2}, torch::device({torch::kCUDA,0}));
//     for(int i=0;i<1000;i++){
//         //std::this_thread::sleep_for(std::chrono::seconds(1));
//         {
//             at::cuda::CUDAStreamGuard stream_guard(*stream);
//             //at::cuda::setCurrentCUDAStream(*stream);
//             tensor1.sub(tensor2);
//         }
//     }
//     //tensor1.sum();
// }

// int main(){
//     pthread_t thread1[100];
//     pthread_t thread2[100];
//     at::cuda::CUDAStream s0 = at::cuda::getStreamFromPool(false, 0);
//     at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool(false, 0);

//     for(int i=0;i<100;i++){
//         pthread_create(&thread1[i],NULL,(void *(*)(void*))pre_Hong,&s0);
//         pthread_create(&thread2[i],NULL,(void *(*)(void*))pre_Cho,&s1);
//     }
//     for (int i = 0; i < 100;i++){
//         pthread_join(thread1[i], NULL);
//         pthread_join(thread2[i], NULL);
//     }
//     at::cuda::device_synchronize();
// }