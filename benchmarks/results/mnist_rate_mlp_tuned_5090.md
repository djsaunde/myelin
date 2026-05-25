config=device:cuda,compile:True,compile_requested:True,T:25,encoding:poisson,batch:128,hidden:256,epochs:3,lr:0.003,grad_clip:0.1,matmul_precision:highest,surrogate_slope:5.0,hard_forward:True,backend:auto,resolved_backend:triton,checkpoint_size:balanced,resolved_checkpoint_size:7,train_examples:10000,test_examples:4096

| Parameter | Shape | Trainable | Count |
|---|---:|---:|---:|
| hidden.synapse.weight | 784x256 | True | 200704 |
| hidden.synapse.bias | 256 | True | 256 |
| output.synapse.weight | 256x10 | True | 2560 |
| output.synapse.bias | 10 | True | 10 |

total_params=203530
trainable_params=203530

| Step | Epoch | Loss | Train Acc | Val Loss | Val Acc | Step ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2.302585 | 0.1250 | 2.297608 | 0.1016 | 5415.151 |
| 200 | 3 | 1.519323 | 0.9219 |  |  | 2.171 |
| 237 | 3 | 1.523897 | 0.9375 | 1.528416 | 0.9199 | 1.801 |

final_test_loss=1.541005
final_test_accuracy=0.9082
total_training_seconds=10.122
peak_cuda_memory_mb=114.043
average_step_ms=36.262
post_warmup_average_step_ms=13.471
steady_state_average_step_ms=13.478
