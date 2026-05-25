config=device:cuda,compile:True,T:25,encoding:poisson,batch:128,hidden:256,epochs:4,lr:0.003,grad_clip:1.0,matmul_precision:highest,surrogate_slope:5.0,hard_forward:True,synapse_init:fan_in,train_examples:10000,test_examples:4096

| Parameter | Shape | Trainable | Count |
|---|---:|---:|---:|
| features.0.weight | 16x1x3x3 | True | 144 |
| features.0.bias | 16 | True | 16 |
| features.3.weight | 32x16x3x3 | True | 4608 |
| features.3.bias | 32 | True | 32 |
| hidden.synapse.weight | 1568x256 | True | 401408 |
| hidden.synapse.bias | 256 | True | 256 |
| output.synapse.weight | 256x10 | True | 2560 |
| output.synapse.bias | 10 | True | 10 |

total_params=409034
trainable_params=409034

| Step | Epoch | Loss | Train Acc | Val Loss | Val Acc | Step ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2.301965 | 0.1094 | 2.278040 | 0.1582 | 2771.734 |
| 200 | 3 | 1.482435 | 0.9844 |  |  | 4.554 |
| 316 | 4 | 1.463720 | 1.0000 | 1.490990 | 0.9658 | 1.774 |

final_test_loss=1.497434
final_test_accuracy=0.9578
total_training_seconds=8.090
peak_cuda_memory_mb=900.286
average_step_ms=18.860
post_warmup_average_step_ms=10.121
steady_state_average_step_ms=10.108
