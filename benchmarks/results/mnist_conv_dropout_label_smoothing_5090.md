config=device:cuda,compile:True,T:25,encoding:poisson,batch:128,hidden:256,epochs:4,lr:0.003,dropout:0.1,label_smoothing:0.05,grad_clip:1.0,matmul_precision:highest,surrogate_slope:5.0,hard_forward:True,synapse_init:fan_in,train_examples:10000,test_examples:4096

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
| 1 | 1 | 2.301878 | 0.1094 | 2.282216 | 0.1797 | 20530.235 |
| 200 | 3 | 1.522038 | 0.9922 |  |  | 4.471 |
| 316 | 4 | 1.510589 | 1.0000 | 1.528146 | 0.9736 | 3.882 |

final_test_loss=1.534343
final_test_accuracy=0.9646
total_training_seconds=51.206
peak_cuda_memory_mb=899.411
average_step_ms=150.972
post_warmup_average_step_ms=86.276
steady_state_average_step_ms=86.496
