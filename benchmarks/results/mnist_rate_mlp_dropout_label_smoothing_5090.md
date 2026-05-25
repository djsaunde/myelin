config=device:cuda,compile:True,compile_requested:True,T:25,encoding:poisson,batch:128,hidden:512,epochs:4,lr:0.003,dropout:0.1,label_smoothing:0.05,grad_clip:0.1,matmul_precision:highest,surrogate_slope:5.0,hard_forward:True,backend:auto,resolved_backend:triton,checkpoint_size:balanced,resolved_checkpoint_size:7,train_examples:20000,test_examples:10000

| Parameter | Shape | Trainable | Count |
|---|---:|---:|---:|
| hidden.synapse.weight | 784x512 | True | 401408 |
| hidden.synapse.bias | 512 | True | 512 |
| output.synapse.weight | 512x10 | True | 5120 |
| output.synapse.bias | 10 | True | 10 |

total_params=407050
trainable_params=407050

| Step | Epoch | Loss | Train Acc | Val Loss | Val Acc | Step ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2.302585 | 0.0859 | 2.291712 | 0.2783 | 5338.328 |
| 300 | 2 | 1.556593 | 0.9453 |  |  | 1.695 |
| 600 | 4 | 1.530531 | 0.9766 |  |  | 2.878 |
| 628 | 4 | 1.564093 | 0.9688 | 1.545813 | 0.9482 | 2.469 |

final_test_loss=1.544614
final_test_accuracy=0.9551
total_training_seconds=11.868
peak_cuda_memory_mb=145.002
average_step_ms=15.273
post_warmup_average_step_ms=6.783
steady_state_average_step_ms=6.777
