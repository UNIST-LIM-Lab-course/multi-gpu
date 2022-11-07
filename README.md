# Multi-GPU Demos

## Scripts

Single GPU

```shell
$ python3 main.py
```

Data Parallel

```
$ python3 main_dp.py
```

Distributed Data Parallel

```
$ torchrun --standalone --nproc_per_node=NUM_GPUS main_ddp.py
```

or 

```
$ python3 -m torch.distributed.launch --nproc_per_node=NUM_GPUS main_ddp.py
```