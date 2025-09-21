import torch
import torch.distributed as dist
import time

class SimpleDistributedDataParallel:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.sync_grads = False

        for param in self.model.parameters():
            rank_0_param = param.data.clone()
            dist.broadcast(rank_0_param, src=0)
        self._sync_time = 0
        self._sync_calls = 0

    def sync_gradients(self):
        """
        Call before optimizer step
        """
        if not self.sync_grads:
            return # probably gradient accumulation step, skip
        t0 = time.perf_counter()
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self._sync_time += t1 - t0
        self._sync_calls += 1
        
    @property
    def avg_sync_time(self):
        return self._sync_time / self._sync_calls if self._sync_calls > 0 else 0
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def disable_grad_sync(self):
        self.sync_grads = False

    def enable_grad_sync(self):
        self.sync_grads = True

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()