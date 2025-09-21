from sillydp.training_utils.utils import get_tokenized_dataset, get_tokenizer, set_seed_all, create_run_name, ddp_setup
from sillydp.data_parallel.data_parallel import SimpleDistributedDataParallel
from cyclopts import App
import os
import torch.distributed as dist
import wandb
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node # type: ignore

app = App()

@app.default
def train_model(
    seed: int = 1337,
    batch_size: int = 256,
    per_device_train_batch_size: int = 8,
    seq_length: int = 1024,
    warmup_steps: int = 1000,
    total_steps: int = 88_000,
    project: str = "nano-dilco",
    run_topic: str = "ddp-single-node",
    lr: float = 4e-4,
    steps: int = 10_000,
    dataset_path: str = "/mnt/hf-c4-tiny/datasets/PrimeIntellect/c4-tiny/en/save_to_disk",
    llama_config: dict = None, # type: ignore
    wandb_config: dict = None, # type: ignore
):
    set_seed_all(seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    assert batch_size % per_device_train_batch_size == 0
    gradient_accumulation_steps = batch_size / per_device_train_batch_size

    if local_rank == 0:
        run_name = create_run_name("comm-measurement", wandb_config)
        wandb.init(project=project, name=run_name, config=wandb_config)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer=get_tokenizer()
    tokenized_ds = get_tokenized_dataset(dataset_path=dataset_path, tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = split_dataset_by_node(
        tokenized_ds, world_size=world_size, rank=local_rank
    )
    def collate_func(batch):
        padded = tokenizer.pad(
            batch,
            padding="longest",
            max_length=None,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        padded['labels'] = padded['input_ids'].clone()
        return padded

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=collate_func,
        drop_last=True,
        shuffle=True
    )
    # Load model configuration and tokenizer
    config = LlamaConfig(**llama_config)
    model = LlamaForCausalLM(config)
    model.to(local_rank) # type: ignore
    dp_model = SimpleDistributedDataParallel(model) # type: ignore
    optimizer = torch.optim.SGD(dp_model.model.parameters(), lr=lr)
    model.train()
    num_batches = 0
    for (i, batch) in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        if i > 2048:
            break
        if (i + 1) % gradient_accumulation_steps == 0:
            dp_model.enable_grad_sync()
        else:
            dp_model.disable_grad_sync()
        output = dp_model(**batch)
        loss = output.loss / gradient_accumulation_steps
        output.loss.backward()
        if dp_model.sync_grads:
            dp_model.sync_gradients()
            optimizer.step()
            optimizer.zero_grad()
            if local_rank == 0:
                wandb.log({"loss": loss.item() * gradient_accumulation_steps, "step": i, "avg_sync_time_seconds": dp_model.avg_sync_time, "perplexity": torch.exp(loss).item()})
                num_batches += 1
    print(f"Average sync time: {dp_model.avg_sync_time}")
    print("Training finished!")
    wandb.finish()


def main():
    print("Training model...")
    ddp_setup()
    app()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
