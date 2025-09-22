# naive-data-parallel
Implementation of naive data parallel training

## Usage

### Basic Usage
Run with default configurations:
```bash
python -m sillydp.main
```

### Custom Configuration Files
You can provide custom configurations via JSON files:

```bash
python -m sillydp.main --llama-config-file configs/llama_small.json --wandb-config-file configs/wandb_custom.json
```

### Command Line Arguments
Available arguments:
- `--seed`: Random seed (default: 1337)
- `--batch-size`: Global batch size (default: 256)
- `--per-device-train-batch-size`: Batch size per device (default: 8)
- `--lr`: Learning rate (default: 4e-4)
- `--steps`: Number of training steps (default: 10,000)
- `--dataset-path`: Path to tokenized dataset
- `--project`: W&B project name (default: "nano-dilco")
- `--llama-config-file`: Path to Llama config JSON file
- `--wandb-config-file`: Path to W&B config JSON file

### Example Configurations

#### Default Llama Config
```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "hidden_size": 128,
  "intermediate_size": 512,
  "num_attention_heads": 4,
  "num_hidden_layers": 6,
  "rms_norm_eps": 1e-05,
  "use_cache": false
}
```

#### Default W&B Config
```json
{
  "nodes": 1,
  "location": "local",
  "backend": "nccl",
  "measure_comms": true
}
```

### Multi-GPU Training
To run on multiple GPUs, use `torchrun`:
```bash
torchrun --nproc_per_node=2 -m sillydp.main --batch-size 512
```

### Examples

1. **Small model training**:
   ```bash
   python -m sillydp.main --batch-size 128 --lr 1e-3 --steps 5000
   ```

2. **Custom model architecture**:
   Create `configs/large_model.json`:
   ```json
   {
     "architectures": ["LlamaForCausalLM"],
     "hidden_size": 256,
     "intermediate_size": 1024,
     "num_attention_heads": 8,
     "num_hidden_layers": 12,
     "rms_norm_eps": 1e-05,
     "use_cache": false
   }
   ```

   Then run:
   ```bash
   python -m sillydp.main --llama-config-file configs/large_model.json --batch-size 512
   ```

3. **Multi-node training**:
   Create `configs/multi_node.json`:
   ```json
   {
     "nodes": 4,
     "location": "cluster",
     "backend": "nccl",
     "measure_comms": true
   }
   ```

   Then run:
   ```bash
   python -m sillydp.main --wandb-config-file configs/multi_node.json
   ```

## Modal Multi-Node Training

For large-scale distributed training on Modal's cloud infrastructure, use the provided `scripts/train_modal.py`:

### Setup
1. Install Modal CLI: `pip install modal`
2. Set up Modal account: `modal setup`
3. Create W&B secret: `modal secret create wandb-secret WANDB_API_KEY=your_key_here`
4. Create modal volume for hf-c4-tiny dataset: `modal run scripts/setup_data_volume.py`


### Usage Examples

**Default multi-node training** (2 nodes, 4 T4s each):
```bash
modal run -m scripts.train_modal
```

**Small single-node experiment** (2 T4s):
```bash
modal run -m scripts.train_modal::small_single_node
```

**Large multi-node training**:
```bash
modal run -m scripts.train_modal::large_multi_node
```

**Performance benchmark**:
```bash
modal run -m scripts.train_modal::benchmark
```

### Modal Configuration

You can modify `n_nodes` and `n_proc_per_node` in the script for different cluster sizes.
