# Eagle3 Online Training for Slime

This implementation adds Eagle3 drafter online training capabilities to Slime RL framework, enabling continuous optimization of speculative decoding performance during reinforcement learning training.

## Features

- ✅ **Online Training**: Train Eagle3 drafter model asynchronously during RL training
- ✅ **Real-time Updates**: Sync trained weights to SGLang inference engine
- ✅ **Cross-step Data Accumulation**: Collect and use data from multiple RL steps
- ✅ **Memory Efficient**: Support CPU offloading for model and optimizer
- ✅ **Flexible Configuration**: Extensive command-line arguments for customization
- ✅ **Monitoring**: Built-in logging and metrics for training progress

## Quick Start

### 1. Basic Usage

Add these flags to your training command:

```bash
# Enable Eagle3 speculative decoding
--sglang-speculative-algorithm EAGLE3 \
--sglang-speculative-num-steps 3 \
--sglang-speculative-eagle-topk 1 \
--sglang-speculative-num-draft-tokens 1 \
--sglang-speculative-draft-model-path /path/to/eagle3/checkpoint \

# Enable Eagle3 online training
--enable-eagle3-training \
--eagle3-training-interval-steps 10 \
--eagle3-batch-size-per-gpu 2 \
--eagle3-checkpoint-path /path/to/checkpoints
```

### 2. Full Example

See `examples/eagle3_integration/run_eagle3_rl.py` for a complete training script.

## Architecture

### Core Components

1. **Eagle3BackgroundTrainer** (`slime/drafter/eagle3_trainer.py`)
   - Asynchronous training of Eagle3 drafter
   - Data collection and buffer management
   - Checkpoint saving

2. **Eagle3WeightUpdater** (`slime/drafter/eagle3_weight_updater.py`)
   - Updates drafter weights in SGLang engine
   - Bucketed weight updates for efficiency
   - Non-blocking async operations

3. **Eagle3 Models** (`slime/drafter/models/`)
   - Qwen2 Eagle3 model implementation
   - Single-layer transformer with concatenated input
   - Shared embeddings with base model

4. **Integration Layer** (`slime/backends/megatron_utils/`)
   - Patches for training loop
   - Manager for Eagle3 lifecycle
   - Configuration utilities

### Data Flow

```
┌─────────────────┐
│  Rollout Gen    │
│  (SGLang+Eagle3)│
└────────┬────────┘
         │
         ├─> Responses
         │
         └─> Hidden States
                │
                ▼
┌─────────────────┐
│  Data Buffer    │
│  (Cross-step)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Eagle3 Trainer │
│  (Async)        │
└────────┬────────┘
         │
         ├─> Training Step
         │
         └─> Checkpoint Save
                │
                ▼
┌─────────────────┐
│  Weight Updater │
│  (To SGLang)    │
└─────────────────┘
```

## Configuration

### Speculative Decoding

| Argument | Default | Description |
|----------|---------|-------------|
| `--sglang-speculative-algorithm` | EAGLE3 | Speculative algorithm |
| `--sglang-speculative-num-steps` | 3 | Draft steps per iteration |
| `--sglang-speculative-eagle-topk` | 1 | Top-k for sampling |
| `--sglang-speculative-num-draft-tokens` | 1 | Draft tokens per step |
| `--sglang-speculative-draft-model-path` | - | Path to drafter checkpoint |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-eagle3-training` | False | Enable online training |
| `--eagle3-training-interval-steps` | 10 | Train every N rollouts |
| `--eagle3-batch-size-per-gpu` | 2 | Training batch size |
| `--eagle3-max-seq-len` | 8192 | Max sequence length |
| `--eagle3-max-epochs` | 10 | Max training epochs |
| `--eagle3-checkpoint-path` | - | Checkpoint save path |
| `--eagle3-collect-hidden-states` | True | Collect hidden states |

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--eagle3-num-layers` | 1 | Number of layers |
| `--eagle3-hidden-size` | - | Hidden size (auto from base) |
| `--eagle3-intermediate-size` | - | FFN size (auto from base) |
| `--eagle3-num-attention-heads` | - | Attention heads (auto from base) |
| `--eagle3-num-key-value-heads` | - | KV heads (auto from base) |

### Optimizer

| Argument | Default | Description |
|----------|---------|-------------|
| `--eagle3-lr` | 1e-6 | Learning rate |
| `--eagle3-lr-warmup-steps` | 1000 | Warmup steps |
| `--eagle3-weight-decay` | 1e-2 | Weight decay |
| `--eagle3-warmup-style` | constant | LR schedule style |

### Offloading

| Argument | Default | Description |
|----------|---------|-------------|
| `--eagle3-offload-param` | False | Offload model to CPU |
| `--eagle3-offload-optimizer` | False | Offload optimizer to CPU |

### Weight Updates

| Argument | Default | Description |
|----------|---------|-------------|
| `--update-weights-bucket-megabytes` | 512 | Bucket size (MB) |

## Training Objectives

Eagle3 drafter is trained with two objectives:

1. **Probability Loss (ploss)**: Predict next token distribution
   - Weight: 0.1 (default)
   - Helps drafter match base model's token predictions

2. **Value Loss (vloss)**: Predict base model hidden states
   - Weight: 1.0 (default)
   - Helps drafter understand base model's representations

```python
loss = 1.0 * vloss + 0.1 * ploss
```

## Performance Tips

### Improving Acceptance Rate

- Increase training frequency: `--eagle3-training-interval-steps 5`
- Increase batch size: `--eagle3-batch-size-per-gpu 4`
- Adjust learning rate: `--eagle3-lr 5e-7`
- Increase data buffer size

### Reducing Memory Usage

- Enable offloading: `--eagle3-offload-param --eagle3-offload-optimizer`
- Reduce batch size: `--eagle3-batch-size-per-gpu 1`
- Reduce sequence length: `--eagle3-max-seq-len 4096`

### Faster Training

- Train less frequently: `--eagle3-training-interval-steps 20`
- Use CPU offloading for optimizer
- Reduce data buffer max size

## Monitoring

### Logs

The trainer logs:

```
[EagleTrainer rank 0] Training activated
Step 10: loss=0.3456, vloss=0.3210, ploss=0.0246
[EagleTrainer rank 0] Training step 10 completed
Updating Eagle3 drafter weights in rollout engine
Eagle3 drafter weights updated successfully
Checkpoint saved to /path/to/checkpoints/eagle3_step_10
```

### Key Metrics

Monitor these metrics to assess performance:

1. **Acceptance Rate**: How often draft tokens are accepted
2. **Training Loss**: Combined ploss + vloss
3. **Data Buffer Size**: Amount of accumulated data
4. **Training Speed**: Steps per second

## Troubleshooting

### Low Acceptance Rate

**Symptoms**: Draft tokens frequently rejected

**Solutions**:
- Train more frequently (reduce `eagle3_training_interval_steps`)
- Increase learning rate slightly
- Increase batch size for better gradient estimation
- Check if hidden states are being collected correctly

### Out of Memory

**Symptoms**: CUDA OOM during training

**Solutions**:
- Enable CPU offloading: `--eagle3-offload-param`
- Reduce batch size: `--eagle3-batch-size-per-gpu 1`
- Reduce sequence length: `--eagle3-max-seq-len 4096`
- Reduce data buffer size

### Slow Training

**Symptoms**: Training takes too long

**Solutions**:
- Train less frequently (increase `eagle3_training_interval_steps`)
- Enable optimizer offloading
- Use smaller batch size
- Reduce data buffer max samples

### Weight Update Fails

**Symptoms**: Errors during weight synchronization

**Solutions**:
- Check SGLang engine is running
- Verify device mesh configuration
- Reduce bucket size: `--update-weights-bucket-megabytes 256`
- Check network connectivity for distributed training

## Advanced Usage

### Custom Training Schedule

Modify `Eagle3BackgroundTrainer.training_step()` to implement custom logic:

```python
async def _training_step_impl(self, step: int):
    # Custom training logic
    if step % custom_interval == 0:
        # Train with custom batch size
        batch_size = get_dynamic_batch_size(step)
        # ...
```

### Custom Loss Functions

Override loss computation in `_training_step_impl()`:

```python
# Custom loss weights
w_v = custom_vloss_weight(step)
w_p = custom_ploss_weight(step)
loss = w_v * vloss + w_p * ploss
```

### Multiple Drafter Models

Create multiple `Eagle3BackgroundTrainer` instances for different tasks:

```python
trainer1 = Eagle3BackgroundTrainer(...)
trainer2 = Eagle3BackgroundTrainer(...)

# Train both
await trainer1.training_step(step)
await trainer2.training_step(step)
```

## File Structure

```
slime/
├── drafter/
│   ├── __init__.py
│   ├── eagle3_trainer.py          # Background trainer
│   ├── eagle3_weight_updater.py   # Weight updates
│   ├── eagle3_config.py           # Configuration
│   ├── README.md                  # This file
│   └── models/
│       ├── __init__.py
│       └── qwen2_eagle3.py        # Qwen2 Eagle3 model
├── backends/
│   └── megatron_utils/
│       ├── eagle3_integration.py  # Integration layer
│       └── eagle3_actor_patch.py  # Training loop patches
└── examples/
    └── eagle3_integration/
        └── run_eagle3_rl.py       # Training example
```

## References

- [EAGLE Paper](https://arxiv.org/abs/2301.13087)
- [SGLang](https://github.com/sgl-project/sglang)
- [FastRL](https://github.com/volcengine/verl)

## License

This implementation follows the same license as Slime framework.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:

- Open a GitHub issue
- Check the documentation: `docs/eagle3_integration.md`
- Contact the development team
