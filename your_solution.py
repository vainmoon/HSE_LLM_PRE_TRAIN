import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    get_scheduler,
    set_seed,
)


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


TRAINING_CONFIG = {
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 8,
    'save_strategy': 'no',
    'save_total_limit': 1,
    'save_only_model': True,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 30,
    'lr_scheduler_type': 'constant_with_warmup',
    'logging_steps': 1,
    'logging_nan_inf_filter': False,
    'eval_strategy': 'no',
    'load_best_model_at_end': False,
    'prediction_loss_only': True,
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': False,
    'report_to': 'none',
    'seed': 42,
    'data_seed': 42,
}

PROMPTS = ['Москва —', 'Искусственный интеллект —', 'В начале XX века']


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.elapsed_seconds = 0.0
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.elapsed_seconds = elapsed
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Evaluate and save the weights at the time limit.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


class PretrainTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_seen = 0
        self.samples_seen = 0
        self.timer = next(
            callback for callback in self.callback_handler.callbacks
            if isinstance(callback, TimeoutCallback)
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.tokens_seen += inputs[ATTENTION_MASK].sum().item()
        self.samples_seen += inputs[INPUT_IDS].shape[0]
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, start_time=None):
        logs.update(
            elapsed_seconds=self.timer.elapsed_seconds,
            tokens_seen=self.tokens_seen,
            samples_seen=self.samples_seen,
        )
        super().log(logs, start_time)


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def prepare_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(
    examples: dict[str, list[str]], tokenizer: PreTrainedTokenizerBase
) -> BatchEncoding:
    tokens = tokenizer(
        examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH
    )
    # PAD and EOS share an ID, so only mask positions marked as padding.
    tokens[LABELS] = [
        [token if keep else -100 for token, keep in zip(ids, mask)]
        for ids, mask in zip(tokens[INPUT_IDS], tokens[ATTENTION_MASK])
    ]
    return tokens


def save_as_parquets(
    ds: Dataset, output_dir: str = OUTPUT_DIR, num_shards: int = NUM_SHARDS
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard.to_parquet(os.path.join(output_dir, f'{index:05d}.parquet'))


def prepare_dataset() -> None:
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=dataset.column_names,
    )
    save_as_parquets(dataset)


def load_tokenized_dataset(data_dir: str = OUTPUT_DIR) -> Dataset:
    files = sorted(
        entry.path for entry in os.scandir(data_dir)
        if entry.is_file() and entry.name.endswith('.parquet')
    )
    return load_dataset('parquet', data_files=files, split='train')


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


@torch.inference_mode()
def generate_text(
    model, tokenizer, prompts: list[str], seed: int, max_new_tokens: int = 128
) -> dict:
    model.eval()
    samples = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        for mode in ('greedy', 'sample'):
            set_seed(seed)
            sampling = (
                {'temperature': 0.8, 'top_p': 0.95, 'top_k': 0} if mode == 'sample' else {}
            )
            output = model.generate(
                **inputs,
                do_sample=mode == 'sample',
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **sampling,
            )
            text = tokenizer.decode(
                output[0, inputs[INPUT_IDS].shape[1]:], skip_special_tokens=True
            )
            samples.append({'prompt': prompt, 'mode': mode, 'text': text})
    return {
        'seed': seed,
        'max_new_tokens': max_new_tokens,
        'temperature': 0.8,
        'top_p': 0.95,
        'top_k': 0,
        'samples': samples,
    }


def train_model(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES.')
    if args.schedule_steps <= args.warmup_steps:
        raise ValueError('--schedule-steps must exceed --warmup-steps.')

    output_dir = Path(OUTPUT_DIR) / args.run
    result_dir = Path('results') / args.run
    if output_dir.exists() or result_dir.exists():
        raise FileExistsError(f'Run {args.run!r} already exists; choose a new --run.')

    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)
    config = {
        **TRAINING_CONFIG,
        'output_dir': str(output_dir),
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.eval_batch_size,
        'gradient_accumulation_steps': args.grad_accum,
        'learning_rate': args.lr,
        'lr_scheduler_type': args.scheduler,
        'warmup_steps': args.warmup_steps,
        'optim': args.optim,
        'torch_compile': args.compile,
        'tf32': args.tf32,
        'gradient_checkpointing': args.gradient_checkpointing,
        'eval_strategy': 'steps' if args.eval_steps else 'no',
        'eval_steps': args.eval_steps or None,
        'seed': args.seed,
        'data_seed': args.seed,
    }
    training_args = TrainingArguments(**config)
    if training_args.world_size != 1:
        raise RuntimeError('Run with one process, without torchrun.')
    output_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(output_dir.resolve() / 'inductor_cache')
    os.environ['TRITON_CACHE_DIR'] = str(output_dir.resolve() / 'triton_cache')

    set_seed(args.seed)
    model = create_model(tokenizer)
    model.config.use_cache = False
    trainer = PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)],
    )
    # A full Wikipedia epoch is much longer than the timed experiment.
    trainer.create_optimizer()
    trainer.lr_scheduler = get_scheduler(
        args.scheduler,
        optimizer=trainer.optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.schedule_steps,
    )
    # Cosine must not rise again if the timer allows more steps than estimated.
    trainer.lr_scheduler.lr_lambdas = [
        lambda step, schedule=schedule: schedule(min(step, args.schedule_steps))
        for schedule in trainer.lr_scheduler.lr_lambdas
    ]
    gpu = torch.cuda.get_device_properties(0)
    save_json(result_dir / 'config.json', {
        'training_args': training_args.to_dict(),
        'schedule_steps': args.schedule_steps,
        'timeout_seconds': MAX_TRAINING_TIME_SECONDS,
        'model_config': model.config.to_dict(),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'model_dtype': str(model.dtype),
        'tokenizer': TOKENIZER_NAME,
        'dataset': 'wikimedia/wikipedia',
        'dataset_config': '20231101.ru',
        'dataset_fingerprint': dataset._fingerprint,
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'max_length': MAX_LENGTH,
        'gpu': gpu.name,
        'gpu_memory_gb': gpu.total_memory / 2**30,
        'python': platform.python_version(),
        'cuda': torch.version.cuda,
        'versions': {
            name: version(name)
            for name in ('torch', 'transformers', 'datasets', 'accelerate', 'flash-attn')
        },
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    save_json(result_dir / 'generations_before.json', generate_text(
        model, tokenizer, PROMPTS, args.seed
    ))
    set_seed(args.seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started_at

    # The timeout callback already evaluates and saves the final weights.
    eval_results = next(
        entry for entry in reversed(trainer.state.log_history) if 'eval_loss' in entry
    )
    trainer.save_state()
    shutil.copy2(output_dir / 'trainer_state.json', result_dir / 'trainer_state.json')
    checkpoint = output_dir / f'checkpoint-{trainer.state.global_step}'
    metrics = {
        'checkpoint': str(checkpoint),
        'eval_loss': eval_results['eval_loss'],
        'perplexity': math.exp(eval_results['eval_loss']),
        'eval_runtime': eval_results['eval_runtime'],
        'train_loss': train_result.training_loss,
        'steps': trainer.state.global_step,
        'train_seconds': trainer.timer.elapsed_seconds,
        'train_wall_seconds': wall_seconds,
        'tokens_seen': trainer.tokens_seen,
        'samples_seen': trainer.samples_seen,
        'tokens_per_second': trainer.tokens_seen / trainer.timer.elapsed_seconds,
        'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
    }
    save_json(result_dir / 'metrics.json', metrics)
    save_json(result_dir / 'generations_after.json', generate_text(
        model, tokenizer, PROMPTS, args.seed
    ))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def load_run(run: str):
    result_dir = Path('results') / run
    metrics = json.loads((result_dir / 'metrics.json').read_text(encoding='utf-8'))
    checkpoint = metrics['checkpoint']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16
    ).to('cuda')
    return model, tokenizer, result_dir, checkpoint


def evaluate_model(run: str, batch_size: int) -> None:
    model, tokenizer, result_dir, checkpoint = load_run(run)
    config = json.loads((result_dir / 'config.json').read_text(encoding='utf-8'))
    dataset = load_tokenized_dataset().select(range(VALIDATION_SIZE))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(OUTPUT_DIR) / run),
            per_device_eval_batch_size=batch_size,
            bf16=True,
            tf32=config['training_args']['tf32'],
            prediction_loss_only=True,
            report_to='none',
        ),
        eval_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate()
    metrics.update(
        checkpoint=checkpoint,
        eval_batch_size=batch_size,
        perplexity=math.exp(metrics['eval_loss']),
    )
    save_json(result_dir / 'evaluation.json', metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def plot_results(runs: list[str]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    rows = []
    for run in runs:
        result_dir = Path('results') / run
        state = json.loads((result_dir / 'trainer_state.json').read_text(encoding='utf-8'))
        config = json.loads((result_dir / 'config.json').read_text(encoding='utf-8'))
        metrics = json.loads((result_dir / 'metrics.json').read_text(encoding='utf-8'))
        train = [entry for entry in state['log_history'] if 'loss' in entry]
        evaluation = [entry for entry in state['log_history'] if 'eval_loss' in entry]
        axes[0, 0].plot([e['step'] for e in train], [e['loss'] for e in train], label=run)
        axes[0, 1].plot(
            [e['elapsed_seconds'] for e in train], [e['loss'] for e in train], label=run
        )
        axes[1, 0].plot(
            [e['elapsed_seconds'] for e in evaluation], [e['eval_loss'] for e in evaluation],
            marker='o', label=run,
        )
        axes[1, 1].plot(
            [e['step'] for e in train], [e['learning_rate'] for e in train], label=run
        )
        args = config['training_args']
        rows.append({
            'run': run,
            'batch_size': args['per_device_train_batch_size'],
            'grad_accum': args['gradient_accumulation_steps'],
            'effective_batch': (
                args['per_device_train_batch_size'] * args['gradient_accumulation_steps']
            ),
            'lr': args['learning_rate'],
            'scheduler': args['lr_scheduler_type'],
            'schedule_steps': config['schedule_steps'],
            'warmup_steps': args['warmup_steps'],
            'optim': args['optim'],
            'compile': args['torch_compile'],
            'bf16': args['bf16'],
            'tf32': args['tf32'],
            'seed': args['seed'],
            'eval_steps': args['eval_steps'],
            'gradient_checkpointing': args['gradient_checkpointing'],
            **metrics,
        })
    for ax, title, xlabel, ylabel in zip(
        axes.flat,
        ('Train loss / steps', 'Train loss / time', 'Evaluation loss', 'Learning rate'),
        ('Optimizer steps', 'Elapsed training time, s', 'Elapsed training time, s', 'Optimizer steps'),
        ('Loss', 'Loss', 'Loss', 'Learning rate'),
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize='small')
    fig.savefig('results/loss.png', dpi=160)
    plt.close(fig)
    with open('results/summary.csv', 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved results/loss.png and results/summary.csv')


def main() -> None:
    parser = argparse.ArgumentParser(description='Timed Qwen3 pretraining on Russian Wikipedia.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare', help='Tokenize Wikipedia and save parquet shards.')
    train = commands.add_parser('train', help='Train a fresh model for 900 seconds.')
    train.add_argument('--run', required=True)
    train.add_argument(
        '--batch-size', type=int, default=TRAINING_CONFIG['per_device_train_batch_size']
    )
    train.add_argument(
        '--eval-batch-size', type=int, default=TRAINING_CONFIG['per_device_eval_batch_size']
    )
    train.add_argument(
        '--grad-accum', type=int, default=TRAINING_CONFIG['gradient_accumulation_steps']
    )
    train.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'])
    train.add_argument(
        '--scheduler', choices=['constant_with_warmup', 'linear', 'cosine'],
        default=TRAINING_CONFIG['lr_scheduler_type'],
    )
    train.add_argument('--schedule-steps', type=int, default=1000)
    train.add_argument('--warmup-steps', type=int, default=TRAINING_CONFIG['warmup_steps'])
    train.add_argument(
        '--optim', choices=['adamw_torch', 'adamw_torch_fused', 'adafactor'],
        default=TRAINING_CONFIG['optim'],
    )
    train.add_argument(
        '--compile', action=argparse.BooleanOptionalAction, default=TRAINING_CONFIG['torch_compile']
    )
    train.add_argument(
        '--tf32', action=argparse.BooleanOptionalAction, default=TRAINING_CONFIG['tf32']
    )
    train.add_argument('--gradient-checkpointing', action='store_true')
    train.add_argument('--eval-steps', type=int, default=0, help='0: evaluate only at timeout.')
    train.add_argument('--seed', type=int, default=TRAINING_CONFIG['seed'])
    evaluate = commands.add_parser(
        'evaluate', help='Re-evaluate the final checkpoint on all 5000 articles.'
    )
    evaluate.add_argument('--run', required=True)
    evaluate.add_argument('--batch-size', type=int, default=8)
    generate = commands.add_parser('generate', help='Generate continuations from the final checkpoint.')
    generate.add_argument('--run', required=True)
    generate.add_argument('--prompt', action='append', help='Repeat for multiple prompts.')
    generate.add_argument('--seed', type=int, default=42)
    generate.add_argument('--max-new-tokens', type=int, default=128)
    plot = commands.add_parser('plot', help='Plot loss curves and export the experiment table.')
    plot.add_argument('--runs', nargs='+', required=True)

    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_dataset()
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args.run, args.batch_size)
    elif args.command == 'generate':
        model, tokenizer, result_dir, checkpoint = load_run(args.run)
        samples = generate_text(
            model, tokenizer, args.prompt or PROMPTS, args.seed, args.max_new_tokens
        )
        samples['checkpoint'] = checkpoint
        save_json(result_dir / 'generations_custom.json', samples)
        print(json.dumps(samples, ensure_ascii=False, indent=2))
    elif args.command == 'plot':
        plot_results(args.runs)


if __name__ == "__main__":
    main()
