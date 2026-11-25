#!/usr/bin/env python3
"""Evaluate saved target adapters on one ordered, held-out GSM8K split."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from helper.get_QAs import get_QAs_from_path
from helper.rebuttal_metrics import CSVLog, asset_id, digest_rows, json_write, runtime_environment, utc_now
from helper.rewards import accuracy_reward_func
from scripts.benchmark_opd_reflex import collator


def read_snapshots(run, manifest):
    if manifest:
        rows = json.loads(manifest.read_text())
        return [(manifest.parent / item['path'], item) for item in rows]
    paths = list((run / 'checkpoints/evaluation').glob('*/snapshot.json'))
    if not paths:
        raise ValueError('no evaluation snapshots found. Supply --checkpoint-manifest for historical adapters')
    return [(path.parent, json.loads(path.read_text())) for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--target-model')
    parser.add_argument('--checkpoint-manifest', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-prompt-length', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--dtype', choices=('bf16','fp16','fp32'))
    parser.add_argument('--attention-implementation')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=.95)
    parser.add_argument('--max-samples', type=int, default=0, help='Nonzero only for a smoke test')
    args = parser.parse_args()
    if min(args.batch_size,args.max_new_tokens) < 1 or args.max_samples < 0 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error('invalid evaluation batch, length, sampling setting, or sample count')
    config = {} if args.checkpoint_manifest else json.loads((args.run_dir / 'config_resolved.yaml').read_text())
    train = config.get('args', {})
    target = args.target_model or train.get('model_dir')
    if target is None or not (Path(target) / 'config.json').is_file():
        parser.error('target model missing. Set TARGET_MODEL to the original model directory')
    seed = args.seed if args.seed is not None else train.get('evaluation_seed',42)
    max_prompt = args.max_prompt_length or train.get('max_prompt_length',2048)
    dtype_name = args.dtype or train.get('dtype','bf16')
    dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[dtype_name]
    attention = args.attention_implementation or train.get('attn_implementation','sdpa')
    snapshots = sorted(read_snapshots(args.run_dir,args.checkpoint_manifest),
        key=lambda pair: (int(pair[1]['target_optimizer_steps']), pair[1]['cumulative_training_wall_time_s'],pair[0].name))
    for path, meta in snapshots:
        if not (path / 'adapter_config.json').is_file():
            raise FileNotFoundError(f'missing target LoRA adapter_config.json under {path}')
        if int(meta['target_optimizer_steps']) < 0 or float(meta['cumulative_training_wall_time_s']) < 0:
            raise ValueError('snapshot metadata requires nonnegative update index and elapsed training time')
    output = args.output or args.run_dir / 'logs'
    for name in ('evaluation_metrics.csv', 'evaluation_predictions.jsonl', 'evaluation_summary.json'):
        if (output / name).exists():
            raise FileExistsError(f'to preserve evaluation results use a fresh EVALUATION_OUTPUT_DIR: {output / name}')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('select exactly one visible CUDA GPU')
    start_date = utc_now()
    evaluation_start = time.perf_counter()
    examples = get_QAs_from_path(args.dataset_path, 'test')
    if args.max_samples:
        examples = examples[:args.max_samples]
    if not examples or any(not row['answer'] for row in examples):
        raise ValueError('held-out split must be nonempty with gold answers')
    dataset_id = asset_id(args.dataset_path)
    expected = config.get('inputs', {}).get('eval_dataset')
    if expected and dataset_id['sha256'] != expected['sha256']:
        raise ValueError('evaluation dataset differs from the held-out dataset recorded during training')
    device = torch.device('cuda:0')
    base = AutoModelForCausalLM.from_pretrained(target, torch_dtype=dtype, attn_implementation=attention,
                                              local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(target, padding_side='left', local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    limit = getattr(base.config, 'max_position_embeddings', None)
    if limit and max_prompt + args.max_new_tokens > limit:
        raise ValueError('evaluation prompt cap + response cap exceeds target context')
    make_batch = collator(tokenizer, max_prompt)
    generation_settings = dict(do_sample=args.sample, max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    if args.sample:
        generation_settings.update(temperature=args.temperature, top_p=args.top_p, top_k=0)
    generation = GenerationConfig(**generation_settings)
    evaluator_settings = dict(**generation_settings, target=asset_id(target), dataset=dataset_id,
        test_rows_sha256=digest_rows(examples), seed=seed, max_prompt_length=max_prompt, dtype=dtype_name,
        batch_size=args.batch_size, attention_implementation=attention, prompt_protocol='train_math_chat_v1',
        answer_verifier='helper.rewards.accuracy_reward_func', sample_count=len(examples))
    expected_target = config.get('inputs', {}).get('target')
    if expected_target and evaluator_settings['target']['sha256'] != expected_target['sha256']:
        raise ValueError('evaluation model differs from the base target recorded during training')
    output.mkdir(parents=True, exist_ok=True)
    result = CSVLog(output / 'evaluation_metrics.csv')
    summaries = []
    with (output / 'evaluation_predictions.jsonl').open('x', buffering=65536) as predictions:
        for index, (path, snapshot) in enumerate(snapshots):
            io_start = time.perf_counter()
            if index == 0:
                model = PeftModel.from_pretrained(base, path).eval()
            else:
                model.load_adapter(path, adapter_name=f'snapshot_{index}')
                model.set_adapter(f'snapshot_{index}')
                model.delete_adapter('default' if index == 1 else f'snapshot_{index-1}')
                model.eval()
            model.requires_grad_(False)
            checkpoint_id = asset_id(path)
            torch.cuda.synchronize(device)
            io_s = time.perf_counter() - io_start
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            correct_count = 0
            eval_start = time.perf_counter()
            common = dict(
                **{key: snapshot.get(key, config.get('identifiers', {}).get(key))
                   for key in ('experiment','model','dataset','method','seed')},
                snapshot=path.name, kind=snapshot.get('kind','historical'),
                target_optimizer_steps=int(snapshot['target_optimizer_steps']),
                cumulative_training_wall_time_s=float(snapshot['cumulative_training_wall_time_s']))
            with torch.inference_mode():
                for start in range(0,len(examples),args.batch_size):
                    batch = make_batch(examples[start:start+args.batch_size])
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    generated = model.generate(input_ids=ids, attention_mask=mask, generation_config=generation)
                    texts = tokenizer.batch_decode(generated[:,ids.shape[1]:],skip_special_tokens=True)
                    correct = accuracy_reward_func(texts,batch['answers'])
                    correct_count += sum(correct)
                    for offset,(text, gold, label) in enumerate(zip(texts,batch['answers'],correct)):
                        predictions.write(json.dumps(dict(**common, example_index=start+offset, prediction=text,
                            gold=gold, correct=bool(label)), ensure_ascii=False)+'\n')
            torch.cuda.synchronize(device)
            accuracy = correct_count/len(examples)
            row = dict(**common, accuracy=accuracy, pass_at_1=accuracy, sample_count=len(examples),
                       correct_count=int(correct_count), settings_seed=seed,
                       checkpoint_sha256=checkpoint_id['sha256'], checkpoint_path=str(path),
                       checkpoint_load_s=io_s, evaluation_wall_time_s=time.perf_counter()-eval_start)
            result.write(row)
            result.flush()
            predictions.flush()
            summaries.append(row)
    result.close()
    json_write(output / 'evaluation_summary.json', dict(settings=evaluator_settings, started_at=start_date,
        finished_at=utc_now(), total_process_wall_time_s=time.perf_counter()-evaluation_start,
        evaluation_wall_time_s=sum(row['evaluation_wall_time_s'] for row in summaries),
        checkpoint_load_s=sum(row['checkpoint_load_s'] for row in summaries),
        environment=runtime_environment(Path(__file__).resolve().parents[1],device), snapshots=summaries))
    print(f'Evaluated {len(snapshots)} snapshots on {len(examples)} examples. Results: {output}')


if __name__ == '__main__':
    main()
