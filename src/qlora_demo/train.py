from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import build_messages, format_solution
from .labels import EncodedExample, audit, encode_prompt_completion, naive_encode

END_OF_TURN = "<|im_end|>"


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B"
    output_dir: str = "outputs"
    max_length: int = 512
    train_examples: int = 7473
    epochs: float = 1.0
    per_device_batch_size: int = 8
    grad_accum: int = 2
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    time_budget_min: float = 28.0
    quantize_4bit: bool = True
    seed: int = 0
    audit_lengths: str = "128,192,256,384,512"
    eos_token: str = ""


def build_dataset(
    tokenizer: Any, rows: list[dict[str, str]], max_length: int, eos_token: str
) -> tuple[list[EncodedExample], dict[str, int]]:
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    kept: list[EncodedExample] = []
    naive: list[EncodedExample] = []
    dropped = 0
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            build_messages(row["question"]), tokenize=False, add_generation_prompt=True
        )
        completion = format_solution(row["answer"]) + eos_token
        naive.append(naive_encode(tokenizer, prompt, completion, max_length))
        encoded = encode_prompt_completion(tokenizer, prompt, completion, max_length, eos_id)
        if encoded is None:
            dropped += 1
        else:
            kept.append(encoded)
    naive_audit = audit(naive, eos_id)
    stats = {
        "rows": len(rows),
        "kept": len(kept),
        "dropped_overlength": dropped,
        "naive_broken": naive_audit["broken"],
        "naive_fully_masked": naive_audit["fully_masked"],
        "kept_broken": audit(kept, eos_id)["broken"],
    }
    return kept, stats


def truncation_audit(
    tokenizer: Any, rows: list[dict[str, str]], lengths: list[int], eos_token: str
) -> dict[str, Any]:
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    pairs = [
        (
            tokenizer.apply_chat_template(build_messages(r["question"]), tokenize=False, add_generation_prompt=True),
            format_solution(r["answer"]) + eos_token,
        )
        for r in rows
    ]
    report: dict[str, Any] = {}
    for length in lengths:
        naive = audit([naive_encode(tokenizer, p, c, length) for p, c in pairs], eos_id)
        guarded = [encode_prompt_completion(tokenizer, p, c, length, eos_id) for p, c in pairs]
        report[str(length)] = {
            "naive_broken": naive["broken"],
            "naive_fully_masked": naive["fully_masked"],
            "guarded_dropped": sum(e is None for e in guarded),
            "guarded_broken": audit([e for e in guarded if e is not None], eos_id)["broken"],
        }
    return report


def save_chat_ready(model: Any, tokenizer: Any, path: Path, eos_ids: list[int]) -> None:
    model.generation_config.eos_token_id = eos_ids
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def main(argv: list[str] | None = None) -> None:
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    parser = argparse.ArgumentParser(description="QLoRA fine-tune on GSM8K.")
    for name, value in asdict(TrainConfig()).items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    cfg = TrainConfig(**vars(parser.parse_args(argv)))
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token = cfg.eos_token or tokenizer.eos_token
    stop_ids = sorted({tokenizer.convert_tokens_to_ids(END_OF_TURN), tokenizer.convert_tokens_to_ids(eos_token)})

    base_ref = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16)
    save_chat_ready(base_ref, tokenizer, out / "base-chat", stop_ids)
    del base_ref

    rows = [dict(r) for r in load_dataset("openai/gsm8k", "main", split="train")]
    rows = rows[: cfg.train_examples]
    examples, data_stats = build_dataset(tokenizer, rows, cfg.max_length, eos_token)
    audit_report = truncation_audit(tokenizer, rows, [int(x) for x in cfg.audit_lengths.split(",")], eos_token)
    print(json.dumps({"data": data_stats, "truncation_audit": audit_report}), flush=True)
    train_ds = Dataset.from_list([{"input_ids": e.input_ids, "labels": e.labels} for e in examples])

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if cfg.quantize_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0}
    )
    if cfg.quantize_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    budget_s = cfg.time_budget_min * 60

    class TimeBudget(TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.start = time.perf_counter()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if time.perf_counter() - self.start > budget_s:
                control.should_training_stop = True
            return control

    sft_args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_length=cfg.max_length,
        packing=False,
        report_to="none",
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[TimeBudget()],
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    train_s = time.perf_counter() - start
    trainer.model.save_pretrained(out / "adapter")

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    summary = {
        "config": asdict(cfg),
        "supervised_eos_token": eos_token,
        "data": data_stats,
        "truncation_audit": audit_report,
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": round(train_s, 1),
        "global_steps": result.global_step,
        "epochs_completed": round(trainer.state.epoch or 0.0, 3),
        "first_logged_loss": losses[0] if losses else None,
        "last_logged_loss": losses[-1] if losses else None,
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "loss_curve": losses,
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "trl": __import__("trl").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__ if cfg.quantize_4bit else None,
        },
    }
    del trainer, model
    torch.cuda.empty_cache()

    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
    save_chat_ready(merged, tokenizer, out / "merged", stop_ids)
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "loss_curve"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
