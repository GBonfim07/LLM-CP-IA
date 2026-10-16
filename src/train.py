import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DADOS, MODELO_PADRAO, RUNS, alvo_json, exigir_gpu, ler_jsonl,
                    montar_mensagens, precisao)


def args_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="nome do experimento")
    p.add_argument("--modelo", default=MODELO_PADRAO)
    p.add_argument("--quant", choices=["4bit", "none"], default="4bit", help="4bit = QLoRA; none = LoRA em fp16/bf16")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=None, help="padrão: 2 x rank")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--limite_treino", type=int, default=None, help="usa só N exemplos (modo rápido)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permitir_cpu", action="store_true", help="só para depuração")
    return p.parse_args()


class ChamadosDataset(Dataset):
    def __init__(self, exemplos, tok, max_len):
        self.itens = []
        for ex in exemplos:
            msgs = montar_mensagens(ex["input"])
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            completo = tok.apply_chat_template(
                msgs + [{"role": "assistant", "content": alvo_json(ex["output"])}], tokenize=False)
            ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
            ids = tok(completo, add_special_tokens=False)["input_ids"][:max_len]
            labels = list(ids)
            n_mask = min(len(ids_prompt), len(ids))
            labels[:n_mask] = [-100] * n_mask
            self.itens.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, i):
        return self.itens[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, lote):
        m = max(len(x["input_ids"]) for x in lote)
        ids, labels, att = [], [], []
        for x in lote:
            pad = m - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            att.append([1] * len(x["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(att)}


def main():
    a = args_cli()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    device = exigir_gpu(a.permitir_cpu)
    dtype = precisao()
    usar_4bit = a.quant == "4bit" and device == "cuda"
    saida = RUNS / a.run
    saida.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"[{a.run}] modelo={a.modelo} quant={a.quant} dtype={dtype} rank={a.rank} lr={a.lr} epochs={a.epochs}")
    tok = AutoTokenizer.from_pretrained(a.modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = {"dtype": dtype}
    if usar_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(a.modelo, **kw)
    if not usar_4bit:
        model.to(device)
    model.config.use_cache = False

    if usar_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=a.rank, lora_alpha=a.alpha or 2 * a.rank, lora_dropout=a.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {treinaveis:,} de {total:,} ({100*treinaveis/total:.2f}%)")

    treino = ler_jsonl(DADOS / "train.jsonl")
    if a.limite_treino:
        treino = treino[:a.limite_treino]
    val = ler_jsonl(DADOS / "val.jsonl")
    ds_treino = ChamadosDataset(treino, tok, a.max_len)
    ds_val = ChamadosDataset(val, tok, a.max_len)
    tam = [len(x["input_ids"]) for x in ds_treino.itens]
    print(f"Treino: {len(ds_treino)} exemplos | Val: {len(ds_val)} | tokens/exemplo: média {sum(tam)/len(tam):.0f}, máx {max(tam)}")

    passos_epoca = math.ceil(len(ds_treino) / (a.batch * a.grad_accum))
    passos_total = math.ceil(passos_epoca * a.epochs)
    args = TrainingArguments(
        output_dir=str(saida / "tmp"),
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * passos_total)),
        logging_steps=max(1, passos_epoca // 8),
        eval_strategy="steps",
        eval_steps=max(1, passos_epoca // 2),
        save_strategy="no",
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        optim="paged_adamw_8bit" if usar_4bit else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        weight_decay=0.0,
        report_to="none",
        seed=a.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=(device == "cuda"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_treino, eval_dataset=ds_val,
                      data_collator=Collator(tok.pad_token_id))

    t0 = time.time()
    resultado = trainer.train()
    tempo = time.time() - t0
    metricas_val = trainer.evaluate()

    model.save_pretrained(saida / "adapter")
    tok.save_pretrained(saida / "adapter")
    (saida / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    info = {
        "run": a.run, "modelo": a.modelo, "quant": a.quant, "dtype": str(dtype).replace("torch.", ""),
        "rank": a.rank, "alpha": a.alpha or 2 * a.rank, "dropout": a.dropout, "lr": a.lr,
        "epochs": a.epochs, "batch": a.batch, "grad_accum": a.grad_accum,
        "batch_efetivo": a.batch * a.grad_accum, "passos": trainer.state.global_step,
        "n_treino": len(ds_treino), "params_treinaveis": treinaveis, "params_total": total,
        "pct_treinavel": round(100 * treinaveis / total, 3),
        "train_loss_final": round(resultado.training_loss, 4),
        "eval_loss_final": round(metricas_val["eval_loss"], 4),
        "tempo_treino_s": round(tempo, 1),
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    (saida / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
