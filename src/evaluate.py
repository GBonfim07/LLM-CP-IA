import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DADOS, MODELO_PADRAO, RESULTADOS, RUNS, calcular_metricas,
                    carregar_modelo, extrair_json, gerar, ler_jsonl, montar_mensagens)


def args_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--modo", choices=["zero_shot", "few_shot", "finetuned"], required=True)
    p.add_argument("--run", help="nome do experimento (obrigatório no modo finetuned)")
    p.add_argument("--modelo", default=None, help="padrão: o do train_info.json ou Qwen2.5-1.5B")
    p.add_argument("--quant", choices=["4bit", "none"], default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--limite", type=int, default=None, help="avalia só N exemplos")
    p.add_argument("--permitir_cpu", action="store_true")
    return p.parse_args()


def main():
    a = args_cli()
    adapter, info_treino = None, {}
    if a.modo == "finetuned":
        if not a.run:
            raise SystemExit("--run é obrigatório no modo finetuned")
        adapter = RUNS / a.run / "adapter"
        info_treino = json.loads((RUNS / a.run / "train_info.json").read_text())
    modelo = a.modelo or info_treino.get("modelo", MODELO_PADRAO)
    quant = a.quant or info_treino.get("quant", "4bit")
    nome = a.run if a.modo == "finetuned" else f"base_{a.modo}"

    teste = ler_jsonl(DADOS / "test.jsonl")
    if a.limite:
        teste = teste[:a.limite]

    model, tok = carregar_modelo(modelo, quant=quant, adapter=adapter, permitir_cpu=a.permitir_cpu)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    respostas, tokens, t0 = [], 0, time.time()
    for i in range(0, len(teste), a.batch):
        lote = teste[i:i + a.batch]
        msgs = [montar_mensagens(e["input"], few_shot=(a.modo == "few_shot")) for e in lote]
        textos, n = gerar(model, tok, msgs, max_new_tokens=a.max_new_tokens)
        respostas += textos
        tokens += n
        print(f"  {min(i + a.batch, len(teste))}/{len(teste)}", end="\r")
    tempo = time.time() - t0
    preds = [extrair_json(r) for r in respostas]

    m = calcular_metricas(teste, preds)
    if "template_novo" in teste[0]:
        for flag, rotulo in [(True, "template_novo"), (False, "template_visto")]:
            idx = [i for i, e in enumerate(teste) if e["template_novo"] == flag]
            if idx:
                sub = calcular_metricas([teste[i] for i in idx], [preds[i] for i in idx])
                m[rotulo] = {k: sub[k] for k in ("n", "exact_match", "acc_categoria", "json_valido")}
    m.update({
        "nome": nome, "modo": a.modo, "modelo": modelo, "quant": quant,
        "tempo_total_s": round(tempo, 1),
        "latencia_media_ms": round(1000 * tempo / len(teste), 1),
        "tokens_por_s": round(tokens / tempo, 1),
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None,
    })

    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / f"metrics_{nome}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    with open(RESULTADOS / f"preds_{nome}.jsonl", "w", encoding="utf-8") as f:
        for e, r, p in zip(teste, respostas, preds):
            f.write(json.dumps({"id": e["id"], "input": e["input"], "resposta": r, "pred": p,
                                "gold": e["output"], "template_novo": e.get("template_novo")},
                               ensure_ascii=False) + "\n")

    print(f"\n[{nome}] exact_match={m['exact_match']:.1%}  json_valido={m['json_valido']:.1%}  "
          f"acc_categoria={m['acc_categoria']:.1%}  f1_cat={m['f1_macro_categoria']:.3f}  "
          f"latência={m['latencia_media_ms']:.0f} ms/ex")


if __name__ == "__main__":
    main()
