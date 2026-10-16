import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CAMPOS, RAIZ, RESULTADOS, RUNS, ler_jsonl

IMG = RAIZ / "docs" / "img"
README = RAIZ / "README.md"
INI, FIM = "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIM -->"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def fmt_lr(lr):
    return "—" if lr is None else f"{lr:.0e}".replace("e-0", "e-")


def carregar():
    metricas, infos, logs = {}, {}, {}
    for f in sorted(RESULTADOS.glob("metrics_*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        metricas[m["nome"]] = m
    for d in sorted(RUNS.glob("*")) if RUNS.exists() else []:
        if (d / "train_info.json").exists():
            infos[d.name] = json.loads((d / "train_info.json").read_text(encoding="utf-8"))
        if (d / "log_history.json").exists():
            logs[d.name] = json.loads((d / "log_history.json").read_text(encoding="utf-8"))
    return metricas, infos, logs


def ordenar(nomes):
    base = [n for n in nomes if n.startswith("base_")]
    return sorted(base, key=lambda n: "few" in n) + sorted(n for n in nomes if not n.startswith("base_"))


def tabela(metricas, infos):
    cab = ("| Modelo | Quant. | Rank | LR | Épocas | Train loss | Eval loss | JSON válido | "
           "Exact match | Acc. categoria | F1 categoria | EM template novo | Tempo treino | VRAM treino | Latência/ex |")
    linhas = [cab, "|" + "|".join(["---"] * 15) + "|"]
    for n in ordenar(metricas):
        m, i = metricas[n], infos.get(n, {})
        tn = m.get("template_novo", {}).get("exact_match")
        linhas.append("| " + " | ".join([
            f"**{n}**", i.get("quant", m.get("quant", "—")),
            str(i.get("rank", "—")), fmt_lr(i.get("lr")), f"{i['epochs']:g}" if "epochs" in i else "—",
            str(i.get("train_loss_final", "—")), str(i.get("eval_loss_final", "—")),
            pct(m["json_valido"]), pct(m["exact_match"]), pct(m["acc_categoria"]),
            f"{m['f1_macro_categoria']:.3f}", pct(tn),
            f"{i['tempo_treino_s'] / 60:.1f} min" if i.get("tempo_treino_s") else "—",
            f"{i['vram_pico_gb']} GB" if i.get("vram_pico_gb") else "—",
            f"{m['latencia_media_ms']:.0f} ms",
        ]) + " |")
    return "\n".join(linhas)


def graficos(metricas, logs):
    IMG.mkdir(parents=True, exist_ok=True)
    gerados = []

    if logs:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for n, h in sorted(logs.items()):
            tr = [(x["step"], x["loss"]) for x in h if "loss" in x and "eval_loss" not in x]
            ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
            if tr:
                ax[0].plot(*zip(*tr), label=n, lw=1.6)
            if ev:
                ax[1].plot(*zip(*ev), marker="o", label=n, lw=1.6)
        for a_, t in zip(ax, ["Loss de treino", "Loss de validação"]):
            a_.set_title(t)
            a_.set_xlabel("passo")
            a_.set_yscale("log")
            a_.grid(alpha=.3)
            a_.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(IMG / "curvas_loss.png", dpi=130)
        plt.close(fig)
        gerados.append("curvas_loss.png")

    if metricas:
        nomes = ordenar(metricas)
        chaves = [("json_valido", "JSON válido"), ("exact_match", "Exact match"), ("acc_categoria", "Acc. categoria")]
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(nomes)), 4.8))
        w = 0.8 / len(chaves)
        for k, (c, rot) in enumerate(chaves):
            vals = [100 * metricas[n][c] for n in nomes]
            xs = [i + (k - 1) * w for i in range(len(nomes))]
            barras = ax.bar(xs, vals, w, label=rot)
            ax.bar_label(barras, fmt="%.0f", fontsize=7)
        ax.set_xticks(range(len(nomes)), nomes, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("%")
        ax.set_ylim(0, 108)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
        ax.set_title("Base (zero/few-shot) × modelos fine-tuned — conjunto de teste")
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(IMG / "comparativo_modelos.png", dpi=130)
        plt.close(fig)
        gerados.append("comparativo_modelos.png")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sel = [n for n in ["base_zero_shot", "base_few_shot"] if n in metricas]
        ft = [n for n in nomes if not n.startswith("base_")]
        if ft:
            sel.append(max(ft, key=lambda n: metricas[n]["exact_match"]))
        w = 0.8 / max(1, len(sel))
        for k, n in enumerate(sel):
            vals = [100 * metricas[n][f"acc_{c}"] for c in CAMPOS]
            barras = ax.bar([i + (k - (len(sel) - 1) / 2) * w for i in range(len(CAMPOS))], vals, w, label=n)
            ax.bar_label(barras, fmt="%.0f", fontsize=7)
        ax.set_xticks(range(len(CAMPOS)), CAMPOS)
        ax.set_ylabel("% de acerto")
        ax.set_ylim(0, 108)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
        ax.set_title("Acurácia por campo do JSON")
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(IMG / "acuracia_por_campo.png", dpi=130)
        plt.close(fig)
        gerados.append("acuracia_por_campo.png")
    return gerados


def eval_min_antes_do_fim(h):
    ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
    if len(ev) < 3:
        return None
    melhor = min(ev, key=lambda t: t[1])
    return melhor, ev[-1]


def analise(metricas, infos, logs):
    ft = [n for n in metricas if not n.startswith("base_")]
    linhas = []
    if not ft:
        return "_Ainda não há modelos fine-tuned avaliados._"
    melhor = max(ft, key=lambda n: metricas[n]["exact_match"])
    mb = metricas[melhor]
    reduzidos = sorted({(i["n_treino"], i["epochs"]) for i in infos.values() if i.get("n_treino", 800) < 800})
    if reduzidos:
        linhas.append("- **Observação:** experimentos rodados em modo reduzido (" +
                      ", ".join(f"{n} exemplos de treino, {e:g} época(s)" for n, e in reduzidos) +
                      ") por limite de tempo de GPU; com o dataset completo os números tendem a subir.")
    linhas.append(f"- **Melhor configuração:** `{melhor}` com **{pct(mb['exact_match'])}** de exact match "
                  f"(todos os 5 campos corretos) e {pct(mb['json_valido'])} de JSON válido.")
    for base in ["base_zero_shot", "base_few_shot"]:
        if base in metricas:
            d = 100 * (mb["exact_match"] - metricas[base]["exact_match"])
            linhas.append(f"- Ganho sobre `{base}`: **{d:+.1f} p.p.** de exact match "
                          f"({pct(metricas[base]['exact_match'])} → {pct(mb['exact_match'])}).")

    ranks = [n for n in ["E2_rank4", "E1_baseline_r16", "E3_rank64"] if n in metricas and n in infos]
    if len(ranks) >= 2:
        partes = [f"r={infos[n]['rank']}: {pct(metricas[n]['exact_match'])} EM, "
                  f"{infos[n]['params_treinaveis'] / 1e6:.1f}M params treináveis" for n in ranks]
        linhas.append("- **Efeito do rank LoRA:** " + "; ".join(partes) + ".")
    lrs = [n for n in ["E4_lr_baixo", "E1_baseline_r16", "E5_lr_alto"] if n in metricas and n in infos]
    if len(lrs) >= 2:
        partes = [f"lr={fmt_lr(infos[n]['lr'])}: eval loss {infos[n]['eval_loss_final']}, EM {pct(metricas[n]['exact_match'])}"
                  for n in lrs]
        linhas.append("- **Efeito da learning rate:** " + "; ".join(partes) + ".")
    if "E6_4_epocas" in logs:
        r = eval_min_antes_do_fim(logs["E6_4_epocas"])
        if r:
            (pm, vm), (pf, vf) = r
            if vf > vm * 1.02:
                linhas.append(f"- **Overfitting (4 épocas):** a loss de validação foi mínima no passo {pm} "
                              f"({vm:.4f}) e subiu para {vf:.4f} no fim do treino.")
            else:
                linhas.append(f"- **4 épocas:** a loss de validação não subiu de forma relevante "
                              f"(mín {vm:.4f}, final {vf:.4f}); sem sinal claro de overfitting.")
    if "E7_lora_sem_quant" in infos and "E1_baseline_r16" in infos:
        q, f = infos["E1_baseline_r16"], infos["E7_lora_sem_quant"]
        linhas.append(f"- **QLoRA 4-bit × LoRA sem quantização:** VRAM {q['vram_pico_gb']} GB × {f['vram_pico_gb']} GB; "
                      f"tempo {q['tempo_treino_s'] / 60:.1f} × {f['tempo_treino_s'] / 60:.1f} min; EM "
                      f"{pct(metricas.get('E1_baseline_r16', {}).get('exact_match'))} × "
                      f"{pct(metricas.get('E7_lora_sem_quant', {}).get('exact_match'))}.")
    if "template_novo" in mb and "template_visto" in mb:
        linhas.append(f"- **Generalização:** no melhor modelo, EM em frases de templates vistos no treino = "
                      f"{pct(mb['template_visto']['exact_match'])}; em templates **nunca vistos** = "
                      f"{pct(mb['template_novo']['exact_match'])}.")

    preds = ler_jsonl(RESULTADOS / f"preds_{melhor}.jsonl")
    erros = Counter()
    confusoes = Counter()
    for p in preds:
        pr = p["pred"] if isinstance(p["pred"], dict) else {}
        for c in CAMPOS:
            if c == "valor":
                ok = (pr.get(c) is None and p["gold"][c] is None) or (
                    isinstance(pr.get(c), (int, float)) and p["gold"][c] is not None
                    and abs(pr[c] - p["gold"][c]) < .01)
            else:
                ok = pr.get(c) == p["gold"][c]
            if not ok:
                erros[c] += 1
        if pr.get("categoria") != p["gold"]["categoria"]:
            confusoes[f"{p['gold']['categoria']} → {pr.get('categoria')}"] += 1
    if erros:
        linhas.append("- **Análise de erro do melhor modelo:** erros por campo: " +
                      ", ".join(f"{c} = {n}" for c, n in erros.most_common()) + ".")
    if confusoes:
        linhas.append("- Confusões de categoria mais comuns: " +
                      "; ".join(f"`{k}` ({v}x)" for k, v in confusoes.most_common(3)) + ".")
    return "\n".join(linhas), melhor


def exemplos(melhor):
    arq_zs = RESULTADOS / "preds_base_zero_shot.jsonl"
    arq_ft = RESULTADOS / f"preds_{melhor}.jsonl"
    if not (arq_zs.exists() and arq_ft.exists()):
        return ""
    zs = {p["id"]: p for p in ler_jsonl(arq_zs)}
    ft = ler_jsonl(arq_ft)
    escolhidos = [p for p in ft if p["pred"] == p["gold"] and zs[p["id"]]["pred"] != p["gold"]
                  and p.get("template_novo")][:3]
    errado = [p for p in ft if p["pred"] != p["gold"]][:1]
    blocos = []
    for p in escolhidos + errado:
        r_zs = zs[p["id"]]["resposta"].replace("\n", " ")
        r_zs = r_zs[:220] + ("…" if len(r_zs) > 220 else "")
        tag = "✅ fine-tuned acertou" if p in escolhidos else "❌ erro do fine-tuned (análise de erro)"
        blocos.append(
            f"**Entrada:** _{p['input']}_  ({tag})\n\n"
            f"- Base zero-shot: `{r_zs}`\n"
            f"- `{melhor}`: `{p['resposta'][:220]}`\n"
            f"- Gabarito: `{json.dumps(p['gold'], ensure_ascii=False)}`\n")
    return "\n".join(blocos)


def main():
    metricas, infos, logs = carregar()
    if not metricas:
        print("Nenhum resultado em results/. Rode src/run_experiments.py antes.")
        return
    tab = tabela(metricas, infos)
    imgs = graficos(metricas, logs)
    texto_analise, melhor = analise(metricas, infos, logs) if any(
        not n.startswith("base_") for n in metricas) else ("_Sem modelos fine-tuned ainda._", None)
    ex = exemplos(melhor) if melhor else ""
    gpu = ""
    gi = RESULTADOS / "gpu_info.json"
    if gi.exists():
        g = json.loads(gi.read_text(encoding="utf-8"))
        gpu = (f"Hardware dos testes: **{g.get('gpu_nome')}** ({g.get('vram_total_gb')} GB VRAM, "
               f"compute capability {g.get('compute_capability')}), PyTorch {g.get('torch')} com CUDA "
               f"{g.get('cuda_versao_torch')}.\n\n")

    md = [gpu + "### Tabela comparativa (conjunto de teste, " +
          f"{next(iter(metricas.values()))['n']} chamados)\n", tab, "",
          "> EM = exact match (os 5 campos corretos ao mesmo tempo). \"Template novo\" = frases com "
          "estrutura que o modelo nunca viu no treino.\n", "### Análise\n", texto_analise, ""]
    for img, tit in [("comparativo_modelos.png", "Comparativo de modelos"),
                     ("acuracia_por_campo.png", "Acurácia por campo"),
                     ("curvas_loss.png", "Curvas de loss")]:
        if img in imgs:
            md.append(f"![{tit}](docs/img/{img})\n")
    if ex:
        md += ["### Exemplos lado a lado\n", ex]
    bloco = "\n".join(md)

    (RESULTADOS / "tabela_experimentos.md").write_text(tab + "\n", encoding="utf-8")
    (RESULTADOS / "RESULTADOS.md").write_text("# Resultados\n\n" + bloco, encoding="utf-8")
    (RESULTADOS / "resumo.json").write_text(json.dumps(
        {"melhor_run": melhor, "metricas": metricas, "treino": infos}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    if README.exists():
        r = README.read_text(encoding="utf-8")
        if INI in r and FIM in r:
            r = re.sub(re.escape(INI) + r".*?" + re.escape(FIM),
                       lambda _: f"{INI}\n{bloco}\n{FIM}", r, flags=re.S)
            README.write_text(r, encoding="utf-8")
            print("README.md atualizado com os resultados.")
    print(tab)
    print(f"\nMelhor run: {melhor}")


if __name__ == "__main__":
    main()
