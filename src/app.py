import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MODELO_PADRAO, RESULTADOS, RUNS, SYSTEM_PROMPT, carregar_modelo,
                    extrair_json, gerar, ler_jsonl, montar_mensagens, schema_ok)

EXEMPLOS_FIXOS = [
    "boa tarde, fiz um pix de 3 mil pelo app ontem pro meu fornecedor e até agora nada, é urgente!",
    "apareceu uma compra de R$ 1.299,00 na SHOPEE no meu cartão de crédito final 8812 que eu não fiz",
    "como faço pra pegar a segunda via do boleto pelo site?",
    "to tentando resgatar 20 mil do CDB no aplicativo e não aparece a opção",
    "cobraram tarifa de pacote de serviços de R$ 54,00 sendo que minha conta é isenta",
]


def escolher_run(run_cli):
    if run_cli:
        return run_cli
    resumo = RESULTADOS / "resumo.json"
    if resumo.exists():
        r = json.loads(resumo.read_text(encoding="utf-8")).get("melhor_run")
        if r:
            return r
    padrao = "E1_baseline_r16"
    return padrao if (RUNS / padrao / "adapter").exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--share", action="store_true")
    p.add_argument("--porta", type=int, default=7860)
    p.add_argument("--permitir_cpu", action="store_true")
    a = p.parse_args()

    run = escolher_run(a.run)
    adapter = RUNS / run / "adapter" if run and (RUNS / run / "adapter").exists() else None
    info = json.loads((RUNS / run / "train_info.json").read_text()) if adapter else {}
    modelo_nome = info.get("modelo", MODELO_PADRAO)
    model, tok = carregar_modelo(modelo_nome, quant=info.get("quant", "4bit"),
                                 adapter=adapter, permitir_cpu=a.permitir_cpu)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    tem_ft = adapter is not None
    print(f"Modelo: {modelo_nome} | adapter: {adapter} | GPU: {gpu}")

    exemplos = list(EXEMPLOS_FIXOS)
    teste = RUNS.parent / "data" / "test.jsonl"
    if teste.exists():
        exemplos += [e["input"] for e in ler_jsonl(teste) if e.get("template_novo")][:5]

    def rodar(texto, usar_ft, system, few_shot, temp, top_p, max_tok):
        msgs = [montar_mensagens(texto, few_shot=few_shot, system_prompt=system)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if usar_ft or not tem_ft:
            resp, n = gerar(model, tok, msgs, max_new_tokens=int(max_tok), temperatura=temp, top_p=top_p)
        else:
            with model.disable_adapter():
                resp, n = gerar(model, tok, msgs, max_new_tokens=int(max_tok), temperatura=temp, top_p=top_p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        parsed = extrair_json(resp[0])
        if parsed is None:
            status = "Resposta não é um JSON válido"
        elif not schema_ok(parsed):
            status = "JSON válido, mas fora do formato esperado"
        else:
            status = "JSON válido e no formato esperado"
        return f"**{status}** — {ms:.0f} ms, {n} tokens", parsed, resp[0]

    def analisar(texto, modo, system, few_shot, temp, top_p, max_tok):
        if not texto or not texto.strip():
            vazio = ("Escreva um chamado no campo à esquerda para analisar.", None, "")
            return (*vazio, gr.update(visible=True), *vazio, gr.update(visible=False))
        mostrar_ft = modo in ("Fine-tuned", "Comparar os dois") and tem_ft
        mostrar_base = modo in ("Base", "Comparar os dois") or not tem_ft
        ft = rodar(texto, True, system, few_shot, temp, top_p, max_tok) if mostrar_ft else ("", None, "")
        base = rodar(texto, False, system, few_shot, temp, top_p, max_tok) if mostrar_base else ("", None, "")
        return (*ft, gr.update(visible=mostrar_ft), *base, gr.update(visible=mostrar_base))

    tema = gr.themes.Soft(
        primary_hue="teal", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Figtree"), "ui-sans-serif", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"])
    css = ".gradio-container {max-width: 1180px !important; margin: auto;} #titulo h1 {margin-bottom: 0}"

    tema_no_launch = "theme" in inspect.signature(gr.Blocks.launch).parameters
    kw_blocks = {} if tema_no_launch else {"theme": tema, "css": css}

    with gr.Blocks(title="Triagem de chamados — LLM local", **kw_blocks) as demo:
        gr.Markdown(
            "# Triagem de chamados bancários\n"
            f"Escreva o chamado do cliente e a LLM devolve categoria, produto, urgência, valor e canal em JSON.  \n"
            f"Modelo: `{modelo_nome}` + LoRA `{run or 'nenhum'}` rodando em **{gpu}**.",
            elem_id="titulo")
        with gr.Row():
            with gr.Column(scale=5):
                entrada = gr.Textbox(label="Chamado do cliente", lines=5,
                                     placeholder="Ex.: fiz um pix de 500 reais ontem e não caiu na conta…")
                modo = gr.Radio(["Fine-tuned", "Base", "Comparar os dois"] if tem_ft else ["Base"],
                                value="Fine-tuned" if tem_ft else "Base", label="Modelo")
                with gr.Accordion("Prompt e parâmetros de geração", open=False):
                    system = gr.Textbox(label="Prompt de sistema (editável)", value=SYSTEM_PROMPT, lines=9)
                    few = gr.Checkbox(label="Incluir 3 exemplos no prompt (few-shot)", value=False)
                    temp = gr.Slider(0, 1.5, value=0.0, step=0.05, label="Temperatura (0 = determinístico)")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    max_tok = gr.Slider(32, 512, value=128, step=16, label="Máximo de tokens gerados")
                with gr.Row():
                    btn = gr.Button("Analisar chamado", variant="primary")
                    limpar = gr.Button("Limpar")
                gr.Examples(exemplos, inputs=entrada, label="Chamados de exemplo")
            with gr.Column(scale=6):
                with gr.Column(visible=tem_ft) as grupo_ft:
                    gr.Markdown(f"### Fine-tuned (`{run}`)")
                    st_ft = gr.Markdown()
                    js_ft = gr.JSON(label="JSON extraído")
                    raw_ft = gr.Textbox(label="Resposta bruta", lines=3)
                with gr.Column(visible=not tem_ft) as grupo_base:
                    gr.Markdown("### Base (sem fine-tuning)")
                    st_base = gr.Markdown()
                    js_base = gr.JSON(label="JSON extraído")
                    raw_base = gr.Textbox(label="Resposta bruta", lines=3)

        entradas = [entrada, modo, system, few, temp, top_p, max_tok]
        saidas = [st_ft, js_ft, raw_ft, grupo_ft, st_base, js_base, raw_base, grupo_base]
        btn.click(analisar, entradas, saidas)
        entrada.submit(analisar, entradas, saidas)
        limpar.click(lambda: ("", "", None, "", "", None, ""),
                     None, [entrada, st_ft, js_ft, raw_ft, st_base, js_base, raw_base])

    kw_launch = {"share": a.share, "server_port": a.porta}
    if tema_no_launch:
        kw_launch.update(theme=tema, css=css)
    demo.queue().launch(**kw_launch)


if __name__ == "__main__":
    main()
