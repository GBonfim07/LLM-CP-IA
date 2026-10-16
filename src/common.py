import json
import re
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data"
RUNS = RAIZ / "runs"
RESULTADOS = RAIZ / "results"

MODELO_PADRAO = "Qwen/Qwen2.5-1.5B-Instruct"

CATEGORIAS = ["pix_nao_creditado", "compra_nao_reconhecida", "cartao_bloqueado", "cobranca_indevida",
              "acesso_bloqueado", "emprestimo", "resgate_investimento", "segunda_via_boleto"]
PRODUTOS = ["pix", "cartao_credito", "cartao_debito", "conta_corrente", "acesso_digital",
            "emprestimo", "investimentos", "boleto"]
URGENCIAS = ["alta", "media", "baixa"]
CANAIS = ["app", "internet_banking", "agencia", "caixa_eletronico", "telefone"]
CAMPOS = ["categoria", "produto", "urgencia", "valor", "canal"]

SYSTEM_PROMPT = (
    "Você classifica chamados de clientes de um banco. Responda SOMENTE com um JSON válido, "
    "sem texto antes ou depois, com as chaves:\n"
    f"categoria: {' | '.join(CATEGORIAS)}\n"
    f"produto: {' | '.join(PRODUTOS)}\n"
    "urgencia: alta | media | baixa\n"
    "valor: número em reais (ex: 1500.0) ou null\n"
    f"canal: {' | '.join(CANAIS)} | null (canal citado no texto)\n"
    "Urgência: alta se o cliente pedir urgência, se for compra não reconhecida ou pix não creditado "
    "a partir de R$ 1000; baixa para empréstimo, resgate de investimento e segunda via de boleto; "
    "media nos demais casos."
)

FEW_SHOT = [
    ("fiz um pix de 2 mil pelo app ontem e não caiu na conta do meu irmão",
     {"categoria": "pix_nao_creditado", "produto": "pix", "urgencia": "alta", "valor": 2000.0, "canal": "app"}),
    ("cobraram anuidade do cartão de crédito de R$ 39,90, quero estorno",
     {"categoria": "cobranca_indevida", "produto": "cartao_credito", "urgencia": "media", "valor": 39.9, "canal": None}),
    ("como emito segunda via de boleto pelo internet banking?",
     {"categoria": "segunda_via_boleto", "produto": "boleto", "urgencia": "baixa", "valor": None, "canal": "internet_banking"}),
]


def ler_jsonl(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def montar_mensagens(texto, few_shot=False, system_prompt=SYSTEM_PROMPT):
    msgs = [{"role": "system", "content": system_prompt}]
    if few_shot:
        for ent, sai in FEW_SHOT:
            msgs.append({"role": "user", "content": ent})
            msgs.append({"role": "assistant", "content": json.dumps(sai, ensure_ascii=False)})
    msgs.append({"role": "user", "content": texto})
    return msgs


def alvo_json(output: dict) -> str:
    return json.dumps({k: output[k] for k in CAMPOS}, ensure_ascii=False)


def exigir_gpu(permitir_cpu=False):
    if torch.cuda.is_available():
        return "cuda"
    if permitir_cpu:
        print("[aviso] Sem GPU: rodando em CPU apenas para depuração.")
        return "cpu"
    raise SystemExit(
        "ERRO: nenhuma GPU CUDA encontrada. O trabalho exige GPU.\n"
        "No Colab: Ambiente de execução > Alterar tipo > GPU T4. "
        "Localmente: verifique driver NVIDIA + PyTorch com CUDA (ver README)."
    )


def precisao():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def carregar_modelo(nome=MODELO_PADRAO, quant="4bit", adapter=None, permitir_cpu=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = exigir_gpu(permitir_cpu)
    dtype = precisao()
    tok = AutoTokenizer.from_pretrained(nome)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"dtype": dtype}
    if quant == "4bit" and device == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(nome, **kwargs)
    if "quantization_config" not in kwargs:
        model.to(device)

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    return model, tok


@torch.no_grad()
def gerar(model, tok, lista_msgs, max_new_tokens=128, temperatura=0.0, top_p=1.0):
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in lista_msgs]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
    if temperatura and temperatura > 0:
        kw.update(do_sample=True, temperature=float(temperatura), top_p=float(top_p))
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    out = model.generate(**enc, **kw)
    novos = out[:, enc["input_ids"].shape[1]:]
    textos = tok.batch_decode(novos, skip_special_tokens=True)
    n_tokens = int((novos != tok.pad_token_id).sum().item())
    return [t.strip() for t in textos], n_tokens


def extrair_json(texto: str):
    if not texto:
        return None
    texto = re.sub(r"```(?:json)?", "", texto)
    ini = texto.find("{")
    if ini < 0:
        return None
    nivel = 0
    for i in range(ini, len(texto)):
        if texto[i] == "{":
            nivel += 1
        elif texto[i] == "}":
            nivel -= 1
            if nivel == 0:
                try:
                    obj = json.loads(texto[ini:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def schema_ok(p: dict) -> bool:
    if not isinstance(p, dict) or set(CAMPOS) - set(p):
        return False
    v = p["valor"]
    return (p["categoria"] in CATEGORIAS and p["produto"] in PRODUTOS and p["urgencia"] in URGENCIAS
            and (p["canal"] is None or p["canal"] in CANAIS)
            and (v is None or isinstance(v, (int, float))))


def campo_correto(campo, pred, gold):
    if not isinstance(pred, dict) or campo not in pred:
        return False
    a, b = pred[campo], gold[campo]
    if campo == "valor":
        if a is None or b is None:
            return a is None and b is None
        try:
            return abs(float(a) - float(b)) < 0.01
        except (TypeError, ValueError):
            return False
    return a == b


def f1_macro(golds, preds, classes):
    f1s = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(golds, preds))
        fp = sum(g != c and p == c for g, p in zip(golds, preds))
        fn = sum(g == c and p != c for g, p in zip(golds, preds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def calcular_metricas(exemplos, preds_json):
    n = len(exemplos)
    golds = [e["output"] for e in exemplos]
    m = {
        "n": n,
        "json_valido": sum(p is not None for p in preds_json) / n,
        "schema_valido": sum(schema_ok(p) for p in preds_json) / n,
    }
    for c in CAMPOS:
        m[f"acc_{c}"] = sum(campo_correto(c, p, g) for p, g in zip(preds_json, golds)) / n
    m["exact_match"] = sum(all(campo_correto(c, p, g) for c in CAMPOS) for p, g in zip(preds_json, golds)) / n
    m["f1_macro_categoria"] = f1_macro(
        [g["categoria"] for g in golds],
        [(p or {}).get("categoria") if isinstance(p, dict) else None for p in preds_json],
        CATEGORIAS)
    return m
