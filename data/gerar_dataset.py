import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

SEED = 42
N_TREINO_POR_CAT = 100
N_VAL_POR_CAT = 12
N_TESTE_NOVO_POR_CAT = 8
N_TESTE_VISTO_POR_CAT = 7
N_HOLDOUT = 2

PASTA = Path(__file__).resolve().parent

CANAIS = {
    "app": ["pelo app", "no aplicativo", "pelo aplicativo do banco", "no app do celular"],
    "internet_banking": ["pelo internet banking", "no site do banco", "pelo computador no internet banking"],
    "agencia": ["na agência", "lá na agência", "presencialmente na agência"],
    "caixa_eletronico": ["no caixa eletrônico", "no caixa 24 horas", "num caixa eletrônico"],
    "telefone": ["pela central de atendimento", "pelo telefone", "ligando na central"],
}
TEMPO = ["ontem", "hoje cedo", "faz 2 dias", "há umas 18 horas", "semana passada",
         "na sexta", "hoje de manhã", "anteontem", "faz uns 3 dias"]
DESDE = ["ontem", "segunda-feira", "semana passada", "o começo do mês", "hoje cedo", "anteontem"]
PESSOA = ["meu irmão", "minha mãe", "um fornecedor", "o dono do apartamento",
          "minha esposa", "um cliente", "meu pai", "uma amiga"]
LOJA = ["AMAZON MKTPLC", "UBER *TRIP", "MERCADOLIVRE*VENDAS", "PAG*JOSEMARIA",
        "IFOOD *RESTAURANTE", "NETFLIX.COM", "SHOPEE BRASIL", "POSTO IPIRANGA",
        "ALIEXPRESS", "PAYPAL *STEAM"]
INVEST = ["CDB", "fundo DI", "tesouro direto", "LCI", "fundo multimercado", "LCA"]
TARIFA = [("tarifa de pacote de serviços", "conta_corrente"),
          ("taxa de manutenção da conta", "conta_corrente"),
          ("seguro que eu não contratei", "conta_corrente"),
          ("tarifa de saque", "conta_corrente"),
          ("anuidade do cartão de crédito", "cartao_credito"),
          ("juros no cartão de crédito", "cartao_credito"),
          ("seguro do cartão de crédito", "cartao_credito")]
URGENCIA_PREFIXO = ["URGENTE: ", "socorro, ", "por favor é urgente, "]
URGENCIA_SUFIXO = [" é urgente!", " preciso resolver isso hoje", " to desesperado",
                   " preciso disso resolvido agora"]
SAUDACOES = ["bom dia, ", "boa tarde, ", "oi, ", "olá, ", "Prezados, ", "ola tudo bem? "]
FECHAMENTOS = [" obrigado", " aguardo retorno", " vlw", " pfv me ajudem", " alguém me responde?"]

VALORES = {
    "pix_nao_creditado": [50, 120, 250, 380, 500, 800, 999.9, 1000, 1500, 2000, 3500, 5000, 12000, 742.35],
    "compra_nao_reconhecida": [29.9, 87.5, 199, 450, 1299, 2500, 4800, 63.47],
    "cobranca_indevida": [9.9, 19.9, 29.9, 39.9, 54, 89.9, 120, 450],
    "emprestimo": [2000, 5000, 10000, 15000, 30000, 7500],
    "resgate_investimento": [1000, 5000, 20000, 50000, 3200, 1500],
    "segunda_via_boleto": [89.9, 150, 320.5, 450, 1200, 2780.15],
}

TEMPLATES = {
    "pix_nao_creditado": [
        ("fiz um pix de {valor} {canal} {tempo} pra {pessoa} e até agora não caiu na conta", ["app", "internet_banking"]),
        ("mandei {valor} via pix {tempo} e o dinheiro saiu da minha conta mas {pessoa} não recebeu", None),
        ("o pix que eu fiz {canal} {tempo} não chegou no destino, já saiu do meu saldo", ["app", "internet_banking"]),
        ("transferi {valor} por pix para {pessoa} {tempo}, aparece como concluído no meu extrato mas não foi creditado", None),
        ("pix enviado {tempo} {canal} no valor de {valor} não compensou do outro lado, o que eu faço?", ["app", "internet_banking"]),
        ("meu pix pra {pessoa} sumiu, debitou {valor} da minha conta e não entrou na conta da pessoa", None),
        ("fiz uma transferência pix {canal} {tempo} e {pessoa} diz que não recebeu nada", ["app", "internet_banking"]),
        ("{pessoa} tá me cobrando porque o pix de {valor} que eu mandei {canal} {tempo} nunca apareceu na conta", ["app", "internet_banking"]),
        ("comprovante do pix diz efetivado mas o valor de {valor} não foi parar na conta de {pessoa}", None),
    ],
    "compra_nao_reconhecida": [
        ("apareceu uma compra de {valor} no meu {cartao} na {loja} que eu não fiz", None),
        ("tem uma cobrança de {loja} no {cartao} que eu não reconheço, valor {valor}", None),
        ("vi {canal} uma compra que não fui eu, {loja}, {valor}, acho que clonaram meu {cartao}", ["app", "internet_banking"]),
        ("não reconheço uma transação de {valor} feita {tempo} no {cartao}", None),
        ("alguém usou meu {cartao} {tempo} pra comprar na {loja}, quero contestar", None),
        ("compra estranha no {cartao}: {loja} de {valor}. Eu nunca comprei nesse lugar", None),
        ("recebi notificação {canal} de uma compra de {valor} que eu não fiz no {cartao}", ["app"]),
        ("quero abrir contestação de uma compra na {loja} que não fiz, saiu do meu {cartao}", None),
        ("fui vítima de golpe? tem {valor} gasto na {loja} {tempo} no meu {cartao} e eu tava em casa", None),
    ],
    "cartao_bloqueado": [
        ("meu {cartao} foi bloqueado do nada, tentei usar {canal} e não passou", ["caixa_eletronico", "app"]),
        ("o {cartao} tá bloqueado desde {desde} e eu não pedi bloqueio nenhum", None),
        ("tentei pagar no mercado e deu cartão bloqueado, é o meu {cartao}", None),
        ("errei a senha {canal} e agora o {cartao} bloqueou, como desbloqueio?", ["caixa_eletronico", "app", "agencia"]),
        ("preciso desbloquear meu {cartao}, ele parou de funcionar {tempo}", None),
        ("meu {cartao} aparece como bloqueado {canal}, o que aconteceu?", ["app", "internet_banking"]),
        ("cartão recusando todas as compras desde {desde}, deve ter bloqueado. é o {cartao}", None),
        ("fui sacar {canal} e o {cartao} foi retido, e agora?", ["caixa_eletronico"]),
        ("o banco bloqueou meu {cartao} por segurança? não consigo usar desde {desde}", None),
    ],
    "cobranca_indevida": [
        ("estão me cobrando {tarifa} de {valor} todo mês e eu nunca autorizei", None),
        ("vi {canal} um débito de {valor} referente a {tarifa}, isso tá errado", ["app", "internet_banking"]),
        ("cobrança indevida: {tarifa}, {valor}. quero estorno", None),
        ("por que tão descontando {tarifa} da minha conta? veio {valor} {tempo}", None),
        ("me cobraram {tarifa} sendo que eu tenho isenção", None),
        ("quero cancelar e receber de volta o que pagaram de {tarifa}, foram {valor}", None),
        ("apareceu {tarifa} no extrato {canal}, não contratei isso", ["app", "internet_banking"]),
        ("acho um absurdo pagar {tarifa}, ninguém me avisou, cobraram {valor}", None),
        ("contestação de cobrança: {tarifa} lançada {tempo} no valor de {valor}", None),
    ],
    "acesso_bloqueado": [
        ("não consigo entrar {canal}, diz que a senha tá errada mas tenho certeza que tá certa", ["app", "internet_banking"]),
        ("meu acesso {canal} foi bloqueado depois que troquei de celular", ["app", "internet_banking"]),
        ("fica dando erro no login {canal} desde {desde}", ["app", "internet_banking"]),
        ("esqueci a senha de acesso e não consigo recuperar {canal}", ["app", "internet_banking"]),
        ("bloquearam meu acesso {canal}, preciso entrar pra pagar umas contas", ["app", "internet_banking"]),
        ("o token não chega e não consigo logar {canal}", ["app", "internet_banking"]),
        ("minha conta digital tá travada, não acesso {canal} desde {desde}", ["app", "internet_banking"]),
        ("tento fazer login {canal} e aparece usuário bloqueado, como resolvo?", ["app", "internet_banking"]),
        ("não recebo o código de verificação pra entrar {canal}", ["app", "internet_banking"]),
    ],
    "emprestimo": [
        ("quero simular um empréstimo de {valor}, quais as taxas?", None),
        ("tem como contratar empréstimo pessoal {canal}? preciso de {valor}", ["app", "agencia", "telefone"]),
        ("gostaria de saber se tenho crédito pré-aprovado pra empréstimo", None),
        ("quanto fica a parcela de um empréstimo de {valor} em 24 vezes?", None),
        ("tentei contratar o consignado {canal} e deu erro", ["app", "telefone"]),
        ("quero antecipar as parcelas do meu empréstimo, como faço {canal}?", ["app", "agencia", "telefone"]),
        ("qual a taxa de juros do empréstimo pessoal hoje? quero uns {valor}", None),
        ("dá pra renegociar meu empréstimo? to pensando em pegar mais {valor}", None),
        ("me ofereceram empréstimo {canal}, queria entender as condições pra {valor}", ["app", "telefone", "agencia"]),
    ],
    "resgate_investimento": [
        ("quero resgatar {valor} do meu {invest} mas a opção não aparece {canal}", ["app", "internet_banking"]),
        ("pedi o resgate do {invest} {tempo} e o dinheiro ainda não caiu na conta", None),
        ("como faço pra resgatar meu {invest}? tentei {canal} e não consegui", ["app", "internet_banking"]),
        ("o resgate de {valor} do {invest} tá pendente desde {desde}", None),
        ("qual o prazo pra cair o resgate do {invest}?", None),
        ("resgatei {valor} do {invest} {canal} e veio valor diferente do esperado", ["app", "internet_banking"]),
        ("não aparece saldo disponível pra resgate no {invest}", None),
        ("preciso tirar {valor} que tá aplicado no {invest}, o resgate é na hora?", None),
        ("solicitei resgate total do {invest} {canal} e sumiu da tela sem confirmação", ["app", "internet_banking"]),
    ],
    "segunda_via_boleto": [
        ("preciso da segunda via do boleto de {valor} que venceu {tempo}", None),
        ("como emito 2ª via de boleto {canal}?", ["app", "internet_banking"]),
        ("perdi o boleto do financiamento, consigo outra via {canal}?", ["app", "internet_banking", "telefone", "agencia"]),
        ("o boleto de {valor} venceu, quero gerar um atualizado", None),
        ("não recebi o boleto deste mês, podem mandar a segunda via?", None),
        ("quero a segunda via do boleto {canal}, valor {valor}", ["app", "internet_banking", "telefone"]),
        ("boleto venceu {tempo}, como pego outro com juros calculado?", None),
        ("o código de barras do boleto tá ilegível, preciso de uma nova via de {valor}", None),
        ("consigo reemitir um boleto vencido {canal}?", ["app", "internet_banking", "telefone", "agencia"]),
    ],
}

PRODUTO_FIXO = {
    "pix_nao_creditado": "pix",
    "acesso_bloqueado": "acesso_digital",
    "emprestimo": "emprestimo",
    "resgate_investimento": "investimentos",
    "segunda_via_boleto": "boleto",
}
URGENCIA_BASE = {
    "compra_nao_reconhecida": "alta",
    "cartao_bloqueado": "media",
    "cobranca_indevida": "media",
    "acesso_bloqueado": "media",
    "emprestimo": "baixa",
    "resgate_investimento": "baixa",
    "segunda_via_boleto": "baixa",
}


def br(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_valor(v: float, rng: random.Random) -> str:
    opcoes = ["R$ " + br(v)]
    if float(v).is_integer():
        iv = int(v)
        opcoes += [f"R${iv}", f"{iv} reais", "R$ " + f"{iv:,}".replace(",", ".")]
        if iv >= 1000 and iv % 1000 == 0:
            opcoes += [f"{iv // 1000} mil", f"{iv // 1000} mil reais"]
        elif iv >= 1000 and iv % 500 == 0:
            opcoes.append(f"{iv / 1000:.1f}".replace(".", ",") + " mil")
        if iv < 1000:
            opcoes.append(f"{iv} conto")
    return rng.choice(opcoes)


def sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def erro_digitacao(s: str, rng: random.Random) -> str:
    palavras = s.split(" ")
    idx = [i for i, p in enumerate(palavras) if len(p) >= 5 and p.isalpha()]
    if not idx:
        return s
    i = rng.choice(idx)
    p = list(palavras[i])
    j = rng.randint(1, len(p) - 3)
    p[j], p[j + 1] = p[j + 1], p[j]
    palavras[i] = "".join(p)
    return " ".join(palavras)


def escolher_cartao(rng: random.Random):
    tipo = rng.choice(["crédito", "crédito", "débito"])
    produto = "cartao_credito" if tipo == "crédito" else "cartao_debito"
    texto = f"cartão de {tipo}"
    if rng.random() < 0.3:
        texto += f" final {rng.randint(1000, 9999)}"
    return texto, produto


def gerar_exemplo(cat: str, t_idx: int, rng: random.Random) -> dict:
    texto, canais_permitidos = TEMPLATES[cat][t_idx]
    out = {"categoria": cat, "produto": PRODUTO_FIXO.get(cat), "urgencia": None,
           "valor": None, "canal": None}

    if "{valor}" in texto:
        v = float(rng.choice(VALORES[cat]))
        out["valor"] = round(v, 2)
        texto = texto.replace("{valor}", formatar_valor(v, rng))

    if "{canal}" in texto:
        if canais_permitidos and rng.random() > 0.25:
            canal = rng.choice(canais_permitidos)
            out["canal"] = canal
            texto = texto.replace("{canal}", rng.choice(CANAIS[canal]))
        else:
            texto = texto.replace("{canal}", "")

    if "{cartao}" in texto:
        ct, prod = escolher_cartao(rng)
        texto = texto.replace("{cartao}", ct)
        out["produto"] = prod
    if "{tarifa}" in texto:
        tarifa, prod = rng.choice(TARIFA)
        texto = texto.replace("{tarifa}", tarifa)
        out["produto"] = prod

    texto = (texto.replace("{tempo}", rng.choice(TEMPO))
                  .replace("{desde}", rng.choice(DESDE))
                  .replace("{pessoa}", rng.choice(PESSOA))
                  .replace("{loja}", rng.choice(LOJA))
                  .replace("{invest}", rng.choice(INVEST)))

    urgente = rng.random() < 0.15
    if cat == "pix_nao_creditado":
        base = "alta" if (out["valor"] or 0) >= 1000 else "media"
    else:
        base = URGENCIA_BASE[cat]
    out["urgencia"] = "alta" if urgente else base

    if urgente:
        if rng.random() < 0.5:
            texto = rng.choice(URGENCIA_PREFIXO) + texto
        else:
            texto = texto.rstrip("?.!") + "," + rng.choice(URGENCIA_SUFIXO)
    if rng.random() < 0.4:
        texto = rng.choice(SAUDACOES) + texto
    if rng.random() < 0.3:
        texto = texto + rng.choice(FECHAMENTOS)
    if rng.random() < 0.35:
        texto = texto.lower()
    if rng.random() < 0.3:
        texto = sem_acento(texto)
    if rng.random() < 0.2:
        texto = erro_digitacao(texto, rng)

    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\s+([,?.!:])", r"\1", texto).strip()
    if texto:
        texto = texto[0].upper() + texto[1:] if rng.random() < 0.5 else texto
    return {"input": texto, "output": out}


def main():
    rng = random.Random(SEED)
    vistos = set()
    splits = {"train": [], "val": [], "test": []}

    def add(split, cat, t_idx, novo=None):
        for _ in range(200):
            ex = gerar_exemplo(cat, t_idx, rng)
            if ex["input"] not in vistos:
                vistos.add(ex["input"])
                ex["id"] = f"{split}_{len(splits[split]):04d}"
                ex["template"] = f"{cat}#{t_idx}"
                if novo is not None:
                    ex["template_novo"] = novo
                splits[split].append(ex)
                return
        raise RuntimeError(f"Não consegui gerar exemplo único para {cat}#{t_idx}")

    for cat, tpls in TEMPLATES.items():
        n = len(tpls)
        vis = list(range(n - N_HOLDOUT))
        hold = list(range(n - N_HOLDOUT, n))
        for i in range(N_TREINO_POR_CAT):
            add("train", cat, vis[i % len(vis)])
        for i in range(N_VAL_POR_CAT):
            add("val", cat, vis[i % len(vis)])
        for i in range(N_TESTE_NOVO_POR_CAT):
            add("test", cat, hold[i % len(hold)], novo=True)
        for i in range(N_TESTE_VISTO_POR_CAT):
            add("test", cat, rng.choice(vis), novo=False)

    for nome, dados in splits.items():
        rng.shuffle(dados)
        with open(PASTA / f"{nome}.jsonl", "w", encoding="utf-8") as f:
            for ex in dados:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        nome: {
            "total": len(d),
            "categoria": dict(Counter(e["output"]["categoria"] for e in d)),
            "urgencia": dict(Counter(e["output"]["urgencia"] for e in d)),
            "com_valor": sum(e["output"]["valor"] is not None for e in d),
            "com_canal": sum(e["output"]["canal"] is not None for e in d),
        }
        for nome, d in splits.items()
    }
    stats["test"]["template_novo"] = sum(e["template_novo"] for e in splits["test"])
    with open(PASTA / "estatisticas.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v["total"] for k, v in stats.items()}, indent=2))
    print("Exemplo:", json.dumps(splits["train"][0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
