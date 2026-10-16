import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTADOS

SRC = Path(__file__).resolve().parent

EXPERIMENTOS = [
    ("E1_baseline_r16", {"rank": 16, "lr": 2e-4, "epochs": 2, "quant": "4bit"}),
    ("E2_rank4",        {"rank": 4,  "lr": 2e-4, "epochs": 2, "quant": "4bit"}),
    ("E3_rank64",       {"rank": 64, "lr": 2e-4, "epochs": 2, "quant": "4bit"}),
    ("E4_lr_baixo",     {"rank": 16, "lr": 2e-5, "epochs": 2, "quant": "4bit"}),
    ("E5_lr_alto",      {"rank": 16, "lr": 1e-3, "epochs": 2, "quant": "4bit"}),
    ("E6_4_epocas",     {"rank": 16, "lr": 2e-4, "epochs": 4, "quant": "4bit"}),
    ("E7_lora_sem_quant", {"rank": 16, "lr": 2e-4, "epochs": 2, "quant": "none"}),
]


def rodar(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--somente", nargs="*")
    p.add_argument("--refazer", action="store_true")
    p.add_argument("--sem_baselines", action="store_true")
    p.add_argument("--modelo", default=None)
    p.add_argument("--permitir_cpu", action="store_true", help="só para depuração")
    a = p.parse_args()
    py = sys.executable
    extra_modelo = (["--modelo", a.modelo] if a.modelo else []) + (["--permitir_cpu"] if a.permitir_cpu else [])
    falhas, t0 = [], time.time()

    def feito(nome):
        return (RESULTADOS / f"metrics_{nome}.json").exists() and not a.refazer

    if not a.sem_baselines:
        for modo in ["zero_shot", "few_shot"]:
            if feito(f"base_{modo}"):
                print(f"[pulando] base_{modo} já avaliado")
                continue
            if not rodar([py, str(SRC / "evaluate.py"), "--modo", modo] + extra_modelo):
                falhas.append(f"base_{modo}")

    for nome, hp in EXPERIMENTOS:
        if a.somente and nome not in a.somente:
            continue
        if feito(nome):
            print(f"[pulando] {nome} já avaliado")
            continue
        cmd = [py, str(SRC / "train.py"), "--run", nome,
               "--rank", str(hp["rank"]), "--lr", str(hp["lr"]),
               "--epochs", str(1 if a.rapido else hp["epochs"]), "--quant", hp["quant"]] + extra_modelo
        if a.rapido:
            cmd += ["--limite_treino", "320"]
        if not rodar(cmd):
            falhas.append(nome)
            continue
        if not rodar([py, str(SRC / "evaluate.py"), "--modo", "finetuned", "--run", nome]
                     + (["--permitir_cpu"] if a.permitir_cpu else [])):
            falhas.append(nome)
        rodar([py, str(SRC / "report.py")])

    rodar([py, str(SRC / "report.py")])
    print(f"\nTempo total: {(time.time() - t0) / 60:.1f} min")
    if falhas:
        print("Experimentos com falha (veja o log acima):", ", ".join(falhas))


if __name__ == "__main__":
    main()
