import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

SAIDA = Path(__file__).resolve().parent.parent / "results" / "gpu_info.json"


def linha(t=""):
    print(t)


def main():
    linha("=" * 64)
    linha("  VERIFICAÇÃO DE GPU / CUDA — LLM Triagem de Chamados")
    linha("=" * 64)
    info = {
        "python": sys.version.split()[0],
        "sistema": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "cuda_disponivel": torch.cuda.is_available(),
        "cuda_versao_torch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    for k, v in info.items():
        linha(f"  {k:<20}: {v}")

    if not torch.cuda.is_available():
        linha("\n  >>> NENHUMA GPU CUDA ENCONTRADA. Veja a seção 'Ambiente e GPU' do README.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    gpu = {
        "gpu_nome": props.name,
        "vram_total_gb": round(props.total_memory / 1024**3, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessadores": props.multi_processor_count,
        "suporta_bf16": torch.cuda.get_device_capability(0)[0] >= 8,
        "qtd_gpus": torch.cuda.device_count(),
    }
    linha("-" * 64)
    for k, v in gpu.items():
        linha(f"  {k:<20}: {v}")

    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 2 * n**3 * reps / dt / 1e12
    gpu["benchmark_matmul_fp16_tflops"] = round(tflops, 1)
    linha("-" * 64)
    linha(f"  Teste matmul 4096x4096 fp16 x{reps}: {dt*1000:.0f} ms  ->  {tflops:.1f} TFLOPS")
    linha(f"  Memória alocada no teste: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")

    linha("-" * 64)
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
        linha(smi)
        info["nvidia_smi"] = smi
    except Exception as e:
        linha(f"  nvidia-smi indisponível: {e}")

    linha("  >>> GPU OK — pronta para o treino.")
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps({**info, **gpu}, ensure_ascii=False, indent=2), encoding="utf-8")
    linha(f"  Informações salvas em {SAIDA.relative_to(SAIDA.parent.parent)}")


if __name__ == "__main__":
    main()
