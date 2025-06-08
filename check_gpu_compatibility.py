#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar compatibilidade e configuração de GPU para o projeto.
Execute antes de iniciar o treinamento com GPU.
"""

import torch
import sys
import os
from typing import Dict, Any

def check_cuda_installation() -> Dict[str, Any]:
    """Verifica instalação do CUDA."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    return info

def check_gpu_devices() -> Dict[str, Any]:
    """Verifica dispositivos GPU disponíveis."""
    if not torch.cuda.is_available():
        return {"gpu_count": 0, "devices": []}
    
    gpu_count = torch.cuda.device_count()
    devices = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "id": i,
            "name": props.name,
            "memory_total_gb": props.total_memory / 1e9,
            "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
            "memory_cached_gb": torch.cuda.memory_reserved(i) / 1e9,
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
        }
        devices.append(device_info)
    
    return {"gpu_count": gpu_count, "devices": devices}

def test_gpu_performance():
    """Testa performance básica da GPU."""
    if not torch.cuda.is_available():
        print("❌ CUDA não disponível - não é possível testar GPU")
        return
    
    print("🧪 Testando performance da GPU...")
    
    # Teste de criação de tensor
    device = torch.device("cuda:0")
    try:
        # Criar tensores de teste
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Teste de multiplicação de matrizes
        import time
        start_time = time.time()
        for _ in range(100):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Aguardar conclusão
        gpu_time = time.time() - start_time
        
        # Teste na CPU para comparação
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        start_time = time.time()
        for _ in range(100):
            c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        
        print(f"✅ Teste de performance concluído:")
        print(f"   GPU: {gpu_time:.4f}s")
        print(f"   CPU: {cpu_time:.4f}s")
        print(f"   Aceleração: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("🚀 GPU está funcionando bem!")
        else:
            print("⚠️ GPU pode não estar otimizada")
            
    except Exception as e:
        print(f"❌ Erro no teste de GPU: {e}")

def check_memory_requirements():
    """Verifica se há memória suficiente para treinamento."""
    if not torch.cuda.is_available():
        return
    
    # Estimativas conservadoras para BERT
    model_sizes = {
        "bert-base": 0.5,  # GB
        "bert-large": 1.3,  # GB
    }
    
    batch_memory_per_sample = {
        128: 0.15,  # GB por amostra para seq_len=128
        256: 0.25,  # GB por amostra para seq_len=256
        512: 0.45,  # GB por amostra para seq_len=512
    }
    
    gpu_info = check_gpu_devices()
    
    for device in gpu_info["devices"]:
        available_memory = device["memory_total_gb"] - device["memory_allocated_gb"]
        
        print(f"\n📊 GPU {device['id']} - {device['name']}:")
        print(f"   Memória total: {device['memory_total_gb']:.1f} GB")
        print(f"   Memória disponível: {available_memory:.1f} GB")
        
        # Calcular batch sizes recomendados
        for seq_len, memory_per_sample in batch_memory_per_sample.items():
            model_memory = model_sizes["bert-base"]
            overhead = 1.0  # GB de overhead
            
            usable_memory = available_memory - model_memory - overhead
            if usable_memory > 0:
                max_batch_size = int(usable_memory / memory_per_sample)
                print(f"   Batch size recomendado (seq_len={seq_len}): {max_batch_size}")
            else:
                print(f"   ⚠️ Memória insuficiente para seq_len={seq_len}")

def check_environment_variables():
    """Verifica variáveis de ambiente importantes."""
    important_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]
    
    print("\n🔧 Variáveis de ambiente CUDA:")
    for var in important_vars:
        value = os.environ.get(var, "Não definida")
        print(f"   {var}: {value}")

def suggest_optimizations():
    """Sugere otimizações baseadas no hardware detectado."""
    gpu_info = check_gpu_devices()
    
    if gpu_info["gpu_count"] == 0:
        print("\n💡 Sugestões:")
        print("   - Instale drivers NVIDIA e CUDA")
        print("   - Verifique se PyTorch foi instalado com suporte CUDA")
        return
    
    print("\n💡 Sugestões de otimização:")
    
    for device in gpu_info["devices"]:
        memory_gb = device["memory_total_gb"]
        
        if memory_gb >= 24:
            print(f"   GPU {device['id']}: GPU de alta capacidade detectada")
            print("     - Use batch_size=32-64")
            print("     - Ative FP16 para maior throughput")
            print("     - Considere usar gradient_accumulation_steps=1")
        elif memory_gb >= 12:
            print(f"   GPU {device['id']}: GPU de capacidade média detectada")
            print("     - Use batch_size=16-32")
            print("     - Ative FP16 para economia de memória")
            print("     - Use gradient_accumulation_steps=2 se necessário")
        elif memory_gb >= 8:
            print(f"   GPU {device['id']}: GPU de capacidade básica detectada")
            print("     - Use batch_size=8-16")
            print("     - Ative FP16 obrigatoriamente")
            print("     - Use gradient_accumulation_steps=4")
            print("     - Considere usar gradient_checkpointing")
        else:
            print(f"   GPU {device['id']}: Memória limitada")
            print("     - Use batch_size=4-8")
            print("     - Ative todas as otimizações de memória")
            print("     - Considere usar CPU em vez de GPU")

def main():
    """Função principal de verificação."""
    print("🔍 VERIFICAÇÃO DE COMPATIBILIDADE GPU")
    print("=" * 50)
    
    # Verificar CUDA
    cuda_info = check_cuda_installation()
    print("\n🎯 Instalação CUDA:")
    print(f"   CUDA disponível: {'✅ Sim' if cuda_info['cuda_available'] else '❌ Não'}")
    print(f"   PyTorch versão: {cuda_info['pytorch_version']}")
    
    if cuda_info['cuda_available']:
        print(f"   CUDA versão: {cuda_info['cuda_version']}")
        print(f"   cuDNN versão: {cuda_info['cudnn_version']}")
    else:
        print("\n❌ CUDA não está disponível!")
        print("Instruções para instalar:")
        print("1. Instale drivers NVIDIA apropriados")
        print("2. Instale CUDA Toolkit")
        print("3. Reinstale PyTorch com suporte CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # Verificar GPUs
    gpu_info = check_gpu_devices()
    print(f"\n🎮 GPUs disponíveis: {gpu_info['gpu_count']}")
    
    for device in gpu_info["devices"]:
        print(f"\n   GPU {device['id']}: {device['name']}")
        print(f"     Memória: {device['memory_total_gb']:.1f} GB")
        print(f"     Compute Capability: {device['compute_capability']}")
        print(f"     Multiprocessors: {device['multiprocessor_count']}")
    
    # Testar performance
    test_gpu_performance()
    
    # Verificar requisitos de memória
    check_memory_requirements()
    
    # Verificar variáveis de ambiente
    check_environment_variables()
    
    # Sugerir otimizações
    suggest_optimizations()
    
    print(f"\n✅ Verificação concluída!")
    print("Execute 'python main.py --train --validate' para iniciar o treinamento com GPU")

if __name__ == "__main__":
    main()