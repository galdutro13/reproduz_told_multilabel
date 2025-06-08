# -*- coding: utf-8 -*-
"""
Configurações e constantes do projeto com suporte a GPU - VERSÃO CORRIGIDA.
"""

import os
import torch
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments
from datetime import datetime

# ---------- Constantes Globais -------------------------------------------
DATASET_PATH = "ToLD-BR.csv"
MODEL_DIR = 'outputs_bert/'
DEFAULT_MODEL_NAME = "google-bert/bert-base-multilingual-cased"
LABELS = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]
NUM_LABELS = len(LABELS)
SEED = 42

# Detecção automática de dispositivo
N_CPU = max(1, multiprocessing.cpu_count() - 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = torch.cuda.is_available()

# ---------- Environment Setup --------------------------------------------
def setup_environment():
    """Configura variáveis de ambiente para otimização com suporte a GPU."""
    import torch
    import warnings
    from transformers import logging as hf_logging
    import logging
    
    # Log informações do dispositivo
    if USE_CUDA:
        print(f"🚀 GPU detectada: {torch.cuda.get_device_name()}")
        print(f"📊 Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔢 Número de GPUs: {torch.cuda.device_count()}")
    else:
        print("💻 Usando CPU - GPU não detectada")
    
    # Configurações de ambiente para CPU (ainda relevantes para alguns componentes)
    if not USE_CUDA:
        os.environ.update({
            "OMP_NUM_THREADS": str(N_CPU),
            "MKL_NUM_THREADS": str(N_CPU),
            "TOKENIZERS_PARALLELISM": "false",
            "ONEDNN_MAX_CPU_ISA": "AVX2",
        })
        torch.set_num_threads(N_CPU)
        torch.set_num_interop_threads(max(1, N_CPU // 2))
    else:
        # Configurações otimizadas para GPU
        os.environ.update({
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_LAUNCH_BLOCKING": "0",  # Para debugging se necessário
        })
    
    # Configurar logging
    hf_logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)
    
    # Ignorar avisos
    warnings.filterwarnings("ignore", message=".*ipex_MKLSGEMM.*")

# ---------- Model Configuration ------------------------------------------
@dataclass
class ModelConfig:
    """Configuração do modelo e treinamento com suporte automático a GPU."""
    model_name: str = DEFAULT_MODEL_NAME
    num_train_epochs: int = 3
    # Batch sizes otimizados automaticamente baseado no dispositivo
    per_device_train_batch_size: int = 16 if USE_CUDA else 8
    per_device_eval_batch_size: int = 32 if USE_CUDA else 16
    warmup_ratio: float = 0.06
    weight_decay: float = 0.0
    learning_rate: float = 4e-5
    max_seq_length: int = 128
    do_lower_case: bool = True
    output_dir: str = MODEL_DIR
    cache_dir: str = "cache_bert/"
    best_model_dir: str = "outputs/best_model"
    evaluate_during_training: bool = True
    eval_strategy: str = "steps"  # ← CORRIGIDO: era evaluation_strategy
    eval_steps: int = 350
    save_steps: int = 350
    logging_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "avg_precision"
    greater_is_better: bool = True
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha_weights: Optional[List[float]] = None
    pos_weight: Optional[List[float]] = None
    # Configurações otimizadas baseado no dispositivo
    dataloader_num_workers: int = 4 if USE_CUDA else N_CPU
    disable_tqdm: bool = False
    logging_first_step: bool = True
    # Configurações específicas para GPU
    use_cuda: bool = USE_CUDA
    fp16: bool = False  # ← DESABILITADO: Conflito com torch.compile no PyTorch 2.7.1
    dataloader_pin_memory: bool = USE_CUDA
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        """Ajustes automáticos após inicialização."""
        if self.use_cuda and torch.cuda.is_available():
            # Ajustar batch size baseado na memória GPU disponível
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory_gb >= 24:  # GPU de alta capacidade (RTX 4090, A100, etc.)
                self.per_device_train_batch_size = max(self.per_device_train_batch_size, 32)
                self.per_device_eval_batch_size = max(self.per_device_eval_batch_size, 64)
            elif gpu_memory_gb >= 12:  # GPU média (RTX 3080, 4070, etc.)
                self.per_device_train_batch_size = max(self.per_device_train_batch_size, 24)
                self.per_device_eval_batch_size = max(self.per_device_eval_batch_size, 48)
            elif gpu_memory_gb >= 8:  # GPU básica (RTX 3060, etc.)
                self.per_device_train_batch_size = max(self.per_device_train_batch_size, 16)
                self.per_device_eval_batch_size = max(self.per_device_eval_batch_size, 32)
            
            print(f"🎯 Batch sizes ajustados para GPU ({gpu_memory_gb:.1f}GB): "
                  f"train={self.per_device_train_batch_size}, eval={self.per_device_eval_batch_size}")
    
    def to_training_args(self) -> TrainingArguments:
        """Converte para TrainingArguments do HF com configurações de GPU - VERSÃO CORRIGIDA."""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            logging_dir=os.path.join(self.output_dir, "runs"),
            logging_steps=self.logging_steps,
            logging_first_step=self.logging_first_step,
            eval_strategy=self.eval_strategy,  # ← CORRIGIDO: era evaluation_strategy
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            seed=SEED,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            remove_unused_columns=False,
            label_names=["labels"],
            disable_tqdm=self.disable_tqdm,
            report_to=["tensorboard"],
            run_name=f"bert_multilabel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            # Configurações específicas para GPU
            use_cpu=not self.use_cuda,
            fp16=self.fp16,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
        )
        
        return training_args

@dataclass
class LossConfig:
    """Configuração para loss functions."""
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha_weights: Optional[List[float]] = None
    pos_weight: Optional[List[float]] = None
    
    def validate(self):
        """Valida configurações conflitantes."""
        if self.use_focal_loss and self.pos_weight:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Focal Loss tem prioridade sobre pos_weight")
            self.pos_weight = None

def get_optimal_batch_size(model_name: str = DEFAULT_MODEL_NAME, max_seq_length: int = 128) -> int:
    """
    Calcula batch size ótimo baseado na GPU disponível.
    
    Args:
        model_name: Nome do modelo
        max_seq_length: Comprimento máximo das sequências
    
    Returns:
        int: Batch size recomendado
    """
    if not torch.cuda.is_available():
        return 8  # Batch size conservador para CPU
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Estimativas baseadas em testes empíricos
    base_memory_per_sample = {
        128: 0.15,  # GB por amostra para seq_len=128
        256: 0.25,  # GB por amostra para seq_len=256
        512: 0.45,  # GB por amostra para seq_len=512
    }
    
    memory_per_sample = base_memory_per_sample.get(max_seq_length, 0.15)
    
    # Reservar 20% da memória para outras operações
    available_memory = gpu_memory_gb * 0.8
    optimal_batch_size = int(available_memory / memory_per_sample)
    
    # Limites práticos
    optimal_batch_size = max(4, min(optimal_batch_size, 64))
    
    return optimal_batch_size

def setup_distributed_training():
    """Configura treinamento distribuído se múltiplas GPUs disponíveis."""
    if torch.cuda.device_count() > 1:
        print(f"🔗 Configurando treinamento distribuído para {torch.cuda.device_count()} GPUs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
        return True
    return False