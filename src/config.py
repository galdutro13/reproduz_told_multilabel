# -*- coding: utf-8 -*-
"""
Configurações e constantes do projeto.
"""

import os
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

# Configuração de ambiente
N_CPU = max(1, multiprocessing.cpu_count() - 1)

# ---------- Environment Setup --------------------------------------------
def setup_environment():
    """Configura variáveis de ambiente para otimização."""
    import torch
    import warnings
    from transformers import logging as hf_logging
    import logging
    
    # Configurações de ambiente
    os.environ.update({
        "OMP_NUM_THREADS": str(N_CPU),
        "MKL_NUM_THREADS": str(N_CPU),
        "TOKENIZERS_PARALLELISM": "false",
        "ONEDNN_MAX_CPU_ISA": "AVX2",
        "IPEX_VERBOSE": "0",
    })
    
    torch.set_num_threads(N_CPU)
    torch.set_num_interop_threads(max(1, N_CPU // 2))
    
    # Configurar logging
    hf_logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)
    
    # Ignorar avisos
    warnings.filterwarnings("ignore", message=".*ipex_MKLSGEMM.*")

# ---------- Model Configuration ------------------------------------------
@dataclass
class ModelConfig:
    """Configuração do modelo e treinamento."""
    model_name: str = DEFAULT_MODEL_NAME
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 100
    warmup_ratio: float = 0.06
    weight_decay: float = 0.0
    learning_rate: float = 4e-5
    max_seq_length: int = 128
    do_lower_case: bool = True
    output_dir: str = MODEL_DIR
    cache_dir: str = "cache_bert/"
    best_model_dir: str = "outputs/best_model"
    evaluate_during_training: bool = True
    evaluation_strategy: str = "steps"
    eval_steps: int = 350
    save_steps: int = 350
    logging_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_avg_precision"  # Changed to include eval_ prefix
    greater_is_better: bool = True
    save_strategy: str = "steps"  # Added explicit save strategy
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha_weights: Optional[List[float]] = None
    pos_weight: Optional[List[float]] = None
    dataloader_num_workers: int = N_CPU
    disable_tqdm: bool = False
    logging_first_step: bool = True
    
    def to_training_args(self) -> TrainingArguments:
        """Converte para TrainingArguments do HF."""
        return TrainingArguments(
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
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,  # Added explicit save strategy
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            seed=SEED,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=False,
            label_names=["labels"],
            disable_tqdm=self.disable_tqdm,
            report_to=["tensorboard"],
            run_name=f"bert_multilabel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

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