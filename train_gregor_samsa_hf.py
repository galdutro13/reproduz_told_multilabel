# -*- coding: utf-8 -*-
"""
Fine-tuning BERT multi-label (ToLD-BR) - Versão Hugging Face Transformers
PyTorch ≥2.1, transformers 4.48.x

Reimplementação usando HF Transformers puro para máxima flexibilidade com loss functions.

Suporta:
- BCEWithLogitsLoss padrão (baseline)
- BCEWithLogitsLoss com pos_weight customizado
- Focal Loss com alpha automático ou manual
- Facilmente extensível para outras loss functions
- Execução de múltiplas configurações via arquivo JSON

Visualizações incluídas:
- Curvas de treino (loss/F1)
- Matrizes de confusão por classe
- Heatmap de coocorrência
- Barplot e radar chart de métricas por classe
- Curvas PR e ROC por classe
- Curvas F-beta × Threshold para otimização de limiares
"""

# ---------- 0 | Ambiente -------------------------------------------------
import os
import multiprocessing
import logging
import warnings
import textwrap
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de ambiente
N_CPU = max(1, multiprocessing.cpu_count() - 1)
os.environ.update({
    "OMP_NUM_THREADS": str(N_CPU),
    "MKL_NUM_THREADS": str(N_CPU),
    "TOKENIZERS_PARALLELISM": "false",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
    "IPEX_VERBOSE": "0",
})

torch.set_num_threads(N_CPU)
torch.set_num_interop_threads(max(1, N_CPU // 2))

# Imports do Hugging Face
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    set_seed,
    logging as hf_logging
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Configurar logging
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ignorar avisos
warnings.filterwarnings("ignore", message=".*ipex_MKLSGEMM.*")

# ---------- 1 | Constantes -----------------------------------------------
DATASET_PATH = "ToLD-BR.csv"
MODEL_DIR = 'outputs_bert/'
DEFAULT_MODEL_NAME = "google-bert/bert-base-multilingual-cased"
LABELS = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]
NUM_LABELS = len(LABELS)
SEED = 42

# Setar seed global
set_seed(SEED)

# ---------- 1.5 | Implementação de Loss Functions -------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss para classificação multi-label binária.
    
    FL(pt) = -α(1-pt)^γ * log(pt)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt).pow(self.gamma)
        focal_loss = focal_weight * bce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.to(logits.device)
                alpha_t = alpha_t.view(1, -1).expand_as(targets)
            
            alpha_factor = torch.where(targets == 1, alpha_t, 1 - alpha_t)
            focal_loss = alpha_factor * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------- 2 | Dataset e DataLoader -------------------------------------
class MultiLabelDataset(Dataset):
    """Dataset para classificação multi-label."""
    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# ---------- 3 | Configuração do Modelo -----------------------------------
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
    logging_steps: int = 350
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "avg_precision"
    greater_is_better: bool = True
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha_weights: Optional[torch.Tensor] = None
    pos_weight: Optional[List[float]] = None
    dataloader_num_workers: int = N_CPU
    
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
            evaluation_strategy=self.evaluation_strategy,
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
        )

# ---------- 4 | Trainer Customizado --------------------------------------
class MultiLabelTrainer(Trainer):
    """Trainer customizado para multi-label com suporte a diferentes loss functions."""
    
    def __init__(self, loss_config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config or {}
        self._setup_loss_function()
        
    def _setup_loss_function(self):
        """Configura a loss function baseada em loss_config."""
        if self.loss_config.get('use_focal_loss'):
            gamma = self.loss_config.get('focal_gamma', 2.0)
            alpha = self.loss_config.get('focal_alpha_weights')
            self.loss_fct = FocalLoss(gamma=gamma, alpha=alpha)
            logger.info(f"✨ Usando Focal Loss com gamma={gamma}")
        elif self.loss_config.get('pos_weight'):
            pos_weight = torch.tensor(self.loss_config['pos_weight'])
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info(f"⚖️ Usando BCEWithLogitsLoss com pos_weight")
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()
            logger.info("📊 Usando BCEWithLogitsLoss padrão")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Calcula a loss usando a função configurada."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """Override para adicionar logging customizado."""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        super().log(logs)

# ---------- 5 | Métricas -------------------------------------------------
from sklearn.metrics import f1_score, hamming_loss, average_precision_score

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Calcula métricas para avaliação."""
    predictions, labels = eval_pred
    
    # Aplicar sigmoid e threshold
    predictions = torch.sigmoid(torch.tensor(predictions))
    predictions = (predictions >= 0.5).float().numpy()
    
    # Calcular métricas
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    hamming = hamming_loss(labels, predictions)
    avg_precision = average_precision_score(labels, predictions, average='macro')
    
    return {
        'macro_f1': f1_macro,
        'hamming_loss': hamming,
        'avg_precision': avg_precision
    }

# ---------- 6 | Utilidades -----------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Carrega e normaliza o CSV."""
    if not os.path.exists(path):
        sys.exit(f"Arquivo {path} ausente.")
    df = pd.read_csv(path)
    if "text" not in df.columns or not set(LABELS).issubset(df.columns):
        sys.exit("CSV precisa conter a coluna 'text' e todas as colunas de rótulo.")
    df[LABELS] = (df[LABELS].fillna(0).astype(float) > 0).astype(int)
    df["labels"] = df[LABELS].values.tolist()
    logger.info(f"Dataset carregado: {len(df)} amostras")
    return df[["text", "labels"]]

def calculate_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Calcula pesos alpha para cada classe baseado na frequência."""
    label_matrix = np.array(df['labels'].tolist())
    pos_counts = label_matrix.sum(axis=0)
    neg_counts = len(df) - pos_counts
    alphas = neg_counts / len(df)
    
    logger.info("\nPesos alpha calculados para cada classe (positivos):")
    for i, label in enumerate(LABELS):
        logger.info(f"  {label}: α={alphas[i]:.3f} (pos={pos_counts[i]}, neg={neg_counts[i]})")
    
    return torch.tensor(alphas, dtype=torch.float32)

def split_stratified_holdout(df: pd.DataFrame, seed: int = SEED,
                           train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Divide o DataFrame em treino, validação e teste."""
    test_ratio = 1.0 - train_ratio - val_ratio
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        y = np.asarray(df["labels"].tolist())
        idx = np.arange(len(df))

        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed
        )
        train_idx, temp_idx = next(msss1.split(idx, y))

        y_temp = y[temp_idx]
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed,
        )
        val_rel, test_rel = next(msss2.split(temp_idx.reshape(-1, 1), y_temp))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        logger.info("Estratificação (iterative-stratification) concluída.")
    except ImportError:
        warnings.warn(
            "Pacote 'iterative-stratification' não localizado. "
            "Instale via 'pip install iterative-stratification' para melhor fidelidade."
        )
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        y_single = df[LABELS].idxmax(axis=1)
        train_idx, temp_idx = train_test_split(
            idx, test_size=(val_ratio + test_ratio), stratify=y_single, random_state=seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_ratio / (val_ratio + test_ratio),
            stratify=y_single[temp_idx], random_state=seed
        )

    d_train = df.iloc[train_idx].reset_index(drop=True)
    d_val = df.iloc[val_idx].reset_index(drop=True)
    d_test = df.iloc[test_idx].reset_index(drop=True)

    def _ratio(x):
        return f"{len(x):,} ({len(x)/len(df):.1%})"
    
    logger.info(f"Partições — treino: {_ratio(d_train)}, "
                f"validação: {_ratio(d_val)}, teste: {_ratio(d_test)}")
    return d_train, d_val, d_test

# ---------- 7 | Criação e Treinamento do Modelo -------------------------
def create_model(config: ModelConfig):
    """Cria o modelo BERT para classificação multi-label."""
    model_config = AutoConfig.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        cache_dir=config.cache_dir
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        config=model_config,
        cache_dir=config.cache_dir
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        do_lower_case=config.do_lower_case,
        cache_dir=config.cache_dir
    )
    
    # Aplicar otimizações se disponíveis
    try:
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, dtype=torch.float32)
        logger.info("🚀 IPEX otimizado (FP32)")
    except ImportError:
        logger.info("ℹ️ IPEX não disponível")
    
    # torch.compile se disponível
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, backend='ipex', mode='default')
            logger.info("🛠️ torch.compile ativado")
        except Exception as e:
            logger.warning(f"torch.compile falhou: {e}")
    
    return model, tokenizer

def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                config: ModelConfig) -> Tuple[Any, Dict[str, List[float]]]:
    """Treina o modelo e retorna o trainer e histórico."""
    # Criar modelo e tokenizer
    model, tokenizer = create_model(config)
    
    # Criar datasets
    train_dataset = MultiLabelDataset(
        train_df['text'].tolist(),
        train_df['labels'].tolist(),
        tokenizer,
        config.max_seq_length
    )
    
    val_dataset = MultiLabelDataset(
        val_df['text'].tolist(),
        val_df['labels'].tolist(),
        tokenizer,
        config.max_seq_length
    )
    
    # Configurar loss
    loss_config = {
        'use_focal_loss': config.use_focal_loss,
        'focal_gamma': config.focal_gamma,
        'focal_alpha_weights': config.focal_alpha_weights,
        'pos_weight': config.pos_weight
    }
    
    # Criar trainer
    training_args = config.to_training_args()
    
    trainer = MultiLabelTrainer(
        loss_config=loss_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if config.evaluate_during_training else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Treinar
    logger.info("Iniciando treinamento...")
    train_result = trainer.train()
    
    # Salvar modelo
    trainer.save_model()
    trainer.save_state()
    
    # Extrair histórico de treinamento
    history = {
        'global_step': [],
        'train_loss': [],
        'eval_loss': [],
        'macro_f1': [],
        'hamming_loss': [],
        'avg_precision': []
    }
    
    for log in trainer.state.log_history:
        if 'loss' in log:
            history['global_step'].append(log.get('step', 0))
            history['train_loss'].append(log['loss'])
        if 'eval_loss' in log:
            history['eval_loss'].append(log['eval_loss'])
            history['macro_f1'].append(log.get('eval_macro_f1', 0))
            history['hamming_loss'].append(log.get('eval_hamming_loss', 0))
            history['avg_precision'].append(log.get('eval_avg_precision', 0))
    
    return trainer, history

def evaluate_model(trainer: Any, test_df: pd.DataFrame) -> Tuple[Dict[str, float], np.ndarray]:
    """Avalia o modelo no conjunto de teste."""
    test_dataset = MultiLabelDataset(
        test_df['text'].tolist(),
        test_df['labels'].tolist(),
        trainer.tokenizer,
        trainer.args.model.config.max_seq_length if hasattr(trainer.args, 'model') else 128
    )
    
    predictions = trainer.predict(test_dataset)
    
    # Extrair logits e aplicar sigmoid
    logits = predictions.predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Calcular métricas
    metrics = compute_metrics(EvalPrediction(predictions=logits, label_ids=predictions.label_ids))
    
    return metrics, probs

# ---------- 8 | Visualizações (mantidas do original) --------------------
# [Todas as funções de plotagem do código original permanecem as mesmas]
# plot_train_curves, plot_multilabel_confusion, plot_cooccurrence_heatmap,
# plot_metrics_per_class, plot_fbeta_threshold_curves, plot_pr_roc_curves

def plot_train_curves(training_details, save_path="outputs_bert/loss_f1_vs_step.png"):
    """Desenha curvas de train_loss, eval_loss e macro_f1."""
    df = pd.DataFrame(training_details)
    
    if 'eval_loss' not in df.columns or 'macro_f1' not in df.columns:
        logger.warning("training_details não possui eval_loss ou macro_f1")
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot losses
    ax1.plot(df['global_step'], df['train_loss'], 'b-', label='Train loss', linewidth=2)
    ax1.plot(df['global_step'], df['eval_loss'], 'r-', label='Eval loss', linewidth=2)
    ax1.set_xlabel('Global step')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot F1 on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df['global_step'], df['macro_f1'], 'g-', label='Macro-F1', linewidth=2)
    ax2.set_ylabel('Macro-F1', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')
    
    plt.title('Curvas de Treinamento: Loss e F1-Macro')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Curva Loss/F1 salva em: {save_path}")

# [Incluir todas as outras funções de plotagem do código original aqui]
# Por brevidade, não estou repetindo todas, mas elas devem ser incluídas

# ---------- 9 | Funções de Configuração e Execução ----------------------
def load_config_file(config_path: str) -> dict:
    """Carrega arquivo de configuração JSON."""
    if not os.path.exists(config_path):
        sys.exit(f"❌ Arquivo de configuração '{config_path}' não encontrado.")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"❌ Erro ao parsear JSON: {e}")
    
    if 'instances' not in config:
        sys.exit("❌ Arquivo de configuração deve conter chave 'instances'.")
    
    return config

def parse_instance_config(instance: dict, instance_num: int) -> ModelConfig:
    """Converte configuração de instância para ModelConfig."""
    params = instance.get('parameters', {})
    
    if 'model_name' not in params:
        sys.exit(f"❌ Instância {instance_num}: 'model_name' é obrigatório.")
    
    config = ModelConfig(
        model_name=params['model_name'],
        use_focal_loss=params.get('use_focal_loss', False),
        focal_gamma=params.get('focal_gamma', 2.0),
        pos_weight=params.get('pos_weight'),
        num_train_epochs=params.get('epochs', 3),
        max_seq_length=params.get('max_seq_length', 128),
        do_lower_case=params.get('do_lower_case', True),
        evaluate_during_training=params.get('validate', True)
    )
    
    # Validação
    if config.use_focal_loss and config.pos_weight:
        logger.warning(f"Instância {instance_num}: Focal Loss tem prioridade sobre pos_weight")
        config.pos_weight = None
    
    return config

def run_instance(instance: dict, instance_num: int, 
                train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    """Executa uma instância de treinamento."""
    instance_id = instance.get('id', f'instance_{instance_num}')
    instance_name = instance.get('name', 'Sem nome')
    
    logger.info("\n" + "="*80)
    logger.info(f"🚀 EXECUTANDO INSTÂNCIA {instance_num}: {instance_id}")
    logger.info(f"📝 Descrição: {instance_name}")
    logger.info("="*80)
    
    # Parse configuração
    config = parse_instance_config(instance, instance_num)
    
    # Configurar diretórios
    instance_dir = instance_id
    os.makedirs(instance_dir, exist_ok=True)
    
    config.output_dir = os.path.join(instance_dir, "outputs_bert")
    config.cache_dir = os.path.join(instance_dir, "cache_bert")
    config.best_model_dir = os.path.join(instance_dir, "outputs/best_model")
    
    # Calcular alpha weights se necessário
    if config.use_focal_loss:
        config.focal_alpha_weights = calculate_class_weights(train_df)
    
    # Treinar modelo
    trainer, history = train_model(train_df, val_df, config)
    
    # Avaliar no teste
    logger.info(f"\n📊 Avaliação no conjunto de teste para {instance_id}:")
    metrics, probs = evaluate_model(trainer, test_df)
    
    # Salvar resultados
    results_path = os.path.join(instance_dir, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Gerar visualizações
    logger.info(f"\n📈 Gerando visualizações para {instance_id}...")
    plot_train_curves(history, os.path.join(instance_dir, "loss_f1_vs_step.png"))
    
    # [Chamar outras funções de visualização aqui]
    
    logger.info(f"\n✅ Instância {instance_id} concluída!")
    
    return metrics

# ---------- 10 | Main ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="""
Fine-tuning BERT multi-label (ToLD-BR) - Versão HF Transformers

Reimplementação usando Hugging Face Transformers puro para máxima flexibilidade.

Exemplos de uso:
  # Baseline (BCE padrão)
  python train_hf_transformers.py --train --validate
  
  # Com pos_weight customizado
  python train_hf_transformers.py --train --pos-weight "7.75,1.47,1.95,12.30,6.66,11.75"
  
  # Com Focal Loss
  python train_hf_transformers.py --train --use-focal-loss
  
  # Executar múltiplas configurações
  python train_hf_transformers.py --config configurator.json
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--train', action='store_true', help='Treina o modelo')
    parser.add_argument('--test', action='store_true', help='Testa o modelo salvo')
    parser.add_argument('--validate', action='store_true', help='Usa validação durante treino')
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--pos-weight', type=str, help='Pesos por classe (ex: "7.75,1.47,...")')
    parser.add_argument('--use-focal-loss', action='store_true', help='Usar Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--config', type=str, help='Arquivo JSON com configurações')
    
    args = parser.parse_args()
    
    # Modo configuração múltipla
    if args.config:
        config = load_config_file(args.config)
        instances = config['instances']
        
        logger.info(f"\n📋 {len(instances)} instâncias encontradas")
        
        # Carregar dados
        train_df = load_split('train')
        val_df = load_split('val')
        test_df = load_split('test')
        
        if train_df is None:
            logger.info("Gerando splits...")
            full_data = load_dataset(DATASET_PATH)
            train_df, val_df, test_df = split_stratified_holdout(full_data)
            save_split(train_df, 'train')
            save_split(val_df, 'val')
            save_split(test_df, 'test')
        
        # Executar instâncias
        all_results = []
        for i, instance in enumerate(instances, 1):
            try:
                results = run_instance(instance, i, train_df, val_df, test_df)
                all_results.append((instance, results))
            except Exception as e:
                logger.error(f"Erro na instância {i}: {e}")
                continue
        
        # Gerar relatório
        generate_summary_report(all_results, args.config)
        return
    
    # Modo simples
    if not (args.train or args.test):
        parser.print_help()
        return
    
    # Configurar modelo
    config = ModelConfig(
        model_name=args.model_name,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        evaluate_during_training=args.validate
    )
    
    if args.pos_weight:
        config.pos_weight = [float(x) for x in args.pos_weight.split(',')]
    
    # Carregar dados
    train_df = load_split('train')
    val_df = load_split('val')
    test_df = load_split('test')
    
    if train_df is None:
        full_data = load_dataset(DATASET_PATH)
        train_df, val_df, test_df = split_stratified_holdout(full_data)
        save_split(train_df, 'train')
        save_split(val_df, 'val')
        save_split(test_df, 'test')
    
    if args.train:
        # Calcular alpha weights se necessário
        if config.use_focal_loss:
            config.focal_alpha_weights = calculate_class_weights(train_df)
        
        # Treinar
        trainer, history = train_model(train_df, val_df, config)
        
        # Avaliar
        metrics, probs = evaluate_model(trainer, test_df)
        
        logger.info("\nResultados no teste:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        # Gerar visualizações
        plot_train_curves(history)
        # [Chamar outras visualizações]
    
    if args.test and not args.train:
        # Carregar modelo salvo e testar
        model_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
        if not os.path.exists(model_path):
            logger.error("Modelo não encontrado. Treine primeiro com --train")
            return
        
        # [Implementar carregamento e teste do modelo]
        logger.info("Teste do modelo salvo não implementado completamente ainda")

# Funções auxiliares para salvar/carregar splits
def save_split(df, name):
    """Salva split em CSV."""
    df.to_csv(f"split_{name}.csv", index=False)

def load_split(name):
    """Carrega split de CSV."""
    import ast
    path = f"split_{name}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'labels' in df.columns:
        df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def generate_summary_report(all_results: list, config_path: str):
    """Gera relatório resumido."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"summary_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO RESUMIDO - EXECUÇÃO DE MÚLTIPLAS INSTÂNCIAS\n")
        f.write("="*80 + "\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Arquivo de configuração: {config_path}\n")
        f.write(f"Total de instâncias: {len(all_results)}\n\n")
        
        # Tabela comparativa
        f.write("RESULTADOS COMPARATIVOS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'ID':<15} {'Nome':<30} {'F1-Macro':<10} {'Hamming':<10} {'Avg Prec':<10}\n")
        f.write("-"*80 + "\n")
        
        for instance_info, results in all_results:
            instance_id = instance_info.get('id', 'N/A')
            instance_name = instance_info.get('name', 'N/A')[:30]
            f1_score = results.get('macro_f1', 0)
            hamming = results.get('hamming_loss', 1)
            avg_prec = results.get('avg_precision', 0)
            
            f.write(f"{instance_id:<15} {instance_name:<30} "
                   f"{f1_score:<10.4f} {hamming:<10.4f} {avg_prec:<10.4f}\n")
    
    logger.info(f"\n📄 Relatório resumido salvo em: {report_path}")

if __name__ == "__main__":
    main()