# -*- coding: utf-8 -*-
"""
Treinamento customizado e callbacks para modelos BERT multi-label.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from transformers import (
    Trainer, TrainerCallback, TrainerControl, TrainerState,
    EvalPrediction, PreTrainedModel, PreTrainedTokenizer
)
import logging

from src.config import ModelConfig, LossConfig
from src.losses import LossFactory
from src.metrics import compute_metrics
from src.data import MultiLabelDataset

logger = logging.getLogger(__name__)

class ProgressCallback(TrainerCallback):
    """Callback customizado para mostrar progresso detalhado."""
    
    def __init__(self):
        self.training_bar = None
        self.epoch_bar = None
        self.current_epoch = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Inicializa barras de progresso."""
        self.epoch_bar = tqdm(
            total=args.num_train_epochs,
            desc="Épocas",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        self.current_epoch = 0
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Início de uma nova época."""
        if state.epoch is not None:
            self.current_epoch = int(state.epoch)
            self.epoch_bar.update(self.current_epoch - self.epoch_bar.n)
            
        # Criar barra para batches
        total_steps = len(kwargs.get('train_dataloader', []))
        self.training_bar = tqdm(
            total=total_steps,
            desc=f"Época {self.current_epoch + 1}/{args.num_train_epochs}",
            position=1,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
    def on_step_end(self, args, state, control, **kwargs):
        """Atualiza barra de progresso após cada step."""
        if self.training_bar is not None:
            # Atualizar com métricas
            current_loss = state.log_history[-1].get('loss', 0) if state.log_history else 0
            lr = state.log_history[-1].get('learning_rate', 0) if state.log_history else 0
            
            self.training_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{lr:.2e}'
            })
            self.training_bar.update(1)
            
    def on_epoch_end(self, args, state, control, **kwargs):
        """Finaliza época."""
        if self.training_bar is not None:
            self.training_bar.close()
            
        # Mostrar métricas de validação se disponíveis
        if state.log_history and 'eval_loss' in state.log_history[-1]:
            eval_metrics = state.log_history[-1]
            logger.info(f"\n📊 Métricas Época {self.current_epoch + 1}: "
                       f"eval_loss={eval_metrics.get('eval_loss', 0):.4f}, "
                       f"macro_f1={eval_metrics.get('eval_macro_f1', 0):.4f}, "
                       f"avg_precision={eval_metrics.get('eval_avg_precision', 0):.4f}")
            
    def on_train_end(self, args, state, control, **kwargs):
        """Finaliza treinamento."""
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        logger.info("\n✅ Treinamento concluído!")

class EarlyStoppingCallback(TrainerCallback):
    """Callback para early stopping baseado em métrica de validação."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 metric: str = "eval_avg_precision", mode: str = "max"):
        """
        Inicializa early stopping.
        
        Args:
            patience: Número de épocas sem melhoria antes de parar
            min_delta: Melhoria mínima considerada significativa
            metric: Métrica para monitorar
            mode: 'max' para maximizar, 'min' para minimizar
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_metric = None
        self.patience_counter = 0
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Verifica se deve aplicar early stopping."""
        if not kwargs.get('logs'):
            return
        
        current_metric = kwargs['logs'].get(self.metric)
        if current_metric is None:
            return
        
        if self.best_metric is None:
            self.best_metric = current_metric
            self.patience_counter = 0
            return
        
        # Verificar melhoria
        if self.mode == "max":
            improved = current_metric > (self.best_metric + self.min_delta)
        else:
            improved = current_metric < (self.best_metric - self.min_delta)
        
        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            logger.info(f"🎯 Nova melhor métrica {self.metric}: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"⏳ Sem melhoria há {self.patience_counter}/{self.patience} épocas")
            
            if self.patience_counter >= self.patience:
                logger.info(f"🛑 Early stopping ativado! Melhor {self.metric}: {self.best_metric:.4f}")
                control.should_training_stop = True

class MultiLabelTrainer(Trainer):
    """Trainer customizado para multi-label com suporte a diferentes loss functions."""
    
    def __init__(self, loss_config: Optional[LossConfig] = None, *args, **kwargs):
        """
        Inicializa trainer customizado.
        
        Args:
            loss_config: Configuração da loss function
        """
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config or LossConfig()
        self.loss_config.validate()
        self._setup_loss_function()
        
    def _setup_loss_function(self):
        """Configura a loss function baseada em loss_config."""
        loss_config_dict = {
            'use_focal_loss': self.loss_config.use_focal_loss,
            'focal_gamma': self.loss_config.focal_gamma,
            'focal_alpha_weights': self.loss_config.focal_alpha_weights,
            'pos_weight': self.loss_config.pos_weight
        }
        
        self.loss_fct = LossFactory.create_loss_function(loss_config_dict)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Calcula a loss usando a função configurada."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Aplicar loss function customizada
        loss = self.loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

class TrainingManager:
    """Gerenciador de treinamento com funcionalidades avançadas."""
    
    def __init__(self, config: ModelConfig, loss_config: LossConfig = None):
        """
        Inicializa gerenciador de treinamento.
        
        Args:
            config: Configuração do modelo
            loss_config: Configuração da loss function
        """
        self.config = config
        self.loss_config = loss_config or LossConfig()
        self.trainer = None
        self.training_history = {}
        
    def setup_trainer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                     train_dataset: MultiLabelDataset, eval_dataset: MultiLabelDataset = None) -> MultiLabelTrainer:
        """
        Configura o trainer.
        
        Args:
            model: Modelo para treinar
            tokenizer: Tokenizer
            train_dataset: Dataset de treino
            eval_dataset: Dataset de validação (opcional)
            
        Returns:
            MultiLabelTrainer: Trainer configurado
        """
        training_args = self.config.to_training_args()
        
        # Configurar callbacks
        callbacks = [ProgressCallback()]
        
        # Early stopping se validação estiver habilitada
        if eval_dataset is not None and self.config.evaluate_during_training:
            early_stopping = EarlyStoppingCallback(
                patience=3,
                metric=f"eval_{self.config.metric_for_best_model}",
                mode="max" if self.config.greater_is_better else "min"
            )
            callbacks.append(early_stopping)
        
        self.trainer = MultiLabelTrainer(
            loss_config=self.loss_config,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=callbacks
        )
        
        return self.trainer
    
    def train(self) -> Tuple[Any, Dict[str, List[float]]]:
        """
        Executa o treinamento.
        
        Returns:
            Tuple: Resultado do treinamento e histórico
        """
        if self.trainer is None:
            raise ValueError("Trainer não foi configurado. Chame setup_trainer() primeiro.")
        
        logger.info("\n🚀 Iniciando treinamento...")
        self._log_training_info()
        
        # Treinar
        train_result = self.trainer.train()
        
        # Salvar modelo
        logger.info("\n💾 Salvando modelo...")
        self.trainer.save_model()
        self.trainer.save_state()
        
        # Extrair histórico
        self.training_history = self._extract_training_history()
        
        # Log resumo
        self._log_training_summary(train_result)
        
        return train_result, self.training_history
    
    def evaluate(self, test_dataset: MultiLabelDataset) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Avalia o modelo no conjunto de teste.
        
        Args:
            test_dataset: Dataset de teste
            
        Returns:
            Tuple: Métricas e probabilidades
        """
        if self.trainer is None:
            raise ValueError("Trainer não foi configurado.")
        
        logger.info("\n🧪 Avaliando modelo no conjunto de teste...")
        logger.info(f"📊 Processando {len(test_dataset)} amostras...")
        
        # Fazer predições
        predictions = self.trainer.predict(
            test_dataset,
            metric_key_prefix="test"
        )
        
        # Extrair logits e calcular probabilidades
        logits = predictions.predictions
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        
        # Calcular métricas
        metrics = compute_metrics(EvalPrediction(predictions=logits, label_ids=predictions.label_ids))
        
        # Adicionar prefixo 'test_' às métricas
        test_metrics = {f"test_{k}": v for k, v in metrics.items()}
        
        return test_metrics, probs
    
    def _log_training_info(self):
        """Log informações sobre o treinamento."""
        if hasattr(self.trainer, 'train_dataset'):
            train_size = len(self.trainer.train_dataset)
            eval_size = len(self.trainer.eval_dataset) if self.trainer.eval_dataset else 0
            
            logger.info(f"📊 Dados: {train_size} amostras de treino, {eval_size} amostras de validação")
        
        logger.info(f"⚙️  Configuração: {self.config.num_train_epochs} épocas, "
                   f"batch_size={self.config.per_device_train_batch_size}, "
                   f"lr={self.config.learning_rate}")
        
        # Calcular steps totais
        if hasattr(self.trainer, 'train_dataset'):
            total_steps = (len(self.trainer.train_dataset) // 
                          self.config.per_device_train_batch_size * 
                          self.config.num_train_epochs)
            logger.info(f"📈 Total de steps: {total_steps}")
            logger.info(f"📊 Avaliação a cada {self.config.eval_steps} steps")
            logger.info(f"💾 Salvamento a cada {self.config.save_steps} steps")
        
        logger.info("\n" + "="*80 + "\n")
    
    def _extract_training_history(self) -> Dict[str, List[float]]:
        """Extrai histórico de treinamento dos logs."""
        history = {
            'global_step': [],
            'train_loss': [],
            'eval_loss': [],
            'macro_f1': [],
            'hamming_loss': [],
            'avg_precision': []
        }
        
        for log in self.trainer.state.log_history:
            if 'loss' in log:
                history['global_step'].append(log.get('step', 0))
                history['train_loss'].append(log['loss'])
            if 'eval_loss' in log:
                # Garantir que temos o mesmo número de steps
                if len(history['eval_loss']) < len(history['global_step']):
                    history['eval_loss'].append(log['eval_loss'])
                    history['macro_f1'].append(log.get('eval_macro_f1', 0))
                    history['hamming_loss'].append(log.get('eval_hamming_loss', 0))
                    history['avg_precision'].append(log.get('eval_avg_precision', 0))
        
        return history
    
    def _log_training_summary(self, train_result):
        """Log resumo do treinamento."""
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMO DO TREINAMENTO")
        logger.info("="*80)
        logger.info(f"✅ Treinamento concluído em {train_result.metrics['train_runtime']:.2f} segundos")
        logger.info(f"📈 Loss final de treino: {train_result.metrics['train_loss']:.4f}")
        
        if self.training_history['eval_loss']:
            logger.info(f"📊 Melhor loss de validação: {min(self.training_history['eval_loss']):.4f}")
            logger.info(f"🎯 Melhor F1-Macro: {max(self.training_history['macro_f1']):.4f}")
            logger.info(f"🎯 Melhor Avg Precision: {max(self.training_history['avg_precision']):.4f}")

class ModelCheckpointer:
    """Gerenciador de checkpoints do modelo."""
    
    @staticmethod
    def save_checkpoint(trainer: Trainer, checkpoint_dir: str, metadata: Dict = None):
        """
        Salva checkpoint com metadados.
        
        Args:
            trainer: Trainer com modelo
            checkpoint_dir: Diretório para salvar
            metadata: Metadados adicionais
        """
        import os
        import json
        from datetime import datetime
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Salvar modelo
        trainer.save_model(checkpoint_dir)
        trainer.save_state()
        
        # Salvar metadados
        if metadata:
            metadata_path = os.path.join(checkpoint_dir, "metadata.json")
            metadata['saved_at'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Checkpoint salvo em: {checkpoint_dir}")
    
    @staticmethod
    def load_checkpoint(checkpoint_dir: str, model_class=None, tokenizer_class=None):
        """
        Carrega checkpoint.
        
        Args:
            checkpoint_dir: Diretório do checkpoint
            model_class: Classe do modelo (opcional)
            tokenizer_class: Classe do tokenizer (opcional)
            
        Returns:
            Tuple: Modelo, tokenizer, metadados
        """
        import os
        import json
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_dir}")
        
        # Carregar modelo
        model_class = model_class or AutoModelForSequenceClassification
        model = model_class.from_pretrained(checkpoint_dir)
        
        # Carregar tokenizer
        tokenizer_class = tokenizer_class or AutoTokenizer
        tokenizer = tokenizer_class.from_pretrained(checkpoint_dir)
        
        # Carregar metadados se existirem
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"📂 Checkpoint carregado de: {checkpoint_dir}")
        
        return model, tokenizer, metadata