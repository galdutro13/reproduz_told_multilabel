# -*- coding: utf-8 -*-
"""
Treinamento customizado e callbacks para modelos BERT multi-label.
"""

import torch
import numpy as np
import pandas as pd # Embora não usado diretamente aqui após a refatoração do history, pode ser usado em outros lugares
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from transformers import (
    Trainer, TrainerCallback, TrainerControl, TrainerState,
    EvalPrediction, PreTrainedModel, PreTrainedTokenizer
)
import logging
import math # Para math.ceil

from src.config import ModelConfig, LossConfig # Supondo que NUM_LABELS não é diretamente usado aqui, mas em config
from src.losses import LossFactory
from src.metrics import compute_metrics
from src.data import MultiLabelDataset # Se MultiLabelDataset for usado apenas aqui para type hinting

logger = logging.getLogger(__name__)

class ProgressCallback(TrainerCallback):
    """Callback customizado para mostrar progresso detalhado."""
    
    def __init__(self):
        self.training_bar = None
        self.epoch_bar = None
        self.current_epoch_display = 0 # Para display do número da época
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Inicializa barras de progresso."""
        self.epoch_bar = tqdm(
            total=args.num_train_epochs,
            desc="Épocas",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        self.current_epoch_display = 0
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Início de uma nova época."""
        # state.epoch é float e 0-indexed. Para display, queremos 1-indexed e inteiro.
        self.current_epoch_display = int(math.floor(state.epoch)) + 1
        if self.epoch_bar.n < self.current_epoch_display -1 : # Se pulou épocas no display (raro, mas possível)
            self.epoch_bar.update(self.current_epoch_display -1 - self.epoch_bar.n)
        
        total_steps_in_epoch = 0
        # train_dataloader pode não estar disponível em todas as chamadas ou setups de trainer
        if "train_dataloader" in kwargs and hasattr(kwargs['train_dataloader'], '__len__'):
            total_steps_in_epoch = len(kwargs['train_dataloader'])
        else: # Fallback se train_dataloader não estiver disponível ou não tiver len()
            if state.max_steps > 0 and args.num_train_epochs > 0: # Se max_steps foi calculado
                 total_steps_in_epoch = math.ceil(state.max_steps / args.num_train_epochs)


        self.training_bar = tqdm(
            total=total_steps_in_epoch if total_steps_in_epoch > 0 else None, # None para barra indeterminada se não souber o total
            desc=f"Época {self.current_epoch_display}/{int(args.num_train_epochs)}",
            position=1,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        
    def on_step_end(self, args, state, control, **kwargs):
        """Atualiza barra de progresso após cada step."""
        if self.training_bar is not None:
            current_log = state.log_history[-1] if state.log_history else {}
            loss = current_log.get('loss')
            lr = current_log.get('learning_rate')
            
            postfix_str = []
            if loss is not None:
                postfix_str.append(f'loss={loss:.4f}')
            if lr is not None:
                postfix_str.append(f'lr={lr:.2e}')
            
            self.training_bar.set_postfix_str(", ".join(postfix_str))
            self.training_bar.update(1)
            
    def on_epoch_end(self, args, state, control, **kwargs):
        """Finaliza época."""
        if self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None # Resetar para a próxima época
        
        # Atualizar a barra de épocas principal
        if self.epoch_bar.n < self.current_epoch_display:
             self.epoch_bar.update(1)

        # Mostrar métricas de validação se disponíveis
        # Encontrar o log de avaliação mais recente para esta época (que acabou de terminar)
        # state.epoch será o float correspondente à época concluída (e.g., 1.0 após a primeira época)
        completed_epoch_float = math.floor(state.epoch) # A época que acabou (0-indexed)
        
        eval_metrics_for_epoch = {}
        for log in reversed(state.log_history):
            # O log de avaliação é registrado com a época correspondente.
            # Ex: após a época 0 (primeira época), log.get('epoch') pode ser ~1.0.
            # Usamos math.isclose para comparar floats.
            if 'eval_loss' in log and log.get('epoch') is not None and \
               math.isclose(log.get('epoch'), completed_epoch_float + 1.0): # O log de epoch é 1-indexed no final da época
                eval_metrics_for_epoch = log
                break
        
        if eval_metrics_for_epoch:
            logger.info(f"\n📊 Métricas Época {self.current_epoch_display}: "
                       f"eval_loss={eval_metrics_for_epoch.get('eval_loss', float('nan')):.4f}, "
                       f"macro_f1={eval_metrics_for_epoch.get('eval_macro_f1', float('nan')):.4f}, "
                       f"avg_precision={eval_metrics_for_epoch.get('eval_avg_precision', float('nan')):.4f}")
            
    def on_train_end(self, args, state, control, **kwargs):
        """Finaliza treinamento."""
        if self.training_bar is not None: # Fechar a barra de steps da última época, se ainda aberta
            self.training_bar.close()
            self.training_bar = None
        if self.epoch_bar is not None:
            # Garantir que a barra de épocas chegue ao final se o treinamento terminar antes (ex: early stopping)
            if self.epoch_bar.n < self.epoch_bar.total:
                 self.epoch_bar.update(self.epoch_bar.total - self.epoch_bar.n)
            self.epoch_bar.close()
            self.epoch_bar = None
        logger.info("\n✅ Treinamento concluído!")

class EarlyStoppingCallback(TrainerCallback):
    """Callback para early stopping baseado em métrica de validação."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 metric: str = "eval_avg_precision", mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_to_monitor = metric 
        self.mode = mode
        self.best_metric_value = None 
        self.patience_counter = 0
        self.stopped_epoch = 0 # Para logar a época em que parou

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        if not logs:
            return
        
        current_metric_value = logs.get(self.metric_to_monitor)
        if current_metric_value is None:
            # O Trainer pode não incluir a métrica se ela for NaN ou não calculada
            logger.warning(f"⚠️ Métrica de early stopping '{self.metric_to_monitor}' não encontrada nos logs de avaliação atuais (step {state.global_step}). Verifique se está sendo calculada e logada.")
            return
        
        logger.debug(f"EarlyStopping: current_metric ({self.metric_to_monitor}) = {current_metric_value:.4f} at step {state.global_step}")

        if self.best_metric_value is None:
            self.best_metric_value = current_metric_value
            self.patience_counter = 0
            logger.info(f"🎯 EarlyStopping: Métrica inicial '{self.metric_to_monitor}' = {self.best_metric_value:.4f} (step {state.global_step}).")
            return
        
        improved = False
        delta = abs(current_metric_value - self.best_metric_value)

        if self.mode == "max":
            if current_metric_value > self.best_metric_value:
                if delta >= self.min_delta:
                    improved = True
                else: # Melhorou, mas não o suficiente (abaixo de min_delta)
                    logger.info(f"🔎 EarlyStopping: Melhoria em '{self.metric_to_monitor}' ({current_metric_value:.4f} vs {self.best_metric_value:.4f}) não excede min_delta ({self.min_delta}). Contando como paciência.")
            
        else: # mode == "min"
            if current_metric_value < self.best_metric_value:
                if delta >= self.min_delta:
                    improved = True
                else: # Melhorou, mas não o suficiente
                    logger.info(f"🔎 EarlyStopping: Melhoria em '{self.metric_to_monitor}' ({current_metric_value:.4f} vs {self.best_metric_value:.4f}) não excede min_delta ({self.min_delta}). Contando como paciência.")

        if improved:
            logger.info(f"🎯 EarlyStopping: Nova melhor métrica '{self.metric_to_monitor}': {current_metric_value:.4f} (anterior: {self.best_metric_value:.4f}, step {state.global_step}). Paciência resetada.")
            self.best_metric_value = current_metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            logger.info(f"⏳ EarlyStopping: Sem melhoria significativa em '{self.metric_to_monitor}' há {self.patience_counter}/{self.patience} avaliações. "
                        f"Atual: {current_metric_value:.4f}, Melhor: {self.best_metric_value:.4f} (step {state.global_step}).")
            
            if self.patience_counter >= self.patience:
                self.stopped_epoch = state.epoch
                logger.info(f"🛑 Early stopping ativado na época {self.stopped_epoch:.2f} (step {state.global_step})! "
                            f"Melhor '{self.metric_to_monitor}': {self.best_metric_value:.4f}.")
                control.should_training_stop = True

class MultiLabelTrainer(Trainer):
    """Trainer customizado para multi-label com suporte a diferentes loss functions."""
    
    def __init__(self, loss_config: Optional[LossConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config if loss_config is not None else LossConfig()
        self.loss_config.validate() 
        self._setup_loss_function()
        
    def _setup_loss_function(self):
        focal_alpha_weights_list = None
        if self.loss_config.focal_alpha_weights is not None:
            focal_alpha_weights_list = torch.tensor(self.loss_config.focal_alpha_weights, device=self.args.device) \
                if isinstance(self.loss_config.focal_alpha_weights, list) \
                else self.loss_config.focal_alpha_weights 
        
        pos_weight_tensor = None
        if self.loss_config.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.loss_config.pos_weight, device=self.args.device) \
                if isinstance(self.loss_config.pos_weight, list) \
                else self.loss_config.pos_weight

        loss_params = { 
            'use_focal_loss': self.loss_config.use_focal_loss,
            'focal_gamma': self.loss_config.focal_gamma,
            'focal_alpha_weights': focal_alpha_weights_list, # Deve ser tensor ou None
            'pos_weight': pos_weight_tensor # Deve ser tensor ou None
        }
        self.loss_fct = LossFactory.create_loss_function(loss_params)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.loss_fct(logits, labels.float()) # labels para float()
        return (loss, outputs) if return_outputs else loss

class TrainingManager:
    """Gerenciador de treinamento com funcionalidades avançadas."""
    
    def __init__(self, config: ModelConfig, loss_config: Optional[LossConfig] = None):
        self.config = config
        self.loss_config = loss_config if loss_config is not None else LossConfig()
        self.trainer: Optional[MultiLabelTrainer] = None
        self.training_history: Dict[str, List[Any]] = {}
        
    def setup_trainer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                     train_dataset: MultiLabelDataset, eval_dataset: Optional[MultiLabelDataset] = None) -> MultiLabelTrainer:
        training_args = self.config.to_training_args()
        callbacks: List[TrainerCallback] = [ProgressCallback()]
        
        if eval_dataset is not None and self.config.evaluate_during_training:
            metric_name_for_early_stopping = self.config.metric_for_best_model
            if not metric_name_for_early_stopping.startswith("eval_"):
                 metric_name_for_early_stopping = f"eval_{self.config.metric_for_best_model}"
            early_stopping = EarlyStoppingCallback(
                patience=self.config.early_stopping_patience if hasattr(self.config, 'early_stopping_patience') else 3, 
                min_delta=self.config.early_stopping_threshold if hasattr(self.config, 'early_stopping_threshold') else 0.001,
                metric=metric_name_for_early_stopping,
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
    
    def train(self) -> Tuple[Any, Dict[str, List[Any]]]:
        if self.trainer is None:
            raise ValueError("Trainer não foi configurado. Chame setup_trainer() primeiro.")
        logger.info("\n🚀 Iniciando treinamento...")
        self._log_training_info()
        train_result = self.trainer.train()
        logger.info("\n💾 Salvando modelo final e estado do trainer...")
        self.trainer.save_model() 
        self.trainer.save_state()
        self.training_history = self._extract_training_history()
        self._log_training_summary(train_result)
        return train_result, self.training_history
    
    def evaluate(self, test_dataset: MultiLabelDataset) -> Tuple[Dict[str, float], np.ndarray]:
        if self.trainer is None:
            raise ValueError("Trainer não foi configurado.")
        logger.info("\n🧪 Avaliando modelo no conjunto de teste...")
        logger.info(f"📊 Processando {len(test_dataset)} amostras...")
        predictions_output = self.trainer.predict(test_dataset, metric_key_prefix="test")
        
        logits = predictions_output.predictions
        if logits is None:
            raise ValueError("Predições não retornaram logits.")
        probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy() # Especificar dtype
        
        test_metrics = predictions_output.metrics if predictions_output.metrics is not None else {}
        logger.info("📊 Métricas de Teste:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        return test_metrics, probs
    
    def _log_training_info(self):
        if self.trainer and self.trainer.train_dataset:
            train_size = len(self.trainer.train_dataset)
            eval_size = len(self.trainer.eval_dataset) if self.trainer.eval_dataset else 0
            logger.info(f"📊 Dados: {train_size} amostras de treino, {eval_size} amostras de validação")
        
        logger.info(f"⚙️  Configuração: {self.config.num_train_epochs} épocas, "
                   f"batch_size={self.config.per_device_train_batch_size}, "
                   f"lr={self.config.learning_rate}")
        if self.trainer: # Trainer deve estar inicializado para ter state.max_steps
            logger.info(f"📈 Total de steps de treinamento: {self.trainer.state.max_steps}")
        logger.info(f"📊 Avaliação a cada {self.config.eval_steps} steps")
        logger.info(f"💾 Salvamento a cada {self.config.save_steps} steps (limite: {self.config.save_total_limit})")
        logger.info(f"🎯 Melhor modelo será salvo baseado em: {self.config.metric_for_best_model} ({'maior' if self.config.greater_is_better else 'menor'} é melhor)")
        logger.info("\n" + "="*80 + "\n")
    
    def _extract_training_history(self) -> Dict[str, List[Any]]:
        """Extrai histórico de treinamento dos logs de forma mais robusta."""
        history: Dict[str, List[Any]] = {
            'train_steps': [], 'train_loss': [], 'train_learning_rate': [],
            'eval_steps': [], 'eval_loss': [],
            'eval_macro_f1': [], 'eval_micro_f1': [], 'eval_weighted_f1': [],
            'eval_hamming_loss': [], 'eval_avg_precision': [],
            'eval_macro_precision': [], 'eval_macro_recall': [],
            # Chaves legadas/compatibilidade
            'global_step': [], 'loss': [], 'learning_rate': [], # Para treino
            'macro_f1': [], 'hamming_loss': [], 'avg_precision': [] # Para eval
        }

        if self.trainer is None or not hasattr(self.trainer, 'state') or not self.trainer.state.log_history:
            logger.warning("Trainer ou log_history não inicializado, histórico de treinamento estará vazio.")
            return history

        for log_entry in self.trainer.state.log_history:
            step = log_entry.get('step')
            epoch = log_entry.get('epoch') # Pode ser float

            # Logs de treinamento (contêm 'loss', podem conter 'learning_rate', mas não 'eval_loss')
            is_train_log = 'loss' in log_entry and 'eval_loss' not in log_entry
            if is_train_log and step is not None:
                history['train_steps'].append(step)
                history['train_loss'].append(log_entry['loss'])
                if 'learning_rate' in log_entry:
                    history['train_learning_rate'].append(log_entry['learning_rate'])
                else: # Adicionar NaN se não houver learning_rate neste log de treino
                    if history['train_learning_rate'] and len(history['train_learning_rate']) < len(history['train_steps']):
                        history['train_learning_rate'].append(float('nan'))


            # Logs de avaliação (contêm 'eval_loss' e outras métricas 'eval_*')
            is_eval_log = 'eval_loss' in log_entry
            if is_eval_log and step is not None: # Os logs de avaliação também têm 'step'
                history['eval_steps'].append(step)
                history['eval_loss'].append(log_entry['eval_loss'])
                history['eval_macro_f1'].append(log_entry.get('eval_macro_f1', float('nan')))
                history['eval_micro_f1'].append(log_entry.get('eval_micro_f1', float('nan')))
                history['eval_weighted_f1'].append(log_entry.get('eval_weighted_f1', float('nan')))
                history['eval_hamming_loss'].append(log_entry.get('eval_hamming_loss', float('nan')))
                history['eval_avg_precision'].append(log_entry.get('eval_avg_precision', float('nan')))
                history['eval_macro_precision'].append(log_entry.get('eval_macro_precision', float('nan')))
                history['eval_macro_recall'].append(log_entry.get('eval_macro_recall', float('nan')))
        
        # Garantir que todas as listas de 'train' tenham o mesmo tamanho de 'train_steps'
        target_train_len = len(history['train_steps'])
        if len(history['train_learning_rate']) < target_train_len:
            history['train_learning_rate'].extend([float('nan')] * (target_train_len - len(history['train_learning_rate'])))

        # Preencher chaves legadas para compatibilidade
        history['global_step'] = list(history['train_steps'])
        history['loss'] = list(history['train_loss']) # 'loss' é ambiguo, mas mantendo por compatibilidade com 'train_loss'
        history['learning_rate'] = list(history['train_learning_rate'])
        
        history['macro_f1'] = list(history['eval_macro_f1'])
        history['hamming_loss'] = list(history['eval_hamming_loss'])
        history['avg_precision'] = list(history['eval_avg_precision'])
            
        return history

    def _log_training_summary(self, train_result):
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMO DO TREINAMENTO")
        logger.info("="*80)
        
        train_metrics = train_result.metrics
        if train_metrics:
            logger.info(f"✅ Treinamento concluído em {train_metrics.get('train_runtime', 0):.2f} segundos ({train_metrics.get('train_samples_per_second', 0):.2f} amostras/seg)")
            logger.info(f"📈 Loss final de treino (época): {train_metrics.get('train_loss', float('nan')):.4f}") # Métrica de fim de época
            # Para loss média de treino de todos os steps, pode-se calcular a partir de history['train_loss']
            if self.training_history.get('train_loss'):
                avg_train_loss_all_steps = np.nanmean([x for x in self.training_history['train_loss'] if isinstance(x, (int, float))])
                logger.info(f"📈 Loss média de treino (todos os steps): {avg_train_loss_all_steps:.4f}")
        else:
            logger.warning("Metrics do resultado do treino não encontradas.")

        # Usar as chaves 'eval_*' do history para o resumo das métricas de validação
        if self.training_history.get('eval_loss') and any(not np.isnan(x) for x in self.training_history['eval_loss']):
            valid_eval_loss = [x for x in self.training_history['eval_loss'] if not np.isnan(x)]
            valid_eval_macro_f1 = [x for x in self.training_history['eval_macro_f1'] if not np.isnan(x)]
            valid_eval_avg_precision = [x for x in self.training_history['eval_avg_precision'] if not np.isnan(x)]

            if valid_eval_loss: logger.info(f"📊 Melhor loss de validação: {min(valid_eval_loss):.4f}")
            if valid_eval_macro_f1: logger.info(f"🎯 Melhor F1-Macro de validação: {max(valid_eval_macro_f1):.4f}")
            if valid_eval_avg_precision: logger.info(f"🎯 Melhor Avg Precision de validação: {max(valid_eval_avg_precision):.4f}")
        else:
            logger.info("ℹ️ Nenhuma métrica de validação registrada ou todas são NaN no histórico para resumo.")


class ModelCheckpointer:
    # ... (ModelCheckpointer permanece o mesmo) ...
    @staticmethod
    def save_checkpoint(trainer: Trainer, checkpoint_dir: str, metadata: Optional[Dict] = None): # metadata pode ser None
        import os
        import json
        from datetime import datetime
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainer.save_model(checkpoint_dir)
        trainer.save_state()
        
        if metadata:
            metadata_to_save = metadata.copy()
            metadata_to_save['saved_at'] = datetime.now().isoformat()
            metadata_path = os.path.join(checkpoint_dir, "training_metadata.json") 
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Checkpoint salvo em: {checkpoint_dir}")
    
    @staticmethod
    def load_checkpoint(checkpoint_dir: str, model_class=None, tokenizer_class=None):
        import os
        import json
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Diretório do checkpoint não encontrado: {checkpoint_dir}")
        
        model_class_to_use = model_class if model_class is not None else AutoModelForSequenceClassification
        model = model_class_to_use.from_pretrained(checkpoint_dir)
        
        tokenizer_class_to_use = tokenizer_class if tokenizer_class is not None else AutoTokenizer
        tokenizer = tokenizer_class_to_use.from_pretrained(checkpoint_dir)
        
        metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
        loaded_metadata = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    loaded_metadata = json.load(f)
                logger.info(f"📄 Metadados de treinamento carregados de: {metadata_path}")
            except Exception as e: # Captura mais genérica para erros de load/decode
                logger.warning(f"⚠️ Erro ao carregar ou decodificar JSON de metadados em: {metadata_path} - {e}")
        logger.info(f"📂 Checkpoint carregado de: {checkpoint_dir}")
        return model, tokenizer, loaded_metadata