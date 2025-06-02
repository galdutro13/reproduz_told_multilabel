# -*- coding: utf-8 -*-
"""
Métricas para avaliação de modelos de classificação multi-label.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    f1_score, hamming_loss, average_precision_score,
    precision_score, recall_score, accuracy_score,
    classification_report, multilabel_confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
from transformers import EvalPrediction
import logging

from src.config import LABELS

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculadora de métricas para classificação multi-label."""
    
    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calcula métricas principais para avaliação durante treinamento.
        
        Args:
            eval_pred: Predições e labels do HuggingFace
            threshold: Limiar para classificação binária
            
        Returns:
            Dict[str, float]: Dicionário com métricas
        """
        predictions, labels = eval_pred
        
        # Aplicar sigmoid e threshold
        predictions = torch.sigmoid(torch.tensor(predictions))
        predictions = (predictions >= threshold).float().numpy()
        
        # Calcular métricas principais
        metrics = {
            'macro_f1': f1_score(labels, predictions, average='macro', zero_division=0),
            'micro_f1': f1_score(labels, predictions, average='micro', zero_division=0),
            'weighted_f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'hamming_loss': hamming_loss(labels, predictions),
            'avg_precision': average_precision_score(labels, predictions, average='macro'),
            'macro_precision': precision_score(labels, predictions, average='macro', zero_division=0),
            'macro_recall': recall_score(labels, predictions, average='macro', zero_division=0)
        }
        
        return metrics

class DetailedMetricsAnalyzer:
    """Analisador detalhado de métricas por classe."""
    
    def __init__(self, labels: List[str] = None):
        """
        Inicializa o analisador.
        
        Args:
            labels: Lista de nomes das classes
        """
        self.labels = labels or LABELS
    
    def calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_probs: np.ndarray = None) -> Dict:
        """
        Calcula métricas detalhadas por classe e globais.
        
        Args:
            y_true: Labels verdadeiros (N, num_classes)
            y_pred: Predições binárias (N, num_classes)
            y_probs: Probabilidades (N, num_classes), opcional
            
        Returns:
            Dict: Métricas detalhadas
        """
        results = {
            'global_metrics': self._calculate_global_metrics(y_true, y_pred, y_probs),
            'per_class_metrics': self._calculate_per_class_metrics(y_true, y_pred, y_probs),
            'confusion_matrices': self._calculate_confusion_matrices(y_true, y_pred)
        }
        
        if y_probs is not None:
            results['curves_data'] = self._calculate_curves_data(y_true, y_probs)
        
        return results
    
    def _calculate_global_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_probs: np.ndarray = None) -> Dict[str, float]:
        """Calcula métricas globais."""
        metrics = {
            # F1 Scores
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            
            # Precision/Recall
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            
            # Outras métricas
            'hamming_loss': hamming_loss(y_true, y_pred),
            'exact_match_ratio': accuracy_score(y_true, y_pred),
        }
        
        if y_probs is not None:
            metrics['avg_precision_macro'] = average_precision_score(y_true, y_probs, average='macro')
            metrics['avg_precision_micro'] = average_precision_score(y_true, y_probs, average='micro')
        
        return metrics
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_probs: np.ndarray = None) -> Dict[str, Dict[str, float]]:
        """Calcula métricas por classe."""
        # Métricas básicas por classe
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class = {}
        for i, label in enumerate(self.labels):
            metrics = {
                'f1': f1_scores[i],
                'precision': precision_scores[i],
                'recall': recall_scores[i],
                'support': y_true[:, i].sum()
            }
            
            if y_probs is not None:
                metrics['avg_precision'] = average_precision_score(y_true[:, i], y_probs[:, i])
                
                # ROC AUC (se houver amostras positivas e negativas)
                if len(np.unique(y_true[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
                    metrics['roc_auc'] = auc(fpr, tpr)
                else:
                    metrics['roc_auc'] = np.nan
            
            per_class[label] = metrics
        
        return per_class
    
    def _calculate_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcula matrizes de confusão por classe."""
        cm_multilabel = multilabel_confusion_matrix(y_true, y_pred)
        
        confusion_matrices = {}
        for i, label in enumerate(self.labels):
            confusion_matrices[label] = cm_multilabel[i]
        
        return confusion_matrices
    
    def _calculate_curves_data(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Dict]:
        """Calcula dados para curvas PR e ROC."""
        curves_data = {}
        
        for i, label in enumerate(self.labels):
            y_true_class = y_true[:, i]
            y_probs_class = y_probs[:, i]
            
            # Curva PR
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                y_true_class, y_probs_class
            )
            
            curves_data[label] = {
                'precision_recall': {
                    'precision': precision_curve,
                    'recall': recall_curve,
                    'thresholds': pr_thresholds
                }
            }
            
            # Curva ROC (se houver amostras positivas e negativas)
            if len(np.unique(y_true_class)) > 1:
                fpr, tpr, roc_thresholds = roc_curve(y_true_class, y_probs_class)
                curves_data[label]['roc'] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': roc_thresholds
                }
        
        return curves_data

class ThresholdOptimizer:
    """Otimizador de limiares para classificação."""
    
    @staticmethod
    def find_optimal_thresholds(y_true: np.ndarray, y_probs: np.ndarray, 
                              metric: str = 'f1', beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encontra limiares ótimos para cada classe.
        
        Args:
            y_true: Labels verdadeiros
            y_probs: Probabilidades preditas
            metric: Métrica para otimizar ('f1', 'precision', 'recall', 'f_beta')
            beta: Parâmetro beta para F-beta score
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Limiares ótimos e scores correspondentes
        """
        num_classes = y_true.shape[1]
        optimal_thresholds = np.zeros(num_classes)
        optimal_scores = np.zeros(num_classes)
        
        for i in range(num_classes):
            thresholds = np.linspace(0.1, 0.9, 50)
            scores = []
            
            for threshold in thresholds:
                y_pred_class = (y_probs[:, i] >= threshold).astype(int)
                
                if metric == 'f1':
                    score = f1_score(y_true[:, i], y_pred_class, zero_division=0)
                elif metric == 'precision':
                    score = precision_score(y_true[:, i], y_pred_class, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_true[:, i], y_pred_class, zero_division=0)
                elif metric == 'f_beta':
                    precision = precision_score(y_true[:, i], y_pred_class, zero_division=0)
                    recall = recall_score(y_true[:, i], y_pred_class, zero_division=0)
                    score = ThresholdOptimizer._f_beta_score(precision, recall, beta)
                else:
                    raise ValueError(f"Métrica '{metric}' não suportada")
                
                scores.append(score)
            
            # Encontrar melhor limiar
            best_idx = np.argmax(scores)
            optimal_thresholds[i] = thresholds[best_idx]
            optimal_scores[i] = scores[best_idx]
        
        return optimal_thresholds, optimal_scores
    
    @staticmethod
    def _f_beta_score(precision: float, recall: float, beta: float) -> float:
        """Calcula F-beta score."""
        if precision + recall == 0:
            return 0.0
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

class MetricsReporter:
    """Gerador de relatórios de métricas."""
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                     labels: List[str] = None) -> str:
        """
        Gera relatório de classificação detalhado.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            labels: Nomes das classes
            
        Returns:
            str: Relatório formatado
        """
        labels = labels or LABELS
        
        report = "=" * 80 + "\n"
        report += "RELATÓRIO DE CLASSIFICAÇÃO MULTI-LABEL\n"
        report += "=" * 80 + "\n\n"
        
        # Métricas globais
        analyzer = DetailedMetricsAnalyzer(labels)
        results = analyzer.calculate_detailed_metrics(y_true, y_pred)
        
        global_metrics = results['global_metrics']
        report += "MÉTRICAS GLOBAIS:\n"
        report += "-" * 40 + "\n"
        for metric, value in global_metrics.items():
            report += f"{metric:20}: {value:.4f}\n"
        
        # Métricas por classe
        per_class = results['per_class_metrics']
        report += "\nMÉTRICAS POR CLASSE:\n"
        report += "-" * 40 + "\n"
        report += f"{'Classe':<15} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Suporte':<8}\n"
        report += "-" * 40 + "\n"
        
        for label in labels:
            metrics = per_class[label]
            report += f"{label:<15} {metrics['f1']:<8.3f} {metrics['precision']:<8.3f} "
            report += f"{metrics['recall']:<8.3f} {int(metrics['support']):<8}\n"
        
        return report
    
    @staticmethod
    def log_metrics_summary(metrics: Dict[str, float], prefix: str = ""):
        """
        Log resumido de métricas.
        
        Args:
            metrics: Dicionário com métricas
            prefix: Prefixo para log
        """
        if prefix:
            logger.info(f"\n📊 {prefix}:")
        else:
            logger.info("\n📊 Métricas:")
        
        # Métricas principais
        main_metrics = ['macro_f1', 'f1_macro', 'avg_precision', 'avg_precision_macro', 'hamming_loss']
        
        for metric in main_metrics:
            if metric in metrics:
                logger.info(f"  {metric}: {metrics[metric]:.4f}")
        
        # Outras métricas se existirem
        other_metrics = {k: v for k, v in metrics.items() if k not in main_metrics}
        if other_metrics:
            logger.info("  Outras métricas:")
            for metric, value in other_metrics.items():
                logger.info(f"    {metric}: {value:.4f}")

# Função auxiliar para compatibility com o código original
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Função de compatibilidade para uso com Trainer do HuggingFace."""
    return MetricsCalculator.compute_metrics(eval_pred)