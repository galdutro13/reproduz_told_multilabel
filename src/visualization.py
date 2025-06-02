# -*- coding: utf-8 -*-
"""
Visualizações e gráficos para análise de modelos multi-label.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from src.config import LABELS

logger = logging.getLogger(__name__)

# Configurações de estilo
plt.style.use('default')
sns.set_palette("husl")

class TrainingVisualizer:
    """Visualizador para curvas de treinamento."""
    
    @staticmethod
    def plot_training_curves(training_history: Dict[str, List[float]], 
                           save_path: str = "training_curves.png"):
        """
        Plota curvas de treinamento (loss e métricas).
        
        Args:
            training_history: Histórico de treinamento
            save_path: Caminho para salvar o gráfico
        """
        df = pd.DataFrame(training_history)
        
        if 'eval_loss' not in df.columns or 'macro_f1' not in df.columns:
            logger.warning("Histórico não possui dados de validação")
            TrainingVisualizer._plot_training_loss_only(df, save_path)
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Losses
        ax1.plot(df['global_step'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(df['global_step'], df['eval_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Global Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Curvas de Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Métricas
        ax2.plot(df['global_step'], df['macro_f1'], 'g-', label='Macro F1', linewidth=2)
        ax2.plot(df['global_step'], df['avg_precision'], 'm-', label='Avg Precision', linewidth=2)
        ax2.set_xlabel('Global Step')
        ax2.set_ylabel('Score')
        ax2.set_title('Métricas de Validação')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Curvas de treinamento salvas em: {save_path}")
    
    @staticmethod
    def _plot_training_loss_only(df: pd.DataFrame, save_path: str):
        """Plota apenas loss de treinamento quando não há validação."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['global_step'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Loss')
        ax.set_title('Curva de Loss de Treinamento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class MetricsVisualizer:
    """Visualizador para métricas de classificação."""
    
    @staticmethod
    def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, 
                              labels: List[str] = None, save_path: str = "confusion_matrices.png"):
        """
        Plota matrizes de confusão para cada classe.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            labels: Nomes das classes
            save_path: Caminho para salvar
        """
        from sklearn.metrics import multilabel_confusion_matrix
        
        labels = labels or LABELS
        cm_multilabel = multilabel_confusion_matrix(y_true, y_pred)
        
        # Calcular layout da grid
        n_classes = len(labels)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_classes == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, label in enumerate(labels):
            cm = cm_multilabel[i]
            
            # Plotar matriz de confusão
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            axes[i].set_title(f'{label}\n(TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]})')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        # Esconder axes vazios
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Matrizes de confusão salvas em: {save_path}")
    
    @staticmethod
    def plot_metrics_per_class(per_class_metrics: Dict[str, Dict[str, float]], 
                             save_path: str = "metrics_per_class.png"):
        """
        Plota métricas por classe em barplot.
        
        Args:
            per_class_metrics: Métricas por classe
            save_path: Caminho para salvar
        """
        # Preparar dados
        classes = list(per_class_metrics.keys())
        metrics_names = ['f1', 'precision', 'recall']
        
        data = []
        for class_name in classes:
            for metric in metrics_names:
                if metric in per_class_metrics[class_name]:
                    data.append({
                        'Class': class_name,
                        'Metric': metric.capitalize(),
                        'Score': per_class_metrics[class_name][metric]
                    })
        
        df_metrics = pd.DataFrame(data)
        
        # Plotar
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.barplot(data=df_metrics, x='Class', y='Score', hue='Metric', ax=ax)
        ax.set_title('Métricas por Classe')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.legend(title='Métrica')
        plt.xticks(rotation=45, ha='right')
        
        # Adicionar valores no topo das barras
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', size=8)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Métricas por classe salvas em: {save_path}")
    
    @staticmethod
    def plot_class_distribution(y_true: np.ndarray, labels: List[str] = None, 
                              save_path: str = "class_distribution.png"):
        """
        Plota distribuição de classes no dataset.
        
        Args:
            y_true: Labels verdadeiros
            labels: Nomes das classes
            save_path: Caminho para salvar
        """
        labels = labels or LABELS
        
        # Calcular contagens
        class_counts = y_true.sum(axis=0)
        total_samples = len(y_true)
        percentages = (class_counts / total_samples) * 100
        
        # Plotar
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico de barras
        bars = ax1.bar(labels, class_counts, color='skyblue', alpha=0.7)
        ax1.set_title('Distribuição de Classes (Contagem)')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Número de Amostras')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Adicionar valores nas barras
        for bar, count in zip(bars, class_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_samples*0.01,
                    f'{int(count)}', ha='center', va='bottom')
        
        # Gráfico de pizza
        wedges, texts, autotexts = ax2.pie(class_counts, labels=labels, autopct='%1.1f%%',
                                          startangle=90)
        ax2.set_title('Distribuição de Classes (Percentual)')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Distribuição de classes salva em: {save_path}")

class CurvesVisualizer:
    """Visualizador para curvas PR e ROC."""
    
    @staticmethod
    def plot_precision_recall_curves(y_true: np.ndarray, y_probs: np.ndarray, 
                                    labels: List[str] = None, 
                                    save_path: str = "precision_recall_curves.png"):
        """
        Plota curvas Precision-Recall para cada classe.
        
        Args:
            y_true: Labels verdadeiros
            y_probs: Probabilidades preditas
            labels: Nomes das classes
            save_path: Caminho para salvar
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        labels = labels or LABELS
        
        # Calcular layout da grid
        n_classes = len(labels)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_classes == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, label in enumerate(labels):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            avg_precision = average_precision_score(y_true[:, i], y_probs[:, i])
            
            axes[i].plot(recall, precision, linewidth=2, 
                        label=f'AP = {avg_precision:.3f}')
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_title(f'Curva PR - {label}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])
        
        # Esconder axes vazios
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Curvas PR salvas em: {save_path}")
    
    @staticmethod
    def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, 
                       labels: List[str] = None, save_path: str = "roc_curves.png"):
        """
        Plota curvas ROC para cada classe.
        
        Args:
            y_true: Labels verdadeiros
            y_probs: Probabilidades preditas
            labels: Nomes das classes
            save_path: Caminho para salvar
        """
        from sklearn.metrics import roc_curve, auc
        
        labels = labels or LABELS
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, label in enumerate(labels):
            # Verificar se há amostras positivas e negativas
            if len(np.unique(y_true[:, i])) <= 1:
                logger.warning(f"Classe '{label}' só tem uma classe, pulando curva ROC")
                continue
            
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'{label} (AUC = {roc_auc:.3f})')
        
        # Linha diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Curvas ROC por Classe')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Curvas ROC salvas em: {save_path}")

class ThresholdVisualizer:
    """Visualizador para análise de limiares."""
    
    @staticmethod
    def plot_threshold_analysis(y_true: np.ndarray, y_probs: np.ndarray, 
                              labels: List[str] = None, 
                              save_path: str = "threshold_analysis.png"):
        """
        Plota análise de F1 vs threshold para cada classe.
        
        Args:
            y_true: Labels verdadeiros
            y_probs: Probabilidades preditas
            labels: Nomes das classes
            save_path: Caminho para salvar
        """
        from sklearn.metrics import f1_score
        
        labels = labels or LABELS
        thresholds = np.linspace(0.1, 0.9, 50)
        
        # Calcular layout da grid
        n_classes = len(labels)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_classes == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        optimal_thresholds = {}
        
        for i, label in enumerate(labels):
            f1_scores = []
            
            for threshold in thresholds:
                y_pred_class = (y_probs[:, i] >= threshold).astype(int)
                f1 = f1_score(y_true[:, i], y_pred_class, zero_division=0)
                f1_scores.append(f1)
            
            # Encontrar threshold ótimo
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            optimal_thresholds[label] = (best_threshold, best_f1)
            
            # Plotar
            axes[i].plot(thresholds, f1_scores, 'b-', linewidth=2)
            axes[i].axvline(best_threshold, color='red', linestyle='--', alpha=0.7)
            axes[i].set_xlabel('Threshold')
            axes[i].set_ylabel('F1 Score')
            axes[i].set_title(f'{label}\nÓtimo: {best_threshold:.3f} (F1={best_f1:.3f})')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim([0.1, 0.9])
        
        # Esconder axes vazios
        for i in range(n_classes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Análise de threshold salva em: {save_path}")
        
        # Log thresholds ótimos
        logger.info("🎯 Thresholds ótimos por classe:")
        for label, (threshold, f1) in optimal_thresholds.items():
            logger.info(f"  {label}: {threshold:.3f} (F1={f1:.3f})")
        
        return optimal_thresholds

class VisualizationSuite:
    """Suite completa de visualizações."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Inicializa suite de visualizações.
        
        Args:
            output_dir: Diretório para salvar visualizações
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_all_plots(self, training_history: Dict, y_true: np.ndarray, 
                          y_pred: np.ndarray, y_probs: np.ndarray, 
                          per_class_metrics: Dict, labels: List[str] = None):
        """
        Gera todas as visualizações.
        
        Args:
            training_history: Histórico de treinamento
            y_true: Labels verdadeiros
            y_pred: Predições binárias
            y_probs: Probabilidades
            per_class_metrics: Métricas por classe
            labels: Nomes das classes
        """
        logger.info(f"\n🎨 Gerando visualizações em: {self.output_dir}")
        
        # Curvas de treinamento
        TrainingVisualizer.plot_training_curves(
            training_history, 
            os.path.join(self.output_dir, "training_curves.png")
        )
        
        # Distribuição de classes
        MetricsVisualizer.plot_class_distribution(
            y_true, labels, 
            os.path.join(self.output_dir, "class_distribution.png")
        )
        
        # Matrizes de confusão
        MetricsVisualizer.plot_confusion_matrices(
            y_true, y_pred, labels,
            os.path.join(self.output_dir, "confusion_matrices.png")
        )
        
        # Métricas por classe
        MetricsVisualizer.plot_metrics_per_class(
            per_class_metrics,
            os.path.join(self.output_dir, "metrics_per_class.png")
        )
        
        # Curvas PR
        CurvesVisualizer.plot_precision_recall_curves(
            y_true, y_probs, labels,
            os.path.join(self.output_dir, "precision_recall_curves.png")
        )
        
        # Curvas ROC
        CurvesVisualizer.plot_roc_curves(
            y_true, y_probs, labels,
            os.path.join(self.output_dir, "roc_curves.png")
        )
        
        # Análise de threshold
        optimal_thresholds = ThresholdVisualizer.plot_threshold_analysis(
            y_true, y_probs, labels,
            os.path.join(self.output_dir, "threshold_analysis.png")
        )
        
        logger.info("✅ Todas as visualizações foram geradas!")
        
        return optimal_thresholds