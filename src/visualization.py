# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, multilabel_confusion_matrix
)
from src.config import LABELS

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")

class PlottingUtils:
    COLORS = ["#FF6347", "#4682B4", "#32CD32", "#FFD700", "#6A5ACD", "#FF69B4", "#00CED1", "#FA8072", "#7B68EE", "#20B2AA"]
    MARKERS = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']

    @staticmethod
    def save_plot(fig, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        try:
            fig.savefig(full_path, bbox_inches='tight', dpi=300)
            logger.info(f"🖼️ Gráfico salvo em: {full_path}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar gráfico {filename}: {e}")
        plt.close(fig)

class TrainingVisualizer:
    @staticmethod
    def plot_training_curves(training_history: Dict[str, List[Any]], output_dir: str):
        logger.info("📈 Plotando curvas de aprendizado...")

        if training_history.get('train_steps') and training_history.get('train_loss'):
            train_steps_clean = training_history['train_steps']
            train_loss_clean = [x for x in training_history['train_loss'] if isinstance(x, (int, float)) and not np.isnan(x)]
            
            if len(train_loss_clean) < len(training_history['train_loss']):
                valid_indices = [i for i, x in enumerate(training_history['train_loss']) if isinstance(x, (int, float)) and not np.isnan(x)]
                train_steps_clean = [training_history['train_steps'][i] for i in valid_indices]

            if train_steps_clean and train_loss_clean and len(train_steps_clean) == len(train_loss_clean):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_steps_clean, train_loss_clean, label='Training Loss', color=PlottingUtils.COLORS[0], marker=PlottingUtils.MARKERS[0 % len(PlottingUtils.MARKERS)], linestyle='-')
                ax.set_title('Curva de Loss do Treinamento')
                ax.set_xlabel('Steps de Treinamento')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                PlottingUtils.save_plot(fig, output_dir, "training_loss_curve.png")
            else:
                logger.warning("⚠️ Não há dados válidos suficientes ou há incompatibilidade de tamanho para plotar a curva de loss de treino.")
        else:
            logger.warning("⚠️ Chaves 'train_steps' ou 'train_loss' não encontradas no histórico para plotagem de loss de treino.")

        if training_history.get('train_steps') and training_history.get('train_learning_rate'):
            lr_steps_clean = training_history['train_steps']
            lr_values_clean = [x for x in training_history['train_learning_rate'] if isinstance(x, (int, float)) and not np.isnan(x)]

            if len(lr_values_clean) < len(training_history['train_learning_rate']):
                valid_indices = [i for i, x in enumerate(training_history['train_learning_rate']) if isinstance(x, (int, float)) and not np.isnan(x)]
                if len(valid_indices) == len(lr_values_clean) and len(lr_steps_clean) >= len(lr_values_clean):
                     lr_steps_clean = [training_history['train_steps'][i] for i in valid_indices]

            if lr_steps_clean and lr_values_clean and len(lr_steps_clean) == len(lr_values_clean):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(lr_steps_clean, lr_values_clean, label='Learning Rate', color=PlottingUtils.COLORS[1 % len(PlottingUtils.COLORS)], marker=PlottingUtils.MARKERS[1 % len(PlottingUtils.MARKERS)], linestyle='-')
                ax.set_title('Curva de Taxa de Aprendizado')
                ax.set_xlabel('Steps de Treinamento')
                ax.set_ylabel('Learning Rate')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                PlottingUtils.save_plot(fig, output_dir, "learning_rate_curve.png")
            else:
                logger.warning("⚠️ Não há dados válidos suficientes ou há incompatibilidade de tamanho para plotar a curva de learning rate.")
        else:
            logger.warning("⚠️ Chaves 'train_steps' ou 'train_learning_rate' não encontradas no histórico para plotagem de learning rate.")

        eval_metrics_to_plot = {
            'eval_loss': 'Validation Loss',
            'eval_avg_precision': 'Validation Average Precision',
            'eval_macro_f1': 'Validation Macro F1-Score',
            'eval_hamming_loss': 'Validation Hamming Loss',
            'eval_micro_f1': 'Validation Micro F1-Score',
            'eval_weighted_f1': 'Validation Weighted F1-Score',
            'eval_macro_precision': 'Validation Macro Precision',
            'eval_macro_recall': 'Validation Macro Recall'
        }

        if training_history.get('eval_steps'):
            eval_steps = training_history['eval_steps']
            if not eval_steps:
                logger.warning("⚠️ 'eval_steps' está vazio. Pulando plot de métricas de avaliação.")
            else:
                for metric_idx, (metric_key, plot_label) in enumerate(eval_metrics_to_plot.items()):
                    if training_history.get(metric_key):
                        metric_values = training_history[metric_key]
                        valid_indices = [i for i, val in enumerate(metric_values) if isinstance(val, (int, float)) and not np.isnan(val)]
                        
                        if not valid_indices:
                            logger.warning(f"⚠️ Não há dados válidos para a métrica '{metric_key}'. Pulando plot.")
                            continue

                        current_eval_steps = [eval_steps[i] for i in valid_indices if i < len(eval_steps)]
                        current_metric_values = [metric_values[i] for i in valid_indices]

                        if not current_eval_steps or len(current_eval_steps) != len(current_metric_values):
                            logger.warning(f"⚠️ Incompatibilidade de tamanho ou ausência de steps válidos para '{metric_key}' após filtrar NaNs. Pulando plot.")
                            continue

                        fig, ax = plt.subplots(figsize=(10, 6))
                        color_idx = (metric_idx + 2) % len(PlottingUtils.COLORS)
                        marker_idx = (metric_idx + 2) % len(PlottingUtils.MARKERS)
                        ax.plot(current_eval_steps, current_metric_values, label=plot_label, color=PlottingUtils.COLORS[color_idx], marker=PlottingUtils.MARKERS[marker_idx], linestyle='-')
                        ax.set_title(f'Curva de {plot_label}')
                        ax.set_xlabel('Steps de Avaliação')
                        ax.set_ylabel(plot_label.split()[-1])
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.7)
                        PlottingUtils.save_plot(fig, output_dir, f"{metric_key}_curve.png")
                    else:
                        logger.warning(f"⚠️ Chave '{metric_key}' não encontrada no histórico para plotagem.")
        else:
            logger.warning("⚠️ Chave 'eval_steps' não encontrada no histórico. Pulando plot de métricas de avaliação.")


class EvaluationVisualizer:
    @staticmethod
    def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, output_dir: str, labels_list: Optional[List[str]] = None):
        if y_true is None or y_probs is None:
            logger.warning("⚠️ y_true ou y_probs é None. Pulando plot de curvas ROC.")
            return
        if y_true.shape != y_probs.shape:
            logger.error(f"❌ Incompatibilidade de shape entre y_true ({y_true.shape}) e y_probs ({y_probs.shape}) para ROC.")
            return
            
        logger.info("📈 Plotando curvas ROC...")
        num_classes = y_true.shape[1]
        current_labels = labels_list if labels_list and len(labels_list) == num_classes else LABELS
        if len(current_labels) != num_classes:
            logger.warning(f"⚠️ Número de LABELS ({len(current_labels)}) não corresponde ao número de classes ({num_classes}). Usando labels genéricos para ROC.")
            current_labels = [f"Classe {i+1}" for i in range(num_classes)]

        fpr, tpr, roc_auc = {}, {}, {}
        fig, ax = plt.subplots(figsize=(12, 9))
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], color=PlottingUtils.COLORS[i % len(PlottingUtils.COLORS)], lw=2,
                    label=f'ROC {current_labels[i]} (AUC = {roc_auc[i]:.3f})')

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        ax.plot(fpr["micro"], tpr["micro"], color='black', linestyle=':', linewidth=3,
                label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
        ax.set_xlim([-0.02, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
        ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
        ax.set_title('Curvas ROC Multi-classe', fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        PlottingUtils.save_plot(fig, output_dir, "roc_curves.png")

    @staticmethod
    def plot_precision_recall_curves(y_true: np.ndarray, y_probs: np.ndarray, output_dir: str, labels_list: Optional[List[str]] = None):
        if y_true is None or y_probs is None:
            logger.warning("⚠️ y_true ou y_probs é None. Pulando plot de curvas Precision-Recall.")
            return
        if y_true.shape != y_probs.shape:
            logger.error(f"❌ Incompatibilidade de shape entre y_true ({y_true.shape}) e y_probs ({y_probs.shape}) para PR.")
            return

        logger.info("📈 Plotando curvas Precision-Recall...")
        num_classes = y_true.shape[1]
        current_labels = labels_list if labels_list and len(labels_list) == num_classes else LABELS
        if len(current_labels) != num_classes:
            logger.warning(f"⚠️ Número de LABELS ({len(current_labels)}) não corresponde ao número de classes ({num_classes}). Usando labels genéricos para PR.")
            current_labels = [f"Classe {i+1}" for i in range(num_classes)]
            
        precision, recall, average_precision = {}, {}, {}
        fig, ax = plt.subplots(figsize=(12, 9))
        for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            average_precision[i] = average_precision_score(y_true[:, i], y_probs[:, i])
            ax.plot(recall[i], precision[i], color=PlottingUtils.COLORS[i % len(PlottingUtils.COLORS)], lw=2,
                    label=f'PR {current_labels[i]} (AP = {average_precision[i]:.3f})')

        precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_probs.ravel())
        average_precision["micro"] = average_precision_score(y_true.ravel(), y_probs.ravel(), average="micro")
        ax.plot(recall["micro"], precision["micro"], color='black', linestyle=':', linewidth=3,
                label=f'Micro-average PR (AP = {average_precision["micro"]:.3f})')
        
        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Curvas Precision-Recall Multi-classe', fontsize=14)
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        PlottingUtils.save_plot(fig, output_dir, "precision_recall_curves.png")

    @staticmethod
    def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str, labels_list: Optional[List[str]] = None):
        if y_true is None or y_pred is None:
            logger.warning("⚠️ y_true ou y_pred é None. Pulando plot de matrizes de confusão.")
            return
        if y_true.shape != y_pred.shape:
            logger.error(f"❌ Incompatibilidade de shape entre y_true ({y_true.shape}) e y_pred ({y_pred.shape}) para matriz de confusão.")
            return
            
        logger.info("📊 Plotando matrizes de confusão...")
        num_classes = y_true.shape[1]
        current_labels = labels_list if labels_list and len(labels_list) == num_classes else LABELS
        if len(current_labels) != num_classes:
            logger.warning(f"⚠️ Número de LABELS ({len(current_labels)}) não corresponde ao número de classes ({num_classes}). Usando labels genéricos para CM.")
            current_labels = [f"Classe {i+1}" for i in range(num_classes)]

        mcm = multilabel_confusion_matrix(y_true, y_pred)
        n_cols = 3
        n_rows = (num_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        axes = axes.flatten() 

        for i in range(num_classes):
            if i >= len(axes): break
            ax = axes[i]
            class_name_short = current_labels[i].split()[0].capitalize()
            sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 10})
            ax.set_title(f'{current_labels[i]}', fontsize=11)
            ax.set_xlabel('Predito', fontsize=9)
            ax.set_ylabel('Verdadeiro', fontsize=9)
            ax.set_xticklabels([f'Não-{class_name_short}', class_name_short], fontsize=8)
            ax.set_yticklabels([f'Não-{class_name_short}', class_name_short], fontsize=8, rotation=0)

        if num_classes < len(axes):
            for j in range(num_classes, len(axes)):
                fig.delaxes(axes[j])

        fig.suptitle('Matrizes de Confusão por Classe (Individuais)', fontsize=14, y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        PlottingUtils.save_plot(fig, output_dir, "confusion_matrices_per_class.png")

    @staticmethod
    def plot_per_class_metrics_comparison(per_class_metrics: Dict[str, Dict[str, float]], output_dir: str, 
                                         metric_names: Optional[List[str]] = None):
        if not per_class_metrics:
            logger.warning("⚠️ Dicionário 'per_class_metrics' está vazio. Pulando plot de comparação de métricas.")
            return

        logger.info("📊 Plotando comparação de métricas por classe...")
        class_names = list(per_class_metrics.keys())
        if not class_names:
            logger.warning("⚠️ Não foi possível extrair nomes das classes de 'per_class_metrics'. Pulando plot.")
            return

        default_metrics = ['f1-score', 'precision', 'recall']
        metrics_to_plot = metric_names if metric_names else default_metrics
        
        available_metrics_in_data = set()
        if isinstance(per_class_metrics.get(class_names[0]), dict):
            available_metrics_in_data.update(per_class_metrics[class_names[0]].keys())
        
        actual_metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics_in_data]
        if not actual_metrics_to_plot:
            logger.warning(f"⚠️ Nenhuma das métricas especificadas/padrão ({metrics_to_plot}) encontrada nos dados. Pulando.")
            return

        plot_data = {metric: [] for metric in actual_metrics_to_plot}
        valid_class_names_for_plot = []

        for class_name in class_names:
            class_metric_dict = per_class_metrics.get(class_name)
            if isinstance(class_metric_dict, dict):
                current_class_metric_values = []
                all_metrics_found_for_class = True
                for metric in actual_metrics_to_plot:
                    value = class_metric_dict.get(metric)
                    if value is None or np.isnan(value):
                        logger.warning(f"Métrica '{metric}' ausente ou NaN para a classe '{class_name}'. Esta classe não será plotada.")
                        all_metrics_found_for_class = False
                        break
                    current_class_metric_values.append(value)
                
                if all_metrics_found_for_class:
                    valid_class_names_for_plot.append(class_name)
                    for i, metric in enumerate(actual_metrics_to_plot):
                        plot_data[metric].append(current_class_metric_values[i])
            else:
                logger.warning(f"Dados de métrica para a classe '{class_name}' não são um dicionário. Pulando esta classe.")
        
        if not valid_class_names_for_plot:
            logger.warning("⚠️ Nenhuma classe com dados de métrica válidos para todas as métricas selecionadas. Pulando plot de comparação.")
            return

        df_metrics = pd.DataFrame(plot_data, index=valid_class_names_for_plot)
        fig, ax = plt.subplots(figsize=(max(10, len(valid_class_names_for_plot) * 0.7), 6))
        df_metrics.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Comparação de Métricas por Classe', fontsize=15)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Classes', fontsize=12)
        ax.legend(title='Métricas', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        
        # --- CORREÇÃO APLICADA AQUI ---
        # Remover 'ha' de tick_params e aplicar rotação e alinhamento aos xticklabels diretamente
        ax.tick_params(axis='x', labelsize=10) # Manter labelsize se desejado
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # --- FIM DA CORREÇÃO ---
        
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        ax.set_ylim(0, 1.05)

        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom',
                           xytext=(0, 5), 
                           textcoords='offset points', fontsize=7, rotation=45)

        PlottingUtils.save_plot(fig, output_dir, "per_class_metrics_comparison.png")

    @staticmethod
    def plot_threshold_tuning_curves(y_true_class: np.ndarray, y_probs_class: np.ndarray, 
                                    class_name: str, output_dir: str) -> float:
        if y_true_class is None or y_probs_class is None or len(y_true_class) == 0 or len(y_probs_class) == 0:
            logger.warning(f"⚠️ y_true ou y_probs é None ou vazio para a classe {class_name}. Pulando plot de threshold tuning.")
            return 0.5 
        
        if len(np.unique(y_true_class)) < 2:
            logger.warning(f"⚠️ A classe {class_name} tem apenas uma classe presente nos dados verdadeiros. "
                           "Não é possível calcular curvas PR/F1 ou encontrar threshold ótimo. Usando 0.5.")
            return 0.5

        precisions, recalls, thresholds_pr = precision_recall_curve(y_true_class, y_probs_class)
        f1_scores = []
        f05_scores = []  # F0.5 gives more weight to precision
        f2_scores = []   # F2 gives more weight to recall
        
        min_len = min(len(precisions), len(recalls), len(thresholds_pr) + 1)

        for i in range(min_len - 1):
            p = precisions[i]
            r = recalls[i]
            
            # F1 score
            if p + r == 0:
                f1_scores.append(0.0)
                f05_scores.append(0.0)
                f2_scores.append(0.0)
            else:
                # F1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(2 * (p * r) / (p + r))
                
                # F0.5 = (1 + 0.5²) * (precision * recall) / (0.5² * precision + recall)
                beta_05 = 0.5
                f05_scores.append((1 + beta_05**2) * (p * r) / ((beta_05**2 * p) + r))
                
                # F2 = (1 + 2²) * (precision * recall) / (2² * precision + recall)
                beta_2 = 2.0
                f2_scores.append((1 + beta_2**2) * (p * r) / ((beta_2**2 * p) + r))
        
        if not f1_scores:
            logger.warning(f"⚠️ Não foi possível calcular F1 scores para a classe {class_name}. Usando threshold padrão 0.5.")
            optimal_threshold = 0.5
            max_f1_score = 0.0
            optimal_f05_threshold = 0.5
            max_f05_score = 0.0
            optimal_f2_threshold = 0.5
            max_f2_score = 0.0
        else:
            # Find optimal thresholds for each F-score
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds_pr[optimal_idx]
            max_f1_score = f1_scores[optimal_idx]
            
            optimal_f05_idx = np.argmax(f05_scores)
            optimal_f05_threshold = thresholds_pr[optimal_f05_idx]
            max_f05_score = f05_scores[optimal_f05_idx]
            
            optimal_f2_idx = np.argmax(f2_scores)
            optimal_f2_threshold = thresholds_pr[optimal_f2_idx]
            max_f2_score = f2_scores[optimal_f2_idx]

        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot curves
        ax.plot(thresholds_pr[:len(f1_scores)], precisions[:len(f1_scores)], 
                label='Precision', color=PlottingUtils.COLORS[0], linewidth=2)
        ax.plot(thresholds_pr[:len(f1_scores)], recalls[:len(f1_scores)], 
                label='Recall', color=PlottingUtils.COLORS[1], linewidth=2)
        ax.plot(thresholds_pr[:len(f1_scores)], f1_scores, 
                label='F1-Score', color=PlottingUtils.COLORS[2], linestyle='--', linewidth=2)
        ax.plot(thresholds_pr[:len(f05_scores)], f05_scores, 
                label='F0.5-Score', color=PlottingUtils.COLORS[3], linestyle=':', linewidth=1.5, alpha=0.8)
        ax.plot(thresholds_pr[:len(f2_scores)], f2_scores, 
                label='F2-Score', color=PlottingUtils.COLORS[4], linestyle='-.', linewidth=1.5, alpha=0.8)
        
        # Add markers for optimal points
        ax.scatter(optimal_threshold, max_f1_score, 
                  marker='o', s=100, color='red', zorder=5, 
                  label=f'Ótimo F1 ({max_f1_score:.3f}) @ thr={optimal_threshold:.3f}')
        
        ax.scatter(optimal_f05_threshold, max_f05_score, 
                  marker='s', s=80, color=PlottingUtils.COLORS[3], zorder=5, 
                  label=f'Ótimo F0.5 ({max_f05_score:.3f}) @ thr={optimal_f05_threshold:.3f}')
        
        ax.scatter(optimal_f2_threshold, max_f2_score, 
                  marker='^', s=80, color=PlottingUtils.COLORS[4], zorder=5, 
                  label=f'Ótimo F2 ({max_f2_score:.3f}) @ thr={optimal_f2_threshold:.3f}')
        
        # Add vertical lines for optimal thresholds (subtle)
        ax.axvline(x=optimal_threshold, color='red', linestyle=':', alpha=0.3)
        ax.axvline(x=optimal_f05_threshold, color=PlottingUtils.COLORS[3], linestyle=':', alpha=0.3)
        ax.axvline(x=optimal_f2_threshold, color=PlottingUtils.COLORS[4], linestyle=':', alpha=0.3)
        
        ax.set_title(f'Ajuste de Threshold para {class_name}', fontsize=14)
        ax.set_xlabel('Threshold de Classificação', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Add a text box with summary
        textstr = f'F0.5: favorece Precision\nF1: balanceado\nF2: favorece Recall'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        
        PlottingUtils.save_plot(fig, output_dir, f"threshold_tuning_{class_name.replace(' ', '_').lower()}.png")
        
        logger.info(f"🎯 Thresholds ótimos para {class_name}:")
        logger.info(f"   F0.5 (precision-focused): {optimal_f05_threshold:.4f} (F0.5: {max_f05_score:.4f})")
        logger.info(f"   F1   (balanced):          {optimal_threshold:.4f} (F1: {max_f1_score:.4f})")
        logger.info(f"   F2   (recall-focused):    {optimal_f2_threshold:.4f} (F2: {max_f2_score:.4f})")
        
        return optimal_threshold

class VisualizationSuite:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)
        self.labels_list = LABELS

    def generate_all_plots(self, 
                           training_history: Dict[str, List[Any]],
                           y_true_test: np.ndarray, 
                           y_pred_test: np.ndarray, 
                           y_probs_test: np.ndarray,
                           per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
                           ) -> Dict[str, float]:
        logger.info(f"\n🎨 Gerando visualizações em: {self.output_base_dir}")

        if training_history:
            TrainingVisualizer.plot_training_curves(training_history, self.output_base_dir)
        else:
            logger.warning("⚠️ Histórico de treinamento não disponível. Pulando plot de curvas de aprendizado.")

        if y_true_test is not None and y_probs_test is not None:
            EvaluationVisualizer.plot_roc_curves(y_true_test, y_probs_test, self.output_base_dir, self.labels_list)
            EvaluationVisualizer.plot_precision_recall_curves(y_true_test, y_probs_test, self.output_base_dir, self.labels_list)
        else:
            logger.warning("⚠️ y_true_test ou y_probs_test não disponíveis. Pulando plots de ROC e PR.")

        if y_true_test is not None and y_pred_test is not None:
            EvaluationVisualizer.plot_confusion_matrices(y_true_test, y_pred_test, self.output_base_dir, self.labels_list)
        else:
            logger.warning("⚠️ y_true_test ou y_pred_test não disponíveis. Pulando plot de matrizes de confusão.")
            
        if per_class_metrics:
            EvaluationVisualizer.plot_per_class_metrics_comparison(per_class_metrics, self.output_base_dir)
        else:
            logger.warning("⚠️ Métricas por classe não disponíveis. Pulando plot de comparação.")

        optimal_thresholds: Dict[str, float] = {}
        if y_true_test is not None and y_probs_test is not None:
            logger.info("📈 Gerando curvas de ajuste de threshold por classe...")
            if y_true_test.ndim == 1:
                 y_true_reshaped = y_true_test.reshape(-1,1)
                 y_probs_reshaped = y_probs_test.reshape(-1,1)
                 num_classes_for_thresh = 1
                 labels_for_thresh = [self.labels_list[0] if self.labels_list else "Classe Única"]
            else:
                 y_true_reshaped = y_true_test
                 y_probs_reshaped = y_probs_test
                 num_classes_for_thresh = y_true_test.shape[1]
                 labels_for_thresh = self.labels_list

            if num_classes_for_thresh == len(labels_for_thresh):
                for i in range(num_classes_for_thresh):
                    class_name = labels_for_thresh[i]
                    optimal_threshold = EvaluationVisualizer.plot_threshold_tuning_curves(
                        y_true_reshaped[:, i],
                        y_probs_reshaped[:, i],
                        class_name,
                        self.output_base_dir
                    )
                    optimal_thresholds[class_name] = optimal_threshold
            else:
                logger.error(f"❌ Disparidade no número de classes ({num_classes_for_thresh}) e labels ({len(labels_for_thresh)}) para ajuste de threshold.")
                for class_name in self.labels_list:
                    optimal_thresholds[class_name] = 0.5
        else:
            logger.warning("⚠️ y_true_test ou y_probs_test não disponíveis. Pulando ajuste de thresholds.")
            for class_name in self.labels_list:
                optimal_thresholds[class_name] = 0.5

        logger.info("✅ Todas as visualizações foram geradas.")
        return optimal_thresholds