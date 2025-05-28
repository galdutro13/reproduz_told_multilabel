# -*- coding: utf-8 -*-
"""
Fine-tuning BERT multi-label (ToLD-BR) em CPU — estável e sem ruído.
PyTorch ≥2.1, transformers 4.48.x, simpletransformers 0.64.x, IPEX 2.7.x.
"""

# ---------- 0 | Ambiente -------------------------------------------------
import os, multiprocessing, logging, warnings, textwrap, sys
import numpy as np
import argparse

N_CPU = max(1, multiprocessing.cpu_count() - 1)
os.environ.update({
    "OMP_NUM_THREADS":            str(N_CPU),
    "MKL_NUM_THREADS":            str(N_CPU),
    "TOKENIZERS_PARALLELISM":     "false",
    "ONEDNN_MAX_CPU_ISA":         "AVX2",   # evita caminhos BF16 parciais
    "IPEX_VERBOSE":               "0",
})
import torch, pandas as pd
torch.set_num_threads(N_CPU)
torch.set_num_interop_threads(max(1, N_CPU // 2))

from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()                     # silencia HF

logging.basicConfig(level=logging.INFO)
logging.getLogger("IPEX").setLevel(logging.ERROR)    # silencia IPEX python

warnings.filterwarnings("ignore", message=".*ipex_MKLSGEMM.*")

# ---------- 1 | Constantes -----------------------------------------------
DATASET_PATH = "ToLD-BR.csv"
MODEL_DIR    = 'outputs_bert/'
MODEL_NAME   = "pablocosta/bertabaporu-base-uncased"
LABELS       = ["homophobia","obscene","insult","racism","misogyny","xenophobia"]
NUM_LABELS   = len(LABELS)
SEED         = 42

# ---------- 2 | Utilidades -----------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Carrega e normaliza o CSV, devolvendo colunas 'text' e 'labels' (lista binária)."""
    if not os.path.exists(path):
        sys.exit(f"Arquivo {path} ausente.")
    df = pd.read_csv(path)
    if "text" not in df.columns or not set(LABELS).issubset(df.columns):
        sys.exit("CSV precisa conter a coluna 'text' e todas as colunas de rótulo.")
    df[LABELS] = (df[LABELS].fillna(0).astype(float) > 0).astype(int)
    df["labels"] = df[LABELS].values.tolist()
    print(f"Dataset carregado: {len(df)} amostras — exemplo: "
          f"{df[['text','labels']].iloc[0].to_dict()}")
    return df[["text", "labels"]]

# ---------- 2b | Métricas ------------------------------------------------
from sklearn.metrics import f1_score, hamming_loss, average_precision_score

def macro_f1(labels, preds, threshold: float = 0.5):
    """
    Converte as probabilidades em 0/1 com limiar (default 0.5)
    e devolve o F1-macro para problema multirrótulo.
    """
    labels = np.asarray(labels, dtype=int)
    preds  = np.asarray(preds)

    # Binarização: 1 se prob ≥ threshold, senão 0
    if preds.dtype != int:
        preds = (preds >= threshold).astype(int)

    return f1_score(labels, preds, average="macro", zero_division=0)

def compute_hamming(labels, preds, threshold: float = 0.5):
    """
    Hamming Loss para multirrótulo:
    converte preds em 0/1 via limiar e devolve hamming_loss do sklearn.
    """
    labels = np.asarray(labels, dtype=int)
    preds  = np.asarray(preds)
    preds  = (preds >= threshold).astype(int)
    return hamming_loss(labels, preds)

def compute_avg_precision(labels, preds, **kwargs):
    """
    Average Precision Score (macro) para multirrótulo:
    recebe preds como probabilidades e devolve average_precision_score.
    """
    labels = np.asarray(labels, dtype=int)
    preds  = np.asarray(preds, dtype=float)
    # average='macro' agrega performance de cada classe
    return average_precision_score(labels, preds, average="macro")


def split_stratified_holdout(df: pd.DataFrame,
                             seed: int = SEED,
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1):
    """
    Divide o DataFrame em treino, validação e teste usando hold-out estratificado
    multi-rótulo (80 / 10 / 10). Requer iterative-stratification; se ausente,
    aplica fallback simplificado.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        y = np.asarray(df["labels"].tolist())
        idx = np.arange(len(df))

        # Primeira divisão: treino × (val+teste)
        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed
        )
        train_idx, temp_idx = next(msss1.split(idx, y))

        # Segunda divisão: validação × teste
        y_temp = y[temp_idx]
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed,
        )
        val_rel, test_rel = next(msss2.split(temp_idx.reshape(-1, 1), y_temp))
        val_idx  = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        print("Estratificação (iterative-stratification) concluída.")
    except ImportError:
        warnings.warn(
            "Pacote 'iterative-stratification' não localizado; realizando "
            "divisão aleatória estratificada simplificada. "
            "Instale-o via 'pip install iterative-stratification' para melhor fidelidade."
        )
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        y_single = df[LABELS].idxmax(axis=1)  # aproximação mono-rótulo para estratificar
        train_idx, temp_idx = train_test_split(
            idx, test_size=(val_ratio + test_ratio), stratify=y_single, random_state=seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_ratio / (val_ratio + test_ratio),
            stratify=y_single[temp_idx], random_state=seed
        )

    d_train = df.iloc[train_idx].reset_index(drop=True)
    d_val   = df.iloc[val_idx].reset_index(drop=True)
    d_test  = df.iloc[test_idx].reset_index(drop=True)

    def _ratio(x): return f"{len(x):,} ({len(x)/len(df):.1%})"
    print(f"Partições — treino: {_ratio(d_train)}, "
          f"validação: {_ratio(d_val)}, teste: {_ratio(d_test)}")
    return d_train, d_val, d_test

# ---------- 3 | Modelo ----------------------------------------------------
def make_model(evaluate_during_training: bool = True):
    args = MultiLabelClassificationArgs()
    args.manual_seed = SEED
    args.process_count = 1
    args.use_multiprocessing = False
    args.use_multiprocessing_for_evaluation = False
    args.dataloader_num_workers = N_CPU
    args.output_dir, args.cache_dir = "outputs_bert/", "cache_bert/"
    args.num_labels = NUM_LABELS
    args.use_cuda   = False
    args.overwrite_output_dir = True
    args.evaluate_during_training = evaluate_during_training
    args.evaluate_during_training_verbose = True
    args.save_eval_checkpoints = True
    args.evaluate_during_training_steps = 150
    args.logging_steps = 150
    args.tensorboard_dir = "runs/"
    args.num_train_epochs = 3
    args.max_seq_length = 80
    args.do_lower_case = True
    

    model = MultiLabelClassificationModel(
        "bert", MODEL_NAME, num_labels=NUM_LABELS, args=args, use_cuda=False, pos_weight=[7.75, 1.47, 1.95, 12.30, 6.66, 11.75]
    )

    # ---- IPEX ------------------------------------------------------------
    try:
        import intel_extension_for_pytorch as ipex
        from packaging.version import parse as V
        import transformers
        if V("4.6.0") <= V(transformers.__version__) <= V("4.48.0"):
            model.model = ipex.optimize(
                model.model,
                dtype=torch.float32,
                inplace=True,
                conv_bn_folding=False,
                linear_bn_folding=False,
                auto_kernel_selection=True,
            )
            print("🚀  IPEX otimizado (FP32).")
        else:
            print("ℹ️  IPEX pulado (transformers > 4.48).")
    except ImportError:
        print("ℹ️  IPEX ausente — usando PyTorch puro.")
    except Exception as e:
        warnings.warn(f"IPEX falhou, prosseguindo: {e}")

    # ---- torch.compile ---------------------------------------
    if hasattr(torch, "compile"):
        try:
            model.model = torch.compile(
                model.model, backend="ipex", dynamic=False, fullgraph=False
            )
            print("🛠️  torch.compile ativado.")
        except Exception as e:
            warnings.warn(f"torch.compile falhou — desativado: {e}")

    return model

# ---------- 4 | Execução --------------------------------------------------
import matplotlib.pyplot as plt              # novo import
from pathlib import Path

def plot_train_curves(training_details, save_path="outputs_bert/loss_f1_vs_step.png"):
    """
    Desenha curvas de train_loss, eval_loss e macro_f1 em função do global_step.

    • Eixo esquerdo  : losses (train e eval)
    • Eixo direito   : Macro-F1
    • Escala x       : global_step (linear)
    """

    import pandas as pd, matplotlib.pyplot as plt
    from pathlib import Path

    df = pd.DataFrame(training_details)

    # ---- nomes de coluna tolerantes a versão ----------------------------
    step_col      = "global_step"
    train_col     = "train_loss" if "train_loss" in df.columns else "loss"
    eval_col      = "eval_loss"  if "eval_loss"  in df.columns else None
    f1_col        = "macro_f1"   if "macro_f1"   in df.columns else None

    # Se não houver eval_loss OU macro_f1 avisa e sai ---------------------
    if eval_col is None or f1_col is None:
        print("⚠️  training_details não possui eval_loss ou macro_f1 — gráfico não gerado.")
        return

    # ---- plot -----------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(df[step_col], df[train_col], label="Train loss", linewidth=1.5)
    ax1.plot(df[step_col], df[eval_col],  label="Eval loss",  linewidth=1.5)
    ax1.set_xlabel("Global step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, ls=":")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(df[step_col], df[f1_col], color="tab:green", label="Macro-F1", linewidth=1.5)
    ax2.set_ylabel("Macro-F1")
    ax2.legend(loc="lower right")

    plt.title("Curvas Loss (train/eval) × Macro-F1")
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅  Curva Loss/F1 salva em: {save_path}")

def train(df_train, df_val, evaluate_during_training: bool = False):
    model = make_model(evaluate_during_training=evaluate_during_training)
    métricas = { 
        "macro_f1":        macro_f1, 
        "hamming_loss":    compute_hamming, 
        "avg_precision":   compute_avg_precision 
        }

    print("Treinando …")
    global_step, training_details = model.train_model(
        df_train, eval_df=df_val, **métricas
    )                                         # ← capturamos training_details
    print("Treino concluído.")
    return model,training_details

def train_and_eval(df_train, df_val):
    model, training_details = train(df_train, df_val, True)

    plot_train_curves(training_details, "outputs_bert/lr_vs_loss.png")
    return model


def predict(model, textos):
    preds, _ = model.predict(textos)
    for t, p in zip(textos, preds):
        lbls = [LABELS[i] for i, f in enumerate(p) if f]
        print(f"\n{textwrap.shorten(t, 80)}\n→ {', '.join(lbls) or 'Nenhuma'}")

# ---------- Utilidades para salvar/carregar splits ------------------------
def save_split(df, name):
    df.to_csv(f"split_{name}.csv", index=False)

def load_split(name):
    import pandas as pd
    import ast
    path = f"split_{name}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'labels' in df.columns:
        df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

# ---------- Função para plotar matrizes de confusão ------------------------
def plot_multilabel_confusion(y_true, y_pred, labels, save_path=f"{MODEL_DIR}confusion_all_labels.png"):
    from sklearn.metrics import multilabel_confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    n_labels = len(labels)
    fig, axes = plt.subplots(2, (n_labels + 1) // 2, figsize=(5 * ((n_labels + 1) // 2), 10))
    axes = axes.flatten()
    # Mapeamento de posições para siglas
    siglas = np.array([["VN", "FP"], ["FN", "VP"]])
    for i, label in enumerate(labels):
        cm = mcm[i]
        ax = axes[i]
        # Cria anotações customizadas com valor e sigla
        annot = np.empty_like(cm).astype(str)
        for r in range(2):
            for c in range(2):
                annot[r, c] = f"{cm[r, c]}\n{siglas[r, c]}"
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                    xticklabels=["Pred 0 (Negativo)", "Pred 1 (Positivo)"],
                    yticklabels=["True 0 (Negativo)", "True 1 (Positivo)"], ax=ax,
                    annot_kws={"fontsize":12, "weight":"bold"})
        ax.set_title(f"{label}", fontsize=14)
        ax.set_ylabel("Real", fontsize=12)
        ax.set_xlabel("Predito", fontsize=12)
    # Remove subplots extras se houver
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Matrizes de Confusão Binária por Classe\n(VN=Verdadeiro Negativo, FP=Falso Positivo, FN=Falso Negativo, VP=Verdadeiro Positivo)", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ Matrizes de confusão salvas em: {save_path}")

# ---------- Função para plotar heatmap de coocorrência ------------------------
def plot_cooccurrence_heatmap(y_true, y_pred, labels, save_path=f"{MODEL_DIR}cooccurrence_heatmap.png"):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_labels = len(labels)
    # Matriz de coocorrência: real (linhas) × predito (colunas)
    cooc = np.zeros((n_labels, n_labels), dtype=int)
    for i in range(len(y_true)):
        true_idx = np.where(y_true[i])[0]
        pred_idx = np.where(y_pred[i])[0]
        for t in true_idx:
            for p in pred_idx:
                cooc[t, p] += 1
    plt.figure(figsize=(2+n_labels, 2+n_labels))
    ax = sns.heatmap(cooc, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Rótulo Predito', fontsize=12)
    ax.set_ylabel('Rótulo Real', fontsize=12)
    plt.title('Heatmap de Coocorrência Predita × Real', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ Heatmap de coocorrência salvo em: {save_path}")

# ---------- Função para calcular e plotar métricas por classe ----------------
def plot_metrics_per_class(y_true, y_pred, labels, save_path_bar=f"{MODEL_DIR}metrics_per_class_bar.png", 
                          save_path_radar=f"{MODEL_DIR}metrics_per_class_radar.png"):
    """
    Calcula e plota precision, recall e F1 por classe.
    Gera tanto barplot agrupado quanto radar chart.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_fscore_support
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcula métricas por classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # DataFrame para facilitar visualização
    metrics_df = pd.DataFrame({
        'Classe': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Ordena por support (quantidade de exemplos positivos) para visualizar desbalanceamento
    metrics_df = metrics_df.sort_values('Support', ascending=False)
    
    # Print das métricas
    print("\n📊 Métricas por Classe (ordenadas por frequência):")
    print("="*70)
    print(f"{'Classe':<15} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-"*70)
    for _, row in metrics_df.iterrows():
        print(f"{row['Classe']:<15} {row['Support']:>8} {row['Precision']:>10.3f} "
              f"{row['Recall']:>10.3f} {row['F1-Score']:>10.3f}")
    print("="*70)
    
    # 1. Barplot Agrupado
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    width = 0.25
    
    # Reordena para a ordem original das labels
    metrics_ordered = metrics_df.set_index('Classe').loc[labels].reset_index()
    
    bars1 = ax.bar(x - width, metrics_ordered['Precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, metrics_ordered['Recall'], width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, metrics_ordered['F1-Score'], width, label='F1-Score', color='#e74c3c')
    
    # Adiciona valores nas barras
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    # Adiciona linha secundária com support
    ax2 = ax.twinx()
    ax2.plot(x, metrics_ordered['Support'], 'k--', marker='o', linewidth=2, 
             markersize=8, label='Support (N positivos)')
    ax2.set_ylabel('Support (número de exemplos positivos)', fontsize=12)
    ax2.legend(loc='upper right')
    
    # Configurações do gráfico
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Métricas', fontsize=12)
    ax.set_title('Métricas de Performance por Classe\n(Classes ordenadas por posição original)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path_bar, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Barplot de métricas por classe salvo em: {save_path_bar}")
    
    # 2. Radar Chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Ângulos para cada classe
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Fecha o círculo
    
    # Dados para o radar
    precision_data = metrics_ordered['Precision'].tolist() + [metrics_ordered['Precision'].iloc[0]]
    recall_data = metrics_ordered['Recall'].tolist() + [metrics_ordered['Recall'].iloc[0]]
    f1_data = metrics_ordered['F1-Score'].tolist() + [metrics_ordered['F1-Score'].iloc[0]]
    
    # Plot
    ax.plot(angles, precision_data, 'o-', linewidth=2, label='Precision', color='#3498db')
    ax.fill(angles, precision_data, alpha=0.25, color='#3498db')
    
    ax.plot(angles, recall_data, 'o-', linewidth=2, label='Recall', color='#2ecc71')
    ax.fill(angles, recall_data, alpha=0.25, color='#2ecc71')
    
    ax.plot(angles, f1_data, 'o-', linewidth=2, label='F1-Score', color='#e74c3c')
    ax.fill(angles, f1_data, alpha=0.25, color='#e74c3c')
    
    # Configurações
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    # Adiciona anotações com support
    for i, (angle, label) in enumerate(zip(angles[:-1], labels)):
        support = metrics_ordered.iloc[i]['Support']
        ax.text(angle, 1.15, f'{label}\n(n={support})', 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Radar Chart: Métricas por Classe\n(com número de exemplos positivos)', 
              fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path_radar, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Radar chart de métricas por classe salvo em: {save_path_radar}")
    
    # Análise de desbalanceamento
    print("\n🔍 Análise de Desbalanceamento:")
    max_support = metrics_df['Support'].max()
    min_support = metrics_df['Support'].min()
    print(f"  - Classe mais frequente: {metrics_df.iloc[0]['Classe']} ({max_support} exemplos)")
    print(f"  - Classe menos frequente: {metrics_df.iloc[-1]['Classe']} ({min_support} exemplos)")
    print(f"  - Razão de desbalanceamento: {max_support/min_support:.1f}:1")
    
    # Identifica classes problemáticas
    problematic_classes = metrics_df[metrics_df['F1-Score'] < 0.5]
    if not problematic_classes.empty:
        print("\n⚠️  Classes com F1-Score < 0.5 (requerem atenção):")
        for _, row in problematic_classes.iterrows():
            print(f"  - {row['Classe']}: F1={row['F1-Score']:.3f}, Support={row['Support']}")
    
    return metrics_df

# ---------- Função para plotar curvas F-beta × Threshold ---------------------
def plot_fbeta_threshold_curves(y_true, y_scores, labels,
                               save_path_individual=f"{MODEL_DIR}fbeta_threshold_per_class.png",
                               save_path_global=f"{MODEL_DIR}fbeta_threshold_global.png"):
    """
    Plota curvas F-beta (F0.5, F1, F2) em função do threshold para cada classe
    e para métricas micro-agregadas (globais).
    
    Args:
        y_true: Array de labels verdadeiros (shape: n_samples, n_classes)
        y_scores: Array de scores/probabilidades (shape: n_samples, n_classes)
        labels: Lista com nomes das classes
        save_path_individual: Caminho para salvar curvas por classe
        save_path_global: Caminho para salvar curvas globais
    
    Returns:
        Dict com thresholds ótimos por classe e beta
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import fbeta_score, precision_recall_fscore_support
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Thresholds para testar (0.1 a 0.9 com passo de 0.01)
    thresholds = np.arange(0.1, 0.91, 0.01)
    
    # Betas para calcular (F0.5, F1, F2)
    betas = [0.5, 1.0, 2.0]
    beta_colors = {'0.5': '#e74c3c', '1.0': '#3498db', '2.0': '#2ecc71'}
    beta_names = {'0.5': 'F₀.₅ (precision-focused)', 
                  '1.0': 'F₁ (balanced)', 
                  '2.0': 'F₂ (recall-focused)'}
    
    # Dicionário para armazenar resultados
    optimal_thresholds = {
        'per_class': {},
        'global': {}
    }
    
    # 1. Curvas F-beta por classe
    n_classes = len(labels)
    fig1 = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig1, hspace=0.3, wspace=0.25)
    
    for class_idx, label in enumerate(labels):
        ax = fig1.add_subplot(gs[class_idx // 2, class_idx % 2])
        
        # Support da classe
        support = int(y_true[:, class_idx].sum())
        
        # Dicionário para esta classe
        optimal_thresholds['per_class'][label] = {}
        
        for beta in betas:
            fbeta_scores = []
            
            # Calcula F-beta para cada threshold
            for thresh in thresholds:
                y_pred_binary = (y_scores[:, class_idx] >= thresh).astype(int)
                
                # Calcula F-beta apenas para esta classe
                if y_true[:, class_idx].sum() > 0 and y_pred_binary.sum() > 0:
                    score = fbeta_score(y_true[:, class_idx], y_pred_binary, 
                                       beta=beta, zero_division=0)
                else:
                    score = 0.0
                
                fbeta_scores.append(score)
            
            # Encontra threshold ótimo
            fbeta_scores = np.array(fbeta_scores)
            optimal_idx = np.argmax(fbeta_scores)
            optimal_thresh = thresholds[optimal_idx]
            optimal_score = fbeta_scores[optimal_idx]
            
            # Armazena threshold ótimo
            optimal_thresholds['per_class'][label][f'F{beta}'] = {
                'threshold': optimal_thresh,
                'score': optimal_score
            }
            
            # Plota curva
            ax.plot(thresholds, fbeta_scores, 
                   color=beta_colors[str(beta)], 
                   linewidth=2.5,
                   label=f'{beta_names[str(beta)]}: best={optimal_thresh:.2f} (F={optimal_score:.3f})')
            
            # Marca ponto ótimo
            ax.scatter(optimal_thresh, optimal_score, 
                      color=beta_colors[str(beta)], 
                      s=100, zorder=5, edgecolors='black', linewidth=2)
        
        # Linha vertical no threshold padrão (0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, 
                  label='Default threshold')
        
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('F-beta Score', fontsize=11)
        ax.set_title(f'{label} (n={support})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle('Curvas F-beta × Threshold por Classe\n'
                 'Pontos marcados indicam thresholds ótimos', 
                 fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.93)  # Ajusta espaço para o título
    plt.savefig(save_path_individual, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Curvas F-beta por classe salvas em: {save_path_individual}")
    
    # 2. Curvas F-beta globais (micro-averaged)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2.1 Micro-averaged (todos os rótulos juntos)
    # Primeiro, calcula precision e recall para todos os thresholds
    precision_scores = []
    recall_scores = []
    
    for thresh in thresholds:
        y_pred_binary = (y_scores >= thresh).astype(int)
        prec, rec, _, _ = precision_recall_fscore_support(
            y_true.ravel(), y_pred_binary.ravel(), 
            average='binary', zero_division=0
        )
        precision_scores.append(prec)
        recall_scores.append(rec)
    
    # Agora calcula F-beta para cada beta
    for beta in betas:
        fbeta_scores_micro = []
        
        for thresh in thresholds:
            # Binariza todas as predições com este threshold
            y_pred_binary = (y_scores >= thresh).astype(int)
            
            # Calcula F-beta micro-averaged
            score = fbeta_score(y_true.ravel(), y_pred_binary.ravel(), 
                               beta=beta, average='binary', zero_division=0)
            fbeta_scores_micro.append(score)
        
        # Encontra threshold ótimo
        fbeta_scores_micro = np.array(fbeta_scores_micro)
        optimal_idx = np.argmax(fbeta_scores_micro)
        optimal_thresh = thresholds[optimal_idx]
        optimal_score = fbeta_scores_micro[optimal_idx]
        
        # Armazena threshold ótimo global
        optimal_thresholds['global'][f'F{beta}'] = {
            'threshold': optimal_thresh,
            'score': optimal_score
        }
        
        # Plota curva
        ax1.plot(thresholds, fbeta_scores_micro, 
                color=beta_colors[str(beta)], 
                linewidth=2.5,
                label=f'{beta_names[str(beta)]}: best={optimal_thresh:.2f} (F={optimal_score:.3f})')
        
        # Marca ponto ótimo
        ax1.scatter(optimal_thresh, optimal_score, 
                   color=beta_colors[str(beta)], 
                   s=120, zorder=5, edgecolors='black', linewidth=2)
    
    # Linha vertical no threshold padrão
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, 
               label='Default threshold')
    
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('F-beta Score', fontsize=12)
    ax1.set_title('Curvas F-beta Globais (Micro-averaged)\n'
                  'Considera todos os rótulos conjuntamente', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2.2 Trade-off Precision × Recall
    ax2.plot(thresholds, precision_scores, 'b-', linewidth=2.5, label='Precision')
    ax2.plot(thresholds, recall_scores, 'g-', linewidth=2.5, label='Recall')
    
    # Marca threshold ótimo para F1
    f1_optimal_thresh = optimal_thresholds['global']['F1.0']['threshold']
    f1_optimal_idx = np.argmin(np.abs(thresholds - f1_optimal_thresh))
    
    ax2.scatter(f1_optimal_thresh, precision_scores[f1_optimal_idx], 
               color='blue', s=120, zorder=5, edgecolors='black', linewidth=2)
    ax2.scatter(f1_optimal_thresh, recall_scores[f1_optimal_idx], 
               color='green', s=120, zorder=5, edgecolors='black', linewidth=2)
    
    ax2.axvline(x=f1_optimal_thresh, color='red', linestyle=':', alpha=0.7,
               label=f'F₁ optimal: {f1_optimal_thresh:.2f}')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, 
               label='Default threshold')
    
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Trade-off Precision × Recall\n'
                  'Mostra como métricas variam com threshold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.suptitle('Análise Global de Thresholds para Multi-label Classification', 
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path_global, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Curvas F-beta globais salvas em: {save_path_global}")
    
    # 3. Análise e recomendações
    print("\n🎯 Análise de Thresholds Ótimos:")
    print("="*80)
    
    # Por classe
    print("\n📊 Thresholds Ótimos por Classe:")
    print("-"*80)
    print(f"{'Classe':<15} {'F₀.₅ (thresh/score)':<20} {'F₁ (thresh/score)':<20} {'F₂ (thresh/score)':<20}")
    print("-"*80)
    
    for label in labels:
        f05 = optimal_thresholds['per_class'][label]['F0.5']
        f1 = optimal_thresholds['per_class'][label]['F1.0']
        f2 = optimal_thresholds['per_class'][label]['F2.0']
        
        print(f"{label:<15} "
              f"{f05['threshold']:.2f} / {f05['score']:.3f}      "
              f"{f1['threshold']:.2f} / {f1['score']:.3f}      "
              f"{f2['threshold']:.2f} / {f2['score']:.3f}")
    
    # Global
    print("\n🌍 Thresholds Ótimos Globais (Micro-averaged):")
    print("-"*80)
    for beta in betas:
        info = optimal_thresholds['global'][f'F{beta}']
        print(f"  {beta_names[str(beta)]:<30}: "
              f"threshold={info['threshold']:.2f}, score={info['score']:.3f}")
    
    # Análise de padrões
    print("\n💡 Insights e Recomendações:")
    
    # Classes que se beneficiam de threshold menor
    low_thresh_classes = []
    for label in labels:
        if optimal_thresholds['per_class'][label]['F1.0']['threshold'] < 0.4:
            low_thresh_classes.append(label)
    
    if low_thresh_classes:
        print(f"\n⬇️  Classes que se beneficiam de threshold BAIXO (<0.4):")
        for cls in low_thresh_classes:
            thresh = optimal_thresholds['per_class'][cls]['F1.0']['threshold']
            print(f"  - {cls}: threshold ótimo = {thresh:.2f}")
        print("  → Geralmente classes raras ou difíceis de detectar")
    
    # Classes que se beneficiam de threshold maior
    high_thresh_classes = []
    for label in labels:
        if optimal_thresholds['per_class'][label]['F1.0']['threshold'] > 0.6:
            high_thresh_classes.append(label)
    
    if high_thresh_classes:
        print(f"\n⬆️  Classes que se beneficiam de threshold ALTO (>0.6):")
        for cls in high_thresh_classes:
            thresh = optimal_thresholds['per_class'][cls]['F1.0']['threshold']
            print(f"  - {cls}: threshold ótimo = {thresh:.2f}")
        print("  → Geralmente classes frequentes ou com muitos falsos positivos")
    
    # Recomendação final
    print("\n📌 Estratégias Recomendadas:")
    print("  1. Use thresholds específicos por classe para maximizar performance")
    print("  2. Para aplicação sensível a falsos positivos: use thresholds de F₀.₅")
    print("  3. Para aplicação sensível a falsos negativos: use thresholds de F₂")
    print(f"  4. Threshold global recomendado: {optimal_thresholds['global']['F1.0']['threshold']:.2f} "
          f"(ao invés do padrão 0.5)")
    
    # Comparação com threshold padrão
    print("\n📈 Ganho potencial vs threshold padrão (0.5):")
    default_scores = []
    for class_idx in range(n_classes):
        y_pred_default = (y_scores[:, class_idx] >= 0.5).astype(int)
        score_default = fbeta_score(y_true[:, class_idx], y_pred_default, 
                                   beta=1.0, zero_division=0)
        score_optimal = optimal_thresholds['per_class'][labels[class_idx]]['F1.0']['score']
        improvement = ((score_optimal - score_default) / max(score_default, 0.001)) * 100
        if improvement > 5:  # Só mostra melhorias significativas
            print(f"  - {labels[class_idx]}: +{improvement:.1f}% "
                  f"(de {score_default:.3f} para {score_optimal:.3f})")
    
    return optimal_thresholds

# ---------- Função para plotar curvas PR e ROC por classe --------------------
def plot_pr_roc_curves(y_true, y_scores, labels, 
                       save_path_pr=f"{MODEL_DIR}pr_curves_per_class.png",
                       save_path_roc=f"{MODEL_DIR}roc_curves_per_class.png"):
    """
    Plota curvas Precision-Recall e ROC para cada classe.
    
    Args:
        y_true: Array de labels verdadeiros (shape: n_samples, n_classes)
        y_scores: Array de scores/probabilidades (shape: n_samples, n_classes)
        labels: Lista com nomes das classes
        save_path_pr: Caminho para salvar curva PR
        save_path_roc: Caminho para salvar curva ROC
    
    Returns:
        Dict com PR-AUC e ROC-AUC por classe
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    import pandas as pd
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Cores distintas para cada classe
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Dicionário para armazenar métricas
    metrics_dict = {
        'Classe': [],
        'PR-AUC': [],
        'ROC-AUC': [],
        'Support': []
    }
    
    # 1. Curvas Precision-Recall
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        # Calcula precision, recall e thresholds
        precision, recall, thresholds_pr = precision_recall_curve(
            y_true[:, i], y_scores[:, i]
        )
        
        # Calcula PR-AUC
        pr_auc = auc(recall, precision)
        
        # Conta o support (número de positivos)
        support = int(y_true[:, i].sum())
        
        # Plota a curva
        ax_pr.plot(recall, precision, color=color, linewidth=2.5,
                   label=f'{label} (AUC={pr_auc:.3f}, n={support})')
        
        # Preenche área sob a curva
        ax_pr.fill_between(recall, precision, alpha=0.2, color=color)
        
        # Armazena métricas
        metrics_dict['Classe'].append(label)
        metrics_dict['PR-AUC'].append(pr_auc)
        metrics_dict['Support'].append(support)
    
    # Linha de baseline (random classifier)
    total_positives = y_true.sum()
    total_samples = y_true.shape[0] * y_true.shape[1]
    baseline = total_positives / total_samples
    ax_pr.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5,
                  label=f'Baseline (random): {baseline:.3f}')
    
    ax_pr.set_xlabel('Recall', fontsize=14)
    ax_pr.set_ylabel('Precision', fontsize=14)
    ax_pr.set_title('Curvas Precision-Recall por Classe\n(Área maior indica melhor performance)', 
                    fontsize=16)
    ax_pr.legend(loc='lower left', fontsize=11)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim([0, 1.05])
    ax_pr.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path_pr, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Curvas PR salvas em: {save_path_pr}")
    
    # 2. Curvas ROC
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        # Calcula FPR, TPR e thresholds
        fpr, tpr, thresholds_roc = roc_curve(y_true[:, i], y_scores[:, i])
        
        # Calcula ROC-AUC
        roc_auc = auc(fpr, tpr)
        
        # Support
        support = int(y_true[:, i].sum())
        
        # Plota a curva
        ax_roc.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f'{label} (AUC={roc_auc:.3f}, n={support})')
        
        # Preenche área sob a curva
        ax_roc.fill_between(fpr, tpr, alpha=0.2, color=color)
        
        # Armazena ROC-AUC
        metrics_dict['ROC-AUC'].append(roc_auc)
    
    # Linha diagonal (random classifier)
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random classifier')
    
    ax_roc.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=14)
    ax_roc.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=14)
    ax_roc.set_title('Curvas ROC por Classe\n(Área maior indica melhor discriminação)', 
                     fontsize=16)
    ax_roc.legend(loc='lower right', fontsize=11)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([0, 1.05])
    ax_roc.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path_roc, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Curvas ROC salvas em: {save_path_roc}")
    
    # 3. Análise de métricas
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df = metrics_df.sort_values('Support', ascending=False)
    
    print("\n📈 Análise de Curvas PR e ROC:")
    print("="*65)
    print(f"{'Classe':<15} {'Support':>8} {'PR-AUC':>10} {'ROC-AUC':>10}")
    print("-"*65)
    for _, row in metrics_df.iterrows():
        print(f"{row['Classe']:<15} {row['Support']:>8} "
              f"{row['PR-AUC']:>10.3f} {row['ROC-AUC']:>10.3f}")
    print("="*65)
    
    # Análise de correlação
    print("\n🔍 Insights sobre Thresholds:")
    
    # Classes com baixo PR-AUC
    low_pr_auc = metrics_df[metrics_df['PR-AUC'] < 0.5]
    if not low_pr_auc.empty:
        print("\n⚠️  Classes com PR-AUC < 0.5 (difíceis de detectar):")
        for _, row in low_pr_auc.iterrows():
            print(f"  - {row['Classe']}: PR-AUC={row['PR-AUC']:.3f}, "
                  f"Support={row['Support']}")
        print("  → Considere ajustar thresholds específicos ou data augmentation")
    
    # Diferença entre ROC-AUC e PR-AUC
    metrics_df['AUC_diff'] = metrics_df['ROC-AUC'] - metrics_df['PR-AUC']
    high_diff = metrics_df[metrics_df['AUC_diff'] > 0.3]
    if not high_diff.empty:
        print("\n📊 Classes com grande diferença ROC-AUC vs PR-AUC:")
        for _, row in high_diff.iterrows():
            print(f"  - {row['Classe']}: ROC={row['ROC-AUC']:.3f}, "
                  f"PR={row['PR-AUC']:.3f} (diff={row['AUC_diff']:.3f})")
        print("  → Indica forte desbalanceamento - PR-AUC é mais confiável")
    
    # Recomendações de threshold
    print("\n💡 Recomendações para ajuste de thresholds:")
    for _, row in metrics_df.iterrows():
        if row['Support'] < 500:  # Classes raras
            print(f"  - {row['Classe']}: Considere threshold < 0.5 "
                  f"(classe rara com {row['Support']} exemplos)")
    
    return metrics_df

# ---------- Main com argparse ---------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning BERT multi-label (ToLD-BR) com splits persistentes.\n\nExemplos:\n  python train_gregor_samsa.py --train\n  python train_gregor_samsa.py --test\n  python train_gregor_samsa.py --train --validate\n",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--train', action='store_true', help='Treina o modelo (gera splits se necessário)')
    parser.add_argument('--test', action='store_true', help='Testa o modelo salvo')
    parser.add_argument('--validate', action='store_true', help='Usa validação durante o treino')
    args = parser.parse_args()

    if not (args.train or args.test):
        print("\n⚠️  Nenhum parâmetro passado! Use --train, --test ou ambos.\n")
        parser.print_help()
        return

    # Carregar ou gerar splits
    train_df = load_split('train')
    val_df   = load_split('val')
    test_df  = load_split('test')
    if train_df is None or test_df is None or (args.validate and val_df is None):
        print("Gerando splits e salvando em disco…")
        full_data = load_dataset(DATASET_PATH)
        train_df, val_df, test_df = split_stratified_holdout(full_data)
        save_split(train_df, 'train')
        save_split(val_df,   'val')
        save_split(test_df,  'test')
    else:
        print("Splits carregados do disco.")

    model = None
    if args.train:
        if args.validate:
            model = train_and_eval(train_df, val_df)
        else:
            # Treina sem validação
            model, _ = train(train_df, val_df)
        # Após treino, sempre testa
        print("\nAvaliação no conjunto de teste:")
        result, model_outputs, wrong_preds = model.eval_model(test_df)
        for k, v in result.items():
            print(f"  {k}: {v:.4f}")
        # Gera e salva as matrizes de confusão após o teste
        y_true = list(test_df['labels'])
        y_pred = (np.array(model_outputs) >= 0.5).astype(int) if isinstance(model_outputs, np.ndarray) or (hasattr(model_outputs, 'shape') and model_outputs is not None) else np.array(model_outputs)
        plot_multilabel_confusion(y_true, y_pred, LABELS)
        plot_cooccurrence_heatmap(y_true, y_pred, LABELS)
        plot_metrics_per_class(y_true, y_pred, LABELS)
        plot_pr_roc_curves(y_true, model_outputs, LABELS)
    if args.test and not args.train:
        # Só testar (carrega modelo salvo)
        model_dir = MODEL_DIR
        model_files = [os.path.join(model_dir, f) for f in ['config.json', 'model.safetensors']]
        if not (os.path.isdir(model_dir) and all(os.path.isfile(f) for f in model_files)):
            print("\n❌ Modelo não encontrado em 'outputs_bert/'.\nTreine o modelo primeiro usando --train antes de testar.")
            return
        # Carrega o modelo salvo usando o construtor padrão
        model = MultiLabelClassificationModel(
            "bert", model_dir, num_labels=NUM_LABELS, args=None, use_cuda=False
        )
        print("\nAvaliação no conjunto de teste:")
        result, model_outputs, wrong_preds = model.eval_model(test_df)
        for k, v in result.items():
            print(f"  {k}: {v:.4f}")
        # Gera e salva as matrizes de confusão após o teste
        y_true = list(test_df['labels'])
        y_pred = (np.array(model_outputs) >= 0.5).astype(int) if isinstance(model_outputs, np.ndarray) or (hasattr(model_outputs, 'shape') and model_outputs is not None) else np.array(model_outputs)
        plot_multilabel_confusion(y_true, y_pred, LABELS)
        plot_cooccurrence_heatmap(y_true, y_pred, LABELS)
        plot_metrics_per_class(y_true, y_pred, LABELS)
        plot_pr_roc_curves(y_true, model_outputs, LABELS)
        plot_fbeta_threshold_curves(y_true, model_outputs, LABELS)
        
if __name__ == "__main__":
    main()