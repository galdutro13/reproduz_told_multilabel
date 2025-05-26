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
from sklearn.metrics import f1_score

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
    

    model = MultiLabelClassificationModel(
        "bert", MODEL_NAME, num_labels=NUM_LABELS, args=args, use_cuda=False
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
    métricas = {"macro_f1": macro_f1}

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
        plot_metrics_per_class(y_true, y_pred, LABELS)
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

if __name__ == "__main__":
    main()