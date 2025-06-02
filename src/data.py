# -*- coding: utf-8 -*-
"""
Gestão de dados, datasets e carregamento.
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
import torch
import warnings
from typing import List, Tuple, Union, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import logging

from src.config import LABELS, SEED

logger = logging.getLogger(__name__)

class MultiLabelDataset(Dataset):
    """Dataset para classificação multi-label."""
    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 tokenizer: PreTrainedTokenizer, max_length: int = 128):
        """
        Inicializa o dataset.
        
        Args:
            texts: Lista de textos
            labels: Lista de listas com labels binários
            tokenizer: Tokenizer do HuggingFace
            max_length: Comprimento máximo das sequências
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(texts) == len(labels), "Textos e labels devem ter o mesmo tamanho"
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retorna um item do dataset.
        
        Args:
            idx: Índice do item
            
        Returns:
            dict: Dicionário com input_ids, attention_mask e labels
        """
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenizar
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

class DataLoader:
    """Classe responsável pelo carregamento e processamento de dados."""
    
    @staticmethod
    def load_dataset(path: str) -> pd.DataFrame:
        """
        Carrega e normaliza o dataset a partir de CSV.
        
        Args:
            path: Caminho para o arquivo CSV
            
        Returns:
            pd.DataFrame: DataFrame processado com colunas 'text' e 'labels'
        """
        if not os.path.exists(path):
            sys.exit(f"❌ Arquivo {path} não encontrado.")
        
        try:
            df = pd.read_csv(path)
        except Exception as e:
            sys.exit(f"❌ Erro ao carregar {path}: {e}")
        
        # Validar colunas obrigatórias
        if "text" not in df.columns:
            sys.exit("❌ CSV deve conter a coluna 'text'.")
        
        missing_labels = set(LABELS) - set(df.columns)
        if missing_labels:
            sys.exit(f"❌ CSV deve conter as colunas de rótulo: {missing_labels}")
        
        # Normalizar labels para 0/1
        df[LABELS] = (df[LABELS].fillna(0).astype(float) > 0).astype(int)
        df["labels"] = df[LABELS].values.tolist()
        
        logger.info(f"✅ Dataset carregado: {len(df)} amostras")
        DataLoader._log_label_distribution(df)
        
        return df[["text", "labels"]]
    
    @staticmethod
    def _log_label_distribution(df: pd.DataFrame):
        """Log da distribuição de labels."""
        logger.info("Distribuição de labels:")
        for label in LABELS:
            count = df[label].sum()
            percentage = (count / len(df)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    @staticmethod
    def split_stratified_holdout(df: pd.DataFrame, seed: int = SEED,
                               train_ratio: float = 0.8, 
                               val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide o DataFrame em treino, validação e teste usando estratificação.
        
        Args:
            df: DataFrame original
            seed: Seed para reprodutibilidade
            train_ratio: Proporção para treino
            val_ratio: Proporção para validação
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test
        """
        test_ratio = 1.0 - train_ratio - val_ratio
        
        if test_ratio < 0:
            raise ValueError("train_ratio + val_ratio não pode ser > 1.0")
        
        try:
            return DataLoader._split_with_iterative_stratification(
                df, seed, train_ratio, val_ratio, test_ratio
            )
        except ImportError:
            logger.warning(
                "📦 Pacote 'iterative-stratification' não encontrado. "
                "Usando estratificação simples. Para melhor qualidade, "
                "instale: pip install iterative-stratification"
            )
            return DataLoader._split_with_sklearn(
                df, seed, train_ratio, val_ratio, test_ratio
            )
    
    @staticmethod
    def _split_with_iterative_stratification(df: pd.DataFrame, seed: int,
                                           train_ratio: float, val_ratio: float, 
                                           test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divisão usando iterative-stratification para multi-label."""
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        
        y = np.asarray(df["labels"].tolist())
        idx = np.arange(len(df))

        # Primeira divisão: train vs (val + test)
        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed
        )
        train_idx, temp_idx = next(msss1.split(idx, y))

        # Segunda divisão: val vs test
        y_temp = y[temp_idx]
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed,
        )
        val_rel, test_rel = next(msss2.split(temp_idx.reshape(-1, 1), y_temp))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        logger.info("✨ Estratificação multi-label concluída")
        return DataLoader._create_splits(df, train_idx, val_idx, test_idx)
    
    @staticmethod
    def _split_with_sklearn(df: pd.DataFrame, seed: int,
                          train_ratio: float, val_ratio: float, 
                          test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divisão usando sklearn com estratificação simples."""
        from sklearn.model_selection import train_test_split
        
        # Usar label mais frequente para estratificação
        y_single = df[LABELS].idxmax(axis=1)
        idx = np.arange(len(df))
        
        # Primeira divisão
        train_idx, temp_idx = train_test_split(
            idx, test_size=(val_ratio + test_ratio), 
            stratify=y_single, random_state=seed
        )
        
        # Segunda divisão
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_ratio / (val_ratio + test_ratio),
            stratify=y_single[temp_idx], random_state=seed
        )

        logger.info("📊 Estratificação simples concluída")
        return DataLoader._create_splits(df, train_idx, val_idx, test_idx)
    
    @staticmethod
    def _create_splits(df: pd.DataFrame, train_idx: np.ndarray, 
                      val_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cria os splits finais e log estatísticas."""
        d_train = df.iloc[train_idx].reset_index(drop=True)
        d_val = df.iloc[val_idx].reset_index(drop=True)
        d_test = df.iloc[test_idx].reset_index(drop=True)

        def _ratio(x):
            return f"{len(x):,} ({len(x)/len(df):.1%})"
        
        logger.info(f"📈 Divisões criadas:")
        logger.info(f"  Treino: {_ratio(d_train)}")
        logger.info(f"  Validação: {_ratio(d_val)}")
        logger.info(f"  Teste: {_ratio(d_test)}")
        
        return d_train, d_val, d_test

class DataPersistence:
    """Gerencia persistência de splits de dados."""
    
    @staticmethod
    def save_split(df: pd.DataFrame, name: str, base_path: str = "."):
        """
        Salva split em CSV.
        
        Args:
            df: DataFrame para salvar
            name: Nome do split ('train', 'val', 'test')
            base_path: Diretório base para salvar
        """
        path = os.path.join(base_path, f"split_{name}.csv")
        df.to_csv(path, index=False)
        logger.info(f"💾 Split '{name}' salvo em: {path}")

    @staticmethod
    def load_split(name: str, base_path: str = ".") -> Optional[pd.DataFrame]:
        """
        Carrega split de CSV.
        
        Args:
            name: Nome do split ('train', 'val', 'test')
            base_path: Diretório base para carregar
            
        Returns:
            pd.DataFrame ou None se não existir
        """
        path = os.path.join(base_path, f"split_{name}.csv")
        
        if not os.path.exists(path):
            return None
        
        try:
            df = pd.read_csv(path)
            
            # Converter coluna 'labels' de string para lista se necessário
            if 'labels' in df.columns and isinstance(df['labels'].iloc[0], str):
                df['labels'] = df['labels'].apply(ast.literal_eval)
            
            logger.info(f"📂 Split '{name}' carregado de: {path}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar split '{name}': {e}")
            return None
    
    @staticmethod
    def load_or_create_splits(dataset_path: str, base_path: str = ".", 
                            **split_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Carrega splits existentes ou cria novos.
        
        Args:
            dataset_path: Caminho para dataset original
            base_path: Diretório para buscar/salvar splits
            **split_kwargs: Argumentos para split_stratified_holdout
            
        Returns:
            Tuple com train, val, test DataFrames
        """
        # Tentar carregar splits existentes
        train_df = DataPersistence.load_split('train', base_path)
        val_df = DataPersistence.load_split('val', base_path)
        test_df = DataPersistence.load_split('test', base_path)
        
        if all(df is not None for df in [train_df, val_df, test_df]):
            logger.info("✅ Splits existentes carregados")
            return train_df, val_df, test_df
        
        # Criar novos splits
        logger.info("🔄 Criando novos splits...")
        full_data = DataLoader.load_dataset(dataset_path)
        train_df, val_df, test_df = DataLoader.split_stratified_holdout(full_data, **split_kwargs)
        
        # Salvar splits
        DataPersistence.save_split(train_df, 'train', base_path)
        DataPersistence.save_split(val_df, 'val', base_path)
        DataPersistence.save_split(test_df, 'test', base_path)
        
        return train_df, val_df, test_df

def calculate_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Calcula pesos alpha para cada classe baseado na frequência.
    
    Args:
        df: DataFrame com coluna 'labels'
        
    Returns:
        torch.Tensor: Pesos alpha para cada classe
    """
    label_matrix = np.array(df['labels'].tolist())
    pos_counts = label_matrix.sum(axis=0)
    neg_counts = len(df) - pos_counts
    alphas = neg_counts / len(df)
    
    logger.info("⚖️ Pesos alpha calculados por classe:")
    for i, label in enumerate(LABELS):
        logger.info(f"  {label}: α={alphas[i]:.3f} (pos={pos_counts[i]}, neg={neg_counts[i]})")
    
    return torch.tensor(alphas, dtype=torch.float32)