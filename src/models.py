# -*- coding: utf-8 -*-
"""
Criação e configuração de modelos BERT para classificação multi-label.
"""

import torch
from typing import Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
import logging

from src.config import ModelConfig, NUM_LABELS

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory para criação de modelos BERT."""
    
    @staticmethod
    def create_model_and_tokenizer(config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Cria modelo BERT e tokenizer para classificação multi-label.
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: Modelo e tokenizer
        """
        logger.info(f"🤖 Criando modelo: {config.model_name}")
        
        # Configurar modelo
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
            cache_dir=config.cache_dir
        )
        
        # Criar modelo
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            config=model_config,
            cache_dir=config.cache_dir
        )
        
        # Criar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            do_lower_case=config.do_lower_case,
            cache_dir=config.cache_dir
        )
        
        # Otimizar modelo se possível
        model = ModelOptimizer.optimize_model(model)
        
        logger.info(f"✅ Modelo criado com {model.num_parameters():,} parâmetros")
        
        return model, tokenizer

class ModelOptimizer:
    """Classe para otimizações do modelo."""
    
    @staticmethod
    def optimize_model(model: PreTrainedModel) -> PreTrainedModel:
        """
        Aplica otimizações ao modelo.
        
        Args:
            model: Modelo a ser otimizado
            
        Returns:
            PreTrainedModel: Modelo otimizado
        """
        # Tentar torch.compile se disponível (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                optimized_model = torch.compile(model, backend='ipex')
                logger.info("🛠️ torch.compile ativado com backend ipex")
                return optimized_model
            except Exception as e:
                logger.warning(f"⚠️ torch.compile falhou: {e}")
                
                # Tentar backend padrão
                try:
                    optimized_model = torch.compile(model)
                    logger.info("🛠️ torch.compile ativado com backend padrão")
                    return optimized_model
                except Exception as e2:
                    logger.warning(f"⚠️ torch.compile com backend padrão falhou: {e2}")
        
        logger.info("📝 Usando modelo sem otimizações torch.compile")
        return model
    
    @staticmethod
    def get_model_info(model: PreTrainedModel) -> dict:
        """
        Retorna informações sobre o modelo.
        
        Args:
            model: Modelo para analisar
            
        Returns:
            dict: Informações do modelo
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': (total_params * 4) / (1024 * 1024),  # Assumindo float32
            'config': model.config.__dict__ if hasattr(model, 'config') else {}
        }
    
    @staticmethod
    def print_model_summary(model: PreTrainedModel, model_name: str = "Model"):
        """
        Imprime resumo do modelo.
        
        Args:
            model: Modelo para analisar
            model_name: Nome do modelo para display
        """
        info = ModelOptimizer.get_model_info(model)
        
        logger.info(f"\n📊 Resumo do {model_name}:")
        logger.info(f"  Total de parâmetros: {info['total_parameters']:,}")
        logger.info(f"  Parâmetros treináveis: {info['trainable_parameters']:,}")
        logger.info(f"  Tamanho estimado: {info['model_size_mb']:.1f} MB")
        
        if info['config']:
            logger.info(f"  Configuração:")
            for key, value in info['config'].items():
                if key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size']:
                    logger.info(f"    {key}: {value}")

class ModelValidator:
    """Validador para verificar consistência do modelo."""
    
    @staticmethod
    def validate_model_config(model: PreTrainedModel, expected_num_labels: int):
        """
        Valida se o modelo está configurado corretamente.
        
        Args:
            model: Modelo para validar
            expected_num_labels: Número esperado de labels
            
        Raises:
            ValueError: Se a configuração estiver incorreta
        """
        if not hasattr(model, 'config'):
            logger.warning("⚠️ Modelo não possui atributo 'config'")
            return
        
        config = model.config
        
        # Verificar número de labels
        if hasattr(config, 'num_labels') and config.num_labels != expected_num_labels:
            raise ValueError(
                f"Modelo configurado para {config.num_labels} labels, "
                f"mas esperado {expected_num_labels}"
            )
        
        # Verificar tipo de problema
        if hasattr(config, 'problem_type') and config.problem_type != "multi_label_classification":
            logger.warning(
                f"⚠️ Modelo configurado para '{config.problem_type}', "
                f"mas esperado 'multi_label_classification'"
            )
        
        logger.info("✅ Configuração do modelo validada")
    
    @staticmethod
    def test_model_forward(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                          sample_text: str = "Texto de teste"):
        """
        Testa forward pass do modelo.
        
        Args:
            model: Modelo para testar
            tokenizer: Tokenizer
            sample_text: Texto de exemplo para teste
        """
        try:
            # Tokenizar texto de exemplo
            inputs = tokenizer(
                sample_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Verificar shape dos logits
            expected_shape = (1, NUM_LABELS)
            actual_shape = outputs.logits.shape
            
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Shape dos logits incorreto: {actual_shape}, esperado {expected_shape}"
                )
            
            logger.info("✅ Teste de forward pass bem-sucedido")
            logger.info(f"   Input shape: {inputs['input_ids'].shape}")
            logger.info(f"   Output shape: {outputs.logits.shape}")
            
        except Exception as e:
            logger.error(f"❌ Erro no teste de forward pass: {e}")
            raise