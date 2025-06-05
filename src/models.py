# -*- coding: utf-8 -*-
"""
Criação e configuração de modelos BERT para classificação multi-label.
VERSÃO COM TORCH.COMPILE MANTIDO + CORREÇÃO DE CONTIGUIDADE
"""

import torch
from typing import Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer
)
import logging

from src.config import ModelConfig, NUM_LABELS

logger = logging.getLogger(__name__)

class ContiguityFixer:
    """Classe para corrigir problemas de contiguidade em modelos compilados."""
    
    @staticmethod
    def make_state_dict_contiguous(state_dict: dict) -> dict:
        """
        Torna todos os tensores de um state_dict contíguos.
        
        Args:
            state_dict: Dicionário de parâmetros do modelo
            
        Returns:
            dict: State dict com tensores contíguos
        """
        contiguous_state_dict = {}
        non_contiguous_count = 0
        
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                if not tensor.is_contiguous():
                    contiguous_state_dict[key] = tensor.contiguous()
                    non_contiguous_count += 1
                    logger.debug(f"Tensor tornado contíguo: {key}")
                else:
                    contiguous_state_dict[key] = tensor
            else:
                contiguous_state_dict[key] = tensor
        
        if non_contiguous_count > 0:
            logger.info(f"🔧 {non_contiguous_count} tensores tornados contíguos para salvamento")
        
        return contiguous_state_dict
    
    @staticmethod
    def patch_trainer_save_method():
        """
        Aplica um patch global ao Trainer para corrigir automaticamente
        problemas de contiguidade durante salvamentos.
        """
        # Salva o método original
        original_save = Trainer._save
        
        def patched_save(self, output_dir=None, state_dict=None):
            """Versão corrigida do _save que torna tensores contíguos"""
            if state_dict is None:
                state_dict = self.model.state_dict()
            
            # Torna todos os tensores contíguos
            contiguous_state_dict = ContiguityFixer.make_state_dict_contiguous(state_dict)
            
            # Chama o método original com tensores corrigidos
            return original_save(self, output_dir, contiguous_state_dict)
        
        # Aplica o patch
        Trainer._save = patched_save
        logger.info("✅ Patch de contiguidade aplicado ao Trainer")

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
        
        # APLICAR PATCH DE CONTIGUIDADE ANTES DE QUALQUER COISA
        ContiguityFixer.patch_trainer_save_method()
        
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
        
        # Otimizar modelo com torch.compile (MANTIDO!)
        model = ModelOptimizer.optimize_model(model)
        
        logger.info(f"✅ Modelo criado com {model.num_parameters():,} parâmetros")
        
        return model, tokenizer

class ModelOptimizer:
    """Classe para otimizações do modelo."""
    
    @staticmethod
    def optimize_model(model: PreTrainedModel) -> PreTrainedModel:
        """
        Aplica otimizações ao modelo MANTENDO torch.compile.
        
        Args:
            model: Modelo a ser otimizado
            
        Returns:
            PreTrainedModel: Modelo otimizado
        """
        # ============================================================
        # TORCH.COMPILE MANTIDO COM CORREÇÃO DE CONTIGUIDADE
        # ============================================================
        
        if hasattr(torch, 'compile'):
            try:
                # Tentar com backend mais estável primeiro
                optimized_model = torch.compile(
                    model, 
                    backend='inductor',  # Backend mais estável que 'ipex'
                    dynamic=False,       # Evita problemas de forma dinâmica
                    fullgraph=False      # Permite fallback para partes não compiláveis
                )
                logger.info("🛠️ torch.compile ativado com backend inductor (estável)")
                logger.info("✅ Correção de contiguidade ativa via patch do Trainer")
                return optimized_model
                
            except Exception as e:
                logger.warning(f"⚠️ torch.compile com inductor falhou: {e}")
                
                # Fallback para backend padrão
                try:
                    optimized_model = torch.compile(model, dynamic=False, fullgraph=False)
                    logger.info("🛠️ torch.compile ativado com backend padrão")
                    logger.info("✅ Correção de contiguidade ativa via patch do Trainer")
                    return optimized_model
                    
                except Exception as e2:
                    logger.warning(f"⚠️ torch.compile com backend padrão falhou: {e2}")
                    
                    # Se torch.compile falhar completamente, usar modelo normal
                    logger.info("📝 Usando modelo sem torch.compile")
                    return model
        else:
            logger.info("📝 torch.compile não disponível nesta versão do PyTorch")
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
        
        # Verificar se modelo foi compilado
        is_compiled = hasattr(model, '_orig_mod')
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': (total_params * 4) / (1024 * 1024),  # Assumindo float32
            'is_compiled': is_compiled,
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
        logger.info(f"  Modelo compilado: {'✅ Sim' if info['is_compiled'] else '❌ Não'}")
        
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

# ===================================================================
# CLASSE ALTERNATIVA: TRAINER CUSTOMIZADO COM CORREÇÃO INTEGRADA
# ===================================================================

class ContiguousMultiLabelTrainer:
    """
    Alternativa: Trainer que já vem com correção de contiguidade integrada.
    Use esta classe em vez do patch global se preferir.
    """
    
    def __init__(self, *args, **kwargs):
        from src.training import MultiLabelTrainer
        super().__init__(*args, **kwargs)
    
    def _save(self, output_dir=None, state_dict=None):
        """Override do _save com correção de contiguidade."""
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        # Aplicar correção de contiguidade
        contiguous_state_dict = ContiguityFixer.make_state_dict_contiguous(state_dict)
        
        # Chama o método pai com state_dict corrigido
        super()._save(output_dir, contiguous_state_dict)
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Override do save_model com correção de contiguidade."""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"💾 Salvando modelo em {output_dir} (com correção de contiguidade)")
        
        # Usa nosso _save customizado
        self._save(output_dir)
        
        # Salva o tokenizer se disponível
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)