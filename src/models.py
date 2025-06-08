# -*- coding: utf-8 -*-
"""
Criação e configuração de modelos BERT para classificação multi-label com suporte a GPU.
VERSÃO CORRIGIDA - Remove conflito entre torch.compile e FP16
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

from src.config import ModelConfig, NUM_LABELS, USE_CUDA, DEVICE

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """Gerenciador de memória GPU."""
    
    @staticmethod
    def clear_gpu_cache():
        """Limpa cache da GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cache da GPU limpo")
    
    @staticmethod
    def get_gpu_memory_info() -> dict:
        """Retorna informações sobre uso de memória GPU."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(device),
            "total_memory_gb": total_memory / 1e9,
            "allocated_memory_gb": allocated_memory / 1e9,
            "cached_memory_gb": cached_memory / 1e9,
            "free_memory_gb": (total_memory - allocated_memory) / 1e9,
            "memory_usage_percent": (allocated_memory / total_memory) * 100
        }
    
    @staticmethod
    def log_gpu_memory_usage(stage: str = ""):
        """Log detalhado do uso de memória GPU."""
        info = GPUMemoryManager.get_gpu_memory_info()
        if info["available"]:
            logger.info(f"🔍 Memória GPU {stage}:")
            logger.info(f"  Dispositivo: {info['device_name']}")
            logger.info(f"  Total: {info['total_memory_gb']:.1f} GB")
            logger.info(f"  Alocada: {info['allocated_memory_gb']:.1f} GB ({info['memory_usage_percent']:.1f}%)")
            logger.info(f"  Cache: {info['cached_memory_gb']:.1f} GB")
            logger.info(f"  Livre: {info['free_memory_gb']:.1f} GB")

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
    """Factory para criação de modelos BERT com suporte automático a GPU."""
    
    @staticmethod
    def create_model_and_tokenizer(config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Cria modelo BERT e tokenizer para classificação multi-label com suporte a GPU.
        
        Args:
            config: Configuração do modelo
            
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: Modelo e tokenizer
        """
        logger.info(f"🤖 Criando modelo: {config.model_name}")
        logger.info(f"📱 Dispositivo alvo: {DEVICE}")
        
        # Log memória GPU inicial
        GPUMemoryManager.log_gpu_memory_usage("antes da criação do modelo")
        
        # APLICAR PATCH DE CONTIGUIDADE
        ContiguityFixer.patch_trainer_save_method()
        
        # Configurar modelo
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
            cache_dir=config.cache_dir
        )
        
        # Configurações específicas para GPU
        if USE_CUDA:
            model_config.use_cache = False  # Economiza memória durante treinamento
        
        # Criar modelo com dtype correto
        torch_dtype = torch.float16 if USE_CUDA and config.fp16 else torch.float32
        
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            config=model_config,
            cache_dir=config.cache_dir,
            torch_dtype=torch_dtype,
        )
        
        # Mover modelo para GPU se disponível
        if USE_CUDA:
            model = model.to(DEVICE)
            logger.info(f"🚀 Modelo movido para GPU: {DEVICE}")
        
        # Criar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            do_lower_case=config.do_lower_case,
            cache_dir=config.cache_dir
        )
        
        # Otimizar modelo
        model = ModelOptimizer.optimize_model(model, config)
        
        # Log memória GPU após criação
        GPUMemoryManager.log_gpu_memory_usage("após criação do modelo")
        
        logger.info(f"✅ Modelo criado com {model.num_parameters():,} parâmetros")
        
        return model, tokenizer

class ModelOptimizer:
    """Classe para otimizações do modelo com suporte a GPU."""
    
    @staticmethod
    def optimize_model(model: PreTrainedModel, config: ModelConfig) -> PreTrainedModel:
        """
        Aplica otimizações ao modelo baseado no dispositivo disponível.
        VERSÃO CORRIGIDA: Evita conflito entre torch.compile e FP16
        
        Args:
            model: Modelo a ser otimizado
            config: Configuração do modelo
            
        Returns:
            PreTrainedModel: Modelo otimizado
        """
        if USE_CUDA:
            return ModelOptimizer._optimize_for_gpu(model, config)
        else:
            return ModelOptimizer._optimize_for_cpu(model)
    
    @staticmethod
    def _optimize_for_gpu(model: PreTrainedModel, config: ModelConfig) -> PreTrainedModel:
        """Otimizações específicas para GPU - VERSÃO CORRIGIDA."""
        logger.info("🚀 Aplicando otimizações para GPU...")
        
        # ===================================================================
        # CORREÇÃO: Evitar torch.compile quando FP16 está ativo
        # ===================================================================
        
        if config.fp16:
            logger.info("⚡ FP16 ativo - torch.compile desabilitado para evitar conflitos")
            logger.info("💾 Gradient checkpointing ativado para economia de memória")
            
            # Ativar gradient checkpointing para economia de memória com FP16
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                
        else:
            # Usar torch.compile apenas quando FP16 está desabilitado
            if hasattr(torch, 'compile'):
                try:
                    optimized_model = torch.compile(
                        model,
                        mode="reduce-overhead",
                        dynamic=True,  # Mudança: dynamic=True para lidar com batch sizes variáveis
                        fullgraph=False
                    )
                    logger.info("🛠️ torch.compile ativado (FP16 desabilitado)")
                    return optimized_model
                except Exception as e:
                    logger.warning(f"⚠️ torch.compile falhou: {e}")
        
        return model
    
    @staticmethod
    def _optimize_for_cpu(model: PreTrainedModel) -> PreTrainedModel:
        """Otimizações específicas para CPU."""
        logger.info("💻 Aplicando otimizações para CPU...")
        
        # Tentar usar IPEX se disponível
        try:
            import intel_extension_for_pytorch as ipex
            from packaging.version import parse as V
            import transformers
            
            if V("4.6.0") <= V(transformers.__version__) <= V("4.48.0"):
                model = ipex.optimize(
                    model,
                    dtype=torch.float32,
                    inplace=True,
                    conv_bn_folding=False,
                    linear_bn_folding=False,
                    auto_kernel_selection=True,
                )
                logger.info("🚀 IPEX otimizado (FP32)")
            else:
                logger.info("ℹ️ IPEX pulado (versão transformers não compatível)")
        except ImportError:
            logger.info("ℹ️ IPEX não disponível - usando PyTorch puro")
        except Exception as e:
            logger.warning(f"IPEX falhou, prosseguindo: {e}")
        
        return model
    
    @staticmethod
    def get_model_info(model: PreTrainedModel) -> dict:
        """
        Retorna informações sobre o modelo incluindo uso de GPU.
        
        Args:
            model: Modelo para analisar
            
        Returns:
            dict: Informações do modelo
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Verificar se modelo foi compilado
        is_compiled = hasattr(model, '_orig_mod')
        
        # Verificar dispositivo do modelo
        device = next(model.parameters()).device if len(list(model.parameters())) > 0 else 'unknown'
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': (total_params * 4) / (1024 * 1024),  # Assumindo float32
            'is_compiled': is_compiled,
            'device': str(device),
            'is_on_gpu': device.type == 'cuda' if hasattr(device, 'type') else False,
            'config': model.config.__dict__ if hasattr(model, 'config') else {}
        }
        
        return info
    
    @staticmethod
    def print_model_summary(model: PreTrainedModel, model_name: str = "Model"):
        """
        Imprime resumo do modelo incluindo informações de GPU.
        
        Args:
            model: Modelo para analisar
            model_name: Nome do modelo para display
        """
        info = ModelOptimizer.get_model_info(model)
        
        logger.info(f"\n📊 Resumo do {model_name}:")
        logger.info(f"  Total de parâmetros: {info['total_parameters']:,}")
        logger.info(f"  Parâmetros treináveis: {info['trainable_parameters']:,}")
        logger.info(f"  Tamanho estimado: {info['model_size_mb']:.1f} MB")
        logger.info(f"  Dispositivo: {info['device']}")
        logger.info(f"  Modelo compilado: {'✅ Sim' if info['is_compiled'] else '❌ Não'}")
        logger.info(f"  Na GPU: {'✅ Sim' if info['is_on_gpu'] else '❌ Não'}")
        
        if info['config']:
            logger.info(f"  Configuração:")
            for key, value in info['config'].items():
                if key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size']:
                    logger.info(f"    {key}: {value}")

class ModelValidator:
    """Validador para verificar consistência do modelo com suporte a GPU."""
    
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
        
        # Verificar se modelo está no dispositivo correto
        model_device = next(model.parameters()).device
        if USE_CUDA and model_device.type != 'cuda':
            logger.warning(f"⚠️ Modelo está em {model_device} mas CUDA está disponível")
        elif not USE_CUDA and model_device.type == 'cuda':
            logger.warning(f"⚠️ Modelo está em GPU mas CUDA não foi configurado para uso")
        
        logger.info("✅ Configuração do modelo validada")
    
    @staticmethod
    def test_model_forward(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                          sample_text: str = "Texto de teste"):
        """
        Testa forward pass do modelo com suporte a GPU.
        
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
            
            # Mover inputs para o mesmo dispositivo do modelo se necessário
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
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
            logger.info(f"   Dispositivo: {device}")
            
        except Exception as e:
            logger.error(f"❌ Erro no teste de forward pass: {e}")
            raise

class MultiGPUManager:
    """Gerenciador para treinamento com múltiplas GPUs."""
    
    @staticmethod
    def setup_multi_gpu(model: PreTrainedModel) -> PreTrainedModel:
        """
        Configura modelo para uso com múltiplas GPUs.
        
        Args:
            model: Modelo a ser configurado
            
        Returns:
            PreTrainedModel: Modelo configurado para multi-GPU
        """
        if torch.cuda.device_count() > 1:
            logger.info(f"🔗 Configurando modelo para {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
            logger.info("✅ DataParallel ativado")
        
        return model
    
    @staticmethod
    def get_gpu_info() -> dict:
        """Retorna informações sobre todas as GPUs disponíveis."""
        if not torch.cuda.is_available():
            return {"gpu_count": 0, "gpus": []}
        
        gpu_count = torch.cuda.device_count()
        gpus = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i,
                "name": props.name,
                "memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return {"gpu_count": gpu_count, "gpus": gpus}

# Classe alternativa com correção integrada
class ContiguousMultiLabelTrainer:
    """
    Trainer que já vem com correção de contiguidade integrada para uso com GPU.
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