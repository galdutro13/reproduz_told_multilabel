# -*- coding: utf-8 -*-
"""
Fine-tuning BERT multi-label (ToLD-BR) - Versão Reestruturada com Suporte a GPU

Reimplementação modular usando princípios de separação de responsabilidades
com detecção automática e otimização para GPU.

Suporta:
- Detecção automática de GPU/CPU
- Otimizações específicas por dispositivo
- Batch sizes adaptativos baseados na memória GPU
- Precisão mista (FP16) automática
- Múltiplas GPUs
- BCEWithLogitsLoss padrão (baseline)
- BCEWithLogitsLoss com pos_weight customizado
- Focal Loss com alpha automático ou manual
- Facilmente extensível para outras loss functions
- Execução de múltiplas configurações via arquivo JSON

Autor: Sistema Reestruturado com Suporte GPU
Data: 2024
"""

import os
import sys
import argparse
import logging
import torch
from typing import Optional

# Configurar ambiente antes dos imports principais
from src.config import setup_environment, USE_CUDA, DEVICE
setup_environment()

# Imports dos módulos
from src.config import ModelConfig, LossConfig, DATASET_PATH, SEED
from src.utils import LoggingSetup, SystemInfo, Timer, validate_requirements
from src.config_manager import BatchExecutor, create_simple_config_example
from src.data import DataPersistence, calculate_class_weights
from src.models import ModelFactory, ModelValidator, GPUMemoryManager
from src.training import TrainingManager
from src.metrics import DetailedMetricsAnalyzer, MetricsReporter
from src.visualization import VisualizationSuite

logger = logging.getLogger(__name__)

def log_system_capabilities():
    """Log detalhado das capacidades do sistema."""
    logger.info("\n" + "="*80)
    logger.info("🔍 CAPACIDADES DO SISTEMA")
    logger.info("="*80)
    
    # Informações básicas do sistema
    SystemInfo.log_system_info()
    
    # Informações específicas de GPU
    if USE_CUDA:
        logger.info("\n🚀 CONFIGURAÇÃO GPU:")
        gpu_info = GPUMemoryManager.get_gpu_memory_info()
        logger.info(f"  Dispositivo principal: {gpu_info['device_name']}")
        logger.info(f"  Memória total: {gpu_info['total_memory_gb']:.1f} GB")
        logger.info(f"  Memória livre: {gpu_info['free_memory_gb']:.1f} GB")
        
        if torch.cuda.device_count() > 1:
            logger.info(f"  Múltiplas GPUs detectadas: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"    GPU {i}: {name} ({memory:.1f} GB)")
    else:
        logger.info("\n💻 CONFIGURAÇÃO CPU:")
        logger.info("  GPU não detectada - usando CPU")

def create_argument_parser() -> argparse.ArgumentParser:
    """Cria parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="""
Fine-tuning BERT multi-label (ToLD-BR) - Versão com Suporte GPU

Reimplementação modular com detecção automática de GPU/CPU e otimizações
específicas para cada dispositivo.

Exemplos de uso:
  # Baseline com auto-detecção de GPU
  python main.py --train --validate
  
  # Forçar uso de CPU mesmo com GPU disponível
  python main.py --train --force-cpu
  
  # Usar batch size específico (substitui auto-detecção)
  python main.py --train --batch-size 32
  
  # Com Focal Loss em GPU
  python main.py --train --use-focal-loss --validate
  
  # Executar múltiplas configurações
  python main.py --config configurator.json
  
  # Verificar compatibilidade GPU
  python main.py --check-gpu
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Ações principais
    parser.add_argument('--train', action='store_true', 
                       help='Treina o modelo')
    parser.add_argument('--test', action='store_true', 
                       help='Testa o modelo salvo')
    parser.add_argument('--validate', action='store_true', 
                       help='Usa validação durante treino')
    
    # Configurações do modelo
    parser.add_argument('--model-name', type=str, 
                       default='google-bert/bert-base-multilingual-cased',
                       help='Nome do modelo HuggingFace')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Número de épocas de treinamento')
    parser.add_argument('--batch-size', type=int, 
                       help='Tamanho do batch (auto-detectado se não especificado)')
    parser.add_argument('--learning-rate', type=float, default=4e-5,
                       help='Taxa de aprendizado')
    parser.add_argument('--max-seq-length', type=int, default=128,
                       help='Comprimento máximo das sequências')
    
    # Configurações de dispositivo
    parser.add_argument('--force-cpu', action='store_true',
                       help='Força uso de CPU mesmo com GPU disponível')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='ID da GPU para usar (padrão: 0)')
    parser.add_argument('--fp16', action='store_true',
                       help='Força uso de precisão mista FP16')
    parser.add_argument('--no-fp16', action='store_true',
                       help='Desabilita precisão mista FP16')
    
    # Configurações de loss
    parser.add_argument('--pos-weight', type=str, 
                       help='Pesos por classe (ex: "7.75,1.47,...")')
    parser.add_argument('--use-focal-loss', action='store_true', 
                       help='Usar Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Parâmetro gamma da Focal Loss')
    
    # Configurações de arquivo
    parser.add_argument('--config', type=str, 
                       help='Arquivo JSON com configurações múltiplas')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH,
                       help='Caminho para dataset CSV')
    parser.add_argument('--output-dir', type=str, default='outputs_bert',
                       help='Diretório de saída')
    
    # Utilitários
    parser.add_argument('--check-gpu', action='store_true',
                       help='Verifica compatibilidade e performance da GPU')
    parser.add_argument('--create-example-config', action='store_true',
                       help='Cria arquivo de configuração de exemplo')
    parser.add_argument('--system-info', action='store_true',
                       help='Mostra informações detalhadas do sistema')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Nível de logging')
    parser.add_argument('--log-file', type=str,
                       help='Arquivo para salvar logs')
    
    return parser

def setup_device_from_args(args):
    """Configura dispositivo baseado nos argumentos."""
    global USE_CUDA, DEVICE
    
    if args.force_cpu:
        USE_CUDA = False
        DEVICE = "cpu"
        # CORREÇÃO: Aplicar mudanças no módulo config também
        import src.config as config_module
        config_module.USE_CUDA = False
        config_module.DEVICE = "cpu"
        logger.info("💻 Forçando uso de CPU conforme solicitado")
    elif torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            logger.error(f"❌ GPU {args.gpu_id} não existe. GPUs disponíveis: 0-{torch.cuda.device_count()-1}")
            sys.exit(1)
        
        USE_CUDA = True
        DEVICE = f"cuda:{args.gpu_id}"
        torch.cuda.set_device(args.gpu_id)
        # CORREÇÃO: Aplicar mudanças no módulo config também
        import src.config as config_module
        config_module.USE_CUDA = True
        config_module.DEVICE = DEVICE
        logger.info(f"🚀 Usando GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        USE_CUDA = False
        DEVICE = "cpu"
        # CORREÇÃO: Aplicar mudanças no módulo config também
        import src.config as config_module
        config_module.USE_CUDA = False
        config_module.DEVICE = "cpu"
        logger.info("💻 GPU não disponível - usando CPU")

def setup_logging_from_args(args):
    """Configura logging baseado nos argumentos."""
    LoggingSetup.setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )

def run_gpu_check():
    """Executa verificação de GPU."""
    logger.info("\n🔍 EXECUTANDO VERIFICAÇÃO DE GPU")
    logger.info("="*50)
    
    # Importar e executar script de verificação
    try:
        import subprocess
        result = subprocess.run([sys.executable, "check_gpu_compatibility.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(result.stderr)
            
    except Exception as e:
        logger.error(f"❌ Erro ao executar verificação de GPU: {e}")
        logger.info("Execute manualmente: python check_gpu_compatibility.py")

def run_single_training(args) -> None:
    """Executa treinamento único baseado em argumentos com suporte a GPU."""
    logger.info("\n🚀 MODO DE TREINAMENTO ÚNICO")
    logger.info("="*50)
    
    # Log memória inicial se GPU
    if USE_CUDA:
        GPUMemoryManager.log_gpu_memory_usage("inicial")
    
    # Determinar configurações automáticas baseadas no dispositivo
    auto_batch_size = None
    auto_fp16 = USE_CUDA
    
    # Sobrescrever com argumentos do usuário se fornecidos
    if args.batch_size:
        auto_batch_size = args.batch_size
    elif USE_CUDA:
        from src.config import get_optimal_batch_size
        auto_batch_size = get_optimal_batch_size(args.model_name, args.max_seq_length)
        logger.info(f"🎯 Batch size auto-detectado: {auto_batch_size}")
    else:
        auto_batch_size = 8  # Conservador para CPU
    
    # Configuração FP16
    if args.fp16:
        auto_fp16 = True
    elif args.no_fp16:
        auto_fp16 = False
    
    logger.info(f"⚡ Precisão mista (FP16): {'Ativada' if auto_fp16 else 'Desativada'}")
    
    # Criar configurações
    model_config = ModelConfig(
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=auto_batch_size,
        per_device_eval_batch_size=auto_batch_size * 2,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        evaluate_during_training=args.validate,
        output_dir=args.output_dir,
        use_cuda=USE_CUDA,
        fp16=auto_fp16,
    )
    
    loss_config = LossConfig(
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma
    )
    
    # Configurar pos_weight se fornecido
    if args.pos_weight:
        try:
            loss_config.pos_weight = [float(x.strip()) for x in args.pos_weight.split(',')]
            logger.info(f"⚖️ Pos weights configurados: {loss_config.pos_weight}")
        except ValueError as e:
            logger.error(f"❌ Erro ao parsear pos_weight: {e}")
            sys.exit(1)
    
    # Validar configurações
    loss_config.validate()
    
    # Carregar dados
    with Timer("Carregamento de dados"):
        train_df, val_df, test_df = DataPersistence.load_or_create_splits(args.dataset)
    
    # Log uso de memória após carregamento de dados
    if USE_CUDA:
        GPUMemoryManager.log_gpu_memory_usage("após carregamento de dados")
    
    # Criar modelo
    with Timer("Criação e configuração do modelo"):
        model, tokenizer = ModelFactory.create_model_and_tokenizer(model_config)
        ModelValidator.validate_model_config(model, 6)  # 6 classes
        ModelValidator.test_model_forward(model, tokenizer)
    
    # Log uso de memória após criação do modelo
    if USE_CUDA:
        GPUMemoryManager.log_gpu_memory_usage("após criação do modelo")
    
    # Calcular alpha weights se necessário
    if loss_config.use_focal_loss:
        logger.info("🧮 Calculando pesos alpha para Focal Loss...")
        loss_config.focal_alpha_weights = calculate_class_weights(train_df)
    
    # Configurar para múltiplas GPUs se disponível
    if USE_CUDA and torch.cuda.device_count() > 1:
        from src.models import MultiGPUManager
        model = MultiGPUManager.setup_multi_gpu(model)
    
    # Executar treinamento
    with Timer("Treinamento"):
        training_manager = TrainingManager(
            model_config, 
            loss_config, 
            use_contiguity_fix=USE_CUDA  # Usar correção de contiguidade para GPU
        )
        
        # Criar datasets
        from src.data import MultiLabelDataset
        train_dataset = MultiLabelDataset(
            train_df['text'].tolist(),
            train_df['labels'].tolist(),
            tokenizer,
            model_config.max_seq_length
        )
        
        val_dataset = None
        if model_config.evaluate_during_training:
            val_dataset = MultiLabelDataset(
                val_df['text'].tolist(),
                val_df['labels'].tolist(),
                tokenizer,
                model_config.max_seq_length
            )
        
        test_dataset = MultiLabelDataset(
            test_df['text'].tolist(),
            test_df['labels'].tolist(),
            tokenizer,
            model_config.max_seq_length
        )
        
        # Treinar
        trainer = training_manager.setup_trainer(model, tokenizer, train_dataset, val_dataset)
        train_result, training_history = training_manager.train()
    
    # Log uso de memória após treinamento
    if USE_CUDA:
        GPUMemoryManager.log_gpu_memory_usage("após treinamento")
    
    # Avaliar
    with Timer("Avaliação"):
        test_metrics, test_probs = training_manager.evaluate(test_dataset)
        
        # Log métricas
        from src.metrics import MetricsReporter
        MetricsReporter.log_metrics_summary(test_metrics, "Resultados no Teste")
    
    # Análise detalhada
    with Timer("Análise detalhada"):
        import numpy as np
        
        y_true = torch.stack([test_dataset[i]['labels'] for i in range(len(test_dataset))]).numpy()
        y_pred = (test_probs >= 0.5).astype(int)
        
        analyzer = DetailedMetricsAnalyzer()
        detailed_metrics = analyzer.calculate_detailed_metrics(y_true, y_pred, test_probs)
        
        # Gerar relatório
        report = MetricsReporter.generate_classification_report(y_true, y_pred)
        print(report)
        
        # Salvar relatório
        report_path = os.path.join(args.output_dir, "classification_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📄 Relatório salvo em: {report_path}")
    
    # Gerar visualizações
    with Timer("Geração de visualizações"):
        viz_suite = VisualizationSuite(os.path.join(args.output_dir, "visualizations"))
        optimal_thresholds = viz_suite.generate_all_plots(
            training_history, y_true, y_pred, test_probs, 
            detailed_metrics['per_class_metrics']
        )
    
    # Log final de memória
    if USE_CUDA:
        GPUMemoryManager.log_gpu_memory_usage("final")
        GPUMemoryManager.clear_gpu_cache()
    
    logger.info("\n✅ Treinamento único concluído com sucesso!")

def run_batch_execution(args) -> None:
    """Executa múltiplas configurações de um arquivo JSON com suporte a GPU."""
    logger.info("\n🚀 MODO DE EXECUÇÃO EM LOTE")
    logger.info("="*50)
    
    if not os.path.exists(args.config):
        logger.error(f"❌ Arquivo de configuração não encontrado: {args.config}")
        sys.exit(1)
    
    with Timer(f"Execução em lote ({args.config})"):
        batch_executor = BatchExecutor(args.dataset)
        results = batch_executor.execute_config_file(args.config)
    
    # Resumo final
    successful = sum(1 for _, result in results if 'error' not in result)
    total = len(results)
    
    logger.info(f"\n📊 RESUMO FINAL:")
    logger.info(f"Total de instâncias: {total}")
    logger.info(f"Bem-sucedidas: {successful}")
    logger.info(f"Com erro: {total - successful}")
    
    if successful > 0:
        # Encontrar melhor modelo
        best_model = max(
            (result for _, result in results if 'error' not in result),
            key=lambda x: x.get('test_metrics', {}).get('test_macro_f1', 0)
        )
        best_f1 = best_model.get('test_metrics', {}).get('test_macro_f1', 0)
        best_id = best_model.get('instance_id', 'N/A')
        
        logger.info(f"🏆 Melhor modelo: {best_id} (F1-Macro: {best_f1:.4f})")
    
    # Limpeza final de memória
    if USE_CUDA:
        GPUMemoryManager.clear_gpu_cache()

def run_model_testing(args) -> None:
    """Executa teste de modelo salvo com suporte a GPU."""
    logger.info("\n🧪 MODO DE TESTE")
    logger.info("="*50)
    
    model_path = os.path.join(args.output_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        logger.error("❌ Modelo não encontrado. Execute o treinamento primeiro com --train")
        sys.exit(1)
    
    # Log memória inicial se GPU
    if USE_CUDA:
        GPUMemoryManager.log_gpu_memory_usage("antes do carregamento do modelo")
    
    # Carregar modelo
    from src.training import ModelCheckpointer
    
    try:
        model, tokenizer, metadata = ModelCheckpointer.load_checkpoint(args.output_dir)
        
        # Mover modelo para GPU se disponível
        if USE_CUDA:
            model = model.to(DEVICE)
            logger.info(f"🚀 Modelo carregado e movido para: {DEVICE}")
        
        logger.info("✅ Modelo carregado com sucesso")
        
        if metadata:
            logger.info(f"📊 Modelo salvo em: {metadata.get('saved_at', 'N/A')}")
        
        # Log memória após carregamento
        if USE_CUDA:
            GPUMemoryManager.log_gpu_memory_usage("após carregamento do modelo")
        
        # Carregar dados de teste
        _, _, test_df = DataPersistence.load_or_create_splits(args.dataset)
        
        # Criar dataset de teste
        from src.data import MultiLabelDataset
        test_dataset = MultiLabelDataset(
            test_df['text'].tolist(),
            test_df['labels'].tolist(),
            tokenizer,
            args.max_seq_length
        )
        
        # Avaliar
        config = ModelConfig(
            output_dir=args.output_dir,
            use_cuda=USE_CUDA,
            fp16=USE_CUDA and not args.no_fp16
        )
        training_manager = TrainingManager(config)
        training_manager.trainer = training_manager.setup_trainer(
            model, tokenizer, test_dataset
        )
        
        test_metrics, test_probs = training_manager.evaluate(test_dataset)
        MetricsReporter.log_metrics_summary(test_metrics, "Resultados do Teste")
        
        # Limpeza de memória
        if USE_CUDA:
            GPUMemoryManager.clear_gpu_cache()
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar modelo: {e}")
        sys.exit(1)

def main():
    """Função principal."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging_from_args(args)
    
    # Header
    logger.info("\n" + "="*80)
    logger.info("🤖 BERT MULTI-LABEL FINE-TUNING - VERSÃO COM SUPORTE GPU")
    logger.info("="*80)
    
    # Configurar dispositivo baseado nos argumentos
    setup_device_from_args(args)
    
    # Validar dependências
    if not validate_requirements():
        sys.exit(1)
    
    # Executar comando específico
    try:
        if args.check_gpu:
            run_gpu_check()
            return
        
        if args.system_info:
            log_system_capabilities()
            return
        
        if args.create_example_config:
            config_path = create_simple_config_example()
            logger.info(f"✅ Configuração de exemplo criada: {config_path}")
            return
        
        # Log informações do sistema
        log_system_capabilities()
        
        # Aviso sobre configurações de GPU
        if USE_CUDA:
            logger.info("\n🔥 CONFIGURAÇÕES DE GPU ATIVAS:")
            logger.info(f"  Dispositivo: {DEVICE}")
            logger.info(f"  Precisão mista (FP16): {'Ativada' if not args.no_fp16 else 'Desativada'}")
            
            if torch.cuda.device_count() > 1:
                logger.info(f"  Múltiplas GPUs: {torch.cuda.device_count()} disponíveis")
            
            # Mostrar configurações recomendadas
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 8:
                logger.warning("⚠️ GPU com pouca memória detectada. Considere:")
                logger.warning("  - Reduzir batch_size")
                logger.warning("  - Usar --fp16 para economia de memória")
                logger.warning("  - Reduzir max_seq_length se possível")
        
        # Modos de execução
        if args.config:
            run_batch_execution(args)
        elif args.train:
            run_single_training(args)
        elif args.test:
            run_model_testing(args)
        else:
            parser.print_help()
            logger.warning("⚠️ Nenhuma ação especificada. Use --help para ver as opções.")
            logger.info("\n💡 Dicas rápidas:")
            logger.info("  - Execute 'python main.py --check-gpu' para verificar GPU")
            logger.info("  - Execute 'python main.py --train --validate' para treinamento básico")
            logger.info(f"  - Dispositivo detectado: {DEVICE}")
    
    except KeyboardInterrupt:
        logger.info("\n⏹️ Execução interrompida pelo usuário")
        if USE_CUDA:
            GPUMemoryManager.clear_gpu_cache()
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Erro fatal: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        
        # Limpeza de memória em caso de erro
        if USE_CUDA:
            GPUMemoryManager.clear_gpu_cache()
        sys.exit(1)
    
    logger.info("\n🎉 Execução concluída!")

if __name__ == "__main__":
    main()