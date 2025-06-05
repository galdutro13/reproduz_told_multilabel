# -*- coding: utf-8 -*-
"""
Fine-tuning BERT multi-label (ToLD-BR) - Versão Reestruturada

Reimplementação modular usando princípios de separação de responsabilidades.

Suporta:
- BCEWithLogitsLoss padrão (baseline)
- BCEWithLogitsLoss com pos_weight customizado
- Focal Loss com alpha automático ou manual
- Facilmente extensível para outras loss functions
- Execução de múltiplas configurações via arquivo JSON

Autor: Sistema Reestruturado
Data: 2024
"""

import os
import sys
import argparse
import logging
from typing import Optional

# Configurar ambiente antes dos imports principais
from src.config import setup_environment
setup_environment()

# Imports dos módulos
from src.config import ModelConfig, LossConfig, DATASET_PATH, SEED
from src.utils import LoggingSetup, SystemInfo, Timer, validate_requirements
from src.config_manager import BatchExecutor, create_simple_config_example
from src.data import DataPersistence, calculate_class_weights
from src.models import ModelFactory, ModelValidator
from src.training import TrainingManager
from src.metrics import DetailedMetricsAnalyzer, MetricsReporter
from src.visualization import VisualizationSuite

logger = logging.getLogger(__name__)

def create_argument_parser() -> argparse.ArgumentParser:
    """Cria parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="""
Fine-tuning BERT multi-label (ToLD-BR) - Versão Reestruturada

Reimplementação modular usando Hugging Face Transformers com separação
de responsabilidades para máxima flexibilidade e manutenibilidade.

Exemplos de uso:
  # Baseline (BCE padrão)
  python main.py --train --validate
  
  # Com pos_weight customizado
  python main.py --train --pos-weight "7.75,1.47,1.95,12.30,6.66,11.75"
  
  # Com Focal Loss
  python main.py --train --use-focal-loss
  
  # Executar múltiplas configurações
  python main.py --config configurator.json
  
  # Criar configuração de exemplo
  python main.py --create-example-config
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
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Tamanho do batch')
    parser.add_argument('--learning-rate', type=float, default=4e-5,
                       help='Taxa de aprendizado')
    parser.add_argument('--max-seq-length', type=int, default=128,
                       help='Comprimento máximo das sequências')
    
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
    parser.add_argument('--create-example-config', action='store_true',
                       help='Cria arquivo de configuração de exemplo')
    parser.add_argument('--system-info', action='store_true',
                       help='Mostra informações do sistema')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Nível de logging')
    parser.add_argument('--log-file', type=str,
                       help='Arquivo para salvar logs')
    
    return parser

def setup_logging_from_args(args):
    """Configura logging baseado nos argumentos."""
    LoggingSetup.setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )

def run_single_training(args) -> None:
    """Executa treinamento único baseado em argumentos."""
    logger.info("\n🚀 MODO DE TREINAMENTO ÚNICO")
    logger.info("="*50)
    
    # Criar configurações
    model_config = ModelConfig(
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        evaluate_during_training=args.validate,
        output_dir=args.output_dir,
        best_model_dir=os.path.join(args.output_dir, "best_model")  # Definir diretório do melhor modelo
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
    
    # Criar modelo
    with Timer("Criação do modelo"):
        model, tokenizer = ModelFactory.create_model_and_tokenizer(model_config)
        ModelValidator.validate_model_config(model, 6)  # 6 classes
        ModelValidator.test_model_forward(model, tokenizer)
    
    # Calcular alpha weights se necessário
    if loss_config.use_focal_loss:
        logger.info("🧮 Calculando pesos alpha para Focal Loss...")
        loss_config.focal_alpha_weights = calculate_class_weights(train_df)
    
    # Executar treinamento
    with Timer("Treinamento"):
        training_manager = TrainingManager(model_config, loss_config)
        
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
    
    # Avaliar usando o MELHOR modelo (não o último)
    with Timer("Avaliação"):
        test_metrics, test_probs = training_manager.evaluate(test_dataset, load_best_model=True)
        
        # Log métricas
        from src.metrics import MetricsReporter
        MetricsReporter.log_metrics_summary(test_metrics, "Resultados no Teste")
    
    # Análise detalhada
    with Timer("Análise detalhada"):
        import torch
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
    
    # NOVO: Copiar melhor modelo para diretório principal se existir
    best_model_dir = model_config.best_model_dir
    if os.path.exists(best_model_dir):
        logger.info(f"\n🏆 Copiando melhor modelo para diretório principal: {args.output_dir}")
        import shutil
        
        # Listar arquivos do melhor modelo
        best_model_files = os.listdir(best_model_dir)
        
        # Copiar cada arquivo
        for filename in best_model_files:
            src = os.path.join(best_model_dir, filename)
            dst = os.path.join(args.output_dir, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        
        # Log informações do melhor modelo se disponível
        metadata_path = os.path.join(best_model_dir, "best_model_metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                best_metadata = json.load(f)
            logger.info(f"📊 Melhor modelo salvo: step={best_metadata.get('best_checkpoint_step')}, "
                       f"{best_metadata.get('best_metric_name')}={best_metadata.get('best_metric_value'):.4f}")
    
    logger.info("\n✅ Treinamento único concluído com sucesso!")

def run_batch_execution(args) -> None:
    """Executa múltiplas configurações de um arquivo JSON."""
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

def run_model_testing(args) -> None:
    """Executa teste de modelo salvo."""
    logger.info("\n🧪 MODO DE TESTE")
    logger.info("="*50)
    
    # Verificar primeiro se existe o melhor modelo
    best_model_dir = os.path.join(args.output_dir, "best_model")
    model_path = os.path.join(args.output_dir, "pytorch_model.bin")
    best_model_path = os.path.join(best_model_dir, "pytorch_model.bin")
    
    # Determinar qual modelo usar
    if os.path.exists(best_model_path):
        logger.info(f"🏆 Melhor modelo encontrado em: {best_model_dir}")
        model_dir_to_use = best_model_dir
        
        # Carregar metadados do melhor modelo se disponível
        metadata_path = os.path.join(best_model_dir, "best_model_metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                best_metadata = json.load(f)
            logger.info(f"📊 Usando melhor modelo: step={best_metadata.get('best_checkpoint_step')}, "
                       f"{best_metadata.get('best_metric_name')}={best_metadata.get('best_metric_value'):.4f}")
    elif os.path.exists(model_path):
        logger.info(f"⚠️ Melhor modelo não encontrado. Usando modelo do diretório principal: {args.output_dir}")
        model_dir_to_use = args.output_dir
    else:
        logger.error("❌ Nenhum modelo encontrado. Execute o treinamento primeiro com --train")
        sys.exit(1)
    
    # Carregar modelo
    from src.training import ModelCheckpointer
    
    try:
        model, tokenizer, metadata = ModelCheckpointer.load_checkpoint(model_dir_to_use)
        logger.info("✅ Modelo carregado com sucesso")
        
        if metadata:
            logger.info(f"📊 Modelo salvo em: {metadata.get('saved_at', 'N/A')}")
            
            # Se há informações do melhor modelo nos metadados
            if 'best_model_info' in metadata:
                best_info = metadata['best_model_info']
                logger.info(f"🏆 Informações do melhor modelo:")
                logger.info(f"   Step: {best_info.get('best_checkpoint_step')}")
                logger.info(f"   {best_info.get('best_metric_name')}: {best_info.get('best_metric_value'):.4f}")
        
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
        config = ModelConfig(output_dir=args.output_dir)
        training_manager = TrainingManager(config)
        training_manager.trainer = training_manager.setup_trainer(
            model, tokenizer, test_dataset
        )
        
        # Avaliar sem tentar recarregar o melhor modelo (já estamos usando ele)
        test_metrics, test_probs = training_manager.evaluate(test_dataset, load_best_model=False)
        MetricsReporter.log_metrics_summary(test_metrics, "Resultados do Teste")
        
        # Análise detalhada adicional
        import torch
        import numpy as np
        from src.metrics import DetailedMetricsAnalyzer
        
        y_true = torch.stack([test_dataset[i]['labels'] for i in range(len(test_dataset))]).numpy()
        y_pred = (test_probs >= 0.5).astype(int)
        
        analyzer = DetailedMetricsAnalyzer()
        detailed_metrics = analyzer.calculate_detailed_metrics(y_true, y_pred, test_probs)
        
        # Gerar e salvar relatório
        report = MetricsReporter.generate_classification_report(y_true, y_pred)
        print(report)
        
        report_path = os.path.join(args.output_dir, "test_classification_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📄 Relatório de teste salvo em: {report_path}")
        
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
    logger.info("🤖 BERT MULTI-LABEL FINE-TUNING - VERSÃO REESTRUTURADA")
    logger.info("="*80)
    
    # Validar dependências
    if not validate_requirements():
        sys.exit(1)
    
    # Executar comando específico
    try:
        if args.system_info:
            SystemInfo.log_system_info()
            return
        
        if args.create_example_config:
            config_path = create_simple_config_example()
            logger.info(f"✅ Configuração de exemplo criada: {config_path}")
            return
        
        # Log informações do sistema
        SystemInfo.log_system_info()
        
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
    
    except KeyboardInterrupt:
        logger.info("\n⏹️ Execução interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Erro fatal: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    logger.info("\n🎉 Execução concluída!")

if __name__ == "__main__":
    main()