# -*- coding: utf-8 -*-
"""
Gerenciamento de configurações múltiplas para execução em lote.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import logging
from datetime import datetime

from src.config import ModelConfig, LossConfig, DATASET_PATH, NUM_LABELS # IMPORTAR NUM_LABELS
# Adicionada vírgula e NUM_LABELS acima

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Gerenciador de configurações múltiplas."""
    
    @staticmethod
    def load_config_file(config_path: str) -> Dict[str, Any]:
        """
        Carrega arquivo de configuração JSON.
        
        Args:
            config_path: Caminho para arquivo JSON
            
        Returns:
            Dict: Configuração carregada
            
        Raises:
            SystemExit: Se arquivo não existir ou for inválido
        """
        if not os.path.exists(config_path):
            sys.exit(f"❌ Arquivo de configuração '{config_path}' não encontrado.")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"❌ Erro ao parsear JSON: {e}")
        except Exception as e:
            sys.exit(f"❌ Erro ao ler arquivo: {e}")
        
        if 'instances' not in config:
            sys.exit("❌ Arquivo de configuração deve conter chave 'instances'.")
        
        logger.info(f"✅ Configuração carregada: {len(config['instances'])} instâncias")
        return config
    
    @staticmethod
    def parse_instance_config(instance: Dict[str, Any], instance_num: int) -> tuple[ModelConfig, LossConfig]:
        """
        Converte configuração de instância para objetos de configuração.
        
        Args:
            instance: Dicionário com configuração da instância
            instance_num: Número da instância para logging
            
        Returns:
            Tuple[ModelConfig, LossConfig]: Configurações parseadas
            
        Raises:
            SystemExit: Se configuração for inválida
        """
        params = instance.get('parameters', {})
        
        # Validação obrigatória
        if 'model_name' not in params:
            sys.exit(f"❌ Instância {instance_num}: 'model_name' é obrigatório.")
        
        # Criar configuração do modelo
        model_config = ModelConfig(
            model_name=params['model_name'],
            num_train_epochs=params.get('epochs', 3),
            max_seq_length=params.get('max_seq_length', 128),
            do_lower_case=params.get('do_lower_case', True),
            evaluate_during_training=params.get('validate', True),
            per_device_train_batch_size=params.get('batch_size', 8),
            learning_rate=params.get('learning_rate', 4e-5),
            warmup_ratio=params.get('warmup_ratio', 0.06),
            weight_decay=params.get('weight_decay', 0.0)
        )
        
        # Criar configuração de loss
        loss_config = LossConfig(
            use_focal_loss=params.get('use_focal_loss', False),
            focal_gamma=params.get('focal_gamma', 2.0),
            pos_weight=params.get('pos_weight')
        )
        
        # Validações de conflito
        ConfigurationManager._validate_config_conflicts(loss_config, instance_num)
        
        return model_config, loss_config
    
    @staticmethod
    def _validate_config_conflicts(loss_config: LossConfig, instance_num: int):
        """Valida conflitos na configuração."""
        if loss_config.use_focal_loss and loss_config.pos_weight:
            logger.warning(f"⚠️ Instância {instance_num}: Focal Loss tem prioridade sobre pos_weight")
            loss_config.pos_weight = None
    
    @staticmethod
    def create_instance_directory(instance_id: str, base_dir: str = ".") -> str:
        """
        Cria diretório para uma instância.
        
        Args:
            instance_id: ID da instância
            base_dir: Diretório base
            
        Returns:
            str: Caminho do diretório criado
        """
        instance_dir = os.path.join(base_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)
        
        # Criar subdiretórios
        subdirs = ['outputs_bert', 'cache_bert', 'visualizations', 'outputs']
        for subdir in subdirs:
            os.makedirs(os.path.join(instance_dir, subdir), exist_ok=True)
        
        return instance_dir
    
    @staticmethod
    def update_config_paths(model_config: ModelConfig, loss_config: LossConfig, 
                          instance_dir: str) -> tuple[ModelConfig, LossConfig]:
        """
        Atualiza caminhos nas configurações.
        
        Args:
            model_config: Configuração do modelo
            loss_config: Configuração de loss
            instance_dir: Diretório da instância
            
        Returns:
            Tuple: Configurações atualizadas
        """
        # Atualizar caminhos no model_config
        model_config.output_dir = os.path.join(instance_dir, "outputs_bert")
        model_config.cache_dir = os.path.join(instance_dir, "cache_bert")
        model_config.best_model_dir = os.path.join(instance_dir, "outputs", "best_model")
        
        return model_config, loss_config

class InstanceExecutor:
    """Executor de instâncias individuais."""
    
    def __init__(self):
        self.results_history = []
    
    def execute_instance(self, instance: Dict[str, Any], instance_num: int,
                        train_df, val_df, test_df) -> Dict[str, Any]:
        """
        Executa uma instância de treinamento.
        
        Args:
            instance: Configuração da instância
            instance_num: Número da instância
            train_df: DataFrame de treino
            val_df: DataFrame de validação
            test_df: DataFrame de teste
            
        Returns:
            Dict: Resultados da execução
        """
        instance_id = instance.get('id', f'instance_{instance_num}')
        instance_name = instance.get('name', 'Sem nome')
        
        logger.info("\n" + "="*80)
        logger.info(f"🚀 EXECUTANDO INSTÂNCIA {instance_num}: {instance_id}")
        logger.info(f"📝 Descrição: {instance_name}")
        logger.info("="*80)
        
        try:
            # Parse configuração
            model_config, loss_config = ConfigurationManager.parse_instance_config(
                instance, instance_num
            )
            
            # Criar diretório da instância
            instance_dir = ConfigurationManager.create_instance_directory(instance_id)
            
            # Atualizar caminhos
            model_config, loss_config = ConfigurationManager.update_config_paths(
                model_config, loss_config, instance_dir
            )
            
            # Executar treinamento
            results = self._run_training_pipeline(
                model_config, loss_config, train_df, val_df, test_df, 
                instance_dir, instance_id
            )
            
            # Salvar configuração usada
            self._save_instance_config(instance, model_config, loss_config, instance_dir)
            
            logger.info(f"✅ Instância {instance_id} concluída com sucesso!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erro na instância {instance_id}: {e}")
            return {'error': str(e), 'instance_id': instance_id}
    
    def _run_training_pipeline(self, model_config: ModelConfig, loss_config: LossConfig,
                             train_df, val_df, test_df, instance_dir: str, 
                             instance_id: str) -> Dict[str, Any]:
        """Executa pipeline completo de treinamento."""
        from src.models import ModelFactory, ModelValidator
        from src.data import MultiLabelDataset, calculate_class_weights
        from src.training import TrainingManager
        from src.metrics import DetailedMetricsAnalyzer
        from src.visualization import VisualizationSuite
        import torch
        
        # Criar modelo e tokenizer
        model, tokenizer = ModelFactory.create_model_and_tokenizer(model_config)
        
        # CORREÇÃO APLICADA AQUI:
        # Usar NUM_LABELS importado de src.config para a validação.
        ModelValidator.validate_model_config(model, NUM_LABELS) 
        
        # Calcular alpha weights se necessário
        if loss_config.use_focal_loss:
            loss_config.focal_alpha_weights = calculate_class_weights(train_df)
        
        # Criar datasets
        train_dataset = MultiLabelDataset(
            train_df['text'].tolist(),
            train_df['labels'].tolist(),
            tokenizer,
            model_config.max_seq_length
        )
        
        val_dataset = MultiLabelDataset(
            val_df['text'].tolist(),
            val_df['labels'].tolist(),
            tokenizer,
            model_config.max_seq_length
        ) if model_config.evaluate_during_training else None
        
        test_dataset = MultiLabelDataset(
            test_df['text'].tolist(),
            test_df['labels'].tolist(),
            tokenizer,
            model_config.max_seq_length
        )
        
        # Treinamento
        training_manager = TrainingManager(model_config, loss_config)
        trainer = training_manager.setup_trainer(model, tokenizer, train_dataset, val_dataset)
        train_result, training_history = training_manager.train()
        
        # Avaliação
        test_metrics, test_probs = training_manager.evaluate(test_dataset)
        
        # Análise detalhada
        y_true = torch.stack([test_dataset[i]['labels'] for i in range(len(test_dataset))]).numpy()
        y_pred = (test_probs >= 0.5).astype(int)
        
        analyzer = DetailedMetricsAnalyzer()
        detailed_metrics = analyzer.calculate_detailed_metrics(y_true, y_pred, test_probs)
        
        # Visualizações
        viz_suite = VisualizationSuite(os.path.join(instance_dir, "visualizations"))
        optimal_thresholds = viz_suite.generate_all_plots(
            training_history, y_true, y_pred, test_probs, 
            detailed_metrics['per_class_metrics']
        )
        
        # Compilar resultados
        results = {
            'instance_id': instance_id,
            'test_metrics': test_metrics,
            'detailed_metrics': detailed_metrics,
            'optimal_thresholds': optimal_thresholds,
            'training_time': train_result.metrics.get('train_runtime', 0),
            'best_eval_metric': max(training_history.get('avg_precision', [0])) if training_history.get('avg_precision') else None
        }
        
        # Salvar resultados
        self._save_results(results, instance_dir)
        
        return results
    
    def _save_instance_config(self, instance: Dict, model_config: ModelConfig, 
                            loss_config: LossConfig, instance_dir: str):
        """Salva configuração completa da instância."""
        config_data = {
            'original_instance': instance,
            'model_config': asdict(model_config),
            'loss_config': asdict(loss_config),
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(instance_dir, "config_used.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def _save_results(self, results: Dict, instance_dir: str):
        """Salva resultados da execução."""
        # Converter tensors numpy para listas para serialização JSON
        def convert_for_json(obj):
            if hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_for_json(results)
        
        results_path = os.path.join(instance_dir, "results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

class BatchExecutor:
    """Executor para múltiplas instâncias."""
    
    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path = dataset_path
        self.all_results = []
    
    def execute_config_file(self, config_path: str) -> List[Dict[str, Any]]:
        """
        Executa todas as instâncias de um arquivo de configuração.
        
        Args:
            config_path: Caminho para arquivo de configuração
            
        Returns:
            List[Dict]: Resultados de todas as instâncias
        """
        from src.data import DataPersistence
        
        # Carregar configuração
        config = ConfigurationManager.load_config_file(config_path)
        instances = config['instances']
        
        logger.info(f"\n📋 Executando {len(instances)} instâncias")
        
        # Carregar ou criar dados
        train_df, val_df, test_df = DataPersistence.load_or_create_splits(self.dataset_path)
        
        # Executar instâncias
        executor = InstanceExecutor()
        self.all_results = []
        
        for i, instance_config in enumerate(instances, 1): # Renomeado 'instance' para 'instance_config' para evitar conflito de nome
            try:
                results = executor.execute_instance(instance_config, i, train_df, val_df, test_df)
                self.all_results.append((instance_config, results))
            except Exception as e:
                logger.error(f"❌ Erro fatal na instância {i}: {e}")
                error_result = {'error': str(e), 'instance_id': instance_config.get('id', f'instance_{i}')}
                self.all_results.append((instance_config, error_result))
                continue
        
        # Gerar relatório resumido
        self._generate_summary_report(config_path)
        
        return self.all_results
    
    def _generate_summary_report(self, config_path: str):
        """Gera relatório resumido comparativo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"summary_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO RESUMIDO - EXECUÇÃO DE MÚLTIPLAS INSTÂNCIAS\n")
            f.write("="*80 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Arquivo de configuração: {config_path}\n")
            f.write(f"Total de instâncias: {len(self.all_results)}\n\n")
            
            # Estatísticas gerais
            successful = sum(1 for _, result in self.all_results if 'error' not in result)
            failed = len(self.all_results) - successful
            
            f.write(f"Execuções bem-sucedidas: {successful}\n")
            f.write(f"Execuções com erro: {failed}\n\n")
            
            if successful > 0:
                # Tabela comparativa
                f.write("RESULTADOS COMPARATIVOS (APENAS SUCESSOS):\n")
                f.write("-"*80 + "\n")
                f.write(f"{'ID':<15} {'Nome':<25} {'F1-Macro':<10} {'Hamming':<10} {'Avg Prec':<10} {'Tempo(s)':<10}\n")
                f.write("-"*80 + "\n")
                
                for instance_info, results in self.all_results:
                    if 'error' in results:
                        continue
                    
                    instance_id = instance_info.get('id', 'N/A')
                    instance_name = instance_info.get('name', 'N/A')[:25]
                    
                    test_metrics = results.get('test_metrics', {})
                    f1_score = test_metrics.get('test_macro_f1', 0)
                    hamming = test_metrics.get('test_hamming_loss', 1)
                    avg_prec = test_metrics.get('test_avg_precision', 0)
                    time_taken = results.get('training_time', 0)
                    
                    f.write(f"{instance_id:<15} {instance_name:<25} "
                           f"{f1_score:<10.4f} {hamming:<10.4f} {avg_prec:<10.4f} {time_taken:<10.1f}\n")
                
                # Identificar melhor modelo
                f.write("\n" + "="*80 + "\n")
                f.write("MELHOR MODELO POR MÉTRICA:\n")
                f.write("="*80 + "\n")
                
                # Filtrar apenas instâncias bem-sucedidas para encontrar o máximo
                successful_results = [res for res in self.all_results if 'error' not in res[1]]
                if successful_results:
                    best_f1 = max(successful_results, 
                                 key=lambda x: x[1].get('test_metrics', {}).get('test_macro_f1', -1)) # Usar -1 para casos onde a métrica pode não existir
                    best_ap = max(successful_results,
                                 key=lambda x: x[1].get('test_metrics', {}).get('test_avg_precision', -1))
                
                    f.write(f"Melhor F1-Macro: {best_f1[0].get('id', 'N/A')} "
                           f"({best_f1[1].get('test_metrics', {}).get('test_macro_f1', 0):.4f})\n")
                    f.write(f"Melhor Avg Precision: {best_ap[0].get('id', 'N/A')} "
                           f"({best_ap[1].get('test_metrics', {}).get('test_avg_precision', 0):.4f})\n")
                else:
                    f.write("Nenhuma execução bem-sucedida para determinar o melhor modelo.\n")

            # Erros se houver
            if failed > 0:
                f.write("\n" + "="*80 + "\n")
                f.write("INSTÂNCIAS COM ERRO:\n")
                f.write("="*80 + "\n")
                
                for instance_info, results in self.all_results:
                    if 'error' in results:
                        instance_id = instance_info.get('id', 'N/A')
                        error_msg = results['error']
                        f.write(f"{instance_id}: {error_msg}\n")
        
        logger.info(f"\n📄 Relatório resumido salvo em: {report_path}")

def create_simple_config_example() -> str:
    """Cria exemplo de configuração simples."""
    config = {
        "instances": [
            {
                "id": "baseline",
                "name": "Baseline BERT multilingual",
                "parameters": {
                    "model_name": "google-bert/bert-base-multilingual-cased",
                    "validate": True,
                    "epochs": 3,
                    "max_seq_length": 128,
                    "do_lower_case": False,
                    "use_focal_loss": False
                }
            },
            {
                "id": "focal_loss",
                "name": "BERT com Focal Loss",
                "parameters": {
                    "model_name": "google-bert/bert-base-multilingual-cased",
                    "validate": True,
                    "epochs": 3,
                    "max_seq_length": 128,
                    "do_lower_case": False,
                    "use_focal_loss": True,
                    "focal_gamma": 2.0
                }
            }
        ]
    }
    
    config_path = "example_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📝 Configuração de exemplo criada: {config_path}")
    return config_path