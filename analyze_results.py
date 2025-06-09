import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CLASSE PRINCIPAL DE ANÁLISE
# ============================================================================

class SimpleMultiModelAnalyzer:
    """Versão simplificada para execução direta."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.labels = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]
        self.models = ["BERT", "Bertimbau", "BertAbaporu"]
        self.loss_types = ["Default", "Focal", "PosWeight"]
        
        # Mapeamento ID -> (Modelo, Loss)
        self.model_mapping = {
            "I001": ("BERT", "Default"),
            "I002": ("BERT", "Focal"),
            "I003": ("BERT", "PosWeight"),
            "I004": ("Bertimbau", "Default"),
            "I005": ("Bertimbau", "Focal"),
            "I006": ("Bertimbau", "PosWeight"),
            "I007": ("BertAbaporu", "Default"),
            "I008": ("BertAbaporu", "Focal"),
            "I009": ("BertAbaporu", "PosWeight")
        }
        
        self.results_data = {}
        self.comparison_df = None
    
    def load_all_results(self):
        """Carrega todos os resultados dos modelos."""
        print("📂 CARREGANDO RESULTADOS DOS MODELOS")
        print("="*50)
        
        for model_id, (model_name, loss_type) in self.model_mapping.items():
            results_path = self.base_path / model_id / "results.json"
            
            if results_path.exists():
                try:
                    with open(results_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    self.results_data[model_id] = {
                        'model_name': model_name,
                        'loss_type': loss_type,
                        'data': data
                    }
                    print(f"   ✅ {model_name:<12} {loss_type:<10} - OK")
                    
                except Exception as e:
                    print(f"   ❌ {model_name:<12} {loss_type:<10} - ERRO: {e}")
            else:
                print(f"   ⚠️  {model_name:<12} {loss_type:<10} - NÃO ENCONTRADO")
        
        print(f"\n📊 TOTAL CARREGADO: {len(self.results_data)}/9 modelos\n")
        
        if len(self.results_data) == 0:
            print("❌ NENHUM RESULTADO ENCONTRADO!")
            print("Verifique se os diretórios I001-I009 existem e contêm results.json")
            return False
        
        return True
    
    def create_comparison_table(self):
        """Cria tabela comparativa principal."""
        print("📋 CRIANDO TABELA COMPARATIVA")
        print("="*50)
        
        comparison_data = []
        
        for model_id, info in self.results_data.items():
            model_name = info['model_name']
            loss_type = info['loss_type']
            data = info['data']
            
            # Métricas principais
            test_metrics = data.get('test_metrics', {})
            
            record = {
                'ID': model_id,
                'Modelo': model_name,
                'Loss': loss_type,
                'Macro_F1': test_metrics.get('test_macro_f1', 0.0),
                'Avg_Precision': test_metrics.get('test_avg_precision', 0.0),
                'Hamming_Loss': test_metrics.get('test_hamming_loss', 1.0),
                'Micro_F1': test_metrics.get('test_micro_f1', 0.0),
                'Tempo_Treino': data.get('training_time', 0.0)
            }
            
            # F1 por classe
            per_class = data.get('detailed_metrics', {}).get('per_class_metrics', {})
            for label in self.labels:
                if label in per_class:
                    record[f'F1_{label}'] = per_class[label].get('f1', 0.0)
                else:
                    record[f'F1_{label}'] = 0.0
            
            comparison_data.append(record)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('Macro_F1', ascending=False)
        
        print("✅ Tabela criada com sucesso!")
        print(f"📊 {len(self.comparison_df)} modelos na comparação\n")
        
        return self.comparison_df
    
    def show_rankings(self):
        """Mostra rankings principais."""
        if self.comparison_df is None:
            self.create_comparison_table()
        
        print("🏆 RANKING GERAL (TOP 5)")
        print("="*50)
        
        top_5 = self.comparison_df.head()
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{emoji} {row['Modelo']:<12} {row['Loss']:<10} | "
                  f"F1: {row['Macro_F1']:.4f} | AP: {row['Avg_Precision']:.4f}")
        
        print("\n📊 PERFORMANCE POR MODELO")
        print("="*50)
        
        model_stats = self.comparison_df.groupby('Modelo')['Macro_F1'].agg(['mean', 'max', 'std']).round(4)
        model_stats = model_stats.sort_values('mean', ascending=False)
        
        for model, stats in model_stats.iterrows():
            print(f"{model:<12}: Média={stats['mean']:.4f} | "
                  f"Melhor={stats['max']:.4f} | Desvio=±{stats['std']:.3f}")
        
        print("\n⚖️  PERFORMANCE POR LOSS")
        print("="*50)
        
        loss_stats = self.comparison_df.groupby('Loss')['Macro_F1'].agg(['mean', 'max', 'std']).round(4)
        loss_stats = loss_stats.sort_values('mean', ascending=False)
        
        for loss, stats in loss_stats.iterrows():
            print(f"{loss:<10}: Média={stats['mean']:.4f} | "
                  f"Melhor={stats['max']:.4f} | Desvio=±{stats['std']:.3f}")
    
    def analyze_problematic_classes(self):
        """Analisa classes com baixa performance."""
        print("\n🎯 ANÁLISE DE CLASSES PROBLEMÁTICAS")
        print("="*50)
        
        # Calcular F1 médio por classe
        class_performance = {}
        for label in self.labels:
            col_name = f'F1_{label}'
            if col_name in self.comparison_df.columns:
                avg_f1 = self.comparison_df[col_name].mean()
                max_f1 = self.comparison_df[col_name].max()
                best_model_idx = self.comparison_df[col_name].idxmax()
                best_model = self.comparison_df.loc[best_model_idx]
                
                class_performance[label] = {
                    'avg_f1': avg_f1,
                    'max_f1': max_f1,
                    'best_model': best_model['Modelo'],
                    'best_loss': best_model['Loss']
                }
        
        # Ordenar por performance (pior para melhor)
        sorted_classes = sorted(class_performance.items(), key=lambda x: x[1]['avg_f1'])
        
        print("Classes ordenadas por dificuldade (mais difícil primeiro):\n")
        
        for label, stats in sorted_classes:
            status = "🔴" if stats['avg_f1'] < 0.2 else "🟡" if stats['avg_f1'] < 0.4 else "🟢"
            print(f"{status} {label:<12}: Média={stats['avg_f1']:.3f} | "
                  f"Melhor={stats['max_f1']:.3f} | "
                  f"Top: {stats['best_model']} ({stats['best_loss']})")
        
        # Identificar classes críticas
        critical_classes = [label for label, stats in class_performance.items() 
                            if stats['avg_f1'] < 0.3]
        
        if critical_classes:
            print(f"\n🚨 CLASSES CRÍTICAS (F1 < 0.3): {', '.join(critical_classes)}")
            print("💡 Recomendações:")
            print("   - Coletar mais dados para essas classes")
            print("   - Usar técnicas de data augmentation")
            print("   - Considerar class weights mais altos")
    
    def create_separated_plots(self, output_dir: str = "analysis_output"):
        """Cria os 4 gráficos separados."""
        print(f"\n🔥 GERANDO GRÁFICOS SEPARADOS")
        print("="*50)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot 1: Heatmap principal
        plt.figure(figsize=(8, 6))
        pivot_data = self.comparison_df.pivot(index='Modelo', columns='Loss', values='Macro_F1')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Macro F1-Score'})
        plt.title('Performance: Modelo × Loss Function')
        plt.ylabel('Modelo Base')
        plt.xlabel('Tipo de Loss')
        heatmap_path = output_path / "heatmap_model_loss.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close() # Fecha a figura para não exibir ou sobrepor no próximo plot
        print(f"📊 Heatmap Model x Loss salvo em: {heatmap_path}")
        
        # Plot 2: Ranking geral
        plt.figure(figsize=(8, 6))
        top_models = self.comparison_df.head(9)
        y_pos = range(len(top_models))
        bars = plt.barh(y_pos, top_models['Macro_F1'])
        plt.yticks(y_pos, [f"{row['Modelo']}\n({row['Loss']})" 
                           for _, row in top_models.iterrows()])
        plt.xlabel('Macro F1-Score')
        plt.title('Ranking Geral (Todos os Modelos)')
        plt.grid(True, alpha=0.3)
        colors = {'BERT': '#FF6B6B', 'Bertimbau': '#4ECDC4', 'BertAbaporu': '#45B7D1'}
        for i, (_, row) in enumerate(top_models.iterrows()):
            bars[i].set_color(colors.get(row['Modelo'], '#95A5A6'))
        ranking_path = output_path / "ranking_geral.png"
        plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Ranking Geral salvo em: {ranking_path}")

        # Plot 3: Performance por classe (classes problemáticas)
        plt.figure(figsize=(8, 6))
        problem_classes = ['racism', 'xenophobia', 'misogyny'] # Exemplo, ajuste conforme sua análise
        class_data = []
        class_labels = []
        
        for label in problem_classes:
            col_name = f'F1_{label}'
            if col_name in self.comparison_df.columns:
                class_data.append(self.comparison_df[col_name].values)
                class_labels.append(label)
        
        if class_data:
            plt.boxplot(class_data, labels=class_labels)
            plt.ylabel('F1-Score')
            plt.title('Distribuição: Classes Problemáticas')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            problematic_classes_path = output_path / "f1_problematic_classes.png"
            plt.savefig(problematic_classes_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 F1 Classes Problemáticas salvo em: {problematic_classes_path}")
        else:
            print("⚠️ Não foi possível gerar o gráfico de classes problemáticas. Verifique as colunas.")

        # Plot 4: Loss function effectiveness
        plt.figure(figsize=(8, 6))
        loss_means = self.comparison_df.groupby('Loss')['Macro_F1'].mean()
        loss_stds = self.comparison_df.groupby('Loss')['Macro_F1'].std()
        
        x_pos = range(len(loss_means))
        plt.bar(x_pos, loss_means.values, yerr=loss_stds.values, 
                capsize=5, alpha=0.7, color=['#FF9999', '#66B2FF', '#99FF99'])
        plt.xticks(x_pos, loss_means.index)
        plt.ylabel('Macro F1-Score')
        plt.title('Efetividade por Loss Function')
        plt.grid(True, alpha=0.3)
        loss_effectiveness_path = output_path / "effectiveness_loss_function.png"
        plt.savefig(loss_effectiveness_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Efetividade da Loss Function salvo em: {loss_effectiveness_path}")
        
        print("\n✅ Todos os gráficos separados foram gerados com sucesso!")
    
    def generate_final_report(self, output_dir: str = "analysis_output"):
        """Gera relatório final."""
        print(f"\n📄 GERANDO RELATÓRIO FINAL")
        print("="*50)
        
        # Criar diretório
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Salvar tabela completa
        csv_path = output_path / "comparison_complete.csv"
        self.comparison_df.to_csv(csv_path, index=False)
        print(f"💾 Tabela completa salva: {csv_path}")
        
        # Relatório em texto
        report_path = output_path / "final_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO FINAL - ANÁLISE MULTI-MODELO BERT\n")
            f.write("="*60 + "\n\n")
            
            # Resumo executivo
            best_model = self.comparison_df.iloc[0]
            f.write("RESUMO EXECUTIVO:\n")
            f.write(f"Melhor modelo geral: {best_model['Modelo']} com {best_model['Loss']}\n")
            f.write(f"Macro F1-Score: {best_model['Macro_F1']:.4f}\n")
            f.write(f"Average Precision: {best_model['Avg_Precision']:.4f}\n\n")
            
            # Top 5
            f.write("TOP 5 MODELOS:\n")
            f.write("-" * 40 + "\n")
            for i, (_, row) in enumerate(self.comparison_df.head().iterrows(), 1):
                f.write(f"{i}. {row['Modelo']} ({row['Loss']}) - F1: {row['Macro_F1']:.4f}\n")
            
            f.write("\n")
            
            # Performance por modelo
            f.write("PERFORMANCE POR MODELO BASE:\n")
            f.write("-" * 40 + "\n")
            model_stats = self.comparison_df.groupby('Modelo')['Macro_F1'].agg(['mean', 'max']).round(4)
            for model, stats in model_stats.sort_values('mean', ascending=False).iterrows():
                f.write(f"{model}: Média={stats['mean']:.4f} | Melhor={stats['max']:.4f}\n")
            
            f.write("\n")
            
            # Performance por loss
            f.write("PERFORMANCE POR LOSS FUNCTION:\n")
            f.write("-" * 40 + "\n")
            loss_stats = self.comparison_df.groupby('Loss')['Macro_F1'].agg(['mean', 'max']).round(4)
            for loss, stats in loss_stats.sort_values('mean', ascending=False).iterrows():
                f.write(f"{loss}: Média={stats['mean']:.4f} | Melhor={stats['max']:.4f}\n")
            
            # Recomendações
            f.write("\nRECOMENDAÇÕES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"1. Use {best_model['Modelo']} com {best_model['Loss']} para máxima performance\n")
            
            # Melhor por categoria
            best_by_model = self.comparison_df.groupby('Modelo')['Macro_F1'].idxmax()
            for model in self.models:
                if model in best_by_model.index:
                    idx = best_by_model[model]
                    row = self.comparison_df.loc[idx]
                    f.write(f"2. Para {model}: use {row['Loss']} (F1: {row['Macro_F1']:.4f})\n")
            
            f.write("3. Foque em melhorar classes problemáticas (racism, xenophobia)\n")
            f.write("4. Implemente early stopping para evitar overfitting\n")
        
        print(f"📄 Relatório salvo: {report_path}")
        
        # Criar README
        readme_path = output_path / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("COMO LER OS RESULTADOS\n")
            f.write("="*30 + "\n\n")
            f.write("1. final_report.txt - Resumo executivo com principais conclusões\n")
            f.write("2. comparison_complete.csv - Dados completos para análise posterior\n")
            f.write("3. heatmap_model_loss.png - Visualização comparativa de Modelo x Função de Perda\n")
            f.write("4. ranking_geral.png - Ranking de todos os modelos por Macro F1-Score\n")
            f.write("5. f1_problematic_classes.png - Desempenho F1-Score em classes problemáticas\n")
            f.write("6. effectiveness_loss_function.png - Efetividade das Funções de Perda\n\n")
            f.write("MÉTRICAS PRINCIPAIS:\n")
            f.write("- Macro F1: Média harmônica de precision e recall (0-1, maior é melhor)\n")
            f.write("- Avg Precision: Área sob curva precision-recall (0-1, maior é melhor)\n")
            f.write("- Hamming Loss: Taxa de erro por classe (0-1, menor é melhor)\n")
        
        print(f"📖 Guia de leitura: {readme_path}")
        print(f"\n✅ Relatório completo salvo em: {output_path}")
    
    def run_complete_analysis(self):
        """Executa análise completa."""
        print("\n" + "="*80)
        print("🚀 ANÁLISE COMPLETA MULTI-MODELO BERT")
        print("="*80)
        
        # Passo 1: Carregar dados
        if not self.load_all_results():
            return False
        
        # Passo 2: Criar tabela comparativa
        self.create_comparison_table()
        
        # Passo 3: Mostrar rankings
        self.show_rankings()
        
        # Passo 4: Analisar classes problemáticas
        self.analyze_problematic_classes()
        
        # Passo 5: Gerar visualizações (agora separadas)
        self.create_separated_plots() # Chame a nova função aqui
        
        # Passo 6: Gerar relatório final
        self.generate_final_report()
        
        print("\n" + "="*80)
        print("🎉 ANÁLISE COMPLETA FINALIZADA!")
        print("="*80)
        print("📁 Verifique a pasta 'analysis_output' para os resultados")
        print("📊 Gráficos salvos individualmente na pasta 'analysis_output'")
        print("📄 Relatório principal em 'analysis_output/final_report.txt'")
        
        return True

# ============================================================================
# FUNÇÃO PRINCIPAL PARA EXECUÇÃO
# ============================================================================

def analyze_my_bert_models(base_path: str = "."):
    """
    FUNÇÃO PRINCIPAL - Execute esta para fazer toda a análise!
    
    Args:
        base_path: Diretório onde estão as pastas I001, I002, etc.
    
    Returns:
        bool: True se bem-sucedido
    """
    analyzer = SimpleMultiModelAnalyzer(base_path)
    return analyzer.run_complete_analysis()

# ============================================================================
# VERIFICAÇÃO E EXECUÇÃO AUTOMÁTICA
# ============================================================================

if __name__ == "__main__":
    print("🔍 VERIFICANDO ARQUIVOS DE RESULTADOS...")
    
    # Verificar se existem diretórios de resultados
    current_dir = Path(".")
    model_dirs = []
    
    for i in range(1, 10):
        dir_name = f"I{i:03d}"
        dir_path = current_dir / dir_name
        results_file = dir_path / "results.json"
        
        if dir_path.exists() and results_file.exists():
            model_dirs.append(dir_name)
            print(f"   ✅ {dir_name} - OK")
        else:
            print(f"   ❌ {dir_name} - NÃO ENCONTRADO")
    
    print(f"\n📊 ENCONTRADOS: {len(model_dirs)}/9 modelos")
    
    if len(model_dirs) > 0:
        print("\n🚀 INICIANDO ANÁLISE AUTOMÁTICA...")
        success = analyze_my_bert_models(".")
        
        if success:
            print("\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
            print("\n📁 RESULTADOS DISPONÍVEIS EM:")
            print("   - analysis_output/final_report.txt")
            print("   - analysis_output/comparison_complete.csv") 
            print("   - analysis_output/heatmap_model_loss.png") # Nome do arquivo alterado
            print("   - analysis_output/ranking_geral.png")
            print("   - analysis_output/f1_problematic_classes.png")
            print("   - analysis_output/effectiveness_loss_function.png")
        else:
            print("\n❌ ERRO NA ANÁLISE!")
    else:
        print("\n⚠️  NENHUM MODELO ENCONTRADO!")
        print("\nPara usar este analisador:")
        print("1. Execute seus treinamentos BERT primeiro")
        print("2. Certifique-se de que existem pastas I001, I002, ..., I009")
        print("3. Cada pasta deve ter um arquivo 'results.json'")
        print("4. Execute este script novamente")
        
        print("\n💡 EXEMPLO DE EXECUÇÃO MANUAL:")
        print("   python analyze_results.py")
        print("   # ou")
        print("   success = analyze_my_bert_models('caminho/para/resultados')")