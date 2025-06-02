# -*- coding: utf-8 -*-
"""
Utilitários gerais para o projeto.
"""

import os
import sys
import time
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class Timer:
    """Classe para medir tempo de execução."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"⏰ Iniciando: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"✅ Concluído: {self.name} em {duration:.2f} segundos")
    
    @property
    def elapsed(self) -> float:
        """Retorna tempo decorrido em segundos."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

class FileManager:
    """Gerenciador de arquivos e diretórios."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Garante que um diretório existe.
        
        Args:
            path: Caminho do diretório
            
        Returns:
            Path: Objeto Path do diretório
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def safe_remove(path: Union[str, Path]) -> bool:
        """
        Remove arquivo ou diretório de forma segura.
        
        Args:
            path: Caminho para remover
            
        Returns:
            bool: True se removido com sucesso
        """
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                path_obj.unlink()
            elif path_obj.is_dir():
                import shutil
                shutil.rmtree(path_obj)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível remover {path}: {e}")
            return False
    
    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """
        Retorna tamanho do arquivo em bytes.
        
        Args:
            path: Caminho do arquivo
            
        Returns:
            int: Tamanho em bytes
        """
        try:
            return Path(path).stat().st_size
        except Exception:
            return 0
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Formata tamanho do arquivo em formato legível.
        
        Args:
            size_bytes: Tamanho em bytes
            
        Returns:
            str: Tamanho formatado
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

class JSONHandler:
    """Manipulador especializado para arquivos JSON."""
    
    @staticmethod
    def save_json(data: Any, path: Union[str, Path], indent: int = 2, 
                  ensure_ascii: bool = False) -> bool:
        """
        Salva dados em arquivo JSON.
        
        Args:
            data: Dados para salvar
            path: Caminho do arquivo
            indent: Indentação para formatação
            ensure_ascii: Se deve garantir ASCII
            
        Returns:
            bool: True se salvo com sucesso
        """
        try:
            # Garantir que o diretório existe
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, 
                         default=JSONHandler._json_serializer)
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao salvar JSON {path}: {e}")
            return False
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Optional[Any]:
        """
        Carrega dados de arquivo JSON.
        
        Args:
            path: Caminho do arquivo
            
        Returns:
            Any: Dados carregados ou None se erro
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠️ Arquivo JSON não encontrado: {path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao decodificar JSON {path}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erro ao carregar JSON {path}: {e}")
            return None
    
    @staticmethod
    def _json_serializer(obj):
        """Serializer customizado para objetos não serializáveis."""
        import numpy as np
        import torch
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif torch.is_tensor(obj):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

class LoggingSetup:
    """Configurador de logging."""
    
    @staticmethod
    def setup_logging(level: str = "INFO", log_file: Optional[str] = None,
                     format_string: Optional[str] = None) -> None:
        """
        Configura sistema de logging.
        
        Args:
            level: Nível de logging
            log_file: Arquivo para salvar logs (opcional)
            format_string: Formato personalizado (opcional)
        """
        # Formato padrão
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configuração básica
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Adicionar handler para arquivo se especificado
        if log_file:
            FileManager.ensure_directory(Path(log_file).parent)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(format_string))
            logging.getLogger().addHandler(file_handler)
        
        # Configurar logging de bibliotecas externas
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    @staticmethod
    def create_session_logger(session_name: str) -> logging.Logger:
        """
        Cria logger específico para uma sessão.
        
        Args:
            session_name: Nome da sessão
            
        Returns:
            logging.Logger: Logger configurado
        """
        logger = logging.getLogger(session_name)
        
        # Handler específico para a sessão
        session_file = f"logs/{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        FileManager.ensure_directory(Path(session_file).parent)
        
        handler = logging.FileHandler(session_file, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

class SystemInfo:
    """Informações do sistema."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Coleta informações do sistema.
        
        Returns:
            Dict: Informações do sistema
        """
        import platform
        import psutil
        import torch
        
        info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation()
            },
            'memory': {
                'total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'used_percent': psutil.virtual_memory().percent
            },
            'cpu': {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_percent': psutil.cpu_percent(interval=1)
            },
            'torch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return info
    
    @staticmethod
    def log_system_info():
        """Log das informações do sistema."""
        info = SystemInfo.get_system_info()
        
        logger.info("\n" + "="*60)
        logger.info("📊 INFORMAÇÕES DO SISTEMA")
        logger.info("="*60)
        logger.info(f"Sistema: {info['platform']['system']} {info['platform']['release']}")
        logger.info(f"Python: {info['python']['version']} ({info['python']['implementation']})")
        logger.info(f"CPU: {info['cpu']['logical_cores']} cores ({info['cpu']['physical_cores']} físicos)")
        logger.info(f"Memória: {info['memory']['available_gb']:.1f}GB disponível de {info['memory']['total_gb']:.1f}GB")
        logger.info(f"PyTorch: {info['torch']['version']}")
        
        if info['torch']['cuda_available']:
            logger.info(f"CUDA: Disponível (versão {info['torch']['cuda_version']}, {info['torch']['device_count']} device(s))")
        else:
            logger.info("CUDA: Não disponível")

class ProgressTracker:
    """Rastreador de progresso para operações longas."""
    
    def __init__(self, total_steps: int, description: str = "Progresso"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = 10  # Log a cada 10 segundos
    
    def update(self, step: int = 1):
        """
        Atualiza progresso.
        
        Args:
            step: Número de steps para avançar
        """
        self.current_step += step
        current_time = time.time()
        
        # Log periódico
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress()
            self.last_log_time = current_time
        
        # Log final
        if self.current_step >= self.total_steps:
            self._log_completion()
    
    def _log_progress(self):
        """Log do progresso atual."""
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f", ETA: {eta:.0f}s"
        else:
            eta_str = ""
        
        logger.info(f"📈 {self.description}: {self.current_step}/{self.total_steps} "
                   f"({percentage:.1f}%, {elapsed:.0f}s decorridos{eta_str})")
    
    def _log_completion(self):
        """Log da conclusão."""
        elapsed = time.time() - self.start_time
        logger.info(f"✅ {self.description} concluído em {elapsed:.2f} segundos")

@contextmanager
def temporary_directory(prefix: str = "temp_"):
    """
    Context manager para diretório temporário.
    
    Args:
        prefix: Prefixo para nome do diretório
    """
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def validate_requirements():
    """
    Valida se todas as dependências estão instaladas.
    
    Returns:
        bool: True se todas as dependências estão ok
    """
    required_packages = [
        'torch', 'transformers', 'sklearn', 'pandas', 
        'numpy', 'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Pacotes faltando: {', '.join(missing_packages)}")
        logger.error("Instale com: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ Todas as dependências estão instaladas")
    return True

def format_duration(seconds: float) -> str:
    """
    Formata duração em formato legível.
    
    Args:
        seconds: Duração em segundos
        
    Returns:
        str: Duração formatada
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Retorna informações sobre memória GPU.
    
    Returns:
        Dict: Informações de memória
    """
    import torch
    
    if not torch.cuda.is_available():
        return {'available': False}
    
    device = torch.cuda.current_device()
    return {
        'available': True,
        'device_name': torch.cuda.get_device_name(device),
        'total_memory_gb': torch.cuda.get_device_properties(device).total_memory / (1024**3),
        'allocated_memory_gb': torch.cuda.memory_allocated(device) / (1024**3),
        'cached_memory_gb': torch.cuda.memory_reserved(device) / (1024**3)
    }

class ConfigValidator:
    """Validador de configurações."""
    
    @staticmethod
    def validate_model_config(config_dict: Dict[str, Any]) -> List[str]:
        """
        Valida configuração de modelo.
        
        Args:
            config_dict: Dicionário de configuração
            
        Returns:
            List[str]: Lista de erros encontrados
        """
        errors = []
        
        # Campos obrigatórios
        required_fields = ['model_name']
        for field in required_fields:
            if field not in config_dict:
                errors.append(f"Campo obrigatório ausente: {field}")
        
        # Validações de valores
        if 'epochs' in config_dict:
            if not isinstance(config_dict['epochs'], int) or config_dict['epochs'] <= 0:
                errors.append("'epochs' deve ser um inteiro positivo")
        
        if 'max_seq_length' in config_dict:
            if not isinstance(config_dict['max_seq_length'], int) or config_dict['max_seq_length'] <= 0:
                errors.append("'max_seq_length' deve ser um inteiro positivo")
        
        if 'learning_rate' in config_dict:
            lr = config_dict['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append("'learning_rate' deve ser um número positivo")
        
        return errors