# -*- coding: utf-8 -*-
"""
Loss functions customizadas para classificação multi-label.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss para classificação multi-label binária.
    
    FL(pt) = -α(1-pt)^γ * log(pt)
    
    Args:
        gamma (float): Parâmetro de focusing. Default: 2.0
        alpha (Union[float, List[float], torch.Tensor], optional): 
            Parâmetro de balanceamento de classe. Default: None
        reduction (str): Especifica a redução a ser aplicada ao output. 
            'mean' | 'sum' | 'none'. Default: 'mean'
    """
    
    def __init__(self, gamma: float = 2.0, 
                 alpha: Optional[Union[float, List[float], torch.Tensor]] = None, 
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da Focal Loss.
        
        Args:
            logits: Tensor de shape (N, C) com logits brutos
            targets: Tensor de shape (N, C) com labels binários (0 ou 1)
            
        Returns:
            torch.Tensor: Loss calculado
        """
        # BCE Loss base
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        
        # Calcular probabilidades
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Aplicar focusing term
        focal_weight = (1 - pt).pow(self.gamma)
        focal_loss = focal_weight * bce_loss
        
        # Aplicar alpha balancing se especificado
        if self.alpha is not None:
            alpha_factor = self._get_alpha_factor(targets, logits.device)
            focal_loss = alpha_factor * focal_loss
        
        # Aplicar redução
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _get_alpha_factor(self, targets: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Calcula o fator alpha para balanceamento de classes."""
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_tensor = torch.tensor(self.alpha, device=device, dtype=torch.float32)
            alpha_t = alpha_tensor.view(1, -1).expand_as(targets)
        
        return torch.where(targets == 1, alpha_t, 1 - alpha_t)

class LossFactory:
    """Factory para criação de loss functions baseado em configuração."""
    
    @staticmethod
    def create_loss_function(loss_config: dict) -> nn.Module:
        """
        Cria loss function baseada na configuração.
        
        Args:
            loss_config: Dicionário com configurações de loss
            
        Returns:
            nn.Module: Loss function configurada
        """
        if loss_config.get('use_focal_loss', False):
            gamma = loss_config.get('focal_gamma', 2.0)
            alpha = loss_config.get('focal_alpha_weights')
            
            logger.info(f"✨ Criando Focal Loss com gamma={gamma}")
            if alpha is not None:
                logger.info(f"   Alpha weights: {alpha}")
            
            return FocalLoss(gamma=gamma, alpha=alpha)
            
        elif loss_config.get('pos_weight') is not None:
            pos_weight = torch.tensor(loss_config['pos_weight'], dtype=torch.float32)
            logger.info(f"⚖️ Criando BCEWithLogitsLoss com pos_weight: {pos_weight.tolist()}")
            
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
        else:
            logger.info("📊 Usando BCEWithLogitsLoss padrão")
            return nn.BCEWithLogitsLoss()

def calculate_pos_weights(label_counts: torch.Tensor, total_samples: int) -> torch.Tensor:
    """
    Calcula pos_weight para BCEWithLogitsLoss baseado na frequência das classes.
    
    Args:
        label_counts: Tensor com contagem de amostras positivas por classe
        total_samples: Número total de amostras
        
    Returns:
        torch.Tensor: Pos weights calculados
    """
    neg_counts = total_samples - label_counts
    pos_weights = neg_counts.float() / label_counts.float()
    
    logger.info("Pos weights calculados:")
    for i, weight in enumerate(pos_weights):
        logger.info(f"  Classe {i}: {weight:.3f}")
    
    return pos_weights