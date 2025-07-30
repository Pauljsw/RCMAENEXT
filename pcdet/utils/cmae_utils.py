"""
pcdet/utils/cmae_utils.py

CMAE-3D Contrastive Learning 및 Memory Management 유틸리티

핵심 기능:
1. Memory queue 고급 관리
2. Positive/Negative pair 생성
3. Feature normalization 및 projection
4. Contrastive loss 계산 헬퍼
5. 배치 처리 최적화
6. Teacher-Student 동기화 체크
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class MemoryQueueManager:
    """
    CMAE-3D Memory Queue 고급 관리자
    
    논문 기반 negative sampling을 위한 memory queue 관리:
    - Dynamic queue size adjustment
    - Feature diversity maintenance  
    - Efficient batch update
    """
    
    def __init__(self, feature_dim: int, queue_size: int = 8192, temperature: float = 0.1):
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.temperature = temperature
        
        # Memory queue 초기화
        self.queue = torch.randn(feature_dim, queue_size)
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0
        
        # Queue 품질 관리
        self.diversity_threshold = 0.95  # Feature diversity 임계값
        self.update_frequency = 100      # Diversity check 주기
        self.update_count = 0
        
        print(f"🎯 Memory Queue Manager 초기화")
        print(f"   - Feature dim: {feature_dim}, Queue size: {queue_size}")
        print(f"   - Temperature: {temperature}, Diversity threshold: {self.diversity_threshold}")
    
    def enqueue_batch(self, features: torch.Tensor) -> None:
        """
        배치 단위로 효율적인 queue 업데이트
        
        Args:
            features: [B, D] normalized features
        """
        batch_size = features.shape[0]
        
        # Circular queue update
        if self.queue_ptr + batch_size <= self.queue_size:
            # 연속적으로 업데이트 가능
            self.queue[:, self.queue_ptr:self.queue_ptr + batch_size] = features.T
            self.queue_ptr = (self.queue_ptr + batch_size) % self.queue_size
        else:
            # 끝까지 채우고 처음부터 시작
            remaining = self.queue_size - self.queue_ptr
            self.queue[:, self.queue_ptr:] = features[:remaining].T
            overflow = batch_size - remaining
            if overflow > 0:
                self.queue[:, :overflow] = features[remaining:].T
            self.queue_ptr = overflow
        
        self.update_count += 1
        
        # 주기적으로 queue diversity 체크
        if self.update_count % self.update_frequency == 0:
            self._check_queue_diversity()
    
    def _check_queue_diversity(self) -> None:
        """
        Queue의 feature diversity 체크 및 개선
        
        너무 유사한 features가 많으면 diversity 향상
        """
        # Queue 내 feature 간 유사도 계산
        similarities = torch.mm(self.queue.T, self.queue)  # [Q, Q]
        
        # 대각선 제외한 최대 유사도
        mask = ~torch.eye(self.queue_size, dtype=torch.bool, device=similarities.device)
        max_similarity = similarities[mask].max().item()
        
        if max_similarity > self.diversity_threshold:
            # Diversity가 낮으면 random noise 추가
            noise = torch.randn_like(self.queue) * 0.1
            self.queue = F.normalize(self.queue + noise, dim=0)
            
            print(f"   ⚠️  Queue diversity 개선: max_sim={max_similarity:.3f} → noise 추가")
    
    def get_negatives(self, query: torch.Tensor, exclude_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Query에 대한 negative samples 반환
        
        Args:
            query: [B, D] query features
            exclude_indices: 제외할 queue indices (optional)
            
        Returns:
            negatives: [B, Q, D] negative samples
        """
        # Queue에서 negative samples 추출
        negatives = self.queue.T.unsqueeze(0).expand(query.shape[0], -1, -1)  # [B, Q, D]
        
        # 필요시 특정 indices 제외
        if exclude_indices is not None:
            # 복잡한 masking 로직은 단순화
            pass
            
        return negatives
    
    def compute_contrastive_logits(self, query: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """
        Contrastive learning logits 계산
        
        Args:
            query: [B, D] query features
            positive: [B, D] positive features
            
        Returns:
            logits: [B, 1+Q] contrastive logits (positive + negatives)
        """
        batch_size = query.shape[0]
        
        # Positive similarity
        pos_sim = torch.sum(query * positive, dim=-1, keepdim=True) / self.temperature  # [B, 1]
        
        # Negative similarities
        neg_sim = torch.mm(query, self.queue) / self.temperature  # [B, Q]
        
        # Concatenate positive and negative logits
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1+Q]
        
        return logits


class ContrastivePairGenerator:
    """
    CMAE-3D Positive/Negative Pair 생성기
    
    Teacher-Student 구조에서 효과적인 contrastive pair 생성:
    - Teacher features as positive targets
    - Cross-batch negative sampling
    - Hard negative mining
    """
    
    def __init__(self, hard_negative_ratio: float = 0.3):
        self.hard_negative_ratio = hard_negative_ratio
        print(f"🎯 Contrastive Pair Generator 초기화")
        print(f"   - Hard negative ratio: {hard_negative_ratio}")
    
    def generate_teacher_student_pairs(self, 
                                     student_features: torch.Tensor,
                                     teacher_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-Student 간 positive pair 생성
        
        Args:
            student_features: [B, D] student features
            teacher_features: [B, D] teacher features (detached)
            
        Returns:
            queries: [B, D] student features (normalized)
            positives: [B, D] teacher features (normalized)
        """
        # Feature normalization
        queries = F.normalize(student_features, dim=-1)
        positives = F.normalize(teacher_features.detach(), dim=-1)
        
        return queries, positives
    
    def mine_hard_negatives(self, 
                          query: torch.Tensor, 
                          candidate_negatives: torch.Tensor,
                          num_hard: int) -> torch.Tensor:
        """
        Hard negative mining
        
        Query와 가장 유사한 (어려운) negative samples 선택
        
        Args:
            query: [B, D] query features
            candidate_negatives: [B, N, D] candidate negative features
            num_hard: 선택할 hard negative 개수
            
        Returns:
            hard_negatives: [B, num_hard, D] hard negative features
        """
        # Query와 candidates 간 유사도 계산
        similarities = torch.bmm(query.unsqueeze(1), candidate_negatives.transpose(1, 2))  # [B, 1, N]
        similarities = similarities.squeeze(1)  # [B, N]
        
        # 가장 유사한 (hard) negatives 선택
        _, hard_indices = torch.topk(similarities, num_hard, dim=1)  # [B, num_hard]
        
        # Hard negatives 추출
        batch_indices = torch.arange(query.shape[0]).unsqueeze(1).expand(-1, num_hard)
        hard_negatives = candidate_negatives[batch_indices, hard_indices]  # [B, num_hard, D]
        
        return hard_negatives
    
    def create_cross_batch_negatives(self, features: torch.Tensor) -> torch.Tensor:
        """
        Cross-batch negative sampling
        
        같은 배치 내 다른 샘플들을 negative로 사용
        
        Args:
            features: [B, D] batch features
            
        Returns:
            cross_negatives: [B, B-1, D] cross-batch negatives
        """
        batch_size = features.shape[0]
        
        # 자기 자신 제외한 모든 샘플을 negative로 사용
        cross_negatives = []
        for i in range(batch_size):
            # i번째 샘플 제외
            negatives = torch.cat([features[:i], features[i+1:]], dim=0)  # [B-1, D]
            cross_negatives.append(negatives)
        
        cross_negatives = torch.stack(cross_negatives, dim=0)  # [B, B-1, D]
        
        return cross_negatives


class FeatureProjectionUtils:
    """
    CMAE-3D Feature Projection 유틸리티
    
    Multi-scale features의 효과적인 projection 및 fusion:
    - Adaptive projection based on feature scale
    - Feature fusion strategies
    - Dimensionality reduction
    """
    
    @staticmethod
    def adaptive_projection(features: Dict[str, torch.Tensor], 
                          target_dim: int = 256) -> torch.Tensor:
        """
        Multi-scale features의 adaptive projection
        
        Args:
            features: {'x_conv1': tensor, 'x_conv2': tensor, ...}
            target_dim: 목표 projection dimension
            
        Returns:
            projected_features: [N, target_dim] projected features
        """
        projected_list = []
        
        for scale_name, feat_tensor in features.items():
            if scale_name.startswith('x_conv'):
                # Sparse tensor의 features 추출
                if hasattr(feat_tensor, 'features'):
                    feat = feat_tensor.features  # [N, C]
                else:
                    feat = feat_tensor
                
                # 각 scale별로 adaptive pooling
                if feat.shape[1] != target_dim:
                    # Linear projection으로 차원 통일
                    projector = torch.nn.Linear(feat.shape[1], target_dim).to(feat.device)
                    projected = projector(feat)
                else:
                    projected = feat
                
                projected_list.append(projected)
        
        if len(projected_list) > 0:
            # Multi-scale features 평균 융합
            fused_features = torch.stack(projected_list, dim=0).mean(dim=0)  # [N, target_dim]
            return F.normalize(fused_features, dim=-1)
        else:
            return torch.empty(0, target_dim)
    
    @staticmethod
    def hierarchical_feature_fusion(features: Dict[str, torch.Tensor],
                                  fusion_strategy: str = 'weighted_avg') -> torch.Tensor:
        """
        Hierarchical multi-scale feature fusion
        
        Args:
            features: Multi-scale features dict
            fusion_strategy: 'weighted_avg', 'attention', 'concat'
            
        Returns:
            fused_features: Fused feature representation
        """
        if fusion_strategy == 'weighted_avg':
            # Scale별 가중치 (더 깊은 layer에 더 높은 가중치)
            weights = {'x_conv1': 0.1, 'x_conv2': 0.2, 'x_conv3': 0.3, 'x_conv4': 0.4}
            
            weighted_features = []
            for scale_name, feat_tensor in features.items():
                if scale_name in weights:
                    feat = feat_tensor.features if hasattr(feat_tensor, 'features') else feat_tensor
                    weighted_feat = feat * weights[scale_name]
                    weighted_features.append(weighted_feat)
            
            if weighted_features:
                return torch.cat(weighted_features, dim=-1)
            
        elif fusion_strategy == 'concat':
            # 단순 concatenation
            feat_list = []
            for scale_name, feat_tensor in features.items():
                if scale_name.startswith('x_conv'):
                    feat = feat_tensor.features if hasattr(feat_tensor, 'features') else feat_tensor
                    feat_list.append(feat)
            
            if feat_list:
                return torch.cat(feat_list, dim=-1)
        
        return torch.empty(0)


class TrainingStabilityChecker:
    """
    CMAE-3D Training Stability 체크 유틸리티
    
    Teacher-Student training의 안정성 모니터링:
    - Loss convergence tracking
    - Feature alignment monitoring  
    - EMA update quality check
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_history = []
        self.alignment_history = []
        
        print(f"🎯 Training Stability Checker 초기화")
        print(f"   - Window size: {window_size}")
    
    def check_loss_stability(self, losses: Dict[str, float]) -> Dict[str, bool]:
        """
        손실 함수 안정성 체크
        
        Args:
            losses: {'occupancy': loss, 'mlfr': loss, 'contrastive': loss}
            
        Returns:
            stability_flags: {'occupancy': stable, 'mlfr': stable, ...}
        """
        self.loss_history.append(losses)
        
        if len(self.loss_history) < self.window_size:
            return {key: True for key in losses.keys()}  # 초기에는 안정적이라고 가정
        
        # 최근 window 내 loss 변동성 체크
        stability_flags = {}
        recent_losses = self.loss_history[-self.window_size:]
        
        for loss_name in losses.keys():
            loss_values = [item[loss_name] for item in recent_losses]
            
            # 변동 계수 (CV) 계산
            mean_loss = np.mean(loss_values)
            std_loss = np.std(loss_values)
            cv = std_loss / (mean_loss + 1e-8)
            
            # CV가 0.5 이하면 안정적
            stability_flags[loss_name] = cv < 0.5
            
            if not stability_flags[loss_name]:
                print(f"   ⚠️  {loss_name} loss 불안정: CV={cv:.3f}")
        
        return stability_flags
    
    def check_teacher_student_alignment(self, 
                                      student_features: torch.Tensor,
                                      teacher_features: torch.Tensor) -> float:
        """
        Teacher-Student feature alignment 체크
        
        Args:
            student_features: [B, D] student features
            teacher_features: [B, D] teacher features
            
        Returns:
            alignment_score: 0~1 사이의 alignment 점수 (높을수록 좋음)
        """
        # Cosine similarity 계산
        student_norm = F.normalize(student_features, dim=-1)
        teacher_norm = F.normalize(teacher_features.detach(), dim=-1)
        
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=-1)  # [B]
        alignment_score = cosine_sim.mean().item()
        
        self.alignment_history.append(alignment_score)
        
        # Alignment 히스토리 관리
        if len(self.alignment_history) > self.window_size:
            self.alignment_history.pop(0)
        
        # Alignment trend 체크
        if len(self.alignment_history) >= 10:
            recent_trend = np.mean(self.alignment_history[-10:])
            if recent_trend < 0.3:
                print(f"   ⚠️  Teacher-Student alignment 낮음: {recent_trend:.3f}")
        
        return alignment_score
    
    def get_training_diagnostics(self) -> Dict[str, float]:
        """
        종합적인 training 진단 정보 반환
        
        Returns:
            diagnostics: 진단 정보 dict
        """
        diagnostics = {}
        
        if len(self.loss_history) > 0:
            recent_losses = self.loss_history[-10:] if len(self.loss_history) >= 10 else self.loss_history
            
            # 각 loss의 평균과 추세
            for loss_name in recent_losses[0].keys():
                loss_values = [item[loss_name] for item in recent_losses]
                diagnostics[f'{loss_name}_mean'] = np.mean(loss_values)
                diagnostics[f'{loss_name}_trend'] = np.mean(np.diff(loss_values)) if len(loss_values) > 1 else 0.0
        
        if len(self.alignment_history) > 0:
            diagnostics['alignment_mean'] = np.mean(self.alignment_history[-10:])
            if len(self.alignment_history) > 1:
                diagnostics['alignment_trend'] = np.mean(np.diff(self.alignment_history[-10:]))
        
        return diagnostics


def compute_info_nce_loss(query: torch.Tensor,
                         positive: torch.Tensor, 
                         negatives: torch.Tensor,
                         temperature: float = 0.1) -> torch.Tensor:
    """
    InfoNCE loss 계산 헬퍼 함수
    
    Args:
        query: [B, D] query features
        positive: [B, D] positive features  
        negatives: [B, N, D] negative features
        temperature: 온도 파라미터
        
    Returns:
        loss: InfoNCE loss
    """
    # Positive similarity
    pos_sim = torch.sum(query * positive, dim=-1) / temperature  # [B]
    
    # Negative similarities
    neg_sim = torch.bmm(query.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / temperature  # [B, N]
    
    # Logits: [positive, negative1, negative2, ...]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
    
    # Labels: positive는 항상 index 0
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss


def safe_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    안전한 L2 normalization (0벡터 처리)
    
    Args:
        tensor: 정규화할 tensor
        dim: 정규화 차원
        eps: 수치 안정성을 위한 epsilon
        
    Returns:
        normalized_tensor: 정규화된 tensor
    """
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    return tensor / (norm + eps)


def log_contrastive_metrics(query: torch.Tensor,
                          positive: torch.Tensor,
                          negatives: torch.Tensor) -> Dict[str, float]:
    """
    Contrastive learning 메트릭 로깅
    
    Args:
        query: [B, D] query features
        positive: [B, D] positive features
        negatives: [B, N, D] negative features
        
    Returns:
        metrics: 로깅용 메트릭 dict
    """
    with torch.no_grad():
        # Positive similarity
        pos_sim = torch.sum(query * positive, dim=-1).mean().item()
        
        # Negative similarity (평균)
        neg_sim = torch.bmm(query.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)  # [B, N]
        neg_sim_mean = neg_sim.mean().item()
        neg_sim_max = neg_sim.max().item()
        
        # Similarity gap (positive와 negative 간 차이)
        sim_gap = pos_sim - neg_sim_mean
        
        metrics = {
            'pos_similarity': pos_sim,
            'neg_similarity_mean': neg_sim_mean,
            'neg_similarity_max': neg_sim_max,
            'similarity_gap': sim_gap,
        }
    
    return metrics