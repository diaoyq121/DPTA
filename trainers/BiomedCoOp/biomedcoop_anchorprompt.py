import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
from trainers.attr_prompt import BIOMEDCOOP_ATTR


def StableGreenkhorn(logits, tau_t, gamma, iters_sinkhorn):
    """
    Stable Greenkhorn-Sinkhorn algorithm for optimal transport
    用于生成稳定的伪标签分布

    Args:
        logits: (batch_size, num_classes) - 原始 logits
        tau_t: float - 温度参数，控制分布的平滑度
        gamma: float - Sinkhorn 正则化参数
        iters_sinkhorn: int - Sinkhorn 迭代次数

    Returns:
        plabel: (batch_size, num_classes) - 优化后的伪标签分布
    """
    with torch.no_grad():
        B, C = logits.shape

        # 1. 数值稳定的 softmax：减去最大值避免溢出
        logits_stable = logits - logits.max(dim=1, keepdim=True)[0]

        # 2. 使用温度缩放的 softmax 得到初始分布
        Q = torch.softmax(logits_stable / tau_t, dim=1)  # (B, C)

        # 3. 应用 Sinkhorn-Knopp 算法进行最优传输
        # 使用 log-domain 计算避免数值溢出
        log_K = logits_stable / gamma  # (B, C)

        # 初始化（在 log domain）
        log_u = torch.zeros(B, device=logits.device)  # (B,)
        log_v = torch.zeros(C, device=logits.device)  # (C,)

        # Sinkhorn 迭代（在 log domain 中进行）
        for _ in range(iters_sinkhorn):
            # 更新 log_u
            log_sum_exp = torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)  # (B,)
            log_u = -log_sum_exp

            # 更新 log_v
            log_sum_exp = torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)  # (C,)
            log_v = -log_sum_exp

        # 4. 计算最终的传输矩阵（从 log domain 转回）
        log_plabel = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)  # (B, C)
        plabel = torch.softmax(log_plabel, dim=1)  # 使用 softmax 确保归一化和数值稳定

    return plabel


def sinkhorn_loss(pred_logits, target_prob, epsilon=0.1, max_iter=100):
    """
    计算基于 Sinkhorn 算法的最优传输损失（Wasserstein 距离）

    Args:
        pred_logits: (batch_size, num_classes) - 模型预测的 logits
        target_prob: (batch_size, num_classes) - 目标概率分布
        epsilon: float - 熵正则化参数
        max_iter: int - Sinkhorn 迭代次数

    Returns:
        loss: scalar - Sinkhorn 损失
    """
    # 将 logits 转换为概率分布
    pred_prob = F.softmax(pred_logits, dim=1)  # (B, C)

    B, C = pred_prob.shape
    device = pred_prob.device

    # 计算代价矩阵：使用类别间的距离
    # 这里使用简单的 0-1 代价：对角线为 0，其他为 1
    cost_matrix = 1.0 - torch.eye(C, device=device)  # (C, C)

    # Sinkhorn 算法：计算最优传输计划
    # K = exp(-cost / epsilon)
    K = torch.exp(-cost_matrix / epsilon)  # (C, C)

    # 初始化 dual variables
    u = torch.ones(B, C, device=device) / C  # (B, C)

    # Sinkhorn 迭代
    for _ in range(max_iter):
        # v = target_prob / (K^T @ u)
        v = target_prob / (torch.matmul(u, K.t()) + 1e-8)  # (B, C)
        # u = pred_prob / (K @ v)
        u = pred_prob / (torch.matmul(v, K) + 1e-8)  # (B, C)

    # 计算传输计划: P = diag(u) @ K @ diag(v)
    # 简化计算：P[i,j] = u[i] * K[i,j] * v[j]
    transport = u.unsqueeze(2) * K.unsqueeze(0) * v.unsqueeze(1)  # (B, C, C)

    # 计算 Wasserstein 距离
    loss = torch.sum(transport * cost_matrix.unsqueeze(0), dim=(1, 2)).mean()

    return loss


def sinkhorn_loss_per_sample(pred_logits, target_prob, text_features=None, epsilon=0.1, max_iter=100):
    """
    计算每个样本的 Sinkhorn 损失（支持语义代价矩阵）

    Args:
        pred_logits: (batch_size, num_classes) - 模型预测的 logits
        target_prob: (batch_size, num_classes) - 目标概率分布
        text_features: (num_classes, 512) - 文本特征，用于计算语义代价矩阵
        epsilon: float - 熵正则化参数
        max_iter: int - Sinkhorn 迭代次数

    Returns:
        loss_per_sample: (batch_size,) - 每个样本的损失
    """
    pred_prob = F.softmax(pred_logits, dim=1)
    B, C = pred_prob.shape
    device = pred_prob.device

    # 计算代价矩阵
    if text_features is not None:
        # 使用文本特征的余弦相似度计算语义代价
        text_sim = text_features @ text_features.t()  # (C, C)
        cost_matrix = 1.0 - text_sim  # 相似度高 -> 代价低
        cost_matrix = torch.clamp(cost_matrix, min=0.0, max=2.0)
        # 对角线设为 0（自己到自己代价为 0）
        cost_matrix = cost_matrix * (1 - torch.eye(C, device=device))
    else:
        # 简单的 0-1 代价
        cost_matrix = 1.0 - torch.eye(C, device=device)

    # Sinkhorn 算法
    K = torch.exp(-cost_matrix / epsilon)
    u = torch.ones(B, C, device=device) / C

    for _ in range(max_iter):
        v = target_prob / (torch.matmul(u, K.t()) + 1e-8)
        u = pred_prob / (torch.matmul(v, K) + 1e-8)

        # 数值稳定性检查
        if torch.isnan(u).any() or torch.isinf(u).any():
            break

    # 计算传输计划
    transport = u.unsqueeze(2) * K.unsqueeze(0) * v.unsqueeze(1)  # (B, C, C)

    # 计算每个样本的 Wasserstein 距离
    loss_per_sample = torch.sum(transport * cost_matrix.unsqueeze(0), dim=(1, 2))  # (B,)

    return loss_per_sample


def image_opt(feat, init_classifier, plabel, lr=10, iter=100, tau_i=0.04, alpha=0.6):
    """
    基于图像特征和伪标签优化分类器

    Args:
        feat: (batch_size, dim) - 图像特征
        init_classifier: (dim, n_cls) - 初始分类器权重
        plabel: (batch_size, n_cls) - 伪标签概率分布
        lr: float - 学习率
        iter: int - 迭代次数
        tau_i: float - 温度参数
        alpha: float - 置信度阈值，用于硬化伪标签

    Returns:
        classifier: (dim, n_cls) - 优化后的分类器
    """
    with torch.no_grad():
        ins, dim = feat.shape

        # 1. 硬化高置信度的伪标签
        val, idx = torch.max(plabel, dim=1)  # 获取最大概率和索引
        mask = val > alpha  # 选择置信度高于阈值的样本
        plabel = plabel.clone()  # 复制避免修改原始数据
        plabel[mask, :] = 0  # 清零
        plabel[mask, idx[mask]] = 1  # 设为 one-hot

        # 2. 计算目标基准（特征加权和）
        base = feat.T @ plabel  # (dim, n_cls)

        # 3. 初始化分类器
        classifier = init_classifier.clone()
        pre_norm = float('inf')

        # 4. 梯度下降优化
        for i in range(iter):
            # 前向传播：计算预测概率
            prob = F.softmax(feat @ classifier / tau_i, dim=1)  # (ins, n_cls)

            # 计算梯度
            grad = feat.T @ prob - base  # (dim, n_cls)

            # 自适应学习率
            temp = torch.norm(grad)
            if temp > pre_norm:
                lr = lr / 2.0
            pre_norm = temp

            # 梯度下降更新
            classifier = classifier - (lr / (ins * tau_i)) * grad

            # 归一化分类器
            classifier = F.normalize(classifier, dim=0)

    return classifier


class MultiLayerFeatureExtractor(nn.Module):
    """
    从 ViT 编码器的多个中间层提取特征并融合
    支持从指定的层（如 3, 6, 9, 12）提取特征
    """
    def __init__(self, image_encoder, layer_indices=[3, 6, 9, 12], feature_dim=768, fusion_type='concat'):
        """
        Args:
            image_encoder: ViT 图像编码器 (TimmModel)
            layer_indices: 要提取特征的层索引列表（从1开始计数，12层ViT）
            feature_dim: 特征维度（BiomedCLIP ViT-B/16 为 768）
            fusion_type: 特征融合方式 ('concat', 'mean', 'attention', 'biomedclip_proj')
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.layer_indices = layer_indices
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.n_layers = len(layer_indices)

        # 注册 hooks 用于提取中间层特征
        self.intermediate_features = {}
        self.hooks = []

        # 为每个指定的层注册 forward hook
        # BiomedCLIP 使用 timm 模型，ViT blocks 在 image_encoder.trunk.blocks
        for idx in layer_indices:
            layer = self.image_encoder.trunk.blocks[idx - 1]  # 0-indexed
            hook = layer.register_forward_hook(self._get_hook(idx))
            self.hooks.append(hook)

        # 根据融合方式初始化融合模块
        if fusion_type == 'concat':
            # 拼接后降维到原始维度
            self.fusion_proj = nn.Linear(feature_dim * self.n_layers, 512)
        elif fusion_type == 'attention':
            # 使用注意力机制融合多层特征
            self.layer_attention = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            self.fusion_proj = nn.Linear(feature_dim, 512)
            # if hasattr(image_encoder, 'head') and hasattr(image_encoder.head, 'proj'):
            #     self.fusion_proj = image_encoder.head.proj  # 直接使用预训练的投影层
            #     print("Using BiomedCLIP pretrained projection layer")
            # else:
            #     # 如果找不到，回退到新的线性层
            #     self.fusion_proj = nn.Linear(feature_dim, 512)
            #     print("Warning: BiomedCLIP projection layer not found, using new linear layer")

        elif fusion_type == 'mean':
            # 简单平均，只需要投影到512维
            # self.fusion_proj = nn.Linear(feature_dim, 512)
            if hasattr(image_encoder, 'head') and hasattr(image_encoder.head, 'proj'):
                self.fusion_proj = image_encoder.head.proj  # 直接使用预训练的投影层
                print("Using BiomedCLIP pretrained projection layer")
            else:
                # 如果找不到，回退到新的线性层
                self.fusion_proj = nn.Linear(feature_dim, 512)
                print("Warning: BiomedCLIP projection layer not found, using new linear layer")

        elif fusion_type == 'biomedclip_proj':
            # 使用 BiomedCLIP 预训练的投影层（从 image_encoder.head.proj 提取）
            # 这样可以利用预训练的投影权重
            if hasattr(image_encoder, 'head') and hasattr(image_encoder.head, 'proj'):
                self.fusion_proj = image_encoder.head.proj  # 直接使用预训练的投影层
                print("Using BiomedCLIP pretrained projection layer")
            else:
                # 如果找不到，回退到新的线性层
                self.fusion_proj = nn.Linear(feature_dim, 512)
                print("Warning: BiomedCLIP projection layer not found, using new linear layer")

        print(f"MultiLayerFeatureExtractor initialized: layers={layer_indices}, fusion={fusion_type}")

    def _get_hook(self, layer_idx):
        """创建 hook 函数来捕获中间层输出"""
        def hook(module, input, output):
            # timm ViT 的输出格式: (batch_size, seq_len, feature_dim)
            # 我们需要 [CLS] token 的特征，即第一个 token
            self.intermediate_features[layer_idx] = output[:, 0, :]  # (batch_size, feature_dim)
        return hook

    def forward(self, image):
        """
        提取多层特征并融合
        Args:
            image: (batch_size, 3, 224, 224)
        Returns:
            fused_features: (batch_size, 512) - 融合后的特征
            final_features: (batch_size, 512) - 最后一层的特征（用于兼容）
        """
        # 清空之前的中间特征
        self.intermediate_features = {}

        # 前向传播，hooks 会自动捕获中间层特征
        final_features = self.image_encoder(image)  # (batch_size, 512)

        # 收集所有中间层特征
        multi_layer_features = []
        for idx in self.layer_indices:
            if idx in self.intermediate_features:
                feat = self.intermediate_features[idx]  # (batch_size, feature_dim)
                multi_layer_features.append(feat)

        # 融合多层特征
        if self.fusion_type == 'concat':
            # 拼接所有层的特征
            concat_features = torch.cat(multi_layer_features, dim=1)  # (B, feature_dim * n_layers)
            fused_features = self.fusion_proj(concat_features)  # (B, 512)

        elif self.fusion_type == 'attention':
            # 使用注意力权重融合
            stacked_features = torch.stack(multi_layer_features, dim=1)  # (B, n_layers, feature_dim)
            attn_scores = self.layer_attention(stacked_features).squeeze(-1)  # (B, n_layers)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, n_layers, 1)
            weighted_features = (stacked_features * attn_weights).sum(dim=1)  # (B, feature_dim)
            fused_features = self.fusion_proj(weighted_features)  # (B, 512)

        elif self.fusion_type == 'mean':
            # 简单平均
            stacked_features = torch.stack(multi_layer_features, dim=1)  # (B, n_layers, feature_dim)
            mean_features = stacked_features.mean(dim=1)  # (B, feature_dim)
            fused_features = self.fusion_proj(mean_features)  # (B, 512)

        elif self.fusion_type == 'biomedclip_proj':
            # 使用 BiomedCLIP 预训练的投影层
            # 直接对中间层特征的平均值应用预训练投影
            stacked_features = torch.stack(multi_layer_features, dim=1)  # (B, n_layers, feature_dim)
            mean_features = stacked_features.mean(dim=1)  # (B, feature_dim=768)
            fused_features = self.fusion_proj(mean_features)  # (B, 512) - 使用预训练投影层

        return fused_features, final_features

    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()


class AttributeRetriever(nn.Module):
    """
    属性检索模块：从预定义的属性库中检索与图像最相关的属性
    支持使用类别特定的属性特征
    """
    def __init__(self, top_k=5, attr_concept_features=None):
        super().__init__()
        self.top_k = top_k

        # 使用传入的类别特定属性概念特征
        # attr_concept_features: (n_cls, n_attr_per_cls, 512)
        if attr_concept_features is not None:
            # 将所有类别的属性特征展平为一个大的属性池
            n_cls, n_attr_per_cls, feat_dim = attr_concept_features.shape
            attr_features = attr_concept_features.view(-1, feat_dim)  # (n_cls * n_attr_per_cls, 512)
            attr_features = attr_features / attr_features.norm(dim=-1, keepdim=True)
            print(f"Loaded {attr_features.shape[0]} attribute concept features ({n_cls} classes × {n_attr_per_cls} attrs)")

            # 注册为 buffer（固定的）
            self.register_buffer("attr_features", attr_features.cpu())
        else:
            # 如果没有传入，设置为 None（需要后续设置）
            self.attr_features = None
            print("AttributeRetriever initialized without attr_features (will be set later)")

    def forward(self, image_features):
        """
        Args:
            image_features: (batch_size, 512) - 图像特征
        Returns:
            top_attr_features: (batch_size, top_k, 512) - top-k 属性特征
            top_attr_scores: (batch_size, top_k) - 相似度分数
        """
        # 计算图像与所有属性的相似度
        attr_features = self.attr_features.to(image_features.device)
        similarities = image_features @ attr_features.t()  # (B, n_attrs)

        # 选择 top-k 属性
        top_scores, top_indices = torch.topk(similarities, self.top_k, dim=1)  # (B, top_k)

        # 获取 top-k 属性特征
        batch_size = image_features.shape[0]
        top_attr_features = attr_features[top_indices]  # (B, top_k, 512)

        return top_attr_features, top_scores


class AttributeAggregator(nn.Module):
    """
    使用注意力机制聚合 top-k 属性特征
    """
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()

        # 注意力计算层
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.scale = hidden_dim ** -0.5

    def forward(self, image_features, attr_features, attr_scores):
        """
        Args:
            image_features: (B, 512) - 图像特征作为 query
            attr_features: (B, top_k, 512) - top-k 属性特征
            attr_scores: (B, top_k) - 检索相似度分数
        Returns:
            aggregated_attr: (B, 512) - 聚合后的属性向量
        """
        batch_size = image_features.shape[0]

        # 计算注意力权重
        query = self.query_proj(image_features).unsqueeze(1)  # (B, 1, hidden_dim)
        key = self.key_proj(attr_features)  # (B, top_k, hidden_dim)
        value = self.value_proj(attr_features)  # (B, top_k, 512)

        # Scaled dot-product attention
        attn_scores = torch.bmm(query, key.transpose(1, 2)) * self.scale  # (B, 1, top_k)
        # attn_scores = attn_scores.squeeze(1) + attr_scores  # (B, top_k)

        # 结合检索分数作为先验
        attn_scores = attn_scores.squeeze(1) + attr_scores  # (B, top_k)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(1)  # (B, 1, top_k)

        # 加权聚合
        aggregated_attr = torch.bmm(attn_weights, value).squeeze(1)  # (B, 512)

        return aggregated_attr


# class CrossAttentionFusion(nn.Module):
#     """
#     无可学习参数的属性-图像融合模块
#     属性特征作为 query，通过相似性调制图像特征
#     """
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps

#     def forward(self, image_features, attr_features):
#         """
#         Args:
#             image_features: (B, D)
#             attr_features: (B, D)

#         Returns:
#             enhanced_features: (B, D)
#         """
#         # 1. L2 normalize
#         img = F.normalize(image_features, dim=-1)
#         attr = F.normalize(attr_features, dim=-1)

#         # 2. cosine similarity as attention weight
#         # (B,)
#         alpha = torch.sum(img * attr, dim=-1, keepdim=True)

#         # 3. attribute-guided residual fusion
#         enhanced_features = image_features + alpha * attr_features

#         return enhanced_features
class CrossAttentionFusion(nn.Module):
    """
    无参数“交叉注意力”（在通道维上做 attention）
    输入/输出: (B, D)
    """
    def __init__(self, tau=0.07, use_residual=True):
        super().__init__()
        self.tau = tau
        self.use_residual = use_residual

    def forward(self, image_features, attr_features):
        # normalize for stable correlation
        q = F.normalize(attr_features, dim=-1)    # (B,D)
        k = F.normalize(image_features, dim=-1)   # (B,D)

        # channel-wise logits (B,D)
        logits = (q * k) / self.tau

        # attention over channels (B,D), sum over D == 1
        attn = torch.softmax(logits, dim=-1)

        # use image as V (or attr as V, 见下面注释)
        context = attn * image_features           # (B,D)

        if self.use_residual:
            return image_features + context
        else:
            return context


class QFormer(nn.Module):
    """
    Q-Former: 使用 learnable queries 从图像特征中提取属性信息
    类似于 BLIP-2 的设计，但简化为单层 Transformer decoder
    """
    def __init__(self, n_queries, image_dim, hidden_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_queries = n_queries

        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)

        # 投影层：将图像特征投影到 hidden_dim
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        # Cross-attention layer: queries attend to image features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, image_features):
        """
        Args:
            image_features: (batch_size, image_dim) - 全局图像特征
        Returns:
            attribute_features: (batch_size, n_queries, hidden_dim) - 提取的属性特征
        """
        batch_size = image_features.shape[0]

        # 扩展 queries 到 batch
        queries = self.queries.expand(batch_size, -1, -1)  # (B, n_queries, hidden_dim)

        # 投影图像特征并扩展维度用于 cross-attention
        image_features_proj = self.image_proj(image_features)  # (B, hidden_dim)
        image_features_proj = image_features_proj.unsqueeze(1)  # (B, 1, hidden_dim)

        # Cross-attention: queries attend to image features
        attn_out, _ = self.cross_attn(
            query=queries,
            key=image_features_proj,
            value=image_features_proj
        )
        queries = self.norm1(queries + attn_out)

        # Feed-forward
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)

        return queries


class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.text_model = biomedclip_model.text  # HFTextEncoder
        self.dtype = biomedclip_model.text.transformer.embeddings.word_embeddings.weight.dtype

    def forward(self, prompts, tokenized_prompts=None):
        """
        直接使用 prompt embeddings 进行文本编码
        Args:
            prompts: 已经构造好的 prompt embeddings (n_cls, seq_len, dim)
            tokenized_prompts: tokenized prompts for attention mask (n_cls, seq_len)
        """
        # HFTextEncoder 支持 inputs_embeds 参数
        # forward(self, x, inputs_embeds=False, y=None)
        x = prompts.type(self.dtype)
        text_features = self.text_model(x, inputs_embeds=True, y=tokenized_prompts)

        return text_features


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDCOOP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDCOOP.CTX_INIT
        att_init = cfg.TRAINER.BIOMEDCOOP.ATT_INIT  # 新增：属性初始化文本
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        text_feature_dim = 512  # BiomedCLIP 文本特征维度
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 添加投影层：将文本特征 (512) 投影到 embedding 空间 (768)
        self.feature_to_embedding = nn.Linear(text_feature_dim, ctx_dim, bias=False)
        nn.init.normal_(self.feature_to_embedding.weight, std=0.02)

        # ========== 初始化上下文向量 (CTX) ==========
        if ctx_init and n_ctx == 5:
            # 使用给定文本初始化上下文向量
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 随机初始化
            if cfg.TRAINER.BIOMEDCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context text: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        # prompt = [att_init + " " + name + "." for name in classnames]

        # ========== 初始化属性向量 (ATTR) ==========
        # 支持从文本初始化或随机初始化
        # 首先从配置获取属性数量
        n_att = cfg.TRAINER.BIOMEDCOOP.get('N_ATT', 5)  # 默认4个属性 tokens

        if att_init:
            # 从预定义的属性文本初始化
            att_init = att_init.replace("_", " ")
            att_prompt = self.tokenizer(att_init)
            with torch.no_grad():
                att_embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(att_prompt).type(dtype)
            # 提取属性 tokens（跳过 SOS），使用配置中指定的数量
            att_vectors = att_embedding[0, 1: 1 + n_att, :]
            att_text = att_init
        else:
            # 随机初始化属性向量
            if cfg.TRAINER.BIOMEDCOOP.CSC:
                print("Initializing class-specific attributes")
                att_vectors = torch.empty(n_cls, n_att, ctx_dim, dtype=dtype)
            else:
                print("Initializing generic attributes")
                att_vectors = torch.empty(n_att, ctx_dim, dtype=dtype)
            nn.init.normal_(att_vectors, std=0.02)
            att_text = " ".join(["x"] * n_att)

        print(f'Initial attribute text: "{att_text}"')
        print(f"Number of attribute words (tokens): {n_att}")
        self.att = nn.Parameter(att_vectors)  # 可学习的属性向量

        # ========== 构造完整 prompts: [SOS][CTX][CLS][ATTR][EOS] ==========
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]

        # 构造模板: "{ctx_prefix} {classname} {att_text}."
        prompts = [ prompt_prefix + " " + "x" + " " + name + "." for name in classnames]
        # prompts = [f"{prompt_prefix} {name} {att_text}." for name in classnames]
        # print("Some example prompts: "{prompts}"")
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])  # (n_cls, n_tkn)

        # ========== 创建冻结的 CLIP 作为教师 ==========
        biomedclip_model_temp, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()

        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = biomedclip_model_temp.visual

            # 预计算冻结的多模板文本特征
            all_teacher_features = []
            attr_features = []
            for i in range(cfg.TRAINER.BIOMEDCOOP.N_PROMPTS):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDCOOP_TEMPLATES[classname][i]) for classname in classnames])
                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

            # ========== 创建数据集级别的全局属性池（类别无关）==========
            dataset_name = cfg.DATASET.NAME

            # 定义数据集到所有类别的映射（包括 base 和 novel）
            DATASET_ALL_CLASSES = {
                'BTMRI': ['glioma tumor', 'meningioma tumor', 'normal brain', 'pituitary tumor'],
                'BUSI': ['benign tumor', 'malignant tumor', 'normal scan'],
                'COVID_19': ['covid lungs', 'lung opacity lungs', 'normal lungs', 'viral pneumonia lungs'],
                'CTKidney': ['cyst kidney', 'kidney stone', 'kidney tumor', 'normal kidney'],
                'DermaMNIST': ['actinic keratosis', 'basal cell carcinoma', 'benign keratosis', 'dermatofibroma',
                               'melanoma', 'melanocytic nevus', 'squamous cell carcinoma', 'vascular lesion'],
                'Kvasir': ['dyed lifted polyps', 'dyed resection margins', 'esophagitis', 'normal cecum',
                           'normal pylorus', 'normal z line', 'polyps', 'ulcerative colitis'],
                'LungColon': ['colon adenocarcinoma', 'colon benign tissue', 'lung adenocarcinoma',
                              'lung benign tissue', 'lung squamous cell carcinoma'],
                'RETINA': ['cataract', 'diabetic retinopathy', 'glaucoma', 'normal retina'],
                'KneeXray': ['healthy knee', 'doubtful osteoarthritis', 'minimal osteoarthritis',
                             'moderate osteoarthritis', 'severe osteoarthritis'],
                'OCTMNIST': ['choroidal neovascularization', 'diabetic macular edema', 'drusen', 'normal OCT scan'],
                'CHMNIST': ['basophil', 'eosinophil', 'erythroblast', 'immature granulocytes',
                            'lymphocyte', 'monocyte', 'neutrophil', 'platelet'],
            }

            # 获取当前数据集的所有类别
            dataset_all_classes = DATASET_ALL_CLASSES.get(dataset_name, classnames)

            # 从数据集的所有类别中提取属性，去重
            all_attr_texts = set()
            for cls_name in dataset_all_classes:
                if cls_name in BIOMEDCOOP_ATTR:
                    for attr_text in BIOMEDCOOP_ATTR[cls_name]:
                        all_attr_texts.add(attr_text)

            all_attr_texts = sorted(list(all_attr_texts))
            print(f"Creating dataset-level global attribute pool for {dataset_name}")
            print(f"  Dataset classes: {len(dataset_all_classes)} (base + novel)")
            print(f"  Unique attributes: {len(all_attr_texts)}")

            # 为全局属性池计算特征
            global_attr_features = []
            batch_size = 32
            for i in range(0, len(all_attr_texts), batch_size):
                batch_texts = all_attr_texts[i:i+batch_size]
                batch_tokenized = torch.cat([self.tokenizer(text) for text in batch_texts])
                batch_features = biomedclip_model_temp.encode_text(batch_tokenized.cuda())
                global_attr_features.append(batch_features)

            global_attr_features = torch.cat(global_attr_features, dim=0)
            global_attr_features = global_attr_features / global_attr_features.norm(dim=-1, keepdim=True)
            self.register_buffer("global_attr_features", global_attr_features)

            # 保留类别特定的属性特征（用于其他用途）
            for i in range(30):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDCOOP_ATTR[classname][i]) for classname in classnames])
                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                attr_features.append(text_features.unsqueeze(1))
        # 注册为 buffer（固定的、不需要梯度）
        # 属性概念特征: (n_cls, 30, 512) - 每个类别的30个属性concept特征
        self.attr_concept_features = torch.cat(attr_features, dim=1)

        self.register_buffer("fixed_embeddings", torch.cat(all_teacher_features, dim=1))

        # 计算 fixed_embeddings_mean 作为额外的 token
        with torch.no_grad():
            fixed_embeddings_mean = self.fixed_embeddings.mean(dim=1)  # (n_cls, 512)
            fixed_embeddings_mean = fixed_embeddings_mean / fixed_embeddings_mean.norm(dim=-1, keepdim=True)
            # 移到 CPU 进行投影
            fixed_embeddings_mean = fixed_embeddings_mean.cpu()

            # 投影到 embedding 空间: (n_cls, 512) -> (n_cls, 768)
            # 在 no_grad 下进行投影，确保不追踪梯度
            fixed_embeddings_mean = self.feature_to_embedding(fixed_embeddings_mean.type(dtype))
            # 转换为 token 格式: (n_cls, 1, 768)
            fixed_embeddings_mean = fixed_embeddings_mean.unsqueeze(1)

        # ========== 注册固定的 token buffers ==========
        # 结构: [SOS] | [ATT] | [CTX] | [MEAN] | [EOS]
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_mean", fixed_embeddings_mean)  # MEAN token

        # 提取每个类别的类别名嵌入
        class_embeddings = []
        for i in range(n_cls):
            name_len = name_lens[i]
            # 从 embedding 中提取类别名部分: [1+n_ctx : 1+n_ctx+name_len]
            cls_emb = embedding[i : i + 1, 1 + n_ctx : 1 + n_ctx + name_len, :]
            class_embeddings.append(cls_emb)

        # 为了统一处理，pad 到最大长度
        max_name_len = max(name_lens)
        padded_class_embeddings = []
        for i, cls_emb in enumerate(class_embeddings):
            if cls_emb.shape[1] < max_name_len:
                # 用零向量 pad
                pad_len = max_name_len - cls_emb.shape[1]
                pad = torch.zeros(1, pad_len, ctx_dim, dtype=dtype)
                cls_emb = torch.cat([cls_emb, pad], dim=1)
            padded_class_embeddings.append(cls_emb)

        class_embeddings_tensor = torch.cat(padded_class_embeddings, dim=0)  # (n_cls, max_name_len, ctx_dim)
        self.register_buffer("class_embeddings", class_embeddings_tensor)

        # suffix 只包含 [EOS]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx +1:, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_att = n_att
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"  # 固定使用 end 模式: [SOS][CTX][CLS][ATTR][EOS]

        # 读取 MEAN token 的位置配置
        try:
            self.mean_position = cfg.TRAINER.BIOMEDCOOP.MEAN_POSITION
        except (AttributeError, KeyError):
            self.mean_position = "front"  # 默认：MEAN 在前
        print(f"MEAN token position: {self.mean_position}")


    def construct_prompts(self, ctx, att, prefix, suffix, mean_token, image_att=None):
        """
        构造 prompts，支持不同的 MEAN token 位置

        位置选项（通过 cfg.TRAINER.BIOMEDCOOP.MEAN_POSITION 配置）:
        - "front": [SOS][MEAN][CTX][EOS]  - MEAN 在前，作为主题
        - "end":   [SOS][CTX][MEAN][EOS]  - MEAN 在后，作为总结
        - "middle": [SOS][CTX_half][MEAN][CTX_half][EOS] - MEAN 在中间，平衡

        Args:
            ctx: 上下文向量 (n_cls, n_ctx, ctx_dim) or (n_ctx, ctx_dim)
            att: 属性向量 (n_cls, n_att, ctx_dim) or (n_att, ctx_dim)
            prefix: SOS token (n_cls, 1, ctx_dim)
            suffix: EOS token (n_cls, 1, ctx_dim)
            mean_token: Mean token (n_cls, 1, ctx_dim)
            image_att: 从图像中提取的属性特征 (batch_size, n_att, ctx_dim), 可选
        """
        max_length = 256  # BiomedCLIP 最大序列长度

        # 展开为类别特定的形式
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if att.dim() == 2:
            att = att.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = []
        for i in range(self.n_cls):
            # 获取当前类别的各部分
            name_len = self.name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]  # (1, 1, dim) - SOS
            ctx_i = ctx[i : i + 1, :, :]  # (1, n_ctx, dim) - Context (learnable)
            cls_i = self.class_embeddings[i : i + 1, :name_len, :]  # (1, name_len, dim) - Class name (fixed)
            att_i = att[i : i + 1, :, :]  # (1, n_att, dim) - Attributes (learnable)
            mean_i = mean_token[i : i + 1, :, :]  # (1, 1, dim) - Mean token (fixed)
            suffix_i = suffix[i : i + 1, :, :]  # (1, 1, dim) - EOS

            # 根据配置选择 MEAN 的位置
            # mean_position = self.mean_position  # 从配置读取
            mean_position = 'middle'  # 从配置读取


            if mean_position == "front":
                # 方案 A: [SOS][MEAN][CTX][EOS]
                prompt = torch.cat([
                    prefix_i,  # SOS
                    mean_i,    # Mean token (fixed) - 作为主题
                    ctx_i,     # Context (learnable) - 作为修饰
                    suffix_i,  # EOS
                ], dim=1)

            elif mean_position == "end":
                # 方案 B: [SOS][CTX][MEAN][EOS]
                prompt = torch.cat([
                    prefix_i,  # SOS
                    ctx_i,     # Context (learnable) - 作为引导
                    mean_i,    # Mean token (fixed) - 作为总结
                    suffix_i,  # EOS
                ], dim=1)

            elif mean_position == "middle":
                # 方案 C: [SOS][CTX_half][MEAN][CTX_half][EOS]
                half_n_ctx = self.n_ctx // 2
                ctx_i_half1 = ctx_i[:, :half_n_ctx, :]
                ctx_i_half2 = ctx_i[:, half_n_ctx:, :]
                prompt = torch.cat([
                    prefix_i,     # SOS
                    ctx_i_half1,  # Context 前半部分
                    mean_i,       # Mean token (fixed) - 在中间
                    ctx_i_half2,  # Context 后半部分
                    suffix_i,     # EOS
                ], dim=1)
            else:
                raise ValueError(f"Unknown mean_position: {mean_position}")

            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts

    def forward(self):
        ctx = self.ctx
        att = self.att

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if att.dim() == 2:
            att = att.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        mean_token = self.token_mean

        # 构造完整的 prompts: [SOS][ATT][CTX][MEAN][EOS]
        prompts = self.construct_prompts(ctx, att, prefix, suffix, mean_token)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

        # ========== 初始化多层特征提取器 ==========
        # 从配置中获取多层特征提取参数
        try:
            use_multilayer = cfg.TRAINER.BIOMEDCOOP.USE_MULTILAYER
        except (AttributeError, KeyError):
            use_multilayer = True  # 默认启用多层特征提取

        if use_multilayer:
            try:
                layer_indices = cfg.TRAINER.BIOMEDCOOP.LAYER_INDICES
            except (AttributeError, KeyError):
                layer_indices = [3, 6, 9, 12]  # 默认提取第3, 6, 9, 12层

            try:
                fusion_type = cfg.TRAINER.BIOMEDCOOP.FUSION_TYPE
            except (AttributeError, KeyError):
                fusion_type = 'attention'  # 默认使用注意力融合

            self.multilayer_extractor = MultiLayerFeatureExtractor(
                image_encoder=self.image_encoder,
                layer_indices=layer_indices,
                feature_dim=768,  # ViT-B/16 的隐藏层维度
                fusion_type=fusion_type
            )
            self.use_multilayer = True
        else:
            self.multilayer_extractor = None
            self.use_multilayer = False

        # 初始化属性检索和聚合模块
        # 使用数据集级别的全局属性池（类别无关）
        self.attr_retriever = AttributeRetriever(
            top_k=5,
            attr_concept_features=self.prompt_learner.global_attr_features.unsqueeze(0)  # (1, n_attrs, 512) -> (n_attrs, 512)
        )
        self.attr_aggregator = AttributeAggregator(feature_dim=512, hidden_dim=256)

        # 初始化交叉注意力融合模块：使用属性特征增强图像特征
        self.cross_attn_fusion = CrossAttentionFusion(
            # feature_dim=512,
            # n_heads=8,
            # dropout=0.1
        )

        print("Initialized AttributeRetriever with dataset-level GLOBAL attribute pool (class-agnostic)")
        print(f"  Global attribute pool size: {self.prompt_learner.global_attr_features.shape[0]} attributes")

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()

        # 计算学习后的图像和文本特征
        text_features = self.text_encoder(prompts, tokenized_prompts)  # 传递 tokenized_prompts 用于 attention mask

        # ========== 多层特征提取 ==========
        if self.use_multilayer:
            # 使用多层特征提取器：提取并融合中间层特征
            fused_features, final_features = self.multilayer_extractor(image.type(self.dtype))
            # fused_features: (B, 512) - 融合后的多层特征
            # final_features: (B, 512) - 最后一层特征

            # 归一化融合后的特征
            image_features = fused_features / fused_features.norm(dim=-1, keepdim=True)
            final_features = final_features / final_features.norm(dim=-1, keepdim=True)
        else:
            # 仅使用最后一层特征（原始方法）
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # ========== 属性检索和聚合 ==========
        # 1. 使用 image_features 检索 top-5 相关属性概念
        top_attr_features, top_attr_scores = self.attr_retriever(image_features)  # (B, 5, 512), (B, 5)

        # 2. 使用注意力机制聚合属性特征
        aggregated_attr = self.attr_aggregator(image_features, top_attr_features, top_attr_scores)  # (B, 512)
        aggregated_attr = aggregated_attr / aggregated_attr.norm(dim=-1, keepdim=True)

        # ========== 交叉注意力增强图像特征 ==========
        # 3. 使用交叉注意力融合：属性特征增强图像特征
        image_features_enhanced = self.cross_attn_fusion(final_features, aggregated_attr)  # (B, 512)
        image_features_enhanced = image_features_enhanced / image_features_enhanced.norm(dim=-1, keepdim=True)
        # image_features_enhanced = image_features / image_features.norm(dim=-1, keepdim=True)

        # 4. 将聚合的属性向量与最相似的文本特征融合，生成判别性 prompt
        # aggregated_attr: (B, 512), text_features: (n_cls, 512)
        batch_size = aggregated_attr.shape[0]

        # 扩展维度: (B, n_cls, 512)
        text_features_expanded = text_features.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_cls, 512)
        aggregated_attr_expanded = aggregated_attr.unsqueeze(1).expand(-1, self.n_cls, -1)  # (B, n_cls, 512)

        # 计算属性特征与每个类别文本特征的相似度
        similarity = F.cosine_similarity(
            aggregated_attr_expanded,  # (B, n_cls, 512)
            text_features_expanded,    # (B, n_cls, 512)
            dim=2  # 在特征维度上计算余弦相似度
        )  # (B, n_cls)

        # 使用 Softmax 在类别维度上归一化，强调相似度最高的类别
        # 相似度高的类别会得到更大的权重
        fusion_weights = F.softmax(similarity / 0.1, dim=1).unsqueeze(2)  # (B, n_cls, 1), 温度=0.1 使分布更尖锐

        # 为每个类别生成判别性的属性特征
        # 策略：属性特征根据与各类别的相似度进行加权，相似度高的类别贡献更多
        class_specific_attrs = fusion_weights * aggregated_attr_expanded  # (B, n_cls, 512)

        # 计算图像特征与文本特征的相似度，用于决定文本和属性的融合比例
        img_text_similarity = F.cosine_similarity(
            image_features_enhanced.unsqueeze(1).expand(-1, self.n_cls, -1),  # (B, n_cls, 512)
            text_features_expanded,    # (B, n_cls, 512)
            dim=2
        )  # (B, n_cls)

        # 使用 Sigmoid 进行独立的权重分配
        # 相似度高 -> text_weight 高 -> 更依赖文本特征
        # 相似度低 -> attr_weight 高 -> 更依赖属性特征
        text_weight = torch.sigmoid(img_text_similarity).unsqueeze(2)  # (B, n_cls, 1)
        attr_weight = 1.0 - text_weight  # (B, n_cls, 1)
        # print(f"fusion_weights[0]: {fusion_weights[0].squeeze()}")  # 属性与类别的相似度权重
        # print(f"text_weight[0]: {text_weight[0].squeeze()}")  # 文本 vs 属性的融合权重
        # print(f"attr_weight[0]: {attr_weight[0].squeeze()}")

        # ========== 方案 2: 基于零样本置信度的自适应融合 ==========
        # 使用零样本预测的置信度来判断是 base 还是 novel


        # 路径 1: 增强路径（适合 base）- 使用自适应的文本/属性融合
        enhanced_path = 0.5 * text_features_expanded + 0.5 * class_specific_attrs  # (B, n_cls, 512)

        # 路径 2: 保守路径（适合 novel）- 主要使用原始文本，少量属性
        conservative_path = text_features_expanded   # (B, n_cls, 512)
        with torch.no_grad():
            # 计算零样本 logits（使用原始图像特征和原始文本特征）
            a = enhanced_path.mean(dim=0)  # (B, 512)
            a = a / a.norm(dim=-1, keepdim=True)
            zero_shot_logits = logit_scale * final_features @ a.t()  # (B, n_cls)

            # 计算零样本置信度：最大概率值
            # 高置信度 (接近1) -> base 类别（预训练知识强）
            # 低置信度 (接近1/n_cls) -> novel 类别（预训rained知识弱）
            zero_shot_conf = F.softmax(zero_shot_logits, dim=1).max(dim=1, keepdim=True)[0]  # (B, 1)

            # 使用陡峭的 sigmoid 将置信度映射到权重
            # 当 conf > 0.5 时，base_weight 快速增大
            # 当 conf < 0.5 时，base_weight 快速减小
            base_weight = torch.sigmoid((zero_shot_conf - 0.5) * 10)  # (B, 1)
            novel_weight = 1 - base_weight
        # 基于零样本置信度自适应融合两条路径
        # base_weight 高 -> 更多使用 enhanced_path
        # novel_weight 高 -> 更多使用 conservative_path
        text_features_enhanced = (
            base_weight.unsqueeze(1) * enhanced_path +
            novel_weight.unsqueeze(1) * conservative_path
        )  # (B, n_cls, 512)

        # 归一化
        text_features_enhanced = text_features_enhanced / text_features_enhanced.norm(dim=-1, keepdim=True)

        # 计算 logits：使用增强后的图像特征
        # image_features_enhanced (B, 512) @ text_features_enhanced (n_cls, 512).t() -> (B, n_cls)
        # logits1 = logit_scale * image_features_enhanced @ text_features_enhanced.t()  # (B, n_cls)
        # logits2 = logit_scale * image_features_enhanced @ text_features.t()  # (B, n_cls)
        # # 计算两种logits的置信度
        # conf1 = F.softmax(logits1, dim=1).max(dim=1, keepdim=True)[0]  # (B, 1)
        # conf2 = F.softmax(logits2, dim=1).max(dim=1, keepdim=True)[0]  # (B, 1)

        # # 自适应权重：置信度高的获得更大权重
        # alpha = conf1 / (conf1 + conf2 + 1e-8)  # (B, 1)
        # logits = alpha * logits1 + (1 - alpha) * logits2

        # logits =  logits1 # 平滑融合增强文本特征和原始文本特征的 logits
        logits = []
        for i in range(batch_size):
            img_feat = image_features_enhanced[i:i+1]  # (1, 512) - 使用增强后的图像特征
            txt_feat = text_features_enhanced[i]  # (n_cls, 512)
            # img_feat = final_features[i:i+1]  # (1, 512) - 使用增强后的图像特征
            # txt_feat = text_features_expanded[i]  # (n_cls, 512)
            logit = logit_scale * img_feat @ txt_feat.t()  # (1, n_cls)
            logits.append(logit)
        logits = torch.cat(logits, dim=0)  # (B, n_cls)

        if self.prompt_learner.training:
            with torch.no_grad():
                # 零样本图像特征
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)

                # 获取预训练的固定特征
                fixed_embeddings = self.prompt_learner.fixed_embeddings

                # CAPA: 动态选择有效的提示模板
                # 使用 detach() 避免梯度图被重复使用
                # 使用增强后的图像特征进行提示选择
                image_features_detached = zero_shot_features.detach()
                scores = []
                for i in range(fixed_embeddings.shape[1]):
                    temp_logits = logit_scale * image_features_detached @ fixed_embeddings[:, i, :].cuda().t()
                    max_logits = torch.max(temp_logits, dim=1).values
                    sp = torch.mean(max_logits)
                    scores.append(sp.item())

                # 使用 MAD (Median Absolute Deviation) 过滤异常值
                s_bar = torch.median(torch.tensor(scores))
                d_bar = torch.median(torch.abs(torch.tensor(scores) - s_bar))
                z = (torch.tensor(scores) - s_bar) / d_bar
                tau = self.cfg.TRAINER.BIOMEDCOOP.TAU
                mask = torch.abs((z - torch.mean(z)) / torch.std(z)) <= tau

                # 选择有效的提示特征
                selected_embeddings = fixed_embeddings[:, mask].mean(dim=1)
                selected_embeddings = selected_embeddings / selected_embeddings.norm(dim=-1, keepdim=True)

            # 计算平均的固定特征（用于 SCCM）
            # 使用 detach() 避免梯度追踪
            fixed_embeddings_mean = fixed_embeddings.mean(dim=1).detach().cuda()
            fixed_embeddings_mean = fixed_embeddings_mean / fixed_embeddings_mean.norm(dim=-1, keepdim=True)

            # 零样本 logits（用于知识蒸馏）
            zero_shot_logits = logit_scale * image_features_detached.cuda() @ selected_embeddings.cuda().t()
            # zero_shot_logits = 0.5 * (zero_shot_logits + logits)  # 平滑零样本 logits
            # 使用 Sinkhorn-Knopp 算法优化 zero_shot_logits，生成更稳定的伪标签
            # 获取 Sinkhorn 参数（如果配置中没有，使用默认值）
            tau_t = self.cfg.TRAINER.BIOMEDCOOP.get('TAU_T', 0.1)  # 温度参数
            gamma = self.cfg.TRAINER.BIOMEDCOOP.get('GAMMA', 0.1)  # Sinkhorn 正则化参数
            iters_sinkhorn = self.cfg.TRAINER.BIOMEDCOOP.get('ITERS_SINKHORN', 3)  # Sinkhorn 迭代次数

            # 应用 StableGreenkhorn 算法优化伪标签分布
            # plabel = zero_shot_logits
            plabel = StableGreenkhorn(zero_shot_logits, tau_t, gamma, iters_sinkhorn)

            # ========== 使用 image_opt 优化文本特征 ==========
            # 获取 image_opt 参数
            try:
                use_image_opt = self.cfg.TRAINER.BIOMEDCOOP.USE_IMAGE_OPT
            except (AttributeError, KeyError):
                use_image_opt = False  # 默认关闭，少样本场景下效果不好

            if use_image_opt:
                # 准备输入
                # feat: 使用学习的图像特征 (B, 512)
                # init_classifier: 使用增强的文本特征 (512, n_cls)
                # plabel: 优化后的伪标签 (B, n_cls)

                # 将 text_features_enhanced 转换为分类器格式
                # text_features_enhanced: (B, n_cls, 512) -> 对batch维度取平均得到 (n_cls, 512)
                # 然后转置为 (512, n_cls) 作为初始分类器
                text_features_for_init = text_features_enhanced.mean(dim=0)  # (n_cls, 512)
                init_classifier = text_features_for_init.t()  # (512, n_cls)

                # 获取优化参数
                opt_lr = self.cfg.TRAINER.BIOMEDCOOP.get('IMAGE_OPT_LR', 0.001)
                opt_iter = self.cfg.TRAINER.BIOMEDCOOP.get('IMAGE_OPT_ITER', 1000)
                opt_tau = self.cfg.TRAINER.BIOMEDCOOP.get('IMAGE_OPT_TAU', 0.04)
                opt_alpha = self.cfg.TRAINER.BIOMEDCOOP.get('IMAGE_OPT_ALPHA', 0.6)

                # 应用 image_opt 优化分类器
                optimized_classifier = image_opt(
                    feat=image_features,  # (B, 512)
                    init_classifier=init_classifier,  # (512, n_cls)
                    plabel=plabel,  # (B, n_cls)
                    lr=opt_lr,
                    iter=opt_iter,
                    tau_i=opt_tau,
                    alpha=opt_alpha
                )

                # 将优化后的分类器转换回文本特征格式
                optimized_text_features = optimized_classifier.t()  # (n_cls, 512)
                optimized_text_features = optimized_text_features / optimized_text_features.norm(dim=-1, keepdim=True)

                # 使用优化后的文本特征重新计算 logits
                optimized_logits = logit_scale * image_features @ optimized_text_features.t()  # (B, n_cls)
            else:
                optimized_text_features = text_features_enhanced
                optimized_logits = logits  # 不使用 image_opt 时，使用原始 logits

            # ========== 联合 optimized_logits 和 plabel（方案1：概率空间混合）==========
            # 获取混合权重（可配置）
            try:
                mix_alpha = self.cfg.TRAINER.BIOMEDCOOP.MIX_ALPHA
            except (AttributeError, KeyError):
                mix_alpha = 0.5  # 默认 0.5，平衡模型预测和零样本知识

            # 将 optimized_logits 转换为概率分布
            pred_prob = F.softmax(optimized_logits, dim=1).detach()  # (B, n_cls)

            # 在概率空间混合：结合模型预测和零样本伪标签
            mixed_prob = mix_alpha * pred_prob + (1 - mix_alpha) * plabel  # (B, n_cls)

            # ========== 四个损失函数 ==========
            # 1. 交叉熵损失（分类损失）- 使用原始 logits
            loss_ce = F.cross_entropy(optimized_logits, label)

            # 2. 自一致性约束损失（让学习的属性特征接近对应类别的预训练特征）
            # aggregated_attr: (B, 512) - 每个样本的聚合属性特征
            # fixed_embeddings_mean: (n_cls, 512) - 每个类别的固定文本特征
            # label: (B,) - 每个样本的真实类别标签

            # 根据标签选择对应类别的固定特征
            target_fixed_features = fixed_embeddings_mean[label]  # (B, 512)

            # 计算属性特征与对应类别固定特征的 MSE 损失
            loss_mse = torch.nn.MSELoss()
            loss_sccm = loss_mse(aggregated_attr, target_fixed_features.cuda()) * self.cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA

            # 3. 知识蒸馏损失（使用最优传输损失替代 KL 散度）
            # 检查是否使用最优传输损失
            try:
                use_ot_loss = self.cfg.TRAINER.BIOMEDCOOP.USE_OT_LOSS
            except (AttributeError, KeyError):
                use_ot_loss = True  

            if use_ot_loss:
                # 使用 Sinkhorn 最优传输损失
                try:
                    ot_epsilon = self.cfg.TRAINER.BIOMEDCOOP.OT_EPSILON
                except (AttributeError, KeyError):
                    ot_epsilon = 0.1  # 默认熵正则化参数

                try:
                    ot_max_iter = self.cfg.TRAINER.BIOMEDCOOP.OT_MAX_ITER
                except (AttributeError, KeyError):
                    ot_max_iter = 50  # 默认迭代次数

                try:
                    ot_weight = self.cfg.TRAINER.BIOMEDCOOP.OT_WEIGHT
                except (AttributeError, KeyError):
                    ot_weight = 0.5  # 默认 OT 权重（0.5 表示平衡 OT 和 KL）

                loss_ot = sinkhorn_loss(
                    optimized_logits,
                    mixed_prob,
                    epsilon=ot_epsilon,
                    max_iter=ot_max_iter
                )
                loss_kl = F.kl_div(
                    F.log_softmax(optimized_logits, dim=1),
                    mixed_prob,
                    reduction='batchmean'
                )
                # 混合 OT 损失和 KL 散度损失（可配置权重）
                loss_kdsp = ot_weight * loss_ot + (1 - ot_weight) * loss_kl
                # loss_kdsp = loss_kl
            else:
                # 使用传统的 KL 散度损失
                loss_kdsp = F.kl_div(
                    F.log_softmax(optimized_logits, dim=1),
                    mixed_prob,
                    reduction='batchmean'
                )
            # loss_kdsp2 = F.kl_div(
            #         F.log_softmax(optimized_logits, dim=1),
            #         mixed_prob,
            #         reduction='batchmean'
            # )
            loss_kdsp = loss_kdsp * self.cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA

            return optimized_logits, loss_ce, loss_sccm, loss_kdsp
        else:
            return logits


@TRAINER_REGISTRY.register()
class BiomedCoOp_AnchorPrompt(TrainerX):
    """
    BiomedCoOp with CAPA and Attribute Learning
    Prompt Structure: [SOS][CTX][CLS][ATTR][EOS]
    - CTX: learnable context tokens
    - ATTR: learnable attribute tokens
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
            biomedclip_model.float()

        print("Building custom CLIP with Attribute Learning")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        # 需要优化的参数：ctx, att, AttributeAggregator, multilayer_extractor, cross_attn_fusion
        names_to_update = [
            "prompt_learner.ctx",           # 上下文向量
            # "prompt_learner.att",           # 属性向量
            "attr_aggregator",              # 属性聚合模块（注意力机制）
            "multilayer_extractor",         # 多层特征提取器（注意力融合模块）
            # "cross_attn_fusion",            # 交叉注意力融合模块（属性增强图像特征）
        ]

        for name, param in self.model.named_parameters():
            # 检查参数名是否以 names_to_update 中的任何一个开头
            should_update = any(name.startswith(prefix) for prefix in names_to_update)
            if not should_update:
                param.requires_grad_(False)

        # 检查可训练参数
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # 优化器包含 ctx 和 att
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCOOP.PREC == "amp" else None

        # 多 GPU 支持
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.BIOMEDCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss_ce,loss_sccm, loss_kdsp = model(image, label)

            # 总损失 = 分类损失 + 自一致性损失 + 知识蒸馏损失
            loss = loss_ce + loss_sccm + loss_kdsp
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_sccm": loss_sccm.item(),
            "loss_kdsp": loss_kdsp.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # 默认加载最佳模型
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 删除所有类别特定的 buffers（这些都是基于类别名生成的，不能从 base 加载到 novel）
            # 这些参数的第一个维度都是 n_cls（类别数），Base 和 Novel 的类别不同
            class_specific_buffers = [
                "prompt_learner.token_prefix",      # (n_cls, 1, 768) - SOS token
                "prompt_learner.token_suffix",      # (n_cls, *, 768) - CLS + EOS tokens
                "prompt_learner.token_mean",        # (n_cls, 1, 768) - 每个类别的平均文本特征
                "prompt_learner.class_embeddings",  # (n_cls, max_len, 768) - 类别名嵌入
                "prompt_learner.fixed_embeddings",  # (n_cls, 50, 512) - 预训练的多模板文本特征
            ]

            for key in class_specific_buffers:
                if key in state_dict:
                    del state_dict[key]
                    print(f"Deleted class-specific buffer: {key}")

            # 过滤掉形状不匹配的参数（用于 base-to-novel 评估）
            # model_state_dict = self._models[name].state_dict()
            # filtered_state_dict = {}
            # mismatched_keys = []

            # for key, value in state_dict.items():
            #     if key in model_state_dict:
            #         if value.shape == model_state_dict[key].shape:
            #             filtered_state_dict[key] = value
            #         else:
            #             mismatched_keys.append(key)
            #             print(f"Skipping {key}: checkpoint shape {value.shape} != model shape {model_state_dict[key].shape}")
            #     else:
            #         # 如果 key 不在当前模型中，也跳过
            #         pass

            # if mismatched_keys:
            #     print(f"Warning: Skipped {len(mismatched_keys)} parameters due to shape mismatch")

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # 设置 strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


