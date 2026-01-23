import timm
import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import os
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
                # 构造属性字典
from trainers.atrr import BIOMEDCOOP_attr
import matplotlib.pyplot as plt
import seaborn as sns

# 添加 Transformers 导入
try:
    from transformers import AutoImageProcessor, AutoModel
    import torchvision.transforms as transforms
    TRANSFORMERS_AVAILABLE = True
    print("Transformers successfully imported")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available, please install: pip install transformers")
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# os.environ.pop("HTTPS_PROXY", None)
# 初始化 DINOv2 模型和处理器
if TRANSFORMERS_AVAILABLE:
    try:
        dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        dino_model = AutoModel.from_pretrained('facebook/dinov2-base')
        # 将模型移动到正确的设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dino_model = dino_model.to(device)
        dino_model.eval()
        print("DINOv2 model and processor loaded successfully")
    except Exception as e:
        print(f"Failed to load DINOv2 model: {e}")
        dino_processor = None
        dino_model = None
else:
    dino_processor = None
    dino_model = None
class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts,tokenized_prompts):

        x = self.model.encode_text(prompts,True,tokenized_prompts)

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDCOOP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDCOOP.CTX_INIT
        att_init = cfg.TRAINER.BIOMEDCOOP.ATT_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        self.n_att = 4
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx==4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.BIOMEDCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        prompt2 = self.tokenizer(att_init)
        with torch.no_grad():
            embedding3 = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt2).type(dtype)
        attr_vectors = embedding3[0, 1: 1 + self.n_att, :]

        prompt3 = [att_init + " " + name + "." for name in classnames]


        # with torch.no_grad():
        #     embedding3 = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt3).type(dtype)
        # attr_vectors = embedding3[0, 1: 1 + self.n_att, :]
        # Tokenize prompts
        self.tokenized_prompts2 = torch.cat([self.tokenizer(p) for p in prompt3])
        
        # Get embeddings

        self.ctx = nn.Parameter(ctx_vectors)
        self.att = nn.Parameter(attr_vectors)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        biomedclip_model_temp,_ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()
        # with torch.no_grad():
        #     self.embedd = biomedclip_model.text.transformer.embeddings.word_embeddings(self.tokenized_prompts3).type(dtype)
        # classnames = ["glioma tumor", "meningioma tumor", "normal brain", "pituitary tumor"]
        # print("classnames:", classnames)
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = biomedclip_model_temp.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []

            for i in range(cfg.TRAINER.BIOMEDCOOP.N_PROMPTS):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDCOOP_TEMPLATES[classname][i]) for classname in classnames])
                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)
        
        # 预计算属性嵌入，类似fixed_embeddings的结构
        from trainers.atrr import BIOMEDCOOP_attr
        
        all_attribute_features = []
        attribute_categories = []
        attribute_descriptions = []
        
        print("Pre-computing attribute embeddings...")
        for category, descriptions in BIOMEDCOOP_attr.items():
            print(f"  Processing attribute category: {category}")
            category_features = []
            
            for desc in descriptions:
                desc_tokens = torch.cat([self.tokenizer(desc)])
                with torch.no_grad():
                    desc_features = biomedclip_model_temp.encode_text(desc_tokens.cuda())
                    desc_features = desc_features / desc_features.norm(dim=-1, keepdim=True)
                    category_features.append(desc_features)
            
            if category_features:
                category_features = torch.cat(category_features, dim=0)  # [n_descriptions, feature_dim]
                all_attribute_features.append(category_features)
                attribute_categories.append(category)
                attribute_descriptions.append(descriptions)
                print(f" Built features for {category}: {category_features.shape}")
        
        if all_attribute_features:
            # 保存为列表而不是拼接，因为不同类别可能有不同数量的描述
            self.attribute_embeddings = all_attribute_features  # List of tensors
            self.attribute_categories = attribute_categories
            self.attribute_descriptions = attribute_descriptions
            print(f"Built attribute embeddings for {len(attribute_categories)} categories")
            for i, category in enumerate(attribute_categories):
                print(f"  {category}: {all_attribute_features[i].shape}")
        else:
            self.attribute_embeddings = None
            self.attribute_categories = []
            self.attribute_descriptions = []
            print("No attribute embeddings were built")
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_att", embedding3[:, 1 + self.n_att:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.tokenized_attr_prompts = self.tokenized_prompts2  # torch.Tensor

        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix
        # import pdb;pdb.set_trace()
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def forward(self):

        ctx = self.ctx
        att = self.att
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            att = att.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        att_suffix = self.token_att
  

        prompts = self.construct_prompts(ctx, prefix, suffix)
        # prompts_attr = self.construct_attr_prompts(ctx, prefix, suffix)
        prompts_attr = self.construct_prompts(att, prefix, att_suffix)

        return prompts, prompts_attr

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.tokenized_prompts2 = self.prompt_learner.tokenized_prompts2

        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.biomedclip_model = biomedclip_model
        self.n_cls = len(classnames) 
        # 动态获取属性类别数和特征维度
        self.n_attr = len(BIOMEDCOOP_attr)
        self.feature_dim = 512  # 你可以根据实际属性特征维度调整
        self.n_classes = len(classnames)
        # 属性特征加权参数（可学习）
        self.attr_weights = nn.Parameter(torch.empty(self.n_classes, 40*self.n_classes))
        nn.init.kaiming_normal_(self.attr_weights, mode='fan_in', nonlinearity='relu')
        # self.attr_weights = nn.Parameter(torch.randn(self.n_cls, 120))
        self.dino_adapter = nn.Linear(768, self.feature_dim)  # DINOv2-base输出768维
        self.dino_proj = nn.Linear(768, 768 * 14 * 14)
        self.out_features_proj = nn.Linear(512, 512 * 14 * 14)

        # 分类头
        self.tumor_classifier = nn.Linear(768, self.n_classes)
        self.dino_classifier = nn.Linear(512, self.n_classes)  # 512是BiomedCLIP的输出维度
        self.adapter = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )
    
    def extract_last_block_features(self, image):
        """
        提取BiomedCLIP图像编码器最后一层Block的特征
        """
        # 获取图像编码器的trunk部分（通常是Vision Transformer）
        trunk = self.image_encoder.trunk
        
        # 获取patch embedding
        x = trunk.patch_embed(image)
        batch_size, num_patches, embed_dim = x.shape
        
        # 添加class token
        cls_token = trunk.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加位置编码
        x = trunk.pos_drop(x + trunk.pos_embed)
        
        # 前向传播到倒数第二层
        for i, block in enumerate(trunk.blocks[:-1]):
            x = block(x)
        
        # 获取最后一层Block的输入特征
        last_block_input = x.clone()
        
        # 通过最后一层Block
        last_block_output = trunk.blocks[-1](x)
        
        # 应用最终的norm
        final_features = trunk.norm(last_block_output)
        
        # 提取CLS token作为最终特征
        cls_features = final_features[:, 0, :]  # [batch_size, embed_dim]
        
        # 通过head得到最终输出
        final_output = self.image_encoder.head(cls_features)
        
        return final_output, cls_features, last_block_input, last_block_output
    
    def cf_prompt(self, P_cls, P_attr, labels, alpha=0.5):
        """
        P_cls:  [C, 512]
        P_attr: [C, 512]
        labels: [B]
        """
        B = labels.size(0)
        C = P_cls.size(0)

        P_true_list = []
        P_cf_list = []

        for i in range(B):
            y = labels[i].item()

            P_true = alpha * P_cls[y] + (1-alpha) * P_attr[y]  # [512]
            P_true_list.append(P_true)
            k = torch.randint(0, C, (1,), device=labels.device).item()
            while k == y:
                k = torch.randint(0, C, (1,), device=labels.device).item()
            j = torch.randint(0, C, (1,), device=labels.device).item()
            while j == y:
                j = torch.randint(0, C, (1,), device=labels.device).item()

            P_cf = 0.5*P_cls[y] + 0.5*P_attr[j]    # [512]
            P_cf_list.append(P_cf)

        P_true = torch.stack(P_true_list)  # [B, 512]
        P_cf = torch.stack(P_cf_list)      # [B, 512]
        return P_true, P_cf
    
    def custom_attn(self, attn_layer, x, ex_feats=None, beta=1.2, gamma=3.0, token_size=(16, 16)):
        # x: (seq_len, batch, embed_dim)  timm的forward格式
        seq_len, bsz, embed_dim = x.size()
        num_heads = attn_layer.num_heads
        head_dim = embed_dim // num_heads

        # 1. 获取qkv（timm Attention用qkv，而不是in_proj_weight/bias）
        # 输入需(B, N, C)
        qkv = attn_layer.qkv(x.transpose(0, 1))  # (B, N, 3*embed_dim)
        qkv = qkv.reshape(bsz, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        # 2. 只处理 patch tokens的v（去掉class token），并reshape成空间格式
        v = v[:, :, 1:, :]  # (B, heads, n_patches, head_dim)
        H, W = token_size
        n_patches = v.shape[2]  # 实际的patch数量
        
        # 检查patch数量是否与token_size匹配
        expected_patches = H * W
        if n_patches != expected_patches:
            # 如果patch数量不匹配，进行插值调整
            v_reshaped = v.permute(1, 0, 2, 3).reshape(num_heads * bsz, n_patches, head_dim)
            v_reshaped = v_reshaped.permute(0, 2, 1).unsqueeze(3)  # (num_heads*bsz, head_dim, n_patches, 1)
            
            # 计算当前的空间尺寸
            current_size = int(np.sqrt(n_patches))
            v_spatial = v_reshaped.view(num_heads * bsz, head_dim, current_size, current_size)
            
            # 插值到目标尺寸
            v_spatial = F.interpolate(v_spatial, size=(H, W), mode='bilinear', align_corners=False)
            v = v_spatial.view(num_heads * bsz, H * W, head_dim).permute(0, 2, 1).view(num_heads * bsz, H, W, head_dim).permute(0, 3, 1, 2)
        else:
            v = v.permute(1, 0, 2, 3).reshape(num_heads * bsz, H, W, head_dim).permute(0, 3, 1, 2)  # (num_heads*B, head_dim, H, W)

        # 3. 对齐到DINO特征空间
        B, C, H2, W2 = ex_feats.shape
        v = F.interpolate(v, size=(H2, W2), mode='bilinear', align_corners=False)
        v = v.permute(0, 2, 3, 1).reshape(num_heads * bsz, H2 * W2, head_dim)

        # 4. 生成DINO空间自注意力mask
        q_k = F.normalize(ex_feats.flatten(2, 3), dim=1)  # (B, C, H2*W2)
        similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k)  # (B, H2*W2, H2*W2)
        similarity = (similarity - torch.mean(similarity) * beta) * gamma
        similarity[similarity < 0.0] = float('-inf')
        mask = similarity.to(v.dtype).unsqueeze(1).repeat(1, num_heads, 1, 1)  # (B, num_heads, H2*W2, H2*W2)
        mask = mask.reshape(num_heads * B, H2 * W2, H2 * W2)
        attn_weights = F.softmax(mask, dim=-1)

        # 5. 空间加权融合
        attn_output = torch.bmm(attn_weights, v)  # (num_heads*B, H2*W2, head_dim)
        attn_output = attn_output.reshape(num_heads, bsz, H2 * W2, head_dim).permute(1, 0, 2, 3).reshape(bsz, H2 * W2, embed_dim)

        # 6. 拼回class token（用0向量或原始class token，这里用0向量）
        cls_token = torch.zeros(bsz, 1, embed_dim, device=attn_output.device, dtype=attn_output.dtype)
        attn_output = torch.cat([cls_token, attn_output], dim=1)  # (B, 1+H2*W2, embed_dim)
        attn_output = attn_output.transpose(0, 1)  # (1+H2*W2, B, embed_dim)

        # 7. out_proj
        attn_output = attn_layer.proj(attn_output)
        
        # 返回完整的序列，让后续的层继续处理
        # attn_output现在是 (seq_len, batch, embed_dim) 格式
        return attn_output
    def cf_loss(self, F_img, P_true, P_cf, margin=0.5):
        """
        F_img:  [B, 512]
        P_true: [B, 512]
        P_cf:   [B, 512]
        """
        sim_t = torch.cosine_similarity(F_img, P_true, dim=-1)  # [B]
        sim_f = torch.cosine_similarity(F_img, P_cf, dim=-1)    # [B]
        # print("sim_t:", sim_t)
        # print("sim_f:", sim_f)
        loss = F.relu(margin + sim_f - sim_t)                   # [B]
        return loss.mean()
    def info_nce_loss(self,F_img, P_true, P_cf, temperature=0.07):
        """
        F_img:  [B, D]       -- anchor（图像特征）
        P_true: [B, D]       -- positive
        P_cf:   [B, D]       -- negative
        """
        B = F_img.size(0)
        D = F_img.size(1)

        # L2 normalize
        F_img = F_img / F_img.norm(dim=1, keepdim=True)
        P_true = P_true / P_true.norm(dim=1, keepdim=True)
        P_cf = P_cf / P_cf.norm(dim=1, keepdim=True)

        # 计算相似度
        pos_sim = torch.exp(torch.sum(F_img * P_true, dim=1) / temperature)  # [B]
        neg_sim = torch.exp(torch.sum(F_img * P_cf, dim=1) / temperature)    # [B]

        # InfoNCE 损失
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        return loss.mean()

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        tokenized_prompts2 = self.tokenized_prompts2

        logit_scale = self.logit_scale.exp()

        prompts, prompts_attr = self.prompt_learner()
        
        image = image.type(self.dtype)

        image_features = self.image_encoder(image.type(self.dtype))

        # 为out_features生成空间注意力，然后融合
        batch_size = image_features.shape[0]
        
        # 将out_features投影到空间格式用于生成注意力
        out_features_spatial = self.out_features_proj(image_features)  # [batch, 512*14*14]
        out_features_spatial = out_features_spatial.view(batch_size, 512, 14, 14)  # [batch, 512, 14, 14]
        
        # 生成空间注意力mask
        spatial_attention = F.normalize(out_features_spatial.flatten(2, 3), dim=1)  # [batch, 512, 196]
        spatial_similarity = torch.einsum("b c m, b c n -> b m n", spatial_attention, spatial_attention)  # [batch, 196, 196]
        
        # 应用温度参数和阈值
        temperature = 0.1
        threshold = 0.5
        spatial_similarity = spatial_similarity / temperature
        spatial_similarity[spatial_similarity < threshold] = float('-inf')
        spatial_mask = F.softmax(spatial_similarity, dim=-1)  # [batch, 196, 196]
        
        # 将空间注意力mask应用到out_features_spatial上
        out_spatial_flat = out_features_spatial.flatten(2, 3)  # [batch, 512, 196]
        out_spatial_masked = torch.bmm(out_spatial_flat, spatial_mask)  # [batch, 512, 196]
        out_features_enhanced = out_spatial_masked.view(batch_size, 512, 14, 14)  # [batch, 512, 14, 14]
        
        # 将增强后的空间特征压缩回512维
        out_features_enhanced_flat = out_features_enhanced.mean(dim=(2, 3))  # [batch, 512] - 空间平均池化
        # 或者使用自适应池化
        # out_features_enhanced_flat = F.adaptive_avg_pool2d(out_features_enhanced, (1, 1)).squeeze(-1).squeeze(-1)  # [batch, 512]
        
        # 加权融合image_features和增强后的out_features
        alpha = 0.8  # 可学习的权重参数
        alpha_sigmoid = torch.sigmoid(self.alpha)
        fused = alpha * image_features + (1 - alpha) * out_features_enhanced_flat  # [batch, 512]
        


        text_features1 = self.text_encoder(prompts, tokenized_prompts)
        text_features2 = self.text_encoder(prompts_attr, tokenized_prompts2)

        # text_features = alpha_sigmoid * text_features1 + (1-alpha_sigmoid) * text_features2
        text_features = text_features2 
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        dino_cls = fused / fused.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) 
        # Compute the prompted logits
        logits = logit_scale * dino_cls @ text_features.t()
        logits_fix = logit_scale * dino_cls @ text_features1.t()
        if label is not None:
            P_true, P_cf = self.cf_prompt(text_features1, text_features2, label, alpha=alpha_sigmoid)
            loss_cf = self.cf_loss(dino_cls, P_true, P_cf) * 10.0
        else:
            # 验证时不计算cf_loss
            loss_cf = None



        if hasattr(self.prompt_learner, 'attribute_embeddings') and self.prompt_learner.attribute_embeddings is not None:
                attribute_embeddings = self.prompt_learner.attribute_embeddings  # List of tensors
                attribute_categories = self.prompt_learner.attribute_categories

        attribute_embeddings = [emb / emb.norm(dim=-1, keepdim=True) for emb in attribute_embeddings]
        image_attribute_similarities = {}
        for i, category in enumerate(attribute_categories):
            category_embeddings = attribute_embeddings[i]  # [n_descriptions, feature_dim]
                        # 计算图像特征与每个属性描述的相似度
            similarities = torch.matmul(image_features, category_embeddings.t())  # [batch_size, n_descriptions]
            image_attribute_similarities[category] = similarities
        selected_attr_features = []

        fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        # perm1 = fixed_embeddings.reshape(-1, fixed_embeddings.shape[-1])
            # 融合
            # fused_attr_feature = torch.stack(selected_attr_features, dim=0).sum(dim=0)  # [batch_size, feature_dim]
        attribute_embeddings = torch.cat(attribute_embeddings, dim=0)  # [102, 512]     
            # attr_weights = F.softmax(self.attr_weights, dim=1)  # [n_cls, 102]
       
        reshaped_embeddings = fixed_embeddings.view(-1, 512)
     
        fused_attr_feature = F.softmax(self.attr_weights, dim=1) @ reshaped_embeddings  # [n_cls, 512]       
               # fused_attr_feature = self.attr_weights @ attribute_embeddings
            # 分类头
            # tumor_logits = self.tumor_classifier(fused_attr_feature)       # [batch_size, n_classes]
        tumor_logits = logit_scale * image_features @ fused_attr_feature.t()
        if self.prompt_learner.training:           
            loss_ce = F.cross_entropy(logits,label)
            # loss_ce = F.cross_entropy(logits,label)

            
            loss_mse = torch.nn.MSELoss()
            loss_sccm = loss_mse(text_features1, fused_attr_feature.cuda()) 

            loss_kdsp = F.kl_div(
                F.log_softmax(logits_fix, dim=1),
                F.log_softmax(tumor_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel()
            loss_kdsp = loss_kdsp * self.cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA


            if str(self.cfg.TRAINER.BIOMEDCOOP.STAGE) == "1":
                tumor_ce = F.cross_entropy(tumor_logits, label)
                # forward 的返回值必须和 trainer 对应
                return tumor_logits, tumor_ce, loss_sccm, loss_kdsp
            # tumor_ce = F.cross_entropy(tumor_logits, label)  # 不加 .item()

            # 使用dino_cls进行分类
            # dino_classifier = nn.Linear(dino_cls.shape[-1], self.n_classes).to(dino_cls.device)
            # dino_logits = dino_classifier(dino_cls)
            # tumor_ce2 = F.cross_entropy(log, label)  # 不加 .item()

            # 分类语义结构
            sim_text = text_features @ text_features.t()
            sim_attr = fused_attr_feature @ fused_attr_feature.t()

            p_text = F.softmax(sim_text / 0.07, dim=-1)
            p_attr = F.softmax(sim_attr / 0.07, dim=-1)

            # JS divergence
            m = 0.5 * (p_text + p_attr)
            loss_align = 0.5 * (
                F.kl_div(p_text.log(), m, reduction='batchmean') +
                F.kl_div(p_attr.log(), m, reduction='batchmean')
            ) * self.cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA


            loss_kdsp2 = F.kl_div(
                F.log_softmax(tumor_logits, dim=1),
                F.log_softmax(logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel()
            loss_kdsp2 = loss_kdsp2 * self.cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA
            loss_sccm2 = loss_mse(text_features, fused_attr_feature.cuda()) * self.cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA

            return logits, loss_ce, loss_sccm, loss_kdsp, loss_kdsp2, loss_align, loss_cf#, tumor_ce2 #, loss_kdsp2
        else:
            # Stage1 推理返回 tumor_logits
            if str(self.cfg.TRAINER.BIOMEDCOOP.STAGE) == "1":
                return tumor_logits
            return logits  


@TRAINER_REGISTRY.register()
class BiomedCoOp_DPTA(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        # classnames = ["glioma tumor", "meningioma tumor", "normal brain", "pituitary tumor"]

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
            biomedclip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        # ===========================
        #  Two-Stage Training Control
        # ===========================
        stage = cfg.TRAINER.BIOMEDCOOP.STAGE

        if str(self.cfg.TRAINER.BIOMEDCOOP.STAGE) == "1":
            print("======== Stage 1: Training ONLY attr_weights ========")
            names_to_update = [
                "attr_weights",
                # "out_features_proj.weight", "out_features_proj.bias"
            ]

        elif str(self.cfg.TRAINER.BIOMEDCOOP.STAGE) == "2":
            print("======== Stage 2: Training ONLY prompt_learner.att ========")
            names_to_update = [
                # "attr_weights",
                # "prompt_learner.ctx",
                "alpha",
                "prompt_learner.att",
                "out_features_proj.weight", "out_features_proj.bias"
            ]

        else:
            raise ValueError("STAGE must be 1 or 2")


        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        if cfg.MODEL.INIT_WEIGHTS:
            print(f"Loading pretrained weights from {cfg.MODEL.INIT_WEIGHTS}")
            try:
                load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
                print("Pretrained weights loaded successfully.")
            except FileNotFoundError as e:
                print(f"Failed to load pretrained weights: {e}")
            except Exception as e:
                print(f"An error occurred while loading pretrained weights: {e}")
        self.model.to(self.device)  # 确保模型在正确的设备上
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCOOP.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def visualize_attr_weights(self, save_path="attr_comparison_results.png"):
        initial_weights = nn.Parameter(torch.empty(4, 120))
        nn.init.kaiming_normal_(initial_weights, mode='fan_in', nonlinearity='relu')

        # 1. 提取当前模型中的权重
        model_to_vis = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        current_weights = model_to_vis.attr_weights.detach().cpu().float().numpy()
        init_weights = initial_weights.detach().cpu().float().numpy()

        # 2. 创建画布：2行1列，长宽比设为横向长条 (18x10)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

        # 绘图公共参数：模仿蓝色网格图风格
        style_args = {
            'cmap': 'Blues',
            'linewidths': 1.0,
            'linecolor': 'white',
            'cbar_kws': {'label': 'Intensity', 'shrink': 0.8},
            'vmin': 0,
            'vmax': 1, # 假设权重经过归一化或在0-1之间激活
            'yticklabels': [f"Class {i+1}" for i in range(4)]
        }

        # 3. 绘制上方图：原始初始化 (Original)
        sns.heatmap(init_weights, ax=ax1, **style_args)
        ax1.set_title("Original Concept Activation (Initial Kaiming Weights)", fontsize=16, pad=15)
        ax1.set_ylabel("Class Index", fontsize=12)

        # 4. 绘制下方图：稀疏激活 (Sparse)
        sns.heatmap(current_weights, ax=ax2, **style_args)
        ax2.set_title("Sparse Concept Activation (Trained Weights)", fontsize=16, pad=15)
        ax2.set_xlabel("Attribute Index", fontsize=12)
        ax2.set_ylabel("Class Index", fontsize=12)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison heatmap (4x120) saved to: {save_path}")  
    def analyze_prompt_nearest_tokens(self, topk=5,save_path="attribute_prompt_tokens.csv"):
        """
        Analyze nearest vocabulary tokens for learned ATTRIBUTE prompts
        using BiomedCLIP (PubMedBERT) word embeddings.
        """

        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model.eval()

        # -------------------------------------------------
        # 1. Get attribute prompt vectors
        # -------------------------------------------------
        ctx = model.prompt_learner.ctx.detach().cpu().float()  # [n_att, dim]

        att = model.prompt_learner.att.detach().cpu().float()  # [n_att, dim]


        if att.dim() == 3:   # CSC case
            att = att.mean(dim=0)

        n_att, dim = att.shape
        print(f"[Attribute Prompt Analysis] {n_att} tokens, dim={dim}")

        # -------------------------------------------------
        # 2. Get vocabulary word embeddings (KEY FIX)
        # -------------------------------------------------
        word_embed = (
            model.biomedclip_model
            .text
            .transformer
            .embeddings
            .word_embeddings
            .weight
            .detach()
            .cpu()
            .float()
        )  # [vocab_size, dim]

        # -------------------------------------------------
        # 3. Normalize for cosine similarity
        # -------------------------------------------------
        # att = F.normalize(att, dim=1)
        # ctx = F.normalize(ctx, dim=1)
        word_embed = F.normalize(word_embed, dim=1)
        att_mean = att.mean(dim=0, keepdim=True)  # [1, dim]
        ctx_mean = ctx.mean(dim=0, keepdim=True)  # [1, dim]

        # 2. 归一化
        att_mean_norm = F.normalize(att_mean, dim=1)  # [1, dim]
        ctx_mean_norm = F.normalize(ctx_mean, dim=1)  # [1, dim]

        # 3. 计算余弦相似度（标量）
        overall_sim = torch.matmul(att_mean_norm, ctx_mean_norm.T).item()

        print(f"att 和 ctx 的总体余弦相似度: {overall_sim:.4f}")
        # -------------------------------------------------
        # 4. Compute cosine distance
        # -------------------------------------------------
        sim = att @ word_embed.t()
        dist = 1.0 - sim
        att_expanded = att.unsqueeze(1)          # [n_att, 1, dim]
        word_embed_expanded = word_embed.unsqueeze(0)  # [1, vocab_size, dim]

        l1_dist = torch.abs(att_expanded - word_embed_expanded).sum(dim=2)  # [n_att, vocab_size]
        dist_matrix = torch.norm(att_expanded - word_embed_expanded, p=2, dim=2)
        values, indices = dist_matrix.topk(k=topk, largest=False, dim=1)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        # tokens = tokenizer.convert_ids_to_tokens(ids)
        # tokenizer = model.prompt_learner.tokenizer
        
        print("\n========== Attribute Prompt Nearest Tokens ==========")
        for i in range(n_att):
            token_ids = indices[i].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            dists = values[i].tolist()

            pairs = [f"{t} ({d:.4f})" for t, d in zip(tokens, dists)]
            print(f"Attribute Token #{i+1}: " + ", ".join(pairs))
        print("====================================================")


    def visualize_all_comparisons_mako(self, save_path="mako_comp"):

        # 获取当前模型
        model_to_vis = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        
        # 提取当前数据
        curr_ctx = model_to_vis.prompt_learner.ctx.detach().cpu().float()
        curr_att = model_to_vis.prompt_learner.att.detach().cpu().float()
        
        # 处理维度 (针对 CSC 模式取均值)
        if curr_ctx.dim() == 3: curr_ctx = curr_ctx.mean(dim=0)
        if curr_att.dim() == 3: curr_att = curr_att.mean(dim=0)

        def plot_mako_pair(data_init, data_curr, title, file_name, is_att=False):
            # 为 att 矩阵设置更扁长的画布 (18x8)，为 ctx 设置标准比例 (18x10)
            fig_h = 8 if is_att else 10
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, fig_h))
            
            # 统一参数配置
            kwargs = {
                'cmap': 'RdBu_r',
                'linewidths': 0.8 if is_att else 0, # att 开启细网格模仿参考图
                'linecolor': '#202020' if is_att else None, # 深色网格线配合 mako
                'cbar_kws': {'label': 'Activation Intensity'},
                'center': 0 if not is_att else None # ctx 通常包含正负值，设置中心
            }

            # 绘制初始化
            sns.heatmap(data_init.numpy(), ax=ax1, **kwargs)
            ax1.set_title(f"category prompt", fontsize=16, color='black')
            ax1.set_ylabel("Token Index")

            # 绘制训练后
            sns.heatmap(data_curr.numpy(), ax=ax2, **kwargs)
            ax2.set_title(f"attribute prompt", fontsize=16, color='black')
            ax2.set_ylabel("Token Index")
            # ax2.set_xlabel("Dimension / Feature Index" if not is_att else "Attribute Index (120)")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Mako comparison saved to: {save_path}")

        # 1. 对比 Context Embeddings (ctx)
        plot_mako_pair(curr_ctx, curr_att, "Two Prompt Context Embeddings", "ctx", is_att=False)

    def visualize_prompt(self, save_path="prompt_heatmap.png"):

            # 1. 提取模型
            model_to_vis = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            
            try:
                # 根据 Stage 自动选择数据 attr_weights
                if str(self.cfg.TRAINER.BIOMEDCOOP.STAGE) == "2" and hasattr(model_to_vis.prompt_learner, 'att'):
                    # data = model_to_vis.attr_weights.att.detach().cpu().float()
                    data = model_to_vis.prompt_learner.att.detach().cpu().float()

                    title_prefix = "Attention Matrix"
                else:
                    data = model_to_vis.prompt_learner.ctx.detach().cpu().float()
                    title_prefix = "Context Embeddings"

                if data.dim() == 3:
                    data = data.mean(dim=0)

                # 2. 动态调整画布高度
                # 如果 Token 数 > 50，高度自动设为 10，否则设为 5
                num_tokens = data.shape[0]
                height = 10 if num_tokens > 50 else 5
                plt.figure(figsize=(18, height)) 

                # 3. 更换颜色与样式优化
                # 推荐颜色：'coolwarm' (蓝红), 'RdBu_r' (经典红蓝), 'mako' (深绿蓝)
                sns.heatmap(
                    data.numpy(), 
                    cmap='mako',       # 换成经典的红蓝对比色，中性值为白色
                    center=0,            # 强制 0 为颜色中心点
                    cbar_kws={'label': 'Value Range'},
                    yticklabels=max(1, num_tokens // 20), # 如果 token 太多，每隔几个显示一个标签
                    xticklabels=100
                )

                plt.title(f"{title_prefix} | Tokens: {num_tokens} | Stage: {self.cfg.TRAINER.BIOMEDCOOP.STAGE}")
                plt.xlabel("Embedding Dimension (768)")
                plt.ylabel("Token Index")

                # 4. 保存
                                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"Heatmap (Tokens: {num_tokens}) saved to: {save_path}")

            except Exception as e:
                print(f"Visualization error: {e}")
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        # visualize_prompt_ctx(model, title="DPC Prompt Visualization")
        prec = self.cfg.TRAINER.BIOMEDCOOP.PREC
        stage = self.cfg.TRAINER.BIOMEDCOOP.STAGE
        if str(self.cfg.TRAINER.BIOMEDCOOP.STAGE) == "1":
            logits, tumor_ce,loss_sccm, loss_kdsp = model(image, label)
            loss = tumor_ce + 0.5* loss_kdsp #+ loss_sccm     # Stage1 只有 tumor_ce
        else:
            logits, loss_ce, loss_sccm, loss_kdsp, loss_kdsp2, loss_sccm2, loss_triplet = model(image, label)
            loss = loss_ce + loss_kdsp2 + loss_sccm2 + loss_triplet
            # loss = loss_ce #+ loss_sccm2 + loss_triplet

        # if stage == "1":
        #     loss =tumor_ce
        # else:
        #     logits, loss_ce, loss_sccm, loss_kdsp, tumor_ce, loss_kdsp2, loss_sccm2,loss_triplet = model(image, label)

        #     loss = loss_ce + loss_kdsp2 + loss_sccm2 + loss_triplet

        # if prec == "amp":
        #     with autocast():
        #         loss = model(image, label)
        #     optim.zero_grad()
        #     scaler.scale(loss).backward()
        #     scaler.step(optim)
        #     scaler.update()
        # else:
        #     logits, loss_ce, loss_sccm, loss_kdsp, tumor_ce, loss_kdsp2, loss_sccm2,loss_triplet = model(image, label)
            
        #     loss = loss_ce + loss_kdsp2 + loss_sccm2 + loss_triplet + tumor_ce
            # loss = tumor_ce
        self.model_backward_and_update(loss)

        loss_summary = {
            # "loss_ce": loss_ce.item(),
            # "loss_sccm2": loss_sccm2.item(),
            # "loss_kdsp2": loss_kdsp2.item(),
            # "loss_cl": loss_triplet.item(),
            "loss": loss.item(),
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

        # By default, the best model is loaded
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

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    # def after_epoch(self):
    #     curr_result = self.test(split="val")
    #     is_best = not hasattr(self, "best_result") or curr_result > getattr(self, "best_result", float('-inf'))
    #     last_epoch = (self.epoch + 1) == self.max_epoch  # 注意 self.max_epoch 是否有定义
    #     if is_best:
    #         self.best_result = curr_result
    #         self.save_model(self.epoch, self.output_dir, is_best=True, model_name="model-best.pth.tar")
    #         print(f"[after_epoch] Best model updated at epoch {self.epoch}, val={curr_result}")
        # if last_epoch:
        #     # 总是保存最后一个 epoch 的模型
        #     self.save_model(self.epoch, self.output_dir)


