from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput
from transformers import AutoConfig
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32

def model_init(model_args):
    config = AutoConfig.from_pretrained(model_args["pretrained_model_name_or_path"])
    return ModelWithBlackhole.from_pretrained(
        model_args["pretrained_model_name_or_path"],
        config=config,
        prototype_embeddings=None,
        threshold=0.85,
        rank=8,
        torch_dtype=get_dtype(model_args)
    )

class Blackhole(nn.Module):
    def __init__(self, hidden_dim, rank=8):
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(hidden_dim, rank, bias=False),
            nn.ReLU(),
            nn.Linear(rank, hidden_dim, bias=False)
        )
        # self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states):
        # mask = torch.sigmoid(self.gate(hidden_states))
        hidden_states = self.lora(hidden_states)
        return hidden_states

class PrototypeSuppressor(nn.Module):
    def __init__(self, prototype_embeddings: torch.Tensor, threshold=0.85):
        super().__init__()
        self.prototypes = prototype_embeddings
        self.threshold = threshold

    def forward(self, hidden_states: torch.Tensor, logits: torch.Tensor):
        # 计算每个样本的平均隐藏状态
        batch_embeddings = hidden_states.mean(dim=1)  # [B, D]
        
        # 计算每个样本与每个原型的余弦相似度
        # batch_embeddings: [B, D], self.prototypes: [N_proto, D]
        similarities = []
        for i in range(batch_embeddings.size(0)):  # 遍历每个样本
            sample_similarities = []
            for j in range(self.prototypes.size(0)):  # 遍历每个原型
                sim = F.cosine_similarity(
                    batch_embeddings[i].unsqueeze(0),  # [1, D]
                    self.prototypes[j].unsqueeze(0),   # [1, D]
                    dim=-1
                )  # [1]
                sample_similarities.append(sim.item())
            # 计算该样本与所有原型的平均相似度
            avg_sim = torch.tensor(sample_similarities).mean()
            similarities.append(avg_sim)
        
        # 转换为张量
        sim = torch.stack(similarities)  # [B]
        print('sim', sim)
        
        should_suppress = sim > self.threshold

        if should_suppress.any():
            mask_indices = torch.topk(logits, k=20, dim=-1).indices
            for b in range(logits.size(0)):
                if should_suppress[b]:
                    logits[b].scatter_(2, mask_indices[b], -100.0)
        return logits

# class ModelWithBlackhole(nn.Module):
#     #     target_layer.register_forward_hook(forward_hook)
#     def __init__(self, base_model, config,
#                  forget_protos=None, retain_protos=None,
#                  forget_cov=None, retain_cov=None,
#                  tau=0.0, rank=8):
#         super().__init__()
#         self.config = config
#         self.forget_protos = forget_protos  # [Kf, D]
#         self.retain_protos = retain_protos  # [Kr, D]
#         self.forget_cov = forget_cov        # [Kf, D] 对角协方差
#         self.retain_cov = retain_cov        # [Kr, D]
#         self.tau = tau
#         self.noise_scale = float(noise_scale)

#         # 若传入的是已带有LoRA的PEFT模型，则直接使用；否则为基础模型注入LoRA
#         if isinstance(base_model, PeftModel):
#             self.model_with_lora = base_model
#             # 尽量拿到底座模型用于保存完整权重
#             self.base_model = getattr(base_model, 'get_base_model', None)() if hasattr(base_model, 'get_base_model') else getattr(base_model, 'base_model', base_model)
#         else:
#             self.base_model = base_model
#             target_modules = ["layers.1.self_attn.q_proj","layers.1.self_attn.v_proj"]
#             self.model_with_lora = get_peft_model(self.base_model, LoraConfig(
#                 r=rank, lora_alpha=rank, target_modules=target_modules,
#                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))

#         # 统一LoRA参数dtype并确保可训练
#         for name, param in self.model_with_lora.named_parameters():
#             if 'lora' in name.lower():
#                 try:
#                     param.data = param.data.to(self.model_with_lora.dtype)
#                 except Exception:
#                     base_dtype = getattr(self.base_model, 'dtype', torch.float32)
#                     param.data = param.data.to(base_dtype)
#                 param.requires_grad_(True)

#         # 训练模式
#         self.model_with_lora.train()

#         self.similarity_scores = None
#         self.q_proj_features = None
#         self._hook_layer()

#     def _mahalanobis(self, x, protos, cov):
#         if protos is None:
#             # 使用大但有限的常数，避免 inf - inf 导致 NaN
#             big = torch.tensor(1e6, device=x.device, dtype=torch.float32)
#             return big.expand(x.size(0))

#         x32 = x.to(torch.float32).unsqueeze(1)            # [B,1,D]
#         p32 = protos.to(torch.float32).unsqueeze(0)       # [1,K,D]
#         if cov is not None:
#             c32 = cov.to(torch.float32)                   # [K,D]
#             c32 = torch.clamp(c32, min=1e-6)              # 防止 1/0
#             inv = 1.0 / c32.unsqueeze(0)                  # [1,K,D]
#         else:
#             inv = 1.0                                     # 标量广播

#         diff = x32 - p32                                  # [B,K,D]
#         m2 = (diff * diff * inv).sum(-1)                  # [B,K]
#         # 清理 NaN/Inf
#         m2 = torch.nan_to_num(m2, nan=1e6, posinf=1e6, neginf=1e6)
#         dmin = m2.min(dim=1).values                       # [B]
#         return dmin

#     def _hook_layer(self):
#         layers = self.model_with_lora.model.model.layers
#         attn = layers[1].self_attn

#         def _compute_mask_from_feat(feat):  # feat: [B,T,D]
#             pooled = feat.mean(dim=1)  # [B,D]
#             d_f = self._mahalanobis(pooled, self.forget_protos, self.forget_cov)
#             # print('d_f', d_f)
#             d_r = self._mahalanobis(pooled, self.retain_protos, self.retain_cov)
#             # print('d_r', d_r)
#             score = d_r - d_f
#             self.similarity_scores = score.detach()
#             mask = (score > self.tau).to(feat.dtype).view(-1,1,1)

#             if (self.forget_protos is not None) or (self.retain_protos is not None):
#                 print(score)
#             self._blackhole_mask = mask
#             return mask

#         def hook_q(module, inp, out):
#             # Cache raw q_proj features for external use (e.g., prototype clustering)
#             self.q_proj_features = out.detach()
#             if self.forget_protos is None:
#                return out
#             mask = _compute_mask_from_feat(out)
#             noise = torch.randn_like(out, dtype=out.dtype) * out.new_tensor(self.noise_scale)
#             return out + mask * noise

#         def hook_v(module, inp, out):
#             mask = getattr(self, "_blackhole_mask", None)
#             if self.forget_protos is None:
#                return out
#             noise = torch.randn_like(out, dtype=out.dtype) * out.new_tensor(self.noise_scale)
#             return out + mask * noise

#         attn.q_proj.register_forward_hook(hook_q)
#         attn.v_proj.register_forward_hook(hook_v)

#     def forward(self, input_ids=None, attention_mask=None, labels=None, *args, **kwargs):
#         self.similarity_scores = None
#         self.q_proj_features = None
        
#         outputs = self.model_with_lora(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             **kwargs
#         )
        
#         loss = outputs.loss
        
#         custom_output = CausalLMOutput(
#             loss=loss,
#             logits=outputs.logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions
#         )
        
#         # 添加自定义属性
#         custom_output.similarity_scores = self.similarity_scores
#         custom_output.q_proj_features = self.q_proj_features
        
#         return custom_output

class ModelWithBlackhole(nn.Module):
    # target_layer.register_forward_hook(forward_hook)
    def __init__(self, base_model, config,
                 forget_protos=None, retain_protos=None,
                 forget_cov=None, retain_cov=None,
                 tau=None, rank=8, noise_scale: float = 10):
        super().__init__()
        self.config = config
        self.forget_protos = forget_protos  # [Kf, D]
        self.retain_protos = retain_protos  # [Kr, D]
        self.forget_cov = forget_cov        # [Kf, D] 对角协方差
        self.retain_cov = retain_cov        # [Kr, D]
        self.tau = tau
        # noise_scale 已不再使用，保留参数为兼容

        if isinstance(base_model, PeftModel):
            self.model_with_lora = base_model
            self.base_model = getattr(base_model, 'get_base_model', None)() \
                if hasattr(base_model, 'get_base_model') else getattr(base_model, 'base_model', base_model)
        else:
            self.base_model = base_model
            # 只在所有层的v_proj上添加LoRA
            target_modules = ["layers.1.self_attn.v_proj"]
            self.model_with_lora = get_peft_model(self.base_model, LoraConfig(
                r=rank, lora_alpha=rank, target_modules=target_modules,
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))

        # 统一LoRA参数dtype并确保可训练
        for name, param in self.model_with_lora.named_parameters():
            if 'lora' in name.lower():
                try:
                    param.data = param.data.to(self.model_with_lora.dtype)
                except Exception:
                    base_dtype = getattr(self.base_model, 'dtype', torch.float32)
                    param.data = param.data.to(base_dtype)
                param.requires_grad_(True)

        # 训练模式
        self.model_with_lora.train()

        self.similarity_scores = None
        self.v_proj_features = None  # 改为基于v_proj的特征
        
        # 全局黑洞状态管理
        self.global_blackhole_mask = None  # 存储当前批次的全局抑制状态
        self.blackhole_decision_made = False  # 标记是否已经做出抑制决定
        
        self._hook_multiple_layers()

    def _mahalanobis(self, x, protos, cov):
        if protos is None:
            big = torch.tensor(1e6, device=x.device, dtype=torch.float32)
            return big.expand(x.size(0))

        x32 = x.to(torch.float32).unsqueeze(1)            # [B,1,D]
        p32 = protos.to(torch.float32).unsqueeze(0)       # [1,K,D]
        if cov is not None:
            c32 = cov.to(torch.float32)                   # [K,D]
            c32 = torch.clamp(c32, min=1e-6)              # 防止 1/0
            inv = 1.0 / c32.unsqueeze(0)                  # [1,K,D]
        else:
            inv = 1.0                                     # 标量广播

        diff = x32 - p32                                  # [B,K,D]
        m2 = (diff * diff * inv).sum(-1)                  # [B,K]
        # 清理 NaN/Inf
        m2 = torch.nan_to_num(m2, nan=1e6, posinf=1e6, neginf=1e6)
        dmin = m2.min(dim=1).values                       # [B]
        return dmin

    def _hook_multiple_layers(self):
        # 在多个attention层添加黑洞机制
        layers = self.model_with_lora.model.model.layers
        num_layers = len(layers)
        
        # 选择要添加黑洞的层（更密集的覆盖以增强遗忘效果）
        target_layer_indices = [
            i for i in range(num_layers)
        ]
        
        # 确保索引在有效范围内
        target_layer_indices = [i for i in target_layer_indices if 0 <= i < num_layers]
        target_layer_indices = list(set(target_layer_indices))  # 去重
        
        print(f"在以下层添加黑洞机制: {target_layer_indices}")
        
        for layer_idx in target_layer_indices:
            attn = layers[layer_idx].self_attn
            self._setup_blackhole_hooks(attn, layer_idx)

    def _setup_blackhole_hooks(self, attn, layer_idx):
        
        def _compute_global_mask_from_feat(feat, is_decision_layer=False):
            """计算全局抑制mask，只有决策层会真正计算，其他层直接使用"""
            if is_decision_layer and not self.blackhole_decision_made:
                # 只有第一个决策层计算抑制决定
               
                pooled = feat.mean(dim=1)  # [B,D]
                
                d_f = self._mahalanobis(pooled, self.forget_protos, self.forget_cov)
                d_r = self._mahalanobis(pooled, self.retain_protos, self.retain_cov)
               
                score = d_r - d_f
                self.similarity_scores = score.detach()
                
                # # 增强黑洞触发：使用更积极的阈值策略
                # forget_similarity = torch.exp(-d_f)  # 距离越小，相似度越高
                # retain_similarity = torch.exp(-d_r)
                
                # 多重判定条件
                print(f'score:', score)
                if self.tau is not None:
                    mask = score > self.tau  # 原始条件
                    mask = mask.view(-1,1,1)
                else:
                    mask = torch.zeros(score.size(0), device=score.device, dtype=torch.bool).view(-1,1,1)
                
                # 保存全局决定
                self.global_blackhole_mask = mask
                self.blackhole_decision_made = True
                
                if mask.any():
                    print(f'黑洞决策: {mask.sum().item()}/{mask.size(0)} 样本被抑制')
                
                return mask
            elif self.blackhole_decision_made and self.global_blackhole_mask is not None:
                # 非决策层直接使用全局决定，但需要调整批次大小
                mask = self.global_blackhole_mask
                current_batch_size = feat.size(0)
                
                # 检查并调整批次大小
                if mask.size(0) != current_batch_size:
                    # print(f'[DEBUG] 调整mask批次大小: {mask.size(0)} -> {current_batch_size}')
                    if mask.size(0) > current_batch_size:
                        # 截断mask
                        mask = mask[:current_batch_size]
                    else:
                        # 扩展mask（用False填充）
                        padding_size = current_batch_size - mask.size(0)
                        padding = torch.zeros(padding_size, 1, 1, device=mask.device, dtype=torch.bool)
                        mask = torch.cat([mask, padding], dim=0)
                
                # if mask.any():
                #     print(f'Layer {layer_idx} 黑洞跟随: {mask.sum().item()}/{mask.size(0)} 样本被抑制')
                return mask
            # else:
            #     # 如果还没有决定或没有原型，不抑制
            #     batch_size = feat.size(0)
            #     return torch.zeros(batch_size, 1, 1, device=feat.device, dtype=torch.bool)

        def _mix_output(module, x, out, do_score, cache_feat=False, use_cached_v_features=False, layer_idx=None):
            if not do_score:
                if cache_feat:
                    self.v_proj_features = out.detach()
                return out

            # 决定用于计算mask的特征
            if use_cached_v_features and hasattr(self, 'v_proj_features') and self.v_proj_features is not None:
                # 使用缓存的v_proj特征来确保维度一致性
                feat_for_mask = self.v_proj_features
            else:
                # 使用当前的输出特征
                feat_for_mask = out

            # 用特征打分（使用全局协调机制）
            is_decision_layer = (layer_idx == 0)  # 第0层是主要决策层
            mask = _compute_global_mask_from_feat(feat_for_mask, is_decision_layer=is_decision_layer)
            
            # 调试信息
            # print(f"[DEBUG] Layer {layer_idx}: mask shape={mask.shape}, out shape={out.shape}")
            
            if cache_feat:
                self.v_proj_features = out.detach()

            # 计算"无LoRA"的基座输出
            if hasattr(module, "base_layer"):
                base_out = module.base_layer(x)
                base_out = base_out.to(dtype=out.dtype)
            else:
                # 极端兼容：退化为当前out（少见）
                base_out = out

            # 广播到 [B,T,1]/[B,1,1] 适配线性输出
            mask_bool = mask.bool()
            
            # 不仅置零，还添加随机噪声
            # 根据层深度调整噪声强度，越深层噪声越强
            base_noise_scale = 0.2
            layer_factor = (layer_idx + 1) / 10.0  # 根据层深度调整
            noise_scale = base_noise_scale + layer_factor * 0.1
            noise = torch.randn_like(out) * noise_scale
            disrupted_out = noise  # 用噪声替代原输出
            
            return torch.where(mask_bool, disrupted_out, base_out)

        # 为当前层创建专用的hook函数
        def hook_v(module, inp, out):
            x = inp[0]
            do_score = (self.forget_protos is not None) and (self.retain_protos is not None)
            
            # 只有第一层缓存v_proj特征
            cache_feat = (layer_idx == 0)
            # 在决策层使用当前特征，其他层使用缓存的v_proj特征
            use_cached_v_features = (layer_idx != 0)
            return _mix_output(module, x, out, do_score, cache_feat=cache_feat, 
                             use_cached_v_features=use_cached_v_features, layer_idx=layer_idx)

        # 只在v_proj上注册hook，q_proj保持原样
        attn.v_proj.register_forward_hook(hook_v)

    @property
    def device(self):
        """获取模型设备"""
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        """获取模型数据类型"""
        return next(self.parameters()).dtype

    def generate(self, *args, **kwargs):
        """生成方法，委托给内部的LoRA模型"""
        return self.model_with_lora.generate(*args, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, labels=None, *args, **kwargs):
        # 重置所有状态
        self.similarity_scores = None
        self.v_proj_features = None  # 改为v_proj特征
        
        # 重置全局黑洞状态（每次前向传播开始时）
        self.global_blackhole_mask = None
        self.blackhole_decision_made = False

        outputs = self.model_with_lora(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        loss = outputs.loss

        custom_output = CausalLMOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

        custom_output.similarity_scores = self.similarity_scores
        custom_output.v_proj_features = self.v_proj_features  # 改为v_proj特征

        return custom_output


    @classmethod
    def from_pretrained(cls, model_path, config=None, prototype_embeddings=None, threshold=0.85, rank=8, *args, **kwargs):
        # 先加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, *args, **kwargs)

        # 如果目录下包含LoRA适配器，则将其附加到基础模型
        adapter_cfg = os.path.join(model_path, "adapter_config.json")
        adapter_bin = os.path.join(model_path, "adapter_model.bin")
        if os.path.exists(adapter_cfg) or os.path.exists(adapter_bin):
            try:
                base_model = PeftModel.from_pretrained(base_model, model_path)
            except Exception as e:
                logger.warning(f"Failed to attach LoRA adapters from {model_path}: {e}. Proceeding with base model only.")

        # 载入自定义状态
        state_path = os.path.join(model_path, "blackhole_state.pt")
        
        state = torch.load(state_path, map_location="cpu") if os.path.exists(state_path) else {}
        
        return cls(
            base_model,
            config,
            forget_protos=state.get("forget_protos"),
            retain_protos=state.get("retain_protos"),
            forget_cov=state.get("forget_cov"),
            retain_cov=state.get("retain_cov"),
            tau=state.get("tau", None),
            rank=rank,
            # noise_scale=state.get("noise_scale", 0.5),
        )
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # 1) 保存基础模型完整权重
        try:
            # 若model_with_lora提供获取base模型的方法，优先使用
            base_to_save = self.base_model
            if isinstance(self.model_with_lora, PeftModel):
                try:
                    base_to_save = self.model_with_lora.get_base_model()
                except Exception:
                    base_to_save = self.base_model
            base_to_save.save_pretrained(save_directory)
        except Exception as e:
            logger.warning(f"Saving base model failed: {e}")
            raise

        # 2) 保存LoRA适配器权重（与基础模型共存于同一目录）
        try:
            self.model_with_lora.save_pretrained(save_directory)
        except Exception as e:
            logger.warning(f"Saving LoRA adapters failed: {e}")

        # 3) 保存自定义状态（原型/阈值等）
        state = {
            "tau": self.tau,
            "forget_protos": self.forget_protos,
            "retain_protos": self.retain_protos,
            "forget_cov": self.forget_cov,
            "retain_cov": self.retain_cov,
        }
        torch.save(state, os.path.join(save_directory, "blackhole_state.pt"))

def get_model(model_cfg: DictConfig, trainer=None):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    
    # 检查是否请求黑洞模型
    if trainer=='MyUnlearning':
        model = model_init(model_args)
    else:
        model_cls = MODEL_REGISTRY[model_handler]
        with open_dict(model_args):
            model_path = model_args.pop("pretrained_model_name_or_path", None)
        try:
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                **model_args,
                cache_dir=hf_home,
            )
            # 若目录包含LoRA适配器，则自动装载
            try:
                adapter_cfg = os.path.join(model_path, "adapter_config.json") if isinstance(model_path, str) else None
                adapter_bin = os.path.join(model_path, "adapter_model.bin") if isinstance(model_path, str) else None
                if adapter_cfg and (os.path.exists(adapter_cfg) or (adapter_bin and os.path.exists(adapter_bin))):
                    model = PeftModel.from_pretrained(model, model_path)
            except Exception as e:
                logger.warning(f"Attaching LoRA adapters to model from {model_path} failed: {e}")
        except Exception as e:
            logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
            raise ValueError(
                f"Error {e} while fetching model using {model_handler}.from_pretrained()."
            )
    
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
_register_model(ModelWithBlackhole)  # 注册黑洞模型
