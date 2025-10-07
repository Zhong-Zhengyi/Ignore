import logging
import torch
import numpy as np
import requests
import json
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, Any
import os

from evals.metrics.utils import run_batchwise_evals
from evals.metrics.base import unlearning_metric
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("evaluator")


class QwenAPIEvaluator:
    """使用qwen-max-2025-01-25模型API进行语义评估"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/api/v1"):#https://dashscope.aliyuncs.com/compatible-mode/v1
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "qwen-max-2025-01-25"
        
    def call_api(self, prompt: str, max_retries: int = 3) -> str:
        """调用qwen-max-2025-01-25 API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 100
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/services/aigc/text-generation/generation",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                return result["output"]["text"].strip()
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"API调用最终失败: {e}")
                    return "False"  # 默认返回False
    
    def evaluate_semantic_match(self, model_answer: str, ground_truth: str) -> bool:
        """评估模型答案与真实答案的语义匹配度"""
        prompt = f"""请评估以下两个答案在语义上是否匹配。

模型生成的答案: {model_answer}
真实答案: {ground_truth}

请仔细比较这两个答案的语义内容。如果它们在语义上基本一致或表达相同的意思，请回答"True"；如果不一致或表达不同的意思，请回答"False"。

只回答True或False，不要解释。"""

        response = self.call_api(prompt)
        
        # 解析响应
        response_lower = response.lower().strip()
        if "true" in response_lower:
            return True
        elif "false" in response_lower:
            return False
        else:
            # 如果无法解析，默认返回False
            logger.warning(f"无法解析API响应: {response}")
            return False


def eval_semantic_accuracy_batch(model, tokenizer, batch, generation_args, api_evaluator):
    """评估一个batch的语义正确率"""
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]#input_ids：模型的输入，包含完整的上下文和问题 labels：模型的目标输出，只包含需要模型生成的答案部分，其余位置为 -100（不参与损失）
    
    # 解码输入文本
    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    
    # 解码真实答案
    tokens = [label[label != -100] for label in labels]
    full_texts = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]
    
    attention_mask = batch["attention_mask"]
    
    # 生成模型答案
    generation_args = generation_args.copy()
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_args,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # 解码生成的答案
    gen_texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    # 评估语义正确率
    results = []
    for input_text, ground_truth, gen_text in zip(input_texts, ground_truths, gen_texts):
        try:
            is_correct = api_evaluator.evaluate_semantic_match(gen_text, ground_truth)
            results.append({
                "is_correct": is_correct,
                "input": input_text,
                "ground_truth": ground_truth,
                "generation": gen_text,
            })
        except Exception as e:
            logger.error(f"评估失败: {e}")
            results.append({
                "is_correct": False,
                "input": input_text,
                "ground_truth": ground_truth,
                "generation": gen_text,
                "error": str(e)
            })
    return results


# @unlearning_metric(name="semantic_accuracy_forget")
# def semantic_accuracy_forget(model, **kwargs):
#     """在TOFU_QA_forget"""
#     tokenizer = kwargs["tokenizer"]
#     collator = kwargs["collators"]
#     batch_size = kwargs.get("batch_size", 8)
#     generation_args = kwargs.get("generation_args", {})
#     api_key = kwargs.get("api_key")
#     data = kwargs["data"]

#     if not api_key:
#         raise ValueError("需要提供qwen-max-2025-01-25的API密钥")

#     api_evaluator = QwenAPIEvaluator(api_key=api_key)

#     results = {}
   
#     forget_dataloader = DataLoader(
#         data,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=collator,
#     )
  

#     fun_args = {
#         "tokenizer": tokenizer,
#         "generation_args": generation_args,
#         "api_evaluator": api_evaluator
#     }

#     # 评估遗忘数据集
#     logger.info(f"评估遗忘数据集，共{len(data)}个样本")
#     scores = run_batchwise_evals(
#         model,
#         forget_dataloader,
#         eval_semantic_accuracy_batch,
#         fun_args,
#         f"计算遗忘数据语义正确率"
#     )
#     correct = sum(
#         1 for evals in scores.values()
#         if evals.get("is_correct", False)
#     )
#     total = len(scores)
#     accuracy = correct / total if total > 0 else 0.0
#     results = {
#         "agg_value": accuracy,
#         "value_by_index": scores,
#         "correct_count": correct,
#         "total_count": total
#     }

#     return results


@unlearning_metric(name="semantic_accuracy_retain")
def semantic_accuracy_retain(model, **kwargs):
    """在TOFU_QA_retain上评估语义正确率"""
    tokenizer = kwargs["tokenizer"]
    collator = kwargs["collators"]
    batch_size = kwargs.get("batch_size", 8)
    generation_args = kwargs.get("generation_args", {})
    api_key = kwargs.get("api_key")
    data = kwargs["data"]

    if not api_key:
        raise ValueError("需要提供qwen-max-2025-01-25的API密钥")

    api_evaluator = QwenAPIEvaluator(api_key=api_key)

    results = {}

    retain_dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    fun_args = {
        "tokenizer": tokenizer,
        "generation_args": generation_args,
        "api_evaluator": api_evaluator
    }

    # 评估剩余数据集
    logger.info(f"评估剩余数据集，共{len(data)}个样本")
    scores = run_batchwise_evals(
        model,
        retain_dataloader,
        eval_semantic_accuracy_batch,
        fun_args,
        f"计算剩余数据语义正确率"
    )
    correct = sum(
        1 for evals in scores.values()
        if evals.get("is_correct", False)
    )
    total = len(scores)
    accuracy = correct / total if total > 0 else 0.0
    results = {
        "agg_value": accuracy,
        "value_by_index": scores,
        "correct_count": correct,
        "total_count": total
    }

    return results