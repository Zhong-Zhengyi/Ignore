import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Any
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from omegaconf import OmegaConf
from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals

logger = logging.getLogger("evaluator")


class SuperficialForgettingEvaluator():
    def __init__(self, **kwargs):
        self.augmented_data_path = kwargs.get("augmented_data_path")
        self.model_path = kwargs.get("model_path")
        
    def load_augmented_data(self):
        """加载增强数据"""
        if not os.path.exists(self.augmented_data_path):
            raise FileNotFoundError(f"增强数据文件不存在: {self.augmented_data_path}")
        
        with open(self.augmented_data_path, 'r', encoding='utf-8') as f:
            augmented_data = json.load(f)
        
        logger.info(f"加载了 {len(augmented_data)} 个增强数据样本")
        return augmented_data
    
    def eval_forget_rouge_batch(self, model, tokenizer, batch, generation_args):
        """计算一个batch的forget_Q_A_ROUGE分数"""
        batch = {k: v.to(model.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
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
            max_new_tokens=512,
            **generation_args,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # 解码生成的答案
        gen_texts = tokenizer.batch_decode(
            output[:, input_ids.shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # 计算ROUGE分数
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        results = []
        
        for ground_truth, gen_text in zip(ground_truths, gen_texts):
            rouge_scores = scorer.score(ground_truth, gen_text)
            # 使用rouge1的recall作为forget_Q_A_ROUGE
            forget_rouge = rouge_scores["rouge1"].recall
            results.append({
                "forget_Q_A_ROUGE": forget_rouge,
                "ground_truth": ground_truth,
                "generation": gen_text,
            })
        
        return results
    
    def eval_augmented_rouge_batch(self, model, tokenizer, batch, generation_args):
        """计算一个batch的增强样本forget_Q_A_ROUGE分数"""
        # for k, v in batch.items():
        #     v=torch.tensor(v)
        #     v=v.to(model.device)
        batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        original_indices = batch.get("original_indices", [])
        
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
            max_new_tokens=512,
            **generation_args,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # 解码生成的答案
        gen_texts = tokenizer.batch_decode(
            output[:, input_ids.shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # 计算ROUGE分数
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        results = []
        
        for i, (ground_truth, gen_text) in enumerate(zip(ground_truths, gen_texts)):
            rouge_scores = scorer.score(ground_truth, gen_text)
            forget_rouge = rouge_scores["rouge1"].recall
            results.append({
                "forget_Q_A_ROUGE": forget_rouge,
                "ground_truth": ground_truth,
                "generation": gen_text,
                "original_idx": original_indices[i] if i < len(original_indices) else i
            })
        
        return results

    def evaluate_augmented_samples(self, model, tokenizer, augmented_data, generation_args, batch_size=8, template_args=None):
        """评估增强样本的forget_Q_A_ROUGE"""
        logger.info("开始评估增强样本...")
        
        # 准备增强样本数据
        augmented_samples = []
        for idx, data in augmented_data.items():
            original_text = data["original_text"]
            for aug_sample in data["augmented_samples"]:
                augmented_samples.append({
                    "input": original_text,
                    "augmented": aug_sample,
                    "original_idx": int(idx)
                })
        
        logger.info(f"共有 {len(augmented_samples)} 个增强样本需要评估")
        
        # 创建数据集类
        class AugmentedDataset:
            def __init__(self, samples, tokenizer, template_args):
                self.samples = samples
                self.tokenizer = tokenizer
                self.template_args = template_args
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                # 使用增强样本作为输入和目标
                text = sample["augmented"]
                
                # 简单的文本处理，将最后一行作为目标答案
                lines = text.strip().split('\n')
                if len(lines) >= 2:
                    input_text = '\n'.join(lines[:-1])
                    target_text = lines[-1]
                else:
                    input_text = text
                    target_text = text
                
                # 编码输入
                input_encoding = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # 编码完整文本（用于生成labels）
                full_encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # 创建labels（只有目标部分参与损失计算）
                input_length = input_encoding["input_ids"].shape[1]
                labels = full_encoding["input_ids"].clone()
                labels[0, :input_length] = -100  # 输入部分不参与损失计算
                
                return {
    "input_ids": full_encoding["input_ids"].squeeze(0),
    "attention_mask": full_encoding["attention_mask"].squeeze(0),
    "labels": labels.squeeze(0),
    "original_idx": sample["original_idx"],  # 用于后续聚合
    "index": idx  # 关键：每条增强样本唯一的索引，避免重复
}
        
        # 创建数据集和数据加载器
        dataset = AugmentedDataset(augmented_samples, tokenizer, template_args or {})
        dataloader = DataLoader(dataset, batch_size=6, shuffle=False, collate_fn=lambda b: self.collate_augmented_batch(b, tokenizer.pad_token_id))
        
        # 评估增强样本
        fun_args = {
            "tokenizer": tokenizer,
            "generation_args": generation_args,
        }
        
        augmented_scores = run_batchwise_evals(
            model,
            dataloader,
            self.eval_augmented_rouge_batch,
            fun_args,
            '计算增强样本forget_Q_A_ROUGE'
        )
        
        # 按原始索引分组结果
        augmented_results = {}
        for result in augmented_scores.values():
            original_idx = int(result.get("original_idx", 0))
            if original_idx not in augmented_results:
                augmented_results[original_idx] = []
            augmented_results[original_idx].append(result["forget_Q_A_ROUGE"])
        
        return augmented_results
    
    def collate_augmented_batch(self, batch, pad_token_id):
        """增强样本的批处理函数，支持序列padding"""
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        original_indices = [item["original_idx"] for item in batch]
        data_indices = [item["index"] for item in batch]
        # 使用pad_sequence统一序列长度
        # input_ids 用 pad_token_id 填充
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        # attention_mask 用 0 填充
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        # labels 用 -100 填充
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        index = torch.tensor(data_indices, dtype=torch.long)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "original_indices": original_indices, "index": index}
    
    def calculate_superficial_forgetting(self, forget_rouge_scores, augmented_rouge_scores):
        """计算superficial forgetting程度"""
        superficial_scores = []
        
        for idx, forget_score in forget_rouge_scores.items():
            if idx in augmented_rouge_scores:
                # 计算该遗忘样本对应的增强样本的平均ROUGE分数
                augmented_scores = augmented_rouge_scores[idx]
                avg_augmented_rouge = np.mean(augmented_scores)
                
                # 计算superficial forgetting程度（差值）
                superficial_score = forget_score - avg_augmented_rouge
                superficial_scores.append(superficial_score)
                
                logger.info(f"样本 {idx}: forget_ROUGE={forget_score:.4f}, "
                          f"avg_augmented_ROUGE={avg_augmented_rouge:.4f}, "
                          f"superficial_forgetting={superficial_score:.4f}")
        
        # 计算整体superficial forgetting程度
        overall_superficial_forgetting = np.mean(superficial_scores) if superficial_scores else 0.0
        
        logger.info(f"整体superficial forgetting程度: {overall_superficial_forgetting:.4f}")
        
        return {
            "overall_superficial_forgetting": overall_superficial_forgetting,
            "individual_scores": superficial_scores,
            "forget_rouge_scores": forget_rouge_scores,
            "augmented_rouge_scores": augmented_rouge_scores
        }
    
    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        """主评估函数"""
        # 设置参数
        overwrite = kwargs.get("overwrite", None) if overwrite is None else overwrite
        model_path = kwargs.get("model_path", None)
        self.eval_epoch = model_path.split("/")[-1] if model_path else "unknown"
        
        # 准备模型
        model.eval()
        
        # 设置输出目录
        output_dir = output_dir if output_dir else kwargs.get("output_dir")
        
        # 加载增强数据
        augmented_data = self.load_augmented_data()
        
        # 获取遗忘数据集
        forget_data = kwargs.get("data")
        if forget_data is None:
            raise ValueError("需要提供遗忘数据集")
        
        # 准备生成参数
        generation_args = kwargs.get("generation_args", {})
        tokenizer = kwargs.get("tokenizer")
        collator = kwargs.get("collators")
        batch_size = kwargs.get("batch_size", 8)
        
        # 评估遗忘样本的forget_Q_A_ROUGE
        print("开始评估遗忘样本...")
        forget_dataloader = DataLoader(
            forget_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        
        fun_args = {
            "tokenizer": tokenizer,
            "generation_args": generation_args,
        }
        
        forget_scores = run_batchwise_evals(
            model,
            forget_dataloader,
            self.eval_forget_rouge_batch,
            fun_args,
            '计算遗忘样本forget_Q_A_ROUGE'
        )
        
        # 提取forget_Q_A_ROUGE分数
        forget_rouge_scores = {int(k): v["forget_Q_A_ROUGE"] for k, v in forget_scores.items() if "forget_Q_A_ROUGE" in v}
        
        # 评估增强样本
        print("开始评估增强样本...")
        template_args = kwargs.get("template_args", {})
        augmented_rouge_scores = self.evaluate_augmented_samples(
            model, tokenizer, augmented_data, generation_args, batch_size, template_args
        )
        
        # 计算superficial forgetting程度
        superficial_results = self.calculate_superficial_forgetting(
            forget_rouge_scores, augmented_rouge_scores
        )

        return superficial_results


@unlearning_metric(name="superficial_forgetting")
def superficial_forgetting(model, **kwargs):
    evaluator = SuperficialForgettingEvaluator(**kwargs)
    return evaluator.evaluate(model, **kwargs)