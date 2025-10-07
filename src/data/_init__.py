from typing import Dict, Any, Union
from omegaconf import DictConfig

from data.qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset, QAwithGenerate
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.unlearn import ForgetRetainDataset
from data.pretraining import PretrainingDataset, CompletionDataset

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, trainer=None, **kwargs):
    dataset_handler_name = dataset_cfg.get("handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    dataset_args = dataset_cfg.args

    # 如果算法为MyUnlearning，加载模型和分词器
    if trainer == "MyUnlearning" and dataset_name == "TOFU_QA_forget_generate":
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        import torch
        import os
        model_path = dataset_args.get("model_path")
        assert model_path is not None, "model_path must be provided in dataset_args for MyUnlearning"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 先加载基础模型
        model = AutoModelForCausalLM.from_pretrained(model_path)
        # 若目录存在LoRA适配器，则附加
        adapter_cfg = os.path.join(model_path, "adapter_config.json")
        adapter_bin = os.path.join(model_path, "adapter_model.bin")
        if os.path.exists(adapter_cfg) or os.path.exists(adapter_bin):
            try:
                model = PeftModel.from_pretrained(model, model_path)
            except Exception as e:
                # 不阻塞数据构造
                print(f"[warn] Failed to attach LoRA adapters from {model_path}: {e}")

        model = model.to(device)

        return dataset_handler(model=model, **dataset_args, **kwargs)

    return dataset_handler(**dataset_args, **kwargs)


def get_datasets(dataset_cfgs: Union[Dict, DictConfig], trainer=None, **kwargs):
    dataset = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        access_name = dataset_cfg.get("access_key", dataset_name)
        dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, trainer, **kwargs)
    if len(dataset) == 1:
        # return a single dataset
        return list(dataset.values())[0]
    # return mapping to multiple datasets
    return dataset


def get_data(data_cfg: DictConfig, trainer=None, mode="train", **kwargs):
    data = {}
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, trainer, **kwargs)
    if mode == "train":
        return data
    elif mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}
        unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
        data["train"] = unlearn_dataset
        for split in unlearn_splits:
            data.pop(split)
    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args = collator_cfg.args
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(QAwithGenerate)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)
_register_data(QAwithAlternateDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)
# Register collators
_register_collator(DataCollatorForSupervisedDataset)
