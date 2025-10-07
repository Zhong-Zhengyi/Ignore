import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import copy
from torch.utils.data import Subset



@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    task_name = cfg.get("task_name", None)
    
    model_cfg = cfg.model

    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    trainer = cfg.trainer.get("handler", None)
    model, tokenizer = get_model(copy.deepcopy(model_cfg), trainer)
    
    if task_name and "Llama-2-7b-hf" in task_name:
        if not isinstance(model, PeftModel):
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
  
    # Load Dataset
    data_cfg = cfg.data
    
    data = get_data(
        data_cfg, trainer, mode=mode, tokenizer=tokenizer, template_args=template_args
    )#在遗忘情形下，可以使用data["train"].forget来获取遗忘数据集，data["train"].retain来获取保留数据集

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )
    holdout_split = cfg.get('holdout_split', None)
    forget_split = cfg.get('forget_split', None)
    retain_split = cfg.get('retain_split', None)
    train_dataset = data.get("train", None)

    # train_dataset = Subset(train_dataset, range(386, len(train_dataset)))

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
        forget_split=forget_split,
        retain_split=retain_split,
        holdout_split=holdout_split,
        task_name=task_name,
        model_args=model_cfg.model_args,
    )
    # 保存预加载模型
    # model.save_pretrained('./saves/finetune/muse_Llama-2-7b-hf')
    # tokenizer.save_pretrained('./saves/finetune/muse_Llama-2-7b-hf')

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")

if __name__ == "__main__":
    main()