# import os
# import sys
import hydra
from omegaconf import DictConfig
from omegaconf import DictConfig, open_dict

# 添加项目根目录到Python路径
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
   
    model_path= model_cfg.model_args.get("pretrained_model_name_or_path", None)
    model, tokenizer = get_model(model_cfg, cfg.trainer.get("handler", None))

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
            "model_args": model_cfg.get("model_args", {}),
            'model_path': model_path,
        }
        _ = evaluator.evaluate(**eval_args)

if __name__ == "__main__":
    main()