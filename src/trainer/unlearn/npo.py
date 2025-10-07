import random
import torch
from src.trainer.utils import compute_dpo_loss
from src.trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        forget_inputs = inputs["forget"]

        # 将ref_model移到GPU进行计算
        try:
            self.ref_model = self.ref_model.to(self.accelerator.device)
        except Exception as e:
            print(f"将ref_model移到GPU时出错: {e}")
            raise
        
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        
        # 将ref_model移到CPU以释放GPU显存
        try:
            self.ref_model = self.ref_model.to("cpu")
            torch.cuda.empty_cache()  # 清理GPU缓存
        except Exception as e:
            print(f"将ref_model移到CPU时出错: {e}")
            # 即使移动失败也继续执行，但记录错误
    
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

        