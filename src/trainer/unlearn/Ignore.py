from src.trainer.unlearn.base import UnlearnTrainer
from src.trainer.utils import compute_dpo_loss
import torch
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Optional
import random
import json
import os
from tqdm import tqdm
import copy
from src.trainer.unlearn.grad_diff import GradDiff
from peft import LoraConfig, get_peft_model, TaskType
import requests
import time
import os
os.environ['QWEN_API_KEY'] = 'your_api_key'


class Ignore(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize parameters
        self.beta = beta
        self.tau = 0.1  # Consistency threshold
        self.num_augmented_samples = 32  # Number of augmented samples
        
        # Qwen API configuration
        self.qwen_api_key = os.getenv('QWEN_API_KEY')
        self.qwen_api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        if not self.qwen_api_key:
            raise ValueError("Please set the QWEN_API_KEY environment variable or provide the API key via the qwen_api_key parameter")
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        # Set chat_template to support dialogue format
        if hasattr(self.processing_class, 'chat_template') and self.processing_class.chat_template is None:
            # Set default chat_template for Llama models
            self.processing_class.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>' }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>' }}\n{% elif message['role'] == 'system' %}\n{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>' }}\n{% endif %}\n{% endfor %}\n{{ '<|im_start|>assistant\n' }}"
      
        # Set save path
        self.save_dir = "your_save_dir"
        os.makedirs(self.save_dir, exist_ok=True)
        task_name_path = copy.deepcopy(self.task_name)
        
        # Initialize variables to store augmented data
        self.augmented_forget_data = {}
        self.text_to_augmented_map = {}  # Hash map for fast lookup
        if 'scal' in task_name_path:
            task_name_path = task_name_path.replace("_scal", "")
        elif 'sust' in task_name_path:
            task_name_path = task_name_path.replace("_sust", "")
        if 'forget_1' in task_name_path:
            task_name_path = task_name_path.replace("_forget_1", "")
        elif 'forget_2' in task_name_path:
            task_name_path = task_name_path.replace("_forget_2", "")
        elif 'forget_3' in task_name_path:
            task_name_path = task_name_path.replace("_forget_3", "")
        elif 'forget_4' in task_name_path:
            task_name_path = task_name_path.replace("_forget_4", "")
        
        print(f"task_name: {task_name_path}")
        
        self.augmented_data_file = os.path.join(self.save_dir, task_name_path+"_augmented_data.json")
        self.qwen_augmented_data_file = os.path.join(self.save_dir, task_name_path+"_qwen_augmented_data.json")
        
        self._preprocess_augmented_data()
        # self._preprocess_qwen_aumented_data()
        
        self._setup_gradient_checkpointing_compatibility()
    
    def _setup_gradient_checkpointing_compatibility(self):
        self._gradient_checkpointing_enabled = False
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self._gradient_checkpointing_enabled = getattr(self.model, 'gradient_checkpointing_enable', False)
    

    def _preprocess_qwen_aumented_data(self):
        if os.path.exists(self.qwen_augmented_data_file):
            print(f"Loaded existing augmented data file: {self.qwen_augmented_data_file}")
            with open(self.qwen_augmented_data_file, 'r', encoding='utf-8') as f:
                self.augmented_forget_data = json.load(f)
            
            for idx, data in self.augmented_forget_data.items():
                original_text = data['original_text']
                text_key = self._get_text_key_for_matching(original_text, max_tokens=200)
                self.text_to_augmented_map[text_key] = data['augmented_samples']
            print(f"Loaded {len(self.augmented_forget_data)} augmented items")
            return
        else:
            print("Start generating augmented samples for forget data...")
            forget_dataset = self.train_dataset.forget
                    
            # Initialize JSON file with an empty dictionary
            with open(self.qwen_augmented_data_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
            for idx, forget_item in enumerate(tqdm(forget_dataset, desc="Generating augmented data")):
            # if idx<=200:
                answer_start_indices = self._find_answer_start_indices(forget_item['input_ids'])
                # Convert forget data to text
                original_text = self.processing_class.decode(forget_item['input_ids'], skip_special_tokens=True)
                original_answer = self.processing_class.decode(forget_item['input_ids'][answer_start_indices:], skip_special_tokens=True)
                
                print(f"Processing {idx+1}/{len(forget_dataset)}")
                
                # Generate augmented samples for this data
                augmented_samples_text = self.qwen_generate_augmented_samples(original_answer)
                augmented_samples = []
                for sample in augmented_samples_text:
                    answer_id = self.processing_class.encode(sample, add_special_tokens=False)
                    # Convert tensor slice and list to tensors, then concatenate
                    prefix_tensor = forget_item['input_ids'][:answer_start_indices]
                    answer_tensor = torch.tensor(answer_id, dtype=prefix_tensor.dtype, device=prefix_tensor.device)
                    augmented_sample_ids = torch.cat([prefix_tensor, answer_tensor])
                    augmented_sample_text = self.processing_class.decode(augmented_sample_ids, skip_special_tokens=True)
                    augmented_samples.append(augmented_sample_text)
                # Save to memory
                current_data = {
                    'original_text': original_text,
                    'augmented_samples': augmented_samples
                }
                self.augmented_forget_data[str(idx)] = current_data
                
                # Also add to the fast lookup table using the first 200 tokens as the key
                text_key = self._get_text_key_for_matching(original_text, max_tokens=200)
                self.text_to_augmented_map[text_key] = augmented_samples

                with open(self.qwen_augmented_data_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                # Add new data
                existing_data[str(idx)] = current_data
                # Write back to file
                with open(self.qwen_augmented_data_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                print(f"Saved item {idx+1} to JSON file")
            
            print(f"Successfully generated augmented samples for {len(self.augmented_forget_data)} forget items")

    def _preprocess_augmented_data(self):
        # Check if the augmented data file already exists
        if os.path.exists(self.augmented_data_file):
            print(f"Loaded existing augmented data file: {self.augmented_data_file}")
            with open(self.augmented_data_file, 'r', encoding='utf-8') as f:
                self.augmented_forget_data = json.load(f)
            # Build a hash map for fast lookup
            for idx, data in self.augmented_forget_data.items():
                original_text = data['original_text']
                # Use the first 200 tokens as the matching key
                text_key = self._get_text_key_for_matching(original_text, max_tokens=200)
                self.text_to_augmented_map[text_key] = data['augmented_samples']
                # self.text_to_augmented_map[original_text].append(original_text)
            print(f"Loaded {len(self.augmented_forget_data)} augmented items")
            return
        else:
            print("Start generating augmented samples for forget data...")
            forget_dataset = self.train_dataset.forget
                
            # Initialize JSON file with an empty dictionary
            with open(self.augmented_data_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
        
            for idx, forget_item in enumerate(tqdm(forget_dataset, desc="Generating augmented data")):
            # if (idx==234) or (idx==258) or (idx==296) or (idx==305) or (idx==339):
                answer_start_indices = self._find_answer_start_indices(forget_item['input_ids'])
                # Convert forget data to text
                original_text = self.processing_class.decode(forget_item['input_ids'], skip_special_tokens=True)
                original_answer = self.processing_class.decode(forget_item['input_ids'][answer_start_indices:], skip_special_tokens=True)
                
                print(f"Processing {idx+1}/{len(forget_dataset)}")
                
                # Generate augmented samples for this data
                augmented_samples_text = self.generate_augmented_samples(original_answer)
                augmented_samples = []
                for sample in augmented_samples_text:
                    answer_id = self.processing_class.encode(sample, add_special_tokens=False)
                    # Convert tensor slice and list to tensors, then concatenate
                    prefix_tensor = forget_item['input_ids'][:answer_start_indices]
                    answer_tensor = torch.tensor(answer_id, dtype=prefix_tensor.dtype, device=prefix_tensor.device)
                    augmented_sample_ids = torch.cat([prefix_tensor, answer_tensor])
                    augmented_sample_text = self.processing_class.decode(augmented_sample_ids, skip_special_tokens=True)
                    augmented_samples.append(augmented_sample_text)
                # Save to memory
                current_data = {
                    'original_text': original_text,
                    'augmented_samples': augmented_samples
                }
                self.augmented_forget_data[str(idx)] = current_data
                
                # Also add to the fast lookup table using the first 200 tokens as the key
                text_key = self._get_text_key_for_matching(original_text, max_tokens=200)
                self.text_to_augmented_map[text_key] = augmented_samples

                # Read existing data
                with open(self.augmented_data_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                # Add new data
                existing_data[str(idx)] = current_data
                # Write back to file
                with open(self.augmented_data_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                print(f"Saved item {idx+1} to JSON file")
            
            print(f"Successfully generated augmented samples for {len(self.augmented_forget_data)} forget items")
    
    def _find_answer_start_indices(self, input_ids: torch.Tensor) -> int:
        answer_prompt = "assistant\n"
        answer_prompt_tokens = self.processing_class.encode(answer_prompt, add_special_tokens=False)

        input_list = input_ids.tolist()
        # Find the position of answer_prompt_tokens in input_list
        found_idx = -1  # Initialize variable
        if 'tofu' in self.task_name:
            for j in range(len(input_list) - len(answer_prompt_tokens) + 1):
                if input_list[j:j+len(answer_prompt_tokens)] == answer_prompt_tokens:
                    found_idx = j + len(answer_prompt_tokens)
                    break 
        else:
            found_idx = 0
        
        return found_idx
            
    def generate_augmented_samples(self, original_text: str) -> List[str]:
            
        augmented_samples = []  # Contains paraphrased samples
        
        # Generate multiple paraphrase prompt templates
        prompt_templates = self._get_paraphrase_prompts(original_text)
        # prompt = f"Rephrase this text while maintaining the exact same meaning:\n{original_text}\nParaphrased text:"
        
        for prompt in prompt_templates:          
            # Use the current template to generate a paraphrase
            paraphrased = self._generate_paraphrase_with_model(prompt)
            augmented_samples.append(paraphrased)

        return augmented_samples
    
    def qwen_generate_augmented_samples(self, original_text: str) -> List[str]:
        augmented_samples = []  # Contains paraphrased samples
        prompt_templates = self._get_paraphrase_prompts(original_text)
        for prompt in prompt_templates:
            paraphrased = self._generate_paraphrase_with_qwen_api(prompt)
            augmented_samples.append(paraphrased)
        return augmented_samples
    
    def _get_paraphrase_prompts(self, text: str) -> List[str]:
        base_prompts = []
        # Generic templates for paraphrasing the entire text
        if 'tofu' in self.task_name:
            base_prompts.extend([
                f"Original text: {text}\nPlease rewrite the above text while preserving the original meaning but using different expressions. Do not make it the same as the original one.",
                f"Please rewrite the following text, requiring semantic consistency but different expressions: {text}. Do not make it the same as the original one.",
                f"Text content: {text}\nPlease provide a paraphrased version of this text with the same meaning but different expressions. Do not make it the same as the original one.",
                f"Please paraphrase the following content:\n{text}\nRequirement: preserve semantics while changing expressions. Do not make it the same as the original one.",
                f"Rephrase this text while maintaining the exact same meaning:\n{text}. Do not make it the same as the original one.",
                f"Given text: {text}\nGenerate an equivalent version using different words and sentence structures. Do not make it the same as the original one.",
            ])
        elif 'muse' in self.task_name:
            base_prompts.extend([
                f"Please carefully read the following text and generate a detailed summary. Requirements: 1) Comprehensively cover the core content and main viewpoints of the text; 2) Maintain the original logical structure and argumentative approach; 3) Include important details, examples, and data; 4) Ensure the summary is rich in content and meaningful, avoiding being too brief; 5) Keep the word count within 600 words. Text content:\n{text}",
                f"Please conduct an in-depth analysis and summary of the following text. Requirements: 1) Accurately understand and convey the complete meaning of the original text; 2) Highlight key information and important concepts; 3) Maintain content coherence and completeness; 4) Avoid oversimplification, ensure the summary has sufficient depth; 5) Word count should not exceed 600 words. Text to summarize:\n{text}",
                f"Please carefully analyze the following content and provide a comprehensive and detailed summary. Requirements: 1) Completely preserve the core ideas and main arguments of the original text; 2) Include necessary background information and context; 3) Ensure the summary content is substantial and can independently convey the main information of the original text; 4) Maintain clear logic and reasonable structure; 5) Word count limited to 600 words. Content as follows:\n{text}",
                f"Please provide a comprehensive summary of the following text. Requirements: 1) Deeply understand various aspects and dimensions of the text; 2) Accurately extract and express main viewpoints and key information; 3) Maintain richness and completeness of content; 4) Ensure the summary fully reflects the depth and breadth of the original text; 5) Word count controlled within 600 words. Text content:\n{text}",
                # f"Please carefully read and summarize the following text. Requirements: 1) Comprehensively grasp the main theme and core content of the text; 2) Retain important details, arguments, and evidence; 3) Ensure the summary content is substantial, avoiding being too brief; 4) Maintain the original argumentative logic and structural hierarchy; 5) Word count should not exceed 600 words. Content to summarize:\n{text}",
                f"Please conduct a detailed summary analysis of the following text. Requirements: 1) Deeply understand the complete meaning and various key points of the text; 2) Comprehensively cover main viewpoints, important information, and key details; 3) Maintain depth and breadth of content, do not omit important content; 4) Ensure the summary has sufficient completeness and readability; 5) Word count limited to 600 words. Text as follows:\n{text}",
                f"Please carefully read and summarize the following text. Requirements: 1) Comprehensively grasp the main theme and core content of the text; 2) Retain important details, arguments, and evidence; 3) Ensure the summary content is substantial, avoiding being too brief; 4) Maintain the original argumentative logic and structural hierarchy; 5) Word count should not exceed 600 words. Content to summarize:\n{text}",
            ])
        return base_prompts
    
    def _generate_paraphrase_with_model(self, prompt: str, max_length: int = 128, temperature: float = 0.8) -> str:
        # Ensure the model is in eval mode
        self.model.eval()
        
        # Use more efficient memory management
        torch.cuda.empty_cache()  # Clear GPU cache
        
        # Avoid deep copying the entire model; reuse the existing instance
        ge_model = self.model  # Directly use the original model
        if torch.cuda.is_available():
            ge_model = ge_model.to("cuda")
        
        print('===prompt===', prompt)
        if 'tofu' in self.task_name:  # tofu uses the Qwen model
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        elif 'muse' in self.task_name:  # muse uses the pirate model
            messages = [
                {"role": "system", "content": "You are a helpful assistant, please summarize the main idea of the following news. It is required to have complete basic elements such as a subject, predicate, object, time, place, cause and result."},
                {"role": "user", "content": prompt}
            ]
        
        text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.processing_class([text], return_tensors="pt").to(ge_model.device)
        
        input_length = model_inputs['input_ids'][0].shape[0]
        if 'llama' in self.task_name.lower():
            max_length = 512
        else:
            max_length = 512
        
        # Use a full set of generation parameters to reduce memory usage
        with torch.no_grad():
            generated_ids = ge_model.generate(
                **model_inputs, 
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.processing_class.eos_token_id,
                eos_token_id=self.processing_class.eos_token_id,
                synced_gpus=False,
                return_dict_in_generate=False,
                use_cache=True,  # Enable cache to reduce memory usage
            )
        
        # Only take the newly generated part
        generated_ids = generated_ids[0][input_length:]
        generated_text = self.processing_class.decode(generated_ids, skip_special_tokens=True)
            # return generated_text, generated_ids, input_length
        # while ('assistant' in generate_text.lower()) or ('summarize' in generate_text.lower()):
        #     generated_text, generated_ids, input_length = generate_text(prompt)
        
        print(f"Input length: {input_length}")
        print(f"Number of generated tokens: {len(generated_ids)}")
        print(f"Generated text: '{generated_text}'")
        
        # Clean up memory
        del generated_ids, model_inputs
        torch.cuda.empty_cache()
        
        # If the generated text is empty or too short, retry with the same parameters
        if not generated_text.strip():
            retry_count = 0
            max_retries = 3 # Maximum retry attempts
            
            # The first generation failed; start retrying until success or hitting max retries
            while not generated_text.strip() and retry_count < max_retries:
                print(f"Warning: Model generated no new content, retrying (attempt {retry_count + 1})...")
                
                # Re-prepare model inputs
                model_inputs = self.processing_class([text], return_tensors="pt").to(ge_model.device)
                input_length = model_inputs['input_ids'][0].shape[0]
                
                with torch.no_grad():
                    generated_ids = ge_model.generate(
                        **model_inputs, 
                        max_new_tokens=max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.processing_class.eos_token_id,
                        eos_token_id=self.processing_class.eos_token_id,
                        synced_gpus=False,
                        return_dict_in_generate=False,
                        use_cache=True,
                    )
                
                generated_ids = generated_ids[0][input_length:]
                generated_text = self.processing_class.decode(generated_ids, skip_special_tokens=True)
                
                print(f"Regenerated text: '{generated_text}'")
                
                # Clean up memory
                del generated_ids, model_inputs
                torch.cuda.empty_cache()
                
                retry_count += 1
            
            print("Warning: Model did not generate new content; generation parameters may need adjustment")
        paraphrase = self._extract_paraphrase_from_generated(generated_text)
        return paraphrase
    
    def _extract_paraphrase_from_generated(self, generated_text: str) -> str:
        if ":" in generated_text:
            paraphrase = generated_text.split(":", 1)[-1].strip()
            print(f"Colon detected, extracted content: '{paraphrase}'")
        else:
            paraphrase = generated_text.strip()
            print(f"No colon detected, returning original: '{paraphrase}'")
        return paraphrase
    
    def _generate_paraphrase_with_qwen_api(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate paraphrased text using the Qwen API
        """
        headers = {
            'Authorization': f'Bearer {self.qwen_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Build API request payload
        data = {
            "model": "qwen-max-2025-01-25",  # Adjust model version as needed
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": 0.8,
                "top_p": 0.9,
                "max_tokens": 512,
                "repetition_penalty": 1.1
            }
        }
        
        for attempt in range(max_retries):
            try:
                print(f"=== API request (attempt {attempt + 1}/{max_retries}) ===")
                print(f"Prompt: {prompt}")
                
                response = requests.post(
                    self.qwen_api_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'output' in result and 'text' in result['output']:
                        generated_text = result['output']['text']
                        print(f"API response success: '{generated_text}'")
                        
                        # Extract paraphrase content
                        paraphrase = self._extract_paraphrase_from_generated(generated_text)
                        return paraphrase
                    else:
                        print(f"Unexpected API response format: {result}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                else:
                    print(f"API request failed, status code: {response.status_code}")
                    print(f"Response body: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                        
            except requests.exceptions.RequestException as e:
                print(f"API request exception: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
            except Exception as e:
                print(f"Error while processing API response: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
        
        # All retries failed; return the original prompt
        print("Warning: All API retries failed, returning original text")
        return prompt
    
    def _get_text_key_for_matching(self, text: str, max_tokens: int = 200) -> str:
        """
        Get a text key for matching using only the first max_tokens tokens
        """
        # Encode text to tokens
        tokens = self.processing_class.encode(text, add_special_tokens=False)
        
        # Take only the first max_tokens tokens
        truncated_tokens = tokens[:max_tokens]
        
        # Decode back to text
        truncated_text = self.processing_class.decode(truncated_tokens, skip_special_tokens=True)
        
        # Clean text
        cleaned_text = " ".join(truncated_text.strip().split())
        
        return cleaned_text

    def compute_gradients_for_sample(self, model, sample_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model.train()
        outputs = model(**sample_inputs)
        loss = outputs.loss

        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        gradients = {n:(g if g is not None else torch.zeros_like(p))
                    for (n,p),g in zip(model.named_parameters(), grads) if p.requires_grad}
        return gradients
    
    
    def compute_consistency_mask(self, all_gradients: List[Dict[str, torch.Tensor]], tau: float = None) -> Dict[str, torch.Tensor]:
        if tau is None:
            tau = self.tau
            
        consistency_masks = {}
        
        for param_name in all_gradients[0].keys():
            # Get gradients for this parameter across all samples
            param_gradients = [grads[param_name] for grads in all_gradients]
            gradient_tensor = torch.stack(param_gradients, dim=0)  # [N, ...param_shape]
            
            # Compute gradient signs
            gradient_signs = torch.sign(gradient_tensor)  # [N, ...param_shape]
            
            # Compute mean sign per parameter dimension
            mean_signs = torch.mean(gradient_signs, dim=0)  # [...param_shape]
           
            # Create consistency mask: a dimension is consistent when |mean_sign| > tau
            consistency_mask = (torch.abs(mean_signs) > tau).float()
            
            consistency_masks[param_name] = consistency_mask
            
        return consistency_masks
    
    def _iter_trainable_params(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and p.numel() > 0:
                yield n, p
    
    def apply_masked_gradient_update(self, model, all_gradients, consistency_masks, learning_rate=1e-5):
        updated_params = 0; total_params = 0
        with torch.no_grad():
            for n, p in self._iter_trainable_params(model):
                if n not in consistency_masks or n not in all_gradients[0]:
                    continue
                g_list = [grads[n] for grads in all_gradients if n in grads]
                if not g_list: 
                    continue
                mean_g = torch.mean(torch.stack(g_list, dim=0), dim=0)
                mask = consistency_masks[n].to(mean_g)
                # Shape alignment (local shard size)
                if mask.shape != mean_g.shape:
                    if mask.numel() == mean_g.numel(): mask = mask.view_as(mean_g)
                    else: continue
                if mean_g.shape != p.shape:
                    if mean_g.numel() == p.numel(): mean_g = mean_g.view_as(p); mask = mask.view_as(p)
                    else: continue

                masked_g = torch.clamp(mean_g * mask, -1.0, 1.0)
                p.add_(masked_g, alpha=-learning_rate)

                updated_params += mask.sum().item()
                total_params   += mask.numel()
        return updated_params, total_params

    def _consistency_regularizer(self, model, all_gradients, masks):
        reg = 0.0; updated = 0; total = 0
        for name, p in self._iter_trainable_params(model):
            if name not in masks or name not in all_gradients[0]:
                continue
            g_list = [gr[name] for gr in all_gradients if name in gr]
            if not g_list:
                continue
            mean_g = torch.mean(torch.stack(g_list, dim=0), dim=0)
            m = masks[name].to(mean_g)
            if m.shape != mean_g.shape:
                if m.numel() == mean_g.numel(): m = m.view_as(mean_g)
                else: continue
            v = torch.clamp(mean_g * m, -1.0, 1.0).detach()
            if v.shape != p.shape:
                if v.numel() == p.numel(): v = v.view_as(p)
                else: continue
            reg = reg + (p * v).sum()     # d(reg)/dp = v
            updated += (m > 0).sum().item()
            total   += m.numel()
        return reg, updated, total
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Check for GPU and move model to device if available
        model = model.to(self.accelerator.device)
        print(f"Current device: {self.accelerator.device}")

        forget_inputs = inputs["forget"]
        self.ref_model = self.ref_model.to(self.accelerator.device)

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )
        # self.ref_model = self.ref_model.to("cpu")
        # torch.cuda.empty_cache()

        retain_inputs = inputs["retain"]
        
        # Use half of each sample's sequence length for training to reduce memory usage
        seq_len = retain_inputs["input_ids"].shape[1]

        
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # Apply embodied gradient ascent to forget data
        total_consistency_loss = 0.0
        batch_size = forget_inputs["input_ids"].shape[0]
        
        # Process each sample in the batch
        for batch_idx in range(batch_size):
            sample_input_ids = forget_inputs["input_ids"][batch_idx]
            sample_text = self.processing_class.decode(sample_input_ids, skip_special_tokens=True)
            
            # Use the hash map to quickly find pre-generated augmented data; match using the first 200 tokens only
            text_key = self._get_text_key_for_matching(sample_text, max_tokens=200)
            print(f"Forget sample {batch_idx+1}/{batch_size} match key: '{text_key}'")
            augmented_texts = self.text_to_augmented_map.get(text_key)
            if augmented_texts is None:
                print(f"Warning: No matching augmented data found; skip consistency computation for this sample.")
           
            if augmented_texts and len(augmented_texts) > 1:
                # Perform embodied gradient ascent for the current sample
                sample_loss, stats = self._compute_consistency_loss_for_sample(model, augmented_texts)
                total_consistency_loss += sample_loss
                
                # Print statistics (optional)
                if batch_idx == 0:  # Print stats only for the first sample
                    print(f"Consistency stats: update_ratio={stats['update_ratio']:.4f}, "
                            f"consistent_params={stats['consistent_params']}, "
                            f"total_params={stats['total_params']}")
        # Average consistency loss
        avg_consistency_loss = total_consistency_loss / batch_size

        loss = 1.0 * forget_loss + 1.0 * retain_loss -  0.5 * avg_consistency_loss
        return (loss, forget_outputs) if return_outputs else loss

    def _compute_consistency_loss_for_sample(self, model, augmented_texts: List[str]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the consistency-gradient-based loss for a single sample
        """
        # 1) Compute gradients for all augmented samples
        all_gradients = []
        
        # Limit the number of augmented samples to avoid memory issues; pick the longest ones
        max_samples = min(6, len(augmented_texts))  # Further reduce if necessary to save memory
        # Sort by text length and select the longest
        selected_texts = sorted(augmented_texts, key=lambda x: len(x), reverse=False)[:max_samples]
        
        for i, text in enumerate(selected_texts):
            print(f"Processing augmented sample {i+1}/{len(selected_texts)}")
            if text and text.strip():
                # Prepare inputs
                sample_inputs = self.processing_class.encode_plus(
                    text, max_length=512, padding='max_length', 
                    truncation=True, return_tensors='pt'
                )
                sample_inputs['labels'] = sample_inputs['input_ids'].clone()
                sample_inputs = {k: v.to(model.device) for k, v in sample_inputs.items()}
                
                # Compute gradients for this sample
                gradients = self.compute_gradients_for_sample(model, sample_inputs)
                
                all_gradients.append(gradients)
                
                # Clean up intermediates to free memory (optional)
                # del sample_inputs
                # torch.cuda.empty_cache()
                
        
        # 2) Compute consistency masks
        consistency_masks = self.compute_consistency_mask(all_gradients, self.tau)
        
        # 3) Apply masked gradient-based regularizer
        reg_loss, updated_params, total_params = self._consistency_regularizer(model, all_gradients, consistency_masks)

        print(f"reg_loss: {reg_loss}")
        
        # 4) Compute statistics
        consistent_params = sum(mask.sum().item() for mask in consistency_masks.values())
        stats = {
            'num_augmented_samples': len(selected_texts),
            'consistency_ratio': consistent_params / total_params if total_params > 0 else 0.0,
            'consistent_params': consistent_params,
            'total_params': total_params,
            'updated_params': updated_params,
            'update_ratio': updated_params / total_params if total_params > 0 else 0.0,
            'tau_threshold': self.tau
        }
        return reg_loss, stats
    