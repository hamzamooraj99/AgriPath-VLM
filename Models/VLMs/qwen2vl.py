from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer
import torch
from datasets import load_dataset
from transformers import Qwen2VLImageProcessorFast
from PIL import Image

# MIN_IMG_SIZE = 224
# MAX_IMG_SIZE = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AgriPathModule:
    def __init__(self, hf_repo:str, processor_name:str):
        self.qwen_processor = Qwen2VLImageProcessorFast.from_pretrained(processor_name)
        self.train_conv_set = [self.convert_to_conversation(sample) for sample in (load_dataset(hf_repo, split='train'))]
        self.val_conv_set = [self.convert_to_conversation(sample) for sample in (load_dataset(hf_repo, split='validation'))]
        # self.test_conv_set = [self.convert_to_conversation(sample) for sample in (load_dataset(hf_repo, split='test'))]

    def preprocess_image(self, image):
        image = image.convert('RGB')
        image = self.qwen_processor(image, return_tensors='pt', device=DEVICE)
        return image

    def convert_to_conversation(self, sample):
        instruction = "You are a botanist expert and have to identify and describe the crop and disease (if any) present in the image provided."
        image = self.preprocess_image(sample['image'])

        conversation = [
            {"role": "user",
            "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image}
                ]
            },
            {"role": "assistant",
            "content": [
                    {"type": "text", "text": f"Class: {sample['crop']}\nDisease: {sample['disease']}"}
                ]
            }
        ]
        return({"messages": conversation})
    
    

class QwenModule:
    def __init__(self, model_name:str, data_module: AgriPathModule):
        self.model_name = model_name
        self.model, self.tokenizer = self.model_selection()
        self.trainer = self.trainer_creator(data_module.train_conv_set, data_module.val_conv_set)

    def model_selection(self):
        model, tokenizer = FastVisionModel.from_pretrained(
            self.model_name,
            load_in_4bit= True,
            use_gradient_checkpointing="unsloth"
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True, finetune_attention_modules=True, finetune_language_layers=True, finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias='none',
            random_state=3407,
            use_rslora=False,
            loftq_config=None
        )

        return model, tokenizer


    def trainer_creator(self, train_set, eval_set):
        # Enable the model for training
        FastVisionModel.for_training(self.model)

        # Trainer Arguments
        args = SFTConfig(
            per_device_train_batch_size=4,  #Each GPU processes 2 samples per batch,
            gradient_accumulation_steps=2,  #Gradients are accumulated for 4 steps before updating model
            warmup_steps=50,                 #Gradually increases learning rate for first n steps to prevent instability
            num_train_epochs=3,             #Parameter to perform full fine-tune (use max_steps=30 for a quick test)
            # Optimisation & Mixed Precision
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),   #Use float16 if GPU does not support bf16
            bf16=is_bf16_supported(),         #Use bfloat16 if GPU supports it (better stability)
            # Optimiser & Weight Decay
            optim="adamw_8bit",
            weight_decay=0.01,              #Regularisation to prevent overfitting
            lr_scheduler_type='linear',     #Decay type for learning rate from learning_rate to 0
            seed=3407,
            output_dir='outputs',
            # Logging & Reporting
            report_to='none',               #Integration with Weights & Biases ('none' disables, 'wandb' enables)
            # Settings for Vision Fine-Tuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=8,             #CPU processes for parallel dataset processing
            max_seq_length=2048             #Maximum token length for input sequence
        )
        
        # Initialise the Trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer), #Data collator ensures text and visual inputs are correctly batched
            train_dataset=train_set,
            eval_dataset=eval_set,
            args=args
        )

        return trainer

def show_current_mem_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    return start_gpu_memory, max_memory

def show_final_mem_stats(start_gpu_memory, max_memory, trainer_stats):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



if __name__ == '__main__':

    qwen_models = {
        "2B": {
            'model': "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
            'processor': "Qwen/Qwen2-VL-2B-Instruct",
            'repo' : "hamzamooraj99/AgriPath-Qwen2-VL-2B"
        },
        "7B": {
            'model': "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
            'processor': "Qwen/Qwen2-VL-7B-Instruct",
            'repo' : "hamzamooraj99/AgriPath-Qwen2-VL-7B"
        },
        "72B": {
            'model': "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
            'processor': "Qwen/Qwen2-VL-72B-Instruct",
            'repo' : "hamzamooraj99/AgriPath-Qwen2-VL-72B"
        },
    }

    hf_repo = "hamzamooraj99/AgriPath-LF16-30k"

    model_name = qwen_models["2B"]['model']
    processor_name = qwen_models["2B"]['processor']
    model_repo = qwen_models["2B"]['repo']

    ap_module = AgriPathModule(hf_repo, processor_name)
    qwen_module = QwenModule(model_name, ap_module)

    start_mem, max_mem = show_current_mem_stats()

    trainer_stats = qwen_module.trainer.train()

    show_final_mem_stats(start_mem, max_mem, trainer_stats)

    qwen_module.model.push_to_hub(model_repo)
    qwen_module.tokenizer.push_to_hub(model_repo)