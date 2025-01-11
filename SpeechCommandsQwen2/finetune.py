import triton
from datasets import load_dataset
from huggingface_hub import login
from PIL import Image
from io import BytesIO
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import wandb
import neptune
import os

login(token="T") #Huggingface login token: Change it to your huggingface token
hf_username = "X" #Huggingface user name: Change it to your huggingface username
repo_name = "Y" #Huggingface user name: Given any name to your repository

#Nepture Token: Change it to your neptune token
os.environ["NEPTUNE_API_TOKEN"] = "TT" 
os.environ["NEPTUNE_PROJECT"] = "P" #Neptune project: Change it to your neptune project


model_id = "Qwen/Qwen2-VL-7B-Instruct"

def format_data(sample):
    system_message = """You are a human speech recognizer. You will be given a spectrogram of audio signal of one second, containing a single word and your job is to predict the word. The list of possible words and corresponding phonemes are given below:
    bed: [B, EH, D], bird: [B, ER, D], cat: [K, AE, T], dog: [D, AO, G], down: [D, AW, N], eight: [EY, T], five: [F, AY, V], four: [F, AO, R], go: [G, OW], happy: [HH, AE, P, IY], house: [HH, AW, S], left: [L, EH, F, T], marvin: [M, AA, R, V, IH, N], nine: [N, AY], no: [N, OW], off: [AO, F], on: [AA, N], one: [W, AH, N], right: [R, AY, T], seven: [S, EH, V, AH, N], sheila: [SH, IY, L, AH], six: [S, IH, K], stop: [S, T, AA, P], three: [TH, R, IY], tree: [T, R, IY], two: [T, UW], up: [AH, P], wow: [W, AW], yes: [Y, EH, S], zero: [Z, IH, R, OW]
    Focus on identifying the word from the sequence of phonemes in the spectrogram image and provide response in one word only which is the predicted word."""

    image_bytes = sample["image"]['bytes']
    image_bytes = BytesIO(image_bytes)
    image = Image.open(image_bytes)
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "Predict the word from the spectrogram of the audio signal given as image.",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]

def get_dataset():
    train_dataset, test_dataset = load_dataset("SpeechClassification/data_short", split=["train", "test"])
    train_valid_split = train_dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = train_valid_split["train"]
    eval_dataset = train_valid_split["test"]

    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]

    return train_dataset, eval_dataset, test_dataset

def get_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)

    return model, processor


def train(model, processor, train_dataset, eval_dataset):
    def collate_fn(examples):
        texts = [processor.apply_chat_template(example, tokenize=False) for example in examples] 

        image_inputs = [process_vision_info(example)[0] for example in examples]
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone() 
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        if isinstance(processor, Qwen2VLProcessor):  # Set the image token IDs for Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655] 
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels 

        return batch 
    
    target_modules = [
        "q_proj", 
        "v_proj",
        "o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",

        # Patch Embedding
        "visual.patch_embed.proj",

        # Attention Mechanisms in Vision Blocks
        "visual.blocks.*.attn.qkv",
        "visual.blocks.*.attn.proj",
        "visual.blocks.*.mlp.fc1",
        "visual.blocks.*.mlp.fc2",

        # Merger Components
        "visual.merger.mlp.0",
        "visual.merger.mlp.2"
    ]
    peft_config = LoraConfig(
        lora_alpha=16, #scaling factor of ∆W: (α/r)
        lora_dropout=0.05,
        r=32,  #low rank dimension
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    training_args = SFTConfig(
        output_dir="qwen2-7b-instruct-trl-sft-SpeechCommands", 
        num_train_epochs=2,  
        per_device_train_batch_size=6, 
        per_device_eval_batch_size=6,  
        gradient_accumulation_steps=1,  
        gradient_checkpointing=True,  
       
        optim="adamw_torch_fused",  
        learning_rate=1e-5,  
        lr_scheduler_type="constant",  
      
        logging_steps=16,  
        eval_steps=128,  
        eval_strategy="steps",  
        save_strategy="steps",  
        save_steps=128, 
        metric_for_best_model="eval_loss",  
        greater_is_better=False, 
        load_best_model_at_end=True,  
       
        bf16=True,  
        tf32=True,  
        max_grad_norm=0.3, 
        warmup_ratio=0.03,  
        
        push_to_hub=False, 
        report_to="wandb",  
        
        gradient_checkpointing_kwargs={"use_reentrant": False},  
       
        dataset_text_field="",  
        dataset_kwargs={"skip_prepare_dataset": True}, 
    )

    training_args.remove_unused_columns = False 

    wandb.init(
        project="P",  # change this
        name="N",  # change this
        config=training_args,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    model_params = {
        'model_name': model_id,
    }
    run = neptune.init_run()
    params = {**model_params}
    run['parameters'] = params

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train_dataset, eval_dataset, test_dataset = get_dataset()
    model, processor = get_model()

    train(model, processor, train_dataset, eval_dataset)
