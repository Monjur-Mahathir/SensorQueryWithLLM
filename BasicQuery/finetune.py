import random 
import torch
import numpy as np 
import pandas as pd
from textwrap import dedent
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from huggingface_hub import login
import neptune
import os

"""
System variables such as huggingface login tokens, usernames and repository names. 
"""
login(token="<YOUR HUGGINGFACE TOKEN>") #Huggingface login token: Change it to your huggingface token
hf_username = "mmonjur" #Huggingface user name: Change it to your huggingface username
repo_name = "UCI_HAR_LLAMA_3_8B" #Huggingface user name: Given any name to your repository

os.environ["NEPTUNE_API_TOKEN"] = "<YOUR NEPTUNE TOKEN" #Nepture Token: Change it to your neptune token
os.environ["NEPTUNE_PROJECT"] = "mmonjur/Llama-3-8b-finetune-tutorial" #Neptune project: Change it to your neptune project

activity_list = {'1': 'WALKING', '2': 'WALKING_UPSTAIRS', '3': 'WALKING_DOWNSTAIRS', '4': 'SITTING', '5': 'STANDING', '6': 'LAYING'} #Activity list in UCI HAR dataset
#Indices of the features used for training
feature_indices = [1, 31, 37, 47, 48, 49, 54, 57, 73, 82, 96, 101, 109, 113, 124, 127, 138, 145, 148, 163, 178, 180, 220, 228, 230, 233, 234, 246, 255, 277, 287, 291, 309, 314, 315, 317, 321, 332, 338, 341, 359, 403, 406, 419, 421, 422, 426, 428, 430, 433, 439, 442, 453, 466, 469, 487, 492, 493, 503, 510, 522, 532, 544, 559]
OUTPUT_DIR = "TrainingOutputs" #Output directory for saving the checkpoints
seed = 42 

"""
Meta uses a decoder-only architecture for Llama 3. Decoder models use only the decoder part of the transformer architecture. At each stage, for a given word,
the attention layers can only access the words that came before it. This is also called autoregressive generation.

Grouped-query attention (GQA) instead of Multi-head attention.

In self attention, we assume that for every token, we will have query, key and values. In multi-head attention, several self-attention heads are computed in 
parallel and concatenated. The heads are independent and do not share params.

In Multi-query attention, the keys and values are shared across all queries. "I will create keys and values in such a way that they will provide answers to all
queries". It decreases the number of parameters and computation but doesn't perform well in practice. "Its hard to create key and value matrices that provide good
answers to all queries".

In GQA, the queries are grouped into groups and the keys and values are shared. "If we can't find such good keys and values that provide good answers to all queries,
we can group the queries and create keys and values that provide good answers to small groups of queries". This is the idea behind GQA.

LoRA: LLMs only use a subset of parameters for a specific task and could thus be represented by a projection to a lower-dimensional space without loss of performance.

foriginal(x) = W0 * x,
flora(x) = W0 * x + (α/r)*∆W*x = W0*x + (α/r)*BAx
Here, ∆W is the fine-tuned correction
B and A represent a low-rank decomposition of the ∆W matrix, where
A is an n x r matrix, B is an r x m matrix, m and n are the original weight 
matrix’ dimensions, r <<<n, m is the lower rank.
"""

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
seed_all(seed)

def get_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, 
            quantization_config=quantization_config, device_map='auto')
    
    PAD_TOKEN = "<|pad|>"
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.padding_side = "right"

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    return model, tokenizer
    
def read_feature_file(filepath, indices_of_selected_features):
    """
    filepath: Path of the feature file in text format where each line contains F features separated by space
    indices_of_selected_features: Index of the subset of the features used for training. We select len(indices_of_selected_features) number of features for training, which is a subset of F.
    """
    Xs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            data = np.array([float(i) for i in data])
            data = data[indices_of_selected_features]
            
            out_str = ""
            for d in data:
                out_str += str(d) + ", "
            out_str = out_str[:-2]
            Xs.append(out_str)
    f.close()
    Xs = np.array(Xs)
    return Xs

def read_label_file(filepath):
    """
    filepath: Path of the label file in text format where each line contains a label
    """
    ys = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = activity_list[line.split()[0]]
            ys.append(label)
    f.close()
    ys = np.array(ys)
    return ys

def process_dataset(train_feature_file, train_label_file, test_feature_file, test_label_file, tokenizer):
    def format_example(row):
        prompt = dedent(
            f"""
            {row["features"]}
            """
        )
        system_prompt = """
        You are human activity recognition assistant, who is given a set of features and asked to predict the activity. 
        The features selected for this task come from the accelerometer and gyroscope 3-axial raw signals tAcc-XYZ and tGyro-XYZ. These time domain signals (prefix 't' to denote time) were captured at a constant rate of 50 Hz. Then they were filtered using a median filter and a 3rd order low pass Butterworth filter with a corner frequency of 20 Hz to remove noise. Similarly, the acceleration signal was then separated into body and gravity acceleration signals (tBodyAcc-XYZ and tGravityAcc-XYZ) using another low pass Butterworth filter with a corner frequency of 0.3 Hz. 
        Subsequently, the body linear acceleration and angular velocity were derived in time to obtain Jerk signals (tBodyAccJerk-XYZ and tBodyGyroJerk-XYZ). Also the magnitude of these three-dimensional signals were calculated using the Euclidean norm (tBodyAccMag, tGravityAccMag, tBodyAccJerkMag, tBodyGyroMag, tBodyGyroJerkMag). 
        Finally a Fast Fourier Transform (FFT) was applied to some of these signals producing fBodyAcc-XYZ, fBodyAccJerk-XYZ, fBodyGyro-XYZ, fBodyAccJerkMag, fBodyGyroMag, fBodyGyroJerkMag. (Note the 'f' to indicate frequency domain signals). 
        These signals were used to estimate variables of the feature vector for each pattern: ('-XYZ' is used to denote 3-axial signals in the X, Y and Z directions)
        
        tBodyAcc-XYZ, tGravityAcc-XYZ, tBodyAccJerk-XYZ, tBodyGyro-XYZ, tBodyGyroJerk-XYZ, tBodyAccMag, tGravityAccMag, tBodyAccJerkMag ,tBodyGyroMag, tBodyGyroJerkMag, fBodyAcc-XYZ, fBodyAccJerk-XYZ, fBodyGyro-XYZ, fBodyAccMag, fBodyAccJerkMag, fBodyGyroMag, fBodyGyroJerkMag

        The set of variables that were estimated from these signals are: 
        mean(): Mean value
        std(): Standard deviation
        mad(): Median absolute deviation 
        max(): Largest value in array
        min(): Smallest value in array
        sma(): Signal magnitude area
        energy(): Energy measure. Sum of the squares divided by the number of values. 
        iqr(): Interquartile range 
        entropy(): Signal entropy
        arCoeff(): Autorregresion coefficients with Burg order equal to 4
        correlation(): correlation coefficient between two signals
        maxInds(): index of the frequency component with largest magnitude
        meanFreq(): Weighted average of the frequency components to obtain a mean frequency
        skewness(): skewness of the frequency domain signal 
        kurtosis(): kurtosis of the frequency domain signal 
        bandsEnergy(): Energy of a frequency interval within the 64 bins of the FFT of each window.
        angle(): Angle between to vectors.

        Additional vectors obtained by averaging the signals in a signal window sample. These are used on the angle() variable:

        gravityMean, tBodyAccMean, tBodyAccJerkMean, tBodyGyroMean, tBodyGyroJerkMean

        However, from the total set of 561 features, only a subset of 64 features will be given to you. These features are:
        Feature 1: tBodyAcc-mean()-Y, Feature 2: tBodyAcc-arCoeff()-Y,3, Feature 3: tBodyAcc-correlation()-X,Y, Feature 4: tGravityAcc-mad()-Y, Feature 5: tGravityAcc-mad()-Z, Feature 6: tGravityAcc-max()-X, Feature 7: tGravityAcc-min()-Z, Feature 8: tGravityAcc-energy()-Y, Feature 9: tGravityAcc-arCoeff()-Z,1, Feature 10: tBodyAccJerk-mean()-Z, Feature 11: tBodyAccJerk-energy()-X, Feature 12: tBodyAccJerk-iqr()-Z, Feature 13: tBodyAccJerk-arCoeff()-Y,1, Feature 14: tBodyAccJerk-arCoeff()-Z,1, Feature 15: tBodyGyro-std()-Y, Feature 16: tBodyGyro-mad()-Y, Feature 17: tBodyGyro-energy()-Z, Feature 18: tBodyGyro-arCoeff()-X,1, Feature 19: tBodyGyro-arCoeff()-X,4, Feature 20: tBodyGyroJerk-std()-X, Feature 21: tBodyGyroJerk-energy()-Z, Feature 22: tBodyGyroJerk-iqr()-Y, Feature 23: tGravityAccMag-iqr(), Feature 24: tBodyAccJerkMag-mad(), Feature 25: tBodyAccJerkMag-min(), Feature 26: tBodyAccJerkMag-iqr(), Feature 27: tBodyAccJerkMag-entropy(), Feature 28: tBodyGyroMag-iqr(), Feature 29: tBodyGyroJerkMag-max(), Feature 30: fBodyAcc-min()-X, Feature 31: fBodyAcc-entropy()-X, Feature 32: fBodyAcc-maxInds-Y, Feature 33: fBodyAcc-bandsEnergy()-57,64, Feature 34: fBodyAcc-bandsEnergy()-1,24, Feature 35: fBodyAcc-bandsEnergy()-25,48, Feature 36: fBodyAcc-bandsEnergy()-9,16, Feature 37: fBodyAcc-bandsEnergy()-41,48, Feature 38: fBodyAcc-bandsEnergy()-17,24, Feature 39: fBodyAcc-bandsEnergy()-1,16, Feature 40: fBodyAcc-bandsEnergy()-49,64, Feature 41: fBodyAccJerk-sma(), Feature 42: fBodyAccJerk-bandsEnergy()-1,16, Feature 43: fBodyAccJerk-bandsEnergy()-49,64, Feature 44: fBodyAccJerk-bandsEnergy()-33,48, Feature 45: fBodyAccJerk-bandsEnergy()-1,24, Feature 46: fBodyAccJerk-bandsEnergy()-25,48, Feature 47: fBodyGyro-std()-X, Feature 48: fBodyGyro-std()-Z, Feature 49: fBodyGyro-mad()-Y, Feature 50: fBodyGyro-max()-Y, Feature 51: fBodyGyro-energy()-X, Feature 52: fBodyGyro-iqr()-X, Feature 53: fBodyGyro-meanFreq()-Z, Feature 54: fBodyGyro-bandsEnergy()-49,56, Feature 55: fBodyGyro-bandsEnergy()-17,32, Feature 56: fBodyGyro-bandsEnergy()-25,48, Feature 57: fBodyGyro-bandsEnergy()-33,40, Feature 58: fBodyGyro-bandsEnergy()-41,48, Feature 59: fBodyAccMag-std(), Feature 60: fBodyAccMag-entropy(), Feature 61: fBodyBodyAccJerkMag-iqr(), Feature 62: fBodyBodyGyroMag-min(), Feature 63: fBodyBodyGyroJerkMag-max(), Feature 64: angle(Y,gravityMean).
        Now, given these features, you are asked to predict the human activity.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["label"]}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    Xs = read_feature_file(train_feature_file, feature_indices)
    ys = read_label_file(train_label_file)

    train_df = pd.DataFrame(columns=['features', 'label'])
    train_df["features"] = Xs
    train_df["label"] = ys

    train_df["text"] = train_df.apply(format_example, axis=1)
    train, val = train_test_split(train_df, test_size=0.2, random_state=seed)

    Xs = read_feature_file(test_feature_file, feature_indices)
    ys = read_label_file(test_label_file)   

    test_df = pd.DataFrame(columns=['features', 'label'])
    test_df["features"] = Xs
    test_df["label"] = ys

    test_df["text"] = test_df.apply(format_example, axis=1)
    test, _ = train_test_split(test_df, test_size=0.95, random_state=seed)

    train.to_json("train.jsonl", orient="records", lines=True)
    val.to_json("val.jsonl", orient="records", lines=True)
    test.to_json("test.jsonl", orient="records", lines=True)

    dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"})
    return dataset


def train_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Fine-tune the llama model on custon sensor query dataset
    """
    model, tokenizer = get_model(model_id)
    dataset = process_dataset("UCI HAR Dataset/train/X_train.txt", "UCI HAR Dataset/train/y_train.txt", "UCI HAR Dataset/test/X_test.txt", "UCI HAR Dataset/test/y_test.txt", tokenizer)
    response_template = "<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    lora_config = LoraConfig(
        r=32,  #low rank dimension
        lora_alpha=16,  #scaling factor of ∆W: (α/r)
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        ],
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field='text',  # text contains the formatted examples
        max_seq_length=4096,
        num_train_epochs=1, #Increase accordingly
        per_device_train_batch_size=2,  # training batch size
        per_device_eval_batch_size=1,  # evaluation batch size
        gradient_accumulation_steps=2,  # update weights every: batch_size * gradient_accum_steps = 2 * 2 = 4 steps
        optim="paged_adamw_8bit",  #8 bit optimizers reduce the memory requirement by 75%
        eval_strategy='steps',
        eval_steps=0.25,  # evalaute every 25% of the trainig steps
        save_steps=0.2,  # save every 20% of the trainig steps
        logging_steps=10,
        learning_rate=1e-4,
        fp16=True, 
        save_strategy='steps',
        warmup_ratio=0.1,  
        lr_scheduler_type="cosine", 
        save_safetensors=True,  
        dataset_kwargs={
            "add_special_tokens": False,  # we template with special tokens already
            "append_concat_token": False,  # no need to add additional sep token
        },
        seed=seed
    )
    ####Dont need this portion if not using neptune for tracking
    model_params = {
        'model_name': "meta-llama/Meta-Llama-3-8B-Instruct",
        'load_in_4bit': 'True',
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'torch.bfloat16',
    }
    run = neptune.init_run()
    params = {**model_params}
    run['parameters'] = params
    ####

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

def load_saved_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct",checkpoint=735):
    """
    Load the saved model from the checkpoint and push it to the huggingface hub
    """
    new_model = f"{OUTPUT_DIR}/checkpoint-{str(checkpoint)}"
    tokenizer = AutoTokenizer.from_pretrained(new_model)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = PeftModel.from_pretrained(model, new_model)
    model = model.merge_and_unload()

    # Push the model to hugging face
    model.push_to_hub(f"{hf_username}/{repo_name}", tokenizer=tokenizer, max_shard_size="5GB", private=True)
    tokenizer.push_to_hub(f"{hf_username}/{repo_name}", private=True)

if __name__ == "__main__":
    model_id="meta-llama/Meta-Llama-3-8B-Instruct"
    checkpoint=1470 #The checkpoint may be different based on number of epochs, saving frequency etc.
    
    train_model(model_id)
    #load_saved_model(model_id, checkpoint)
