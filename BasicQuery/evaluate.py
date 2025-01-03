import random 
import numpy as np 
import torch
from textwrap import dedent
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from trl import DataCollatorForCompletionOnlyLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login

"""
System variables such as huggingface login tokens, usernames and repository names. 
"""
login(token="<YOUR HUGGINGFACE TOKEN>") #Huggingface login token: Change it to your huggingface token
hf_username = "mmonjur" #Huggingface user name: Change it to your huggingface username
repo_name = "UCI_HAR_LLAMA_3_8B" #Huggingface user name: Given any name to your repository

activity_list = {'1': 'WALKING', '2': 'WALKING_UPSTAIRS', '3': 'WALKING_DOWNSTAIRS', '4': 'SITTING', '5': 'STANDING', '6': 'LAYING'}
feature_indices = [1, 31, 37, 47, 48, 49, 54, 57, 73, 82, 96, 101, 109, 113, 124, 127, 138, 145, 148, 163, 178, 180, 220, 228, 230, 233, 234, 246, 255, 277, 287, 291, 309, 314, 315, 317, 321, 332, 338, 341, 359, 403, 406, 419, 421, 422, 426, 428, 430, 433, 439, 442, 453, 466, 469, 487, 492, 493, 503, 510, 522, 532, 544, 559]

def load_model():
    model_id = f"{hf_username}/{repo_name}"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, 
            quantization_config=quantization_config, device_map='auto')
    return model, tokenizer

def creaet_test_prompt(features):
    prompt = dedent(
        f"""
        {features}
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
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

if __name__ == "__main__":
    model, tokenizer = load_model()
    pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        return_full_text=False
    )

    test_feature_file = "UCI HAR Dataset/test/X_test.txt"
    test_label_file = "UCI HAR Dataset/test/y_test.txt"
    Xs = read_feature_file(test_feature_file, feature_indices)
    ys = read_label_file(test_label_file) 
    
    test_df = pd.DataFrame(columns=['features', 'label'])
    test_df["features"] = Xs
    test_df["label"] = ys

    features = test_df["features"].tolist()
    labels = test_df["label"].tolist()

    acc = 0
    total = 0
    for i in range(len(features)):
        prompt = creaet_test_prompt(features[i])
        result = pipe(prompt)[0]['generated_text'].lower()
        label = labels[i].lower()
        
        if label == "walking" and ( result.__contains__("walk") and not (result.__contains__("up") or result.__contains__("down")) ):
            acc += 1
        elif label == "walking_upstairs" and ( (result.__contains__("walk") and  result.__contains__("up")) and not result.__contains__("down") ):
            acc += 1
        elif label == "walking_downstairs" and ( (result.__contains__("walk") and  result.__contains__("down")) and not result.__contains__("up") ):
            acc += 1
        elif label == "sitting" and result.__contains__("sit"):
            acc += 1
        elif label == "standing" and result.__contains__("stand"):
            acc += 1
        elif label == "laying" and result.__contains__("lay"):
            acc += 1

        total += 1
    print(f"Accuracy: {acc/total*100:.2f}%")
    
    
    
