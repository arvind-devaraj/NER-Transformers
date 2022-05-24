import os, re, math, random, json, string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import TrainerCallback, AdamW, get_cosine_schedule_with_warmup
from transformers import DataCollatorForTokenClassification, PreTrainedModel, RobertaTokenizerFast

from datasets import load_dataset, ClassLabel, Sequence, load_metric



from datetime import date
today = date.today()
log_date = today.strftime("%d-%m-%Y")

RANDOM_SEED = 42

BATCH_SIZES = 8

EPOCHS = 8

# WHICH PRE-TRAINED TRANSFORMER TO FINE-TUNE?
MODEL_BASE = "roberta-base"
MODEL_CHECKPOINT=MODEL_BASE

FEATURE_CLASS_LABELS = "data/feature_class_labels.json"
DATA_FILE = 'data/cuad-v1-annotated.json'
TEMP_MODEL_OUTPUT_DIR = 'temp_model_output_dir'
SAVED_MODEL = f"CUAD-{MODEL_BASE}"  

data_files = DATA_FILE
datasets = load_dataset('json', data_files=data_files, field='data')
print(datasets)

# Check the ner_tags to ensure that these are integers
datasets["train"].features["ner_tags"]


# Open the label list created in pre-processing corresponding to the ner_tag indices
with open(FEATURE_CLASS_LABELS, 'r') as f:
    label_list = json.load(f)

for n in range(len(label_list)):
    print(n, label_list[n])



tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_BASE, add_prefix_space=True)


def tokenize_and_align_labels(examples, label_all_tokens=False):
    tokenized_inputs = tokenizer(examples["split_tokens"],
                                 truncation=True,
                                 is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, load_from_cache_file=True)


model = AutoModelForTokenClassification.from_pretrained(MODEL_BASE, num_labels=len(label_list))

#Optimizer
learning_rate = 0.0000075
lr_max = learning_rate * BATCH_SIZES
weight_decay = 0.05

optimizer = AdamW(
    model.parameters(),
    lr=lr_max,
    weight_decay=weight_decay)

print("The maximum learning rate is: ",lr_max)

# Learning Rate Schedule
num_train_samples = len(datasets["train"])
warmup_ratio = 0.2 # Percentage of total steps to go from zero to max learning rate
num_cycles=0.8 # The cosine exponential rate

num_training_steps = num_train_samples*EPOCHS/BATCH_SIZES
num_warmup_steps = num_training_steps*warmup_ratio

lr_sched = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                           num_warmup_steps=num_warmup_steps,
                                           num_training_steps = num_training_steps,
                                           num_cycles=num_cycles)

args = TrainingArguments(output_dir = TEMP_MODEL_OUTPUT_DIR,
                         learning_rate=lr_max,
                         per_device_train_batch_size=BATCH_SIZES,
                         num_train_epochs=EPOCHS,
                         weight_decay=weight_decay,
                         lr_scheduler_type = 'cosine',
                         warmup_ratio=warmup_ratio,
                         logging_strategy="epoch",
                         save_strategy="epoch",
                         seed=RANDOM_SEED,
                         run_name = MODEL_CHECKPOINT+"-"+log_date
                        )



# Define and instantiate the Trainer...
trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized_datasets["train"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                optimizers=(optimizer, lr_sched)
                )

# Train
trainer.train()                                                                   

trainer.save_model(SAVED_MODEL)