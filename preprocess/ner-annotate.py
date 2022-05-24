import re, json, os, itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

#import matplotlib.pyplot as plt
#import seaborn as sns

import spacy
from spacy.lang.en import English
from spacy.training import offsets_to_biluo_tags # requires spaCy 3.0
from spacy.util import filter_spans


MASTER_PATH = "CUAD-v1/"
JSONL_FILE = 'jsonl_cuadv1.json'
JSONL_FILE_INS = 'jsonl_cuadv1_inspect.json'
FEATURE_CLASS_LABELS = "feature_class_labels.json"
DATA_FILE = 'cuad-v1-annotated.json'

# JSONL is a multi-line json file and requires lines=True parameter
# Bring in both sets of annotations and concatenate vertically 
df1 = pd.read_json (JSONL_FILE, lines=True)
df2 = pd.read_json (JSONL_FILE_INS, lines=True)

df = pd.concat([df1, df2], axis=0)
df = df1 # Use this line to exclude the additional manually checked data
#df = df.drop(['meta', 'annotation_approver', 'comments'], axis=1)
print(df.head())

df_cut = df[df['labels'].map(lambda d: len(d)) > 0].copy()


nlp = English()
df_cut['tokens'] = df_cut['text'].apply(lambda x: nlp(x))


# Check an example of the text indices and labels
row = df_cut.iloc[4]
doc = row['tokens']
for start, end, label in row['labels']:
    print(start, end, label)
print("\n")
print(doc)

 

# Each word must be seperated for the transformer using the IOB format
# Create tags using token.ent_iob_ and add to the DataFrame
# Allow for any character misalignment between spaCy tokenization and Doccano character indices
tags_list_iob = []
for index, row in df_cut.iterrows():
    doc = row['tokens']
    ents=[]
    for start, end, label in row['labels']:
        if doc.char_span(start, end, label) != None:
            ent = doc.char_span(start, end, label)
            ents.append(ent)
        elif doc.char_span(start, end+1, label) != None:
            ent = doc.char_span(start, end+1, label)
            ents.append(ent)
        elif doc.char_span(start+1, end, label) != None:
            ent = doc.char_span(start+1, end, label)
            ents.append(ent)
        elif doc.char_span(start, end-1, label) != None:
            ent = doc.char_span(start, end-1, label)
            ents.append(ent)
        elif doc.char_span(start-1, end, label) != None:
            ent = doc.char_span(start-1, end, label)
            ents.append(ent)

    pat_orig = len(ents)
    filtered = filter_spans(ents) # THIS DOES THE TRICK
    pat_filt =len(filtered)
    doc.ents = filtered

    #doc.ents = ents
    iob_tags = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ != "O" else "O" for t in doc]
    tags_list_iob.append(iob_tags)
df_cut['tags'] = tags_list_iob


# Generate list of the IOB feature class labels from tags
all_tags = list(itertools.chain.from_iterable(tags_list_iob))

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    unique_list.sort()
    return unique_list

feature_class_labels = unique(all_tags)
print(feature_class_labels)

df_cut['ner_tags'] = df_cut['tags'].apply(lambda x: [feature_class_labels.index(tag) for tag in x])
df_cut['split_tokens'] = df_cut['tokens'].apply(lambda x: [tok.text for tok in x])
export_columns = ['id', 'ner_tags', 'split_tokens']
export_df = df_cut[export_columns]
export_df.to_json(DATA_FILE, orient="table", index=False)

# Export Feature Class Labels for use in Transformer fine tuning
with open(FEATURE_CLASS_LABELS, 'w') as f:
    json.dump(feature_class_labels, f, indent=2) 