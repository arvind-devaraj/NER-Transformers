import os, re, math, random, json, string, csv
from tqdm import tqdm 

import fitz 
from pathlib import Path
from collections import defaultdict


# Text cleaning function for standard PDF parsing workflow
def pre_process_doc_common(text):
    text = text.replace("\n", " ")  # Simple replacement for "\n"   
    text = text.replace("\xa0", " ")  # Simple replacement for "\xa0"
    text = text.replace("\x0c", " ")  # Simple replacement for "\x0c"
    
    regex = "\ \.\ "
    subst = "."
    text = re.sub(regex, subst, text, 0)  # Get rid of multiple dots
        
    regex = "_"
    subst = " "
    text = re.sub(regex, subst, text, 0)  # Get rid of underscores
       
    regex = "--+"
    subst = " "
    text = re.sub(regex, subst, text, 0)   # Get rid of multiple dashes
        
    regex = "\*+"
    subst = "*"
    text = re.sub(regex, subst, text, 0)  # Get rid of multiple stars
        
    regex = "\ +"
    subst = " "
    text = re.sub(regex, subst, text, 0)  # Get rid of multiple whitespace
    
    text = text.strip()  #Strip leading and trailing whitespace
    return text



# Function to take in the file list, read each file, clean the text and return all agreements in a list
def pdf_to_text(test_dir, pdf_files):
    text_list = []
    for filename in tqdm(pdf_files):
        print(test_dir+filename)
        agreement = fitz.open(test_dir+filename)
        full_text = ""
        for page in agreement:
            full_text += page.getText('text')#+"\n"
       
        full_text = pre_process_doc_common(full_text)
       
        #text_list.append([filename, full_text, short_text, len_text])
        text_list.append(full_text)
        base_name=Path(filename).stem

        fp=open(base_name+".txt","w", encoding="utf-8")
        fp.write(full_text)
        fp.close()
    return text_list

def text_data(test_dir, txt_files):
    text_list = []
    for filename in txt_files:
        fp=open(test_dir+filename, encoding="utf-8")
        full_text=fp.read()
        fp.close()
        text_list.append(full_text)
        
    return text_list


# Functions to extract each important data point based on the model's labeling of each token

file= open('output.csv', 'w', newline='')
writer = csv.writer(file)

def extract_agreement(file_name,entity_list):
    temp=""
    result=[]
    phrase=defaultdict(list)
    for idx,pred in enumerate(entity_list):
        word = list(pred.keys())[0]
        label = list(pred.values())[0]
        
        #print((word,label))
         
        if label=="B-AGMT_DATE":
            phrase["agmt_date"].append(word)
        if label=="B-EFF_DATE":
            phrase["eff_date"].append(word)
        if label=="B-DURATION":
            phrase["duration"].append(word)
        if label=="B-EXP_DATE":
            phrase["exp_date"].append(word)
        if label=="B-GOV_LAW":
            phrase["gov_law"].append(word)
        
        if label=="B-DOC_NAME":
            phrase["doc_name"].append(word)
        if label=="B-PARTY":
            phrase["party"].append(word)
        
        word= " "+word
        if label=="I-AGMT_DATE" and len(phrase["agmt_date"]):
            phrase["agmt_date"][-1] += word
        if label=="I-EFF_DATE" and len(phrase["eff_date"]):
            phrase["eff_date"][-1] += word
        if label=="I-DURATION" and len(phrase["duration"]):
            phrase["duration"][-1] += word
        if label=="I-EXP_DATE" and len(phrase["exp_date"]):
            phrase["exp_date"][-1] += word
        if label=="I-GOV_LAW" and len(phrase["gov_law"]):
            phrase["gov_law"][-1] += word
        if label=="I-DOC_NAME" and len(phrase["doc_name"]):
            phrase["doc_name"][-1] += word
        if label=="I-PARTY" and len(phrase["party"]):
            phrase["party"][-1] += word



    doc_name=phrase["doc_name"][0] if len(phrase["doc_name"]) else ""
    parties="|".join(phrase["party"]) if len(phrase["party"]) else ""
    agmt_date=phrase["agmt_date"][0] if len(phrase["agmt_date"]) else ""
    eff_date=phrase["eff_date"][0] if len(phrase["eff_date"]) else ""
    writer.writerow([file_name,doc_name,parties,agmt_date,eff_date])
     
    


