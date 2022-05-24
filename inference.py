from simpletransformers.ner import NERModel, NERArgs
from transformers import AutoTokenizer
import pandas as pd


from transformers import RobertaTokenizerFast
from text_utils import *
import csv

FEATURE_CLASS_LABELS = "feature_class_labels.json"

if __name__ == '__main__':    
    model_args = NERArgs()

    with open(FEATURE_CLASS_LABELS, 'r') as f:
        labels = json.load(f)

    model_args.labels_list = labels
    
    MODEL_CHECKPOINT = "roberta-base"
    SAVED_MODEL = f"CUAD-{MODEL_CHECKPOINT}" 
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

    model = NERModel('roberta', SAVED_MODEL,use_cuda=False,args=model_args)

    TEST_FILE_PATH = "./Text_Docs/"
    txt_files = []
    for (dirpath, dirnames, filenames) in os.walk(TEST_FILE_PATH):

        txt_files.extend(filenames)


    samples=text_data(TEST_FILE_PATH,txt_files)
    #print(samples)

    
    

    predictions, _ = model.predict(samples)
    for idx, sample in enumerate(samples):
      for word in predictions[idx]:
        print('{}'.format(word))

    for idx,pred in enumerate(predictions):
        file_name=(txt_files[idx])
        extract_agreement(file_name,pred)
    