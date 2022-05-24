# Import the various libraries
import re, json, os, itertools
import pandas as pd
from tqdm import tqdm

# Path to each individual txt files converted from PDF
TC_PATH = "CUAD-v1/full_contract_txt/"

# Path to folder containing all the CUAD data and files
MASTER_PATH = "CUAD-v1/"

# Name of CSV file containing all the extracted clauses from the Atticus team
MASTER_CLAUSES = 'master_clauses.csv'

# Name of JSON file to export the agreement text and labels for data extraction
JSON_EXPORT = 'jsonl_cuadv1.json'

# Name of JSON file to export the agreement taxt and labels for further inspection
JSON_EXPORT_INSPECT = 'jsonl_cuadv1_inspect.json'


# Walk through all .txt filenames and create a dataframe with the names of the files, sorted alpha/num
text_files = []
for (dirpath, dirnames, filenames) in os.walk(TC_PATH):
    text_files.extend(filenames)

tf_df = pd.DataFrame(data = text_files, columns = ['Text Files'])
tf_df.sort_values('Text Files', axis=0, inplace=True, ignore_index=True) 
tf_df.head()


# Read master clauses CSV into a dataframe, sort by filename to match text file dataframe created above
mc_df = pd.read_csv(MASTER_PATH+MASTER_CLAUSES)

# Cut out the relevant info
mc_df_cut = mc_df[['Filename',
                   'Document Name',
                   'Document Name-Answer',
                   'Parties',
                   'Parties-Answer',
                   'Agreement Date',
                   'Agreement Date-Answer',
                   'Effective Date',
                   'Effective Date-Answer',
                   'Expiration Date',
                   'Expiration Date-Answer',
                   'Renewal Term',
                   'Renewal Term-Answer',
                   'Governing Law',
                   'Governing Law-Answer']].copy()

# Sort the dataframe by filename
mc_df_cut.sort_values('Filename', axis=0, inplace=True, ignore_index=True) 

# Bring in the list of the .txt filenames
mc_df_cut.insert(loc=1, column='Text Files', value=tf_df)

# Create a list of the names of the files, with index num
file_list = [(index, row['Text Files']) for index, row in mc_df_cut.iterrows()]


# Create a function to clean up and pre-process the text.
# This process should be used for any document text inc. train, validation and test sets.
def pre_process_doc_common(text):
    # Simple replacement for "\n"
    text = text.replace("\n", " ")     
    
    # Simple replacement for "\xa0"
    text = text.replace("\xa0", " ")  
    
    # Simple replacement for "\x0c"
    text = text.replace("\x0c", " ")
    
    # Get rid of multiple dots
    regex = "\ \.\ "
    subst = "."
    text = re.sub(regex, subst, text, 0)
    
    # Get rid of underscores
    regex = "_"
    subst = " "
    text = re.sub(regex, subst, text, 0)
    
    # Get rid of multiple dashes
    regex = "--+"
    subst = " "
    text = re.sub(regex, subst, text, 0)
    
    # Get rid of multiple stars
    regex = "\*+"
    subst = "*"
    text = re.sub(regex, subst, text, 0)
    
    # Get rid of multiple whitespace
    regex = "\ +"
    subst = " "
    text = re.sub(regex, subst, text, 0)
    
    #Strip leading and trailing whitespace
    text = text.strip()
    
    return text

# Function to take in the file list, read each file, clean the text and return all agreements in a list
def text_data(file_list, print_text=False, clean_text=True, max_len=3000):
    text_list = []
    for index, filename in tqdm(file_list):
        agreement = open(TC_PATH+filename, "r")
        text = agreement.read()
        if print_text:
            print("Text before cleaning: \n", text)
        
        # Run text through cleansing function
        if clean_text:
            text = pre_process_doc_common(text)
        text = text[:max_len]
        len_text = len(text)
        
        if print_text:
            print("Text after cleaning: \n", text)
        
        text_list.append([index,
                  filename,
                  text,
                  len_text])
        
    return text_list


# Clean text and create dataframe with the text of ech document
data = text_data(file_list, print_text=False, clean_text=True, max_len=1000)
columns = ['ID', 'Documents', 'Text', 'Length_Of_Text']
text_df = pd.DataFrame(data=data, columns=columns)

# Add the two columns to a copy of the main dataframe
mc_df_wk = mc_df_cut.copy()
mc_df_wk = mc_df_wk.join(text_df[['Text', 'Length_Of_Text']])

#Ensure agreement date, doc_name and parties are list objects
mc_df_wk["Agreement Date"] = mc_df_wk["Agreement Date"].apply(eval)
mc_df_wk["Effective Date"] = mc_df_wk["Effective Date"].apply(eval)
mc_df_wk["Expiration Date"] = mc_df_wk["Expiration Date"].apply(eval)

mc_df_wk["Document Name"] = mc_df_wk["Document Name"].apply(eval)
mc_df_wk["Parties"] = mc_df_wk["Parties"].apply(eval)
mc_df_wk["Renewal Term"] = mc_df_wk["Renewal Term"].apply(eval)
mc_df_wk["Governing Law"] = mc_df_wk["Governing Law"].apply(eval)


# Some document name references have more than one entry - remove them for further inspection later
mc_df_wk['Doc_N_Length'] = mc_df_wk['Document Name'].str.len()
mc_df_mul = mc_df_wk[mc_df_wk.Doc_N_Length > 1]
mc_df_wk.drop(mc_df_mul.index, inplace=True)

# Agreement date is an important label. Here we will drop any agreement without a date.
# These will typically be template or specimen agreements which havent been executed
# Prior to dropping, we create a dataframe to manually check and annotate agreement date in a different exercise
mc_df_nul = mc_df_wk[mc_df_wk["Agreement Date-Answer"].isnull()]
mc_df_wk = mc_df_wk.dropna(subset=['Agreement Date-Answer'])
mc_df_wk.info()



# The CUADv1 labels includes the Party definition eg Apple Inc. "Apple", here we keep just the legal entity:
def remove_party_overlaps(labels):
    labels.sort()
    k = []
    for i in range(len(labels)-1):
        l1 = labels[i]
        l2 = labels[i+1]
        if l1[0] == l2[0]:
            len1 = l1[1] - l1[0]
            len2 = l2[1] - l2[0]
            if len1 > len2:
                k.append(l1)
                continue
            else:
                k.append(l2)
                continue
        else:
            k.append(labels[i])
    new_labels = list(k for k,_ in itertools.groupby(k))
    
    return new_labels


# Go through each label and find the label in the text, ensure label is pre-processed same as text.
# If labels don't match, append to a seperate file to check.

clean_text = True
djson = list()
djson_inspect = list()
for index, row in tqdm(mc_df_wk.iterrows()):
    labels = list()
    ids = index
    text = row['Text']
    
    #DOC_NAME
    doc_names = row['Document Name']
    for name in doc_names:
        if clean_text:
            name = pre_process_doc_common(name)
        matches = re.finditer(re.escape(name.lower()), text.lower())
        for m in matches:
            s = m.start()
            e = m.end()
            labels.append([s, e, 'DOC_NAME'])
    
    #AGMT_DATE
    agmt_date = row['Agreement Date']
    for date in agmt_date:
        if clean_text:
            date = pre_process_doc_common(date)
        matches = re.finditer(re.escape(date.lower()), text.lower())
        for m in matches:
            s = m.start()
            e = m.end()
            labels.append([s, e, 'AGMT_DATE'])

    #EFF_DATE
    eff_date = row['Effective Date']
    for date in eff_date:
        if clean_text:
            date = pre_process_doc_common(date)
        matches = re.finditer(re.escape(date.lower()), text.lower())
        for m in matches:
            s = m.start()
            e = m.end()
            labels.append([s, e, 'EFF_DATE'])

     #EXP_DATE
    exp_date = row['Expiration Date']
    for date in exp_date:
        if clean_text:
            date = pre_process_doc_common(date)
        matches = re.finditer(re.escape(date.lower()), text.lower())
        for m in matches:
            s = m.start()
            e = m.end()
            labels.append([s, e, 'EXP_DATE'])


     #
    current_df = row['Renewal Term']
    for name in current_df:
        if clean_text:
            name = pre_process_doc_common(name)
        matches = re.finditer(re.escape(name.lower()), text.lower())
        for m in matches:
            s = m.start()
            e = m.end()
            labels.append([s, e, 'DURATION'])
   

    current_df = row['Governing Law']
    for name in current_df:
        if clean_text:
            name = pre_process_doc_common(name)
        matches = re.finditer(re.escape(name.lower()), text.lower())
        for m in matches:
            s = m.start()
            e = m.end()
            labels.append([s, e, 'GOV_LAW'])
   
    #PARTIES
    parties = row['Parties']
    for party in parties:
        if clean_text:
            party = pre_process_doc_common(party)
        matches = re.finditer(re.escape(party.lower()), text.lower())
        for m in matches:
            s = m.start()
            e = m.end()
            labels.append([s, e, 'PARTY'])
    
    labels = remove_party_overlaps(labels)
    #print(labels)
    
    # Check for incongruous finds, add to inspect file
    flat_list = [item for sublist in labels for item in sublist]

    # if 'DOC_NAME' in flat_list and 'AGMT_DATE' in flat_list and 'PARTY' in flat_list:
    #     djson.append({'id': ids, 'text': text, "labels": labels})
    # else:
    #     djson_inspect.append({'id': ids, 'text': text, "labels": labels})

    djson.append({'id': ids, 'text': text, "labels": labels})
    
# Add to the check JSON file the other documents excluded due to duplicate names and no agreement dates
for index, row in tqdm(mc_df_mul.iterrows()):
    labels = list()
    ids = index
    text = row['Text']
    djson_inspect.append({'id': ids, 'text': text, "labels": labels})

for index, row in tqdm(mc_df_nul.iterrows()):
    labels = list()
    ids = index
    text = row['Text']
    djson_inspect.append({'id': ids, 'text': text, "labels": labels})

# The process above requires the three label types to be present in each agreement extract. This may not
# be the case due to the shortening of the agreememt for example. Let's check how many we are left with
# and how many we need to manually check...
print(f"We are left with {len(djson)} training samples out of 510 to annotate.")
print("Additional agreements to check: ",len(djson_inspect))

# Check for erroneous labels
count = 0
for n in range(len(djson)):
    labs = djson[n]['labels']
    flat_list = [item for sublist in labs for item in sublist]
    if -1 in flat_list:
        count += 1
print(count)


# Export the full datasets for import to Doccano
filepath = JSON_EXPORT
open(filepath, 'w').write("\n".join([json.dumps(e) for e in djson]))

filepath = JSON_EXPORT_INSPECT
open(filepath, 'w').write("\n".join([json.dumps(e) for e in djson_inspect]))