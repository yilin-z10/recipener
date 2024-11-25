#!/usr/bin/env python
# coding: utf-8

# # 1 Ingredients Extraction
# 
# ## step 1 match known_ingredients from the input text

# In[10]:


import pandas as pd
import os
import spacy


# 

# In[2]:


from google.colab import drive
drive.mount('/content/drive')
print(os.listdir('/content/drive/MyDrive/dissertation'))


# In[7]:


pip install spacy


# In[3]:


# Function to find ingredients in the text
def find_ingredients(input_text, known_ingredients):
    matches = {}
    for ingredient in known_ingredients:
        start = 0
        while start < len(input_text):
            start = input_text.find(ingredient, start)
            if start == -1:
                break
            end = start + len(ingredient)
            if ingredient in matches:
                matches[ingredient].append((start, end, "INGREDIENT"))
            else:
                matches[ingredient] = [(start, end, "INGREDIENT")]
            start += len(ingredient)
    return matches

# a small example to test
input_text = "Add the tomatoes to a food processor with a pinch"
known_ingredients = ["tomatoes", "potatoes", "cheese", "grated cheese", "pineapple"]

matches = find_ingredients(input_text, known_ingredients)
matches


# ## step 2 Filter Matches to make sure the whole ingredients name are recognized

# In[4]:


# Function to combine overlapping matches
def combine_overlapping_matches(matches):
    combined = []
    sorted_matches = sorted(matches, key=lambda x: (x[0], x[1]))
    for current in sorted_matches:
        if not combined:
            combined.append(current)
        else:
            last = combined[-1]
            if current[0] <= last[1]:  # overlapping
                new_match = (last[0], max(last[1], current[1]), "INGREDIENT")
                combined[-1] = new_match
            else:
                combined.append(current)
    return combined

# Convert matches dictionary to a flat list of tuples
overcomplete = [item for sublist in matches.values() for item in sublist]
final_matches = combine_overlapping_matches(overcomplete)
final_matches


# ## step 3 Use small test set to test the performance

# In[8]:


input_text = "Preheat the oven to 350F. Place the potatoes on a baking tray. Bake for 40 minutes. Remove and cut each potato into half. Sprinkle grated cheese over the potatoes."
known_ingredients = ["potato", "potatoes", "cheese", "grated cheese", "pineapple"]

matches = find_ingredients(input_text, known_ingredients)
overcomplete = [item for sublist in matches.values() for item in sublist]
final_matches = combine_overlapping_matches(overcomplete)
print(final_matches)


# ## step 4 Use dataset to test the performance

# In[ ]:


# split sentences

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to split text into sentences using spaCy
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Load dataset
path = '/content/drive/MyDrive/dissertation/dataset.csv'
df = pd.read_csv(path)

# Select 'instructions' column and handle missing values
instructions_df = df[['instructions']]
instructions_df['instructions'] = instructions_df['instructions'].fillna("").astype(str)

# Apply sentence splitting
instructions_df['split_instructions'] = instructions_df['instructions'].apply(split_sentences)

# Save the result to a new CSV file
output_path = '/content/drive/MyDrive/dissertation/split_instructions_spacy.csv'
instructions_df.to_csv(output_path, index=False)

output_path


# In[12]:


# Read the instructions data into a DataFrame
path = '/content/drive/MyDrive/dissertation/dataset.csv'
df = pd.read_csv(path)
instructions_df = df[['instructions']]
instructions_df['instructions'] = instructions_df['instructions'].fillna("").astype(str)

# Read the known ingredients
path = '/content/drive/MyDrive/dissertation/fitlered_ingredient_names.csv'
ingredients = pd.read_csv(path, header=None)
known_ingredients = ingredients[1].dropna().tolist()


# In[ ]:


# Process each instruction and find final matches
instructions_df['final_matches'] = instructions_df['instructions'].apply(
    lambda x: combine_overlapping_matches([item for sublist in find_ingredients(x, known_ingredients).values() for item in sublist])
)

# Print or save the DataFrame
print(instructions_df.head())

# Optionally, save the DataFrame to a CSV file
output_path = '/content/drive/MyDrive/dissertation/instructions_with_matches.csv'
instructions_df.to_csv(output_path, index=False)


# In[ ]:





# # 2 Training NER model

# ## Step 1 read and import

# In[ ]:


import pandas as pd
import spacy
from spacy.training import Example
from spacy.tokens import DocBin

# read the data try top 10000 first
path = '/content/drive/MyDrive/dissertation/instructions_with_matches.csv'
df = pd.read_csv(path)
instructions_df = df.head(50000)  # select top 10000

# not null
instructions_df.loc[:, 'instructions'] = instructions_df['instructions'].fillna("").astype(str)


# In[ ]:


get_ipython().system('python -m spacy init config /content/config.cfg --lang en --pipeline ner --optimize accuracy --force')


# ## Step 2 Aligns entity

# In[ ]:


# Facing to Misaligned entity in text warning:
# Aligns entity spans to token boundaries in a spaCy document.
def align_entity_spans(doc, entities):
    new_entities = []
    # Iterate Over Entities and Align Them
    for start, end, label in entities:
        span = doc.char_span(start, end, alignment_mode="expand")
        if span is not None:
            new_entities.append((span.start_char, span.end_char, label))
        else:
            print(f"Could not align entity ({start}, {end}, {label}) in text: {doc.text}")

    # Sort and Filter Overlapping Entities
    new_entities = sorted(new_entities, key=lambda x: (x[0], x[1]))
    filtered_entities = []
    last_end = -1
    # Iterate Over Sorted Entities
    for start, end, label in new_entities:
        if start >= last_end:
            filtered_entities.append((start, end, label))
            last_end = end
        else:
            # overlapping-->longer one
            if end > last_end:
                filtered_entities[-1] = (start, end, label)
                last_end = end
    return filtered_entities


# In[1]:


get_ipython().system('cat /etc/os-release')


# ## Step 3 convert dataframe into spacy for training

# In[ ]:


# convert dataframe into a format that can be used by spaCy for training a NER model
import spacy
def create_training_data(df):
    nlp = spacy.blank("en")
    training_data = []
    for _, row in df.iterrows():
        # Extract Text and Entities
        text = row['instructions']
        entities = eval(row['final_matches'])
        entities = [(start, end, label) for start, end, label in entities]
        doc = nlp.make_doc(text)

        # Align Entities
        aligned_entities = align_entity_spans(doc, entities)

        training_data.append((text, {"entities": aligned_entities}))

    return training_data

training_data = create_training_data(instructions_df)


# In[ ]:


# save the training data in a format that spaCy can use for training
def save_spacy_format(training_data, output_path):
    nlp = spacy.blank("en")
    db = DocBin()
    for text, annot in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annot)
        db.add(example.reference)
    db.to_disk(output_path)

output_path = '/content/drive/MyDrive/dissertation/spacy_training_data.spacy'
save_spacy_format(training_data, output_path)


# # Reduce system RAM

# In[ ]:


import pandas as pd
import spacy
from spacy.training import Example
from spacy.tokens import DocBin
import gc


# In[ ]:


# Define the function to align entities
def align_entity_spans(doc, entities):
    new_entities = []
    for start, end, label in entities:
        span = doc.char_span(start, end, alignment_mode="expand")
        if span is not None:
            new_entities.append((span.start_char, span.end_char, label))
        else:
            print(f"Could not align entity ({start}, {end}, {label}) in text: {doc.text}")

    new_entities = sorted(new_entities, key=lambda x: (x[0], x[1]))
    filtered_entities = []
    last_end = -1
    for start, end, label in new_entities:
        if start >= last_end:
            filtered_entities.append((start, end, label))
            last_end = end
        else:
            if end > last_end:
                filtered_entities[-1] = (start, end, label)
                last_end = end
    return filtered_entities


# In[ ]:


# Function to create training data from DataFrame chunk
def create_training_data(df_chunk):
    nlp = spacy.blank("en")
    training_data = []
    for _, row in df_chunk.iterrows():
        text = row['instructions']
        entities = eval(row['final_matches'])
        entities = [(start, end, label) for start, end, label in entities]
        doc = nlp.make_doc(text)

        aligned_entities = align_entity_spans(doc, entities)

        training_data.append((text, {"entities": aligned_entities}))

    return training_data


# In[ ]:


# Function to save training data in .spacy format
def save_spacy_format(training_data, output_path):
    nlp = spacy.blank("en")
    db = DocBin()
    for text, annot in training_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)


# In[ ]:


# Path to the CSV file
path = '/content/drive/MyDrive/dissertation/instructions_with_matches.csv'

# Define chunk size
chunk_size = 10000
output_path = '/content/drive/MyDrive/dissertation/spacy_training_data.spacy'

# Read and process the CSV file in chunks
for chunk in pd.read_csv(path, chunksize=chunk_size):
    chunk.loc[:, 'instructions'] = chunk['instructions'].fillna("").astype(str)
    training_data_chunk = create_training_data(chunk)
    save_spacy_format(training_data_chunk, output_path)
    del chunk
    del training_data_chunk
    gc.collect()


# ## Step 4 train the model in terminal

# In[ ]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

get_ipython().system('python -m spacy init config /content/config.cfg --lang en --pipeline ner --optimize efficiency --force')


# In[ ]:


#--gpu-id 0

get_ipython().system('python -m spacy train /content/config.cfg --output ./output --paths.train /content/drive/MyDrive/dissertation/spacy_training_data.spacy --paths.dev /content/drive/MyDrive/dissertation/spacy_training_data.spacy  --gpu-id 0')


# ## Step 5 load the trained model and test it simply

# In[ ]:


import pandas as pd
# Define column names
column_names = ['instructions', 'final_matches']

# Load the test data from the uploaded Excel file, specifying column names
test_data_path = '/content/drive/MyDrive/dissertation/test_manually.xlsx'
test_df = pd.read_excel(test_data_path, header=None, names=column_names)


# In[ ]:


# Inspect the first few rows and columns of the dataframe
print(test_df.head())
print(test_df.columns)


# In[ ]:


import spacy
from sklearn.metrics import classification_report

# Load the trained model
model_path = '/content/drive/MyDrive/dissertation/output/model-last'
nlp = spacy.load(model_path)

# Define a function to extract entities from text using the trained model
def get_predictions(text, model):
    doc = model(text)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

# Prepare the test data for evaluation
test_df['predicted_entities'] = test_df['instructions'].apply(lambda x: get_predictions(x, nlp))


# In[ ]:


# load the trained data
model_path = '/content/drive/MyDrive/dissertation/output/model-last'
nlp = spacy.load(model_path)

# test the model simply
text = "Wash beans and break ends off, leaving beans whole. Mix beans, onion, garlic and pecans in a bowl. Coat mixture with olive oil. Spread in baking pan. Bake 20 minutes at 400 degrees, until beans are slightly browned. Enjoy."
# sentences787919: Wash beans and break ends off, leaving beans whole. Mix beans, onion, garlic and pecans in a bowl. Coat mixture with olive oil.
# Spread in baking pan. Bake 20 minutes at 400 degrees, until beans are slightly browned. Enjoy.
# expected result: [(63, 68, 'INGREDIENT'), (70, 76, 'INGREDIENT'), (81, 87, 'INGREDIENT'), (117, 126, 'INGREDIENT')]
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)


# In[ ]:


import spacy
from sklearn.metrics import accuracy_score
import ast

# Load the trained model
model_path = '/content/drive/MyDrive/dissertation/output/model-last'
nlp = spacy.load(model_path)

# Load test data from the provided Excel file
file_path = '/content/drive/MyDrive/dissertation/test_manually.xlsx'
test_data_df = pd.read_excel(file_path, header=None, names=['instructions', 'true_entities'])

# Drop rows with NaN in the true_entities column
test_data_df = test_data_df.dropna(subset=['true_entities'])

# Initialize incorrect label count and total label count
incorrect_labels = 0
total_labels = 0

# Define a function to extract entities from the text using the model
def extract_entities(text, nlp):
    doc = nlp(text)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

# Iterate over the test data
for index, row in test_data_df.iterrows():
    instruction = row['instructions']
    try:
        true_entities = ast.literal_eval(row['true_entities'])  # Convert string representation of list to actual list
        # Check if each entity is a tuple with three elements
        true_entities = [entity for entity in true_entities if isinstance(entity, tuple) and len(entity) == 3]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing true_entities in row {index}: {e}")
        continue  # Skip rows with malformed true_entities
    pred_entities = extract_entities(instruction, nlp)

    # Separate positions and labels for metric calculation
    true_positions = [(start, end) for start, end, label in true_entities]
    true_labels_only = [label for start, end, label in true_entities]

    pred_positions = [(start, end) for start, end, label in pred_entities]
    pred_labels_only = [label for start, end, label in pred_entities]

    # Calculate the total number of labels and the number of incorrect labels for the current row
    total_labels += len(true_labels_only)
    incorrect_labels += len(true_labels_only) - sum(1 for true, pred in zip(true_labels_only, pred_labels_only) if true == pred)

    # Print information about mismatched label counts
    if len(true_labels_only) != len(pred_labels_only):
        print(f"Mismatch in label count for row {index}: {len(true_labels_only)} true labels, {len(pred_labels_only)} predicted labels")

# Calculate error rate
error_rate = incorrect_labels / total_labels

print(f"Incorrect label count: {incorrect_labels}")
print(f"Total label count: {total_labels}")
print(f"Error rate: {error_rate:.2f}")


# In[ ]:


import spacy
import pandas as pd
import ast
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the trained model
model_path = '/content/drive/MyDrive/dissertation/output/model-last'
nlp = spacy.load(model_path)

# Load test data from the provided Excel file
file_path = '/content/drive/MyDrive/dissertation/test_manually.xlsx'
test_data_df = pd.read_excel(file_path, header=None, names=['instructions', 'true_entities'])

# Drop rows with NaN in the true_entities column
test_data_df = test_data_df.dropna(subset=['true_entities'])

# Define a function to extract entities from the text using the model
def extract_entities(text, nlp):
    doc = nlp(text)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

# Initialize lists to hold binary correctness for each sentence
binary_correctness_list = []

# Initialize lists to hold true and predicted binary labels for each sentence
true_binary_labels = []
pred_binary_labels = []

# Iterate over the test data
for index, row in test_data_df.iterrows():
    instruction = row['instructions']
    try:
        true_entities = ast.literal_eval(row['true_entities'])  # Convert string representation of list to actual list
        # Check if each entity is a tuple with three elements
        true_entities = [entity for entity in true_entities if isinstance(entity, tuple) and len(entity) == 3]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing true_entities in row {index}: {e}")
        continue  # Skip rows with malformed true_entities
    pred_entities = extract_entities(instruction, nlp)

    # Separate positions and labels for metric calculation
    true_labels_only = [label for start, end, label in true_entities]
    pred_labels_only = [label for start, end, label in pred_entities]

    # Determine if the prediction is correct at the sentence level
    if len(true_labels_only) == len(pred_labels_only) and all(true == pred for true, pred in zip(true_labels_only, pred_labels_only)):
        binary_correctness_list.append(1)
        true_binary_labels.append(1)
        pred_binary_labels.append(1)
    else:
        binary_correctness_list.append(0)
        true_binary_labels.append(1)  # The ground truth is that we have a sentence to predict
        pred_binary_labels.append(0)  # The prediction is incorrect for this sentence

# Calculate binary accuracy
if binary_correctness_list:
    binary_accuracy = sum(binary_correctness_list) / len(binary_correctness_list)
    print(f"Binary accuracy: {binary_accuracy:.2f}")

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_binary_labels, pred_binary_labels, average='binary')
    recall = recall_score(true_binary_labels, pred_binary_labels, average='binary')
    f1 = f1_score(true_binary_labels, pred_binary_labels, average='binary')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
else:
    print("No valid test data to evaluate binary accuracy.")


# In[18]:


import spacy
import pandas as pd
import ast
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the trained model
model_path = '/content/drive/MyDrive/dissertation/output/model-last'
nlp = spacy.load(model_path)

# Load test data from the provided Excel file
file_path = '/content/drive/MyDrive/dissertation/test_manually.xlsx'
test_data_df = pd.read_excel(file_path, header=None, names=['instructions', 'true_entities'])

# Drop rows with NaN in the true_entities column
test_data_df = test_data_df.dropna(subset=['true_entities'])

# Define a function to extract entities from the text using the model
def extract_entities(text, nlp):
    doc = nlp(text)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

# Initialize lists to hold scores for each sentence
sentence_scores = []

# Iterate over the test data
for index, row in test_data_df.iterrows():
    instruction = row['instructions']
    try:
        true_entities = ast.literal_eval(row['true_entities'])  # Convert string representation of list to actual list
        # Check if each entity is a tuple with three elements
        true_entities = [entity for entity in true_entities if isinstance(entity, tuple) and len(entity) == 3]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing true_entities in row {index}: {e}")
        continue  # Skip rows with malformed true_entities
    pred_entities = extract_entities(instruction, nlp)

    # Separate positions and labels for metric calculation
    true_labels_set = {(start, end, label) for start, end, label in true_entities}
    pred_labels_set = {(start, end, label) for start, end, label in pred_entities}

    # Calculate the score for this sentence
    correct_predictions = len(true_labels_set & pred_labels_set)
    total_true_labels = len(true_labels_set)

    if total_true_labels > 0:
        score = correct_predictions / total_true_labels
    else:
        score = 0

    sentence_scores.append(score)

# Calculate average score
if sentence_scores:
    average_score = sum(sentence_scores) / len(sentence_scores)
    print(f"Average sentence-level score: {average_score:.2f}")

    # Generate binary labels for overall precision, recall, and F1 score
    true_binary_labels = [1 if score > 0 else 0 for score in sentence_scores]
    pred_binary_labels = [1 if score == 1 else 0 for score in sentence_scores]

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_binary_labels, pred_binary_labels)
    recall = recall_score(true_binary_labels, pred_binary_labels)
    f1 = f1_score(true_binary_labels, pred_binary_labels)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
else:
    print("No valid test data to evaluate sentence scores.")





