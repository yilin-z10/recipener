import pandas as pd
import re
import spacy
import os
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from fuzzywuzzy import process

###############################################################################
# 1. Load data
###############################################################################

# Ensure data files exist
if not os.path.exists("datafile/split_instructions_spacy.csv"):
    raise FileNotFoundError("File not found: datafile/split_instructions_spacy.csv")

if not os.path.exists("datafile/fitlered_ingredient_names.csv"):
    raise FileNotFoundError("File not found: datafile/fitlered_ingredient_names.csv")

# Read data
instructions_df = pd.read_csv("datafile/split_instructions_spacy.csv")
ingredients_df = pd.read_csv("datafile/fitlered_ingredient_names.csv", header=None)

# Ensure correct column names
if "instructions" in instructions_df.columns:
    instruction_col = "instructions"
elif "instruction" in instructions_df.columns:
    instruction_col = "instruction"
else:
    raise KeyError("Missing 'instructions' or 'instruction' column in dataset.")

instructions_df["instructions_checked"] = instructions_df[instruction_col].fillna("").astype(str)

# Process ingredient data
if 1 not in ingredients_df.columns:
    raise ValueError("Missing expected ingredient column (index 1) in CSV.")

known_ingredients = ingredients_df[1].dropna().tolist()
if not known_ingredients:
    raise ValueError("known_ingredients list is empty! Check the CSV file.")

# Preload SpaCy model to improve performance
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 2. Define Labeling Functions
###############################################################################

@labeling_function()
def lf_ingredient_lookup(x):
    """Exact ingredient match"""
    for i, ingredient in enumerate(known_ingredients):
        if ingredient.lower() in x.instructions_checked.lower():
            return i
    return INGREDIENT_NOT_FOUND

@labeling_function()
def lf_context(x):
    """Check dependency tree for cooking verbs"""
    if not x.instructions_checked.strip():
        return INGREDIENT_NOT_FOUND
    doc = nlp(x.instructions_checked)
    for token in doc:
        idx = find_ingredient_index_in_list(token.lemma_.lower(), known_ingredients)
        if idx != INGREDIENT_NOT_FOUND:
            return idx
    return INGREDIENT_NOT_FOUND

# Assemble labeling functions
lfs = [lf_ingredient_lookup, lf_context]

###############################################################################
# 3. Snorkel Processing
###############################################################################

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=instructions_df)

# Check the effectiveness of LFs
print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

# Generate final labels
instructions_df["label"] = L_train.max(axis=1)

# Count occurrences of -1 labels
print("Number of -1 labels:", (instructions_df["label"] == -1).sum())

# Ensure `result/` directory exists
os.makedirs("result", exist_ok=True)
instructions_df.to_csv("result/labeled_instructions.csv", index=False)
print("Labeled instructions saved to result/labeled_instructions.csv")
