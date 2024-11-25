
import pandas as pd
import re
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Read the instructions data into a DataFrame
instructions_df = pd.read_csv('/content/drive/MyDrive/dissertation/split_instructions_spacy.csv')
instructions_df['instructions'] = instructions_df['instruction'].fillna("").astype(str)

# Read the known ingredients
ingredients = pd.read_csv('/content/drive/MyDrive/dissertation/fitlered_ingredient_names.csv', header=None)
known_ingredients = ingredients[1].dropna().tolist()

# Define label categories
INGREDIENT = 1
ABSTAIN = -1

# Ingredient list
ingredient_list = known_ingredients

# Define labeling functions
@labeling_function()
def lf_ingredient_lookup(x):
    for ingredient in ingredient_list:
        if ingredient in x.instructions.lower():
            return INGREDIENT
    return ABSTAIN

@labeling_function()
def lf_regex(x):
    pattern = r'\b\d+\s(cups?|tsps?|tablespoons?|ounces?|grams?|pinch)\s\w+\b'
    if re.search(pattern, x.instructions.lower()):
        return INGREDIENT
    return ABSTAIN

@labeling_function()
def lf_context(x):
    doc = nlp(x.instructions)
    for token in doc:
        if token.lemma_ in ingredient_list and token.head.lemma_ in ["add", "mix", "pour"]:
            return INGREDIENT
    return ABSTAIN

@labeling_function()
def lf_dependency_parse(x):
    doc = nlp(x.instructions)
    for token in doc:
        if token.lemma_ in ingredient_list and token.dep_ in ["dobj", "pobj"]:
            return INGREDIENT
    return ABSTAIN

# Apply the labeling functions
lfs = [lf_ingredient_lookup, lf_regex, lf_context, lf_dependency_parse]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=instructions_df)

# Analyze the performance of the labeling functions
summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print(summary)

# Save the labeled results to the DataFrame
instructions_df['label'] = L_train.max(axis=1)  # Use the max label across all label functions for each row
instructions_df.to_csv('/content/drive/MyDrive/dissertation/labeled_instructions.csv', index=False)
