import pandas as pd
import re
import spacy
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis

###############################################################################
# 1. Load data
###############################################################################
instructions_df = pd.read_csv('/datafile/split_instructions_spacy.csv')
instructions_df['instructions'] = instructions_df['instruction'].fillna("").astype(str)

ingredients = pd.read_csv('/datafile/fitlered_ingredient_names.csv', header=None)
known_ingredients = ingredients[1].dropna().tolist()

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 2. Define "no match" constant
###############################################################################
INGREDIENT_NOT_FOUND = -1

###############################################################################
# 3. Helper functions for partial extraction
###############################################################################
re_paren = re.compile(r'\(([^()]*)\)')

def extract_parenthetic_text(text):
    """
    Recursively remove text within parentheses, returning:
      - The cleaned text without parentheses content
      - A list of all extracted parenthetical contents
    """
    match = re_paren.search(text)
    if match:
        span = match.span()
        new_text = text[:span[0]] + text[span[1]:]
        final_text, extracted = extract_parenthetic_text(new_text)
        extracted.insert(0, match.group(1))
        return final_text, extracted
    return text, []

delimiter_re = re.compile(r'[,:;\-]')

def split_at_first_noun_phrase(text):
    """
    Try to find the boundary of the first noun phrase using SpaCy.
    We look for delimiters like commas, colons, semicolons, or dashes,
    and if the root of that segment is NOUN, we split there.
    """
    for delim in delimiter_re.finditer(text):
        split_pos = delim.start()
        left_part = text[:split_pos]
        doc = nlp(left_part)
        try:
            sent = next(doc.sents)
            if sent.root.pos_ == 'NOUN':
                return left_part.strip(), text[split_pos+1:].strip()
        except StopIteration:
            pass
    return text.strip(), ""

# Regex that tries to separate "[quantity/unit]" from "[remaining text]"
re_measure_ext_name_str = r"^\*?\-?\s?([0-9a-zA-Z/\.\-\s]*)\s+(.*)"
re_measure_ext_name = re.compile(re_measure_ext_name_str)

def extract_ingredient_name_only(ingr_text):
    """
    Attempt to extract the main ingredient name from a line of text,
    ignoring possible quantity/unit prefixes and removing parenthetical content.
    Returns the extracted main name in lowercase, or the whole text if extraction fails.
    """
    text = ingr_text.lower().strip()
    match = re_measure_ext_name.search(text)
    if match:
        extended_name = match.group(2).strip()
    else:
        extended_name = text

    # Remove parenthetical parts (e.g. (finely chopped), (optional))
    main_text, _ = extract_parenthetic_text(extended_name)

    # Split off at first noun phrase
    main_name, modifier = split_at_first_noun_phrase(main_text)

    # Remove leading "of " if present
    if main_name.startswith("of "):
        main_name = main_name[3:].strip()

    return main_name if main_name else extended_name

def find_ingredient_index_in_list(word, ingredient_list):
    """
    Return the index of 'word' in 'ingredient_list' (case-insensitive match).
    If not found, return -1.
    """
    word_lower = word.lower()
    for i, ing in enumerate(ingredient_list):
        if ing.lower() == word_lower:
            return i
    return INGREDIENT_NOT_FOUND

###############################################################################
# 4. Define Labeling Functions that return the matched ingredient index
###############################################################################

COOKING_VERBS = {
    "add", "mix", "pour", "stir", "combine", "whisk", "fold", "toss",
    "marinate", "boil", "fry", "roast", "saute", "sauté", "season", "chop",
    "slice", "dice", "peel", "mash", "bake", "blend", "drizzle", "sprinkle",
    "cook", "brown", "sear", "stew", "grill", "preheat", "reduce", "steam",
    "broil", "whip", "mince", "rinse", "soak", "scramble", "beat"
}

def is_governed_by_cooking_verb(token, cooking_verbs):
    """
    Climb up the dependency tree from 'token' until reaching the root.
    If we find a token whose lemma is in cooking_verbs, return True.
    Otherwise, return False.
    """
    current = token
    while current.head != current:
        current = current.head
        if current.lemma_.lower() in cooking_verbs:
            return True
    return False

@labeling_function()
def lf_context(x):
    doc = nlp(x.instructions)
    best_idx = -1
    for token in doc:
        # Check if token is a known ingredient
        idx = find_ingredient_index_in_list(token.lemma_.lower(), known_ingredients)
        if idx != INGREDIENT_NOT_FOUND:
            if is_governed_by_cooking_verb(token, COOKING_VERBS):
                return idx
    return best_idx

@labeling_function()
def lf_ingredient_lookup(x):
    text_lower = x.instructions.lower()
    for i, ingredient in enumerate(known_ingredients):
        if ingredient.lower() in text_lower:
            return i
    return INGREDIENT_NOT_FOUND

@labeling_function()
def lf_spacy_extraction(x):
    candidate_name = extract_ingredient_name_only(x.instructions)
    return find_ingredient_index_in_list(candidate_name, known_ingredients)

@labeling_function()
def lf_dependency_parse(x):
    doc = nlp(x.instructions)
    for token in doc:
        token_lower = token.lemma_.lower()
        idx = find_ingredient_index_in_list(token_lower, known_ingredients)
        if idx != INGREDIENT_NOT_FOUND:
            if token.dep_ in ["dobj", "pobj"]:
                return idx
    return INGREDIENT_NOT_FOUND

@labeling_function()
def lf_simple_regex(x):
    """
    A simple regex approach that tries to detect patterns like:
      "X cup(s) something"
    then checks if 'something' is a known ingredient.
    """
    pattern = r'\b\d+\s(cups?|tsps?|tablespoons?|ounces?|grams?|pinch)\s([a-zA-Z]+)\b'
    match = re.search(pattern, x.instructions.lower())
    if match:
        possible_ing = match.group(2)  # The second capturing group
        return find_ingredient_index_in_list(possible_ing, known_ingredients)
    return INGREDIENT_NOT_FOUND

###############################################################################
# 5. Apply all labeling functions to the DataFrame
###############################################################################
lfs = [
    lf_ingredient_lookup,
    lf_spacy_extraction,
    lf_context,
    lf_dependency_parse,
    lf_simple_regex
]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=instructions_df)

# Convert Snorkel output to BIO format
def convert_to_bio_format(instructions_df, L_train, known_ingredients):
    bio_labels = []

    for idx, row in instructions_df.iterrows():
        doc = nlp(row['instructions'])
        labels = ['O'] * len(doc)  # Start by assuming all tokens are not part of any ingredient

        ingredient_occurrences = []  # Track the position of each ingredient
        for col_idx, label in enumerate(L_train[idx]):
            if label != INGREDIENT_NOT_FOUND:
                ingredient = known_ingredients[label]  # Get the ingredient from index
                ingredient_occurrences.append((ingredient, col_idx))

        for ingredient, token_pos in ingredient_occurrences:
            if token_pos == 0 or labels[token_pos - 1] == 'O':  # First occurrence of ingredient
                labels[token_pos] = 'B-INGREDIENT'
            else:  # Later occurrences of the same ingredient
                labels[token_pos] = 'I-INGREDIENT'

        bio_labels.append(labels)

    return bio_labels

bio_labels = convert_to_bio_format(instructions_df, L_train, known_ingredients)

# Add BIO labels to dataframe
instructions_df['bio_labels'] = bio_labels

# Save to file
instructions_df.to_csv('/result/labeled_instructions_bio.csv', index=False)
print("Labeled instructions with BIO format saved to /result/labeled_instructions_bio.csv")
