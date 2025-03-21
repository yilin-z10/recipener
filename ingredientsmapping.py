import pandas as pd
import re

def find_ingredients(input_text, known_ingredients):
    """Find known ingredients in the input text."""
    matches = {}
    for ingredient in known_ingredients:
        for match in re.finditer(rf"\b{re.escape(ingredient)}\b", input_text, re.IGNORECASE):
            start, end = match.span()
            if ingredient in matches:
                matches[ingredient].append((start, end, "INGREDIENT"))
            else:
                matches[ingredient] = [(start, end, "INGREDIENT")]
    return matches

def combine_overlapping_matches(matches):
    """Combine overlapping ingredient matches."""
    combined = []
    sorted_matches = sorted(matches, key=lambda x: (x[0], x[1]))
    for current in sorted_matches:
        if not combined:
            combined.append(current)
        else:
            last = combined[-1]
            if current[0] <= last[1]:  # Overlapping
                new_match = (last[0], max(last[1], current[1]), "INGREDIENT")
                combined[-1] = new_match
            else:
                combined.append(current)
    return combined

def convert_to_bio_format(instructions_df, known_ingredients):
    """Convert ingredient matches to BIO format."""
    bio_labels = []

    for idx, row in instructions_df.iterrows():
        text = row['instructions']
        doc = text.split()  # Tokenize by spaces (or use SpaCy for more complex tokenization)
        labels = ['O'] * len(doc)  # Start by assuming all tokens are not part of any ingredient

        # Find ingredient matches
        ingredient_matches = find_ingredients(text, known_ingredients)
        matched_positions = []  # Store the position of each match in the text

        # Collect matches and store their token positions
        for ingredient, positions in ingredient_matches.items():
            for start, end, _ in positions:
                start_token = len(text[:start].split())  # Find the start token
                end_token = len(text[:end].split())  # Find the end token
                matched_positions.append((ingredient, start_token, end_token))

        # Convert to BIO format: B-INGREDIENT for the first occurrence, I-INGREDIENT for subsequent occurrences
        ingredient_counts = {}  # Track which ingredient we've seen for BIO labeling
        for ingredient, start_token, end_token in matched_positions:
            for token_idx in range(start_token, end_token):
                if token_idx == start_token:
                    labels[token_idx] = 'B-INGREDIENT'  # First occurrence of ingredient
                else:
                    labels[token_idx] = 'I-INGREDIENT'  # Subsequent occurrence of the same ingredient
        #BIO Format

        bio_labels.append(labels)

    return bio_labels

def main():
    # Load the dataset
    instructions_path = "datafile/split_instructions_spacy.csv"
    ingredients_path = "datafile/fitlered_ingredient_names.csv"

    df = pd.read_csv(instructions_path)
    ingredients_df = pd.read_csv(ingredients_path, header=None)

    # Extract known ingredients
    known_ingredients = ingredients_df[1].dropna().tolist()

    # Convert to BIO format
    df['bio_labels'] = convert_to_bio_format(df, known_ingredients)

    # Save the results
    output_path = "datafile/instructions_with_bio_labels.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed file with BIO labels saved to {output_path}")

if __name__ == "__main__":
    main()
