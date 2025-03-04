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


def main():
    # Load the dataset
    instructions_path = "datafile/split_instructions_spacy.csv"
    ingredients_path = "datafile/fitlered_ingredient_names.csv"

    df = pd.read_csv(instructions_path)
    ingredients_df = pd.read_csv(ingredients_path, header=None)

    # Extract known ingredients
    known_ingredients = ingredients_df[1].dropna().tolist()

    # Apply ingredient matching
    df['final_matches'] = df['instructions'].apply(
        lambda x: combine_overlapping_matches(
            [item for sublist in find_ingredients(x, known_ingredients).values() for item in sublist]
        )
    )

    # Save the results
    output_path = "datafile/instructions_with_matches.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed file saved to {output_path}")


if __name__ == "__main__":
    main()
