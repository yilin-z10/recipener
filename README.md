# recipener
Identifying Ingredients from Recipe Instructions Using a Weakly Supervised Approach

**recipe_database_construction.py** is used to annotate instruction manually. This script processes each original sentence one at a time, ensuring a focused and detailed annotation process.

**ingredientsmapping_SentenceSeg_traditional.py** is a script designed for annotating recipe instructions. It takes a list of known ingredients and instruction sentences as input and maps these ingredients to the sentences by identifying their positions. 
For each instruction, the script:
1. Checks if any known ingredient is present in the text as a substring.
2. Records the start and end indices of each matched ingredient.
3. Stores the results in a dictionary, where:
   - Keys are ingredient names.
   - Values are lists of tuples in the format `(start, end, "INGREDIENT")`.
  

  

