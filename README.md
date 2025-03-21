# recipener
Identifying Ingredients from Recipe Instructions Using a Weakly Supervised Approach




# Recipe Ingredient Extraction Pipeline with Weak Supervision

This project provides a pipeline for extracting ingredients from cooking instructions using NLP techniques, including rule-based matching, weak supervision labeling, and fine-tuned RoBERTa token classification.

## 📂 Project Structure

- `ingredientsmapping.py` - Identifies and extracts known ingredients from text using regex-based matching.
- `labeling_instructions_snorkel.py` - Uses Snorkel to weakly label data by applying various labeling functions.
- `training_robertabase.py` - Fine-tunes a RoBERTa model for NER on labeled ingredient data.


---

## 🛠 Setup Instructions

### 1️⃣ Install Dependencies
Run the following command to install required dependencies:
```bash
git clone https://github.com/yilin-z10/recipener
```



---

## 🔹 Step 1: Ingredient Mapping

### **Run `ingredientmapping.py`**
This script reads cooking instructions, extracts ingredient mentions using regex-based matching, and saves the results.
```bash
python ingredientmapping.py
```

### **Input Files:**
- `datafile/split_instructions_spacy.csv` - CSV containing cooking instructions.
- `datafile/fitlered_ingredient_names.csv` - CSV containing a list of known ingredient names.

### **Output:**
- `datafile/instructions_with_matches.csv` - Cooking instructions with extracted ingredient mentions.

---

## 🔹 Step 2: Labeling with Snorkel

### **Run `labeling_instructions_snorkel.py`**
This script uses Snorkel to apply labeling functions for weak supervision.
```bash
python labeling_instructions_snorkel.py
```

### **Functionality:**
- Applies multiple labeling functions to identify ingredients in text.
- Generates labeled data for training.
- Saves labeled data for later training.

### **Output:**
- `result/labeled_instructions.csv` - Weakly labeled dataset with ingredient annotations.

---

## 🔹 Step 3: Training RoBERTa Model

### **Run `training_robertabase.py`**
This script fine-tunes a RoBERTa-based model for Named Entity Recognition (NER) on ingredient mentions.
```bash
python training_robertabase.py
```

### **Functionality:**
- Preprocesses labeled data.
- Tokenizes text and aligns labels for NER.
- Fine-tunes RoBERTa on the dataset.
- Evaluates the model on test data.
- Saves the trained model.

### **Input Files:**
- `result/labeled_instructions.csv` - Weakly labeled training data.
- `test_manually.xlsx` - Manually labeled test data.

### **Output:**
- Trained model stored in `./trained_model/`.
- Performance metrics printed to console.

---


## 📌 Notes
- Ensure all required datasets are in the correct directories before running scripts.
- Adjust hyperparameters in `trainingroberta.py` for better performance if needed.
- You may need to adjust `labelfunctionsnorkel.py` labeling functions for more accurate weak supervision.

---

## 📧 Contact
For questions or issues, feel free to reach out!

---

**Happy Coding! 🎯**



  

  

