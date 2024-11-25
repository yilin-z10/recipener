import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from ast import literal_eval


# Paths to data
train_data_path = '/content/drive/MyDrive/dissertation/labeled_instructions.csv'  # Training data labeled by Snorkel
test_data_path = '/content/drive/MyDrive/dissertation/test_manually.xlsx'  # Test data


# Step 1: Load the data
# Load labeled training data
train_df = pd.read_csv(train_data_path)

# Load test data
test_df = pd.read_excel(test_data_path, header=None, names=["instructions", "true_entities"])


# Step 2: Define helper functions for data preprocessing
def preprocess_training_data(df, tokenizer, label_map):
    """
    Preprocesses the training data to align tokens with labels for NER training.

    Args:
        df: DataFrame containing training data.
        tokenizer: Tokenizer for tokenizing sentences.
        label_map: Dictionary mapping labels to numerical values.

    Returns:
        Dataset object with input_ids, attention_mask, and labels.
    """
    texts = df["instructions"].tolist()
    labels = []

    for _, row in df.iterrows():
        tokens = tokenizer(row["instructions"], truncation=True, padding=True, return_offsets_mapping=True)
        label_ids = [-100] * len(tokens["input_ids"])  # Default to -100 for ignored tokens
        entity_positions = literal_eval(row["final_matches"]) if isinstance(row["final_matches"], str) else []

        for start, end, _ in entity_positions:
            for idx, (offset_start, offset_end) in enumerate(tokens["offset_mapping"]):
                if offset_start >= start and offset_end <= end:
                    label_ids[idx] = label_map["INGREDIENT"]

        tokens.pop("offset_mapping")  # Remove offsets as they are not needed after labeling
        tokens["labels"] = label_ids
        labels.append(tokens)

    return Dataset.from_list(labels)


def preprocess_test_data(df, tokenizer):
    """
    Preprocesses the test data for evaluation.

    Args:
        df: DataFrame containing test data.
        tokenizer: Tokenizer for tokenizing sentences.

    Returns:
        tokenized_inputs: Tokenized inputs for the test dataset.
        true_labels: True labels in the dataset for evaluation.
    """
    texts = df["instructions"].tolist()
    true_labels = [literal_eval(row) if isinstance(row, str) else [] for row in df["true_entities"]]

    tokenized_inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", is_split_into_words=False)
    return tokenized_inputs, true_labels


# Step 3: Initialize tokenizer and define label map
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
label_map = {"O": 0, "INGREDIENT": 1}  # Map labels to integers
id2label = {v: k for k, v in label_map.items()}

train_dataset = preprocess_training_data(train_df, tokenizer, label_map)
test_inputs, test_true_labels = preprocess_test_data(test_df, tokenizer)


# Step 4: Load the pre-trained RoBERTa model and add a classification head
model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=len(label_map))
model.config.id2label = id2label
model.config.label2id = label_map


# Step 5: Define metrics for evaluation
def compute_metrics(pred):
    """
    Computes precision, recall, F1 score, and accuracy.

    Args:
        pred: Predictions object from the Trainer.

    Returns:
        A dictionary of evaluation metrics.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_labels = [label for label_seq in labels for label in label_seq if label != -100]
    pred_labels = [label for pred_seq in preds for label in pred_seq if label != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted")
    acc = accuracy_score(true_labels, pred_labels)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Step 6: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  
    evaluation_strategy="epoch", 
    learning_rate=5e-5,  
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,  
    num_train_epochs=10,  
    weight_decay=0.01,  
    logging_dir="./logs",  
    logging_steps=10,  
    save_strategy="epoch", 
    load_best_model_at_end=True,  
    metric_for_best_model="f1",  
    save_total_limit=2,  
    optim="adamw_torch",  # Use AdamW
)


# Step 7: Initialize Trainer
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Step 8: Train the model
trainer.train()


# Step 9: Evaluate the model on the test set
test_results = trainer.evaluate()
print("Test results:", test_results)


# Step 10: Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
