import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import KFold
from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import f1_score, accuracy_score
from seqeval.metrics import classification_report


def preprocess_data(df, tokenizer, label_map):
    """
    Preprocesses the data into the format required for token classification.
    """
    dataset = []
    for _, row in df.iterrows():
        text = row["instructions"]
        entities = literal_eval(row["final_matches"]) if isinstance(row["final_matches"], str) else []
        entity_spans = [(start, end) for start, end, _ in entities]

        tokenized = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        labels = []
        offset_mapping = tokenized["offset_mapping"][0].tolist()

        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                labels.append(-100)  # special token
                continue

            label = "O"
            for entity_start, entity_end in entity_spans:
                if start >= entity_start and end <= entity_end:
                    if start == entity_start:
                        label = "B-INGREDIENT"
                    else:
                        label = "I-INGREDIENT"
                    break

            labels.append(label_map[label])

        tokenized_inputs = {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": torch.tensor(labels)
        }

        dataset.append(tokenized_inputs)

    return Dataset.from_list(dataset)


def compute_metrics(pred):
    """
    Computes accuracy, F1 score, and standard deviation of F1 scores.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_labels = []
    pred_labels = []
    f1_scores = []  # List to hold F1 scores for SD calculation

    for i in range(len(labels)):
        true_seq = []
        pred_seq = []
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_seq.append(id2label[labels[i][j]])
                pred_seq.append(id2label[preds[i][j]])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

        # Calculate F1 score for each prediction
        f1_scores.append(f1_score(true_seq, pred_seq, average="weighted"))

    acc = accuracy_score(true_labels, pred_labels)
    f1 = np.mean(f1_scores)  # Average F1 score
    f1_std = np.std(f1_scores)  # Standard deviation of F1 scores
    report = classification_report(true_labels, pred_labels, output_dict=False)

    return {
        "accuracy": acc,
        "f1": f1,
        "f1_std": f1_std,  # Return F1 standard deviation
        "report": report
    }


def main():
    # Load data
    train_data_path = 'datafile/labeled_instructions.csv'
    df = pd.read_csv(train_data_path)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Define label map for BIO format
    label_list = ["O", "B-INGREDIENT", "I-INGREDIENT"]
    label_map = {label: idx for idx, label in enumerate(label_list)}
    id2label = {v: k for k, v in label_map.items()}

    dataset = preprocess_data(df, tokenizer, label_map)

    # Initialize cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores_per_fold = []  # Store F1 scores per fold

    # 5-Fold Cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold + 1}/5...")

        # Split dataset into train and validation sets
        train_subset = dataset.select(train_idx)
        val_subset = dataset.select(val_idx)
        model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=3)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold + 1}",
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_dir=f"./logs/fold_{fold + 1}",
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        results = trainer.evaluate()

        # Store F1 score from this fold
        f1_scores_per_fold.append(results["eval_f1"])

    # Calculate the standard deviation of F1 scores across folds
    f1_mean = np.mean(f1_scores_per_fold)
    f1_std = np.std(f1_scores_per_fold)

    print(f"Mean F1: {f1_mean:.4f}")
    print(f"Standard Deviation of F1: {f1_std:.4f}")


if __name__ == "__main__":
    main()
