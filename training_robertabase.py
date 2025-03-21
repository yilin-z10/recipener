import torch
from datasets import Dataset
from transformers import (
    RobertaForTokenClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import f1_score, accuracy_score
from seqeval.metrics import classification_report


def preprocess_test_data(df, tokenizer, label_map):
    """
    Preprocesses the test data for evaluation in BIO format.
    """
    texts = df["instructions"].tolist()
    true_labels = [literal_eval(row) if isinstance(row, str) else [] for row in df["true_entities"]]

    tokenized_inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", is_split_into_words=False)
    return tokenized_inputs, true_labels


def compute_metrics(pred):
    """
    Computes accuracy, F1 score and classification report for BIO evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_labels = []
    pred_labels = []

    for i in range(len(labels)):
        true_seq = []
        pred_seq = []
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_seq.append(id2label[labels[i][j]])
                pred_seq.append(id2label[preds[i][j]])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=False)

    return {
        "accuracy": acc,
        "f1": f1,
        "report": report
    }


def main():
    # Load data
    train_data_path = 'datafile/labeled_instructions.csv'
    test_data_path = 'datafile/test_manually.xlsx'

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_excel(test_data_path, header=None, names=["instructions", "true_entities"])

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=3)  # BIO, INGREDIENT

    # Define label map for BIO format
    label_list = ["O", "B-INGREDIENT", "I-INGREDIENT"]
    label_map = {label: idx for idx, label in enumerate(label_list)}
    id2label = {v: k for k, v in label_map.items()}

    # Load preprocessed datasets
    train_dataset = Dataset.load_from_disk("datafile/train_bio_format")

    # Initialize Trainer
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
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

    # Evaluate the model
    test_inputs, test_true_labels = preprocess_test_data(test_df, tokenizer, label_map)
    results = trainer.evaluate(test_inputs)

    print("Test Results: ", results)


if __name__ == "__main__":
    main()
