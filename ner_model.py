from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

BS = 4
GRAD_ACC = 8
LR = 5e-5
WD = 0.01
WARMUP = 0.1
N_EPOCHS = 5

def create_model(model_checkpoint: str,
                 num_labels: str
                 ):
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels)

    return model


def create_training_args(
        model_name,
        learning_rate: float = LR,
        per_device_train_batch_size: int = BS,
        per_device_eval_batch_size: int = BS,
        n_epochs: int = N_EPOCHS,
        weight_decay: float = WD,
        gradient_accumulation_steps: int = GRAD_ACC,
        warmup_ratio: float = WARMUP
):
    return TrainingArguments(
        model_name,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=n_epochs,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio
    )