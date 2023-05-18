from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq
from datasets import load_metric
import numpy as np
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


#export CUDA_VISIBLE_DEVICES=0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
model_checkpoint = "KoJLabs/nllb-finetuned-ko2en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang='kor_Hang', tgt_lang='eng_Latn')
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids('eng_Latn')


# dataset
import pandas as pd

train = pd.read_feather('/home/jisukim/online/translation_datasets/train/ko_en_train_aihub_total.feature')
valid = pd.read_feather('/home/jisukim/online/translation_datasets/valid/ko_en_valid_aihub_total.feature')

dataset = DatasetDict({
    "train": Dataset.from_pandas(train),
    "validation": Dataset.from_pandas(valid)
    })

# declare preprocessing code 

max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = examples['ko']
    targets = examples['en']

    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_input_length, truncation=True)

    return model_inputs            


tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



metric = load_metric("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    print(decoded_preds, decoded_labels)
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



args = Seq2SeqTrainingArguments(
    f"nllb-finetuned-allaihub-ko-to-en",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# trainer.train()

# 
print(trainer.evaluate(max_length=max_target_length))

