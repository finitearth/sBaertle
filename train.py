import transformers
import datasets
import pandas as pd
import argparse
from datetime import datetime
import numpy as np
# from datasets import load_dataset, load_metric
from transformers import EncoderDecoderModel
import evaluate

"""
cd s-BERTle
conda activate ds
python train.py
"""

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_checkpoint", type=str, default="Shahm/bart-german") #GermanT5/t5-efficient-gc4-all-german-small-el32
    arg_parser.add_argument("--num_epochs", type=int, default=5)
    arg_parser.add_argument("--batch_size", type=int, default=16)
    arg_parser.add_argument("--to_german", action="store_true")
    arg_parser.add_argument("--max_length", type=int, default=128)
    arg_parser.add_argument("--lr", type=float, default=2e-5)
    arg_parser.add_argument("--weight_decay", type=float, default=0.01)

    args = arg_parser.parse_args()

    return args

    


def preprocess_function(examples):
    source_lan, target_lan = ("schwaebisch", "deutsch") if args.to_german else ("deutsch", "schwaebisch")

    prefix = f"Übersetze von {source_lan} nach {target_lan}: " if "T5" in args.model_checkpoint else ""
    inputs = [prefix + ex for ex in examples[source_lan]]
    data = tokenizer(inputs, max_length=args.max_length, truncation=True)

    targets = examples[target_lan]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=args.max_length, truncation=True)
    data["labels"] = labels["input_ids"]

    return data


def get_datasets():
    df_wiki = pd.read_csv("data/wiki_similarity_filtered_nobrackets.csv")
    df_wiki = df_wiki[df_wiki["score"]>0.75]
    df_dict = pd.read_csv("data/schwaebisches_dictionary.csv")
    df = pd.concat([df_wiki, df_dict], ignore_index=True) #pd.read_csv("data/schwaebisches_dictionary.csv")#
    # df = df.dropna()
    train_df = df.sample(frac=0.8)
    test_df = df.drop(train_df.index)
    
    train_ds = datasets.Dataset.from_pandas(train_df)
    test_ds = datasets.Dataset.from_pandas(test_df)

    train_ds = train_ds.map(preprocess_function, batched=True)
    test_ds = test_ds.map(preprocess_function, batched=True)

    return train_ds, test_ds

def get_model():
    if args.model_checkpoint in ["bert-base-german-cased"]: #"facebook/bart-large-cnn", 
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_checkpoint)
        model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-german-cased", "bert-base-german-cased")
        
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_checkpoint)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    return tokenizer, model

metric = evaluate.load("sacrebleu")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    # perplexity
    return result


def main():
    global tokenizer
    tokenizer, model = get_model()

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

    train_args = transformers.Seq2SeqTrainingArguments(
        "BART-de-schwabish",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        
        bf16=True,
        bf16_full_eval=True,
        evaluation_strategy="epoch",
        save_total_limit=3,
        # dataloader_num_workers=4,
        predict_with_generate=True,
    )

    train_ds, test_ds = get_datasets()

    trainer = transformers.Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    results = trainer.train()

    print("======================================================================")
    print("======================================================================")
    print("========================Results=======================================")
    print(results)


if __name__ == "__main__":
    args = get_args()
    main()
