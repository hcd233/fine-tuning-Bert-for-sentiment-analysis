import os
import datasets
import evaluate
import numpy as np
import pandas as pd
from fire import Fire
from transformers import AutoModelForSequenceClassification, BertTokenizerFast
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator


def main(output_checkpoints="./checkpoints",
         model_path="bert-base-chinese",
         dataset_path="./dataset/weibo_senti_100k.csv",
         learning_rate=2e-5,
         batch_size=80,
         epoch=5,
         weight_decay=0.02,
         warmup_ratio=0.2,
         use_gpu="0",
         ):
    """

    :param output_checkpoints: "the dir where you want to save checkpoints ."
    :param model_path: "name of huggingface repo or a local model dir"
    :param dataset_path:"the dataset dir"
    :param learning_rate:
    :param batch_size:
    :param epoch:
    :param weight_decay:
    :param warmup_ratio:
    :param use_gpu: "the indexes of gpus  you want to use. such as "0,1,2","0","1,3,6" etc."
    :return: None
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu

    def transform_dataset(dataset):
        return tokenizer(dataset['review'], padding="max_length", truncation=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # load model

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    # read data

    df = pd.read_csv(dataset_path).sample(frac=1.0)

    np.random.shuffle(df)

    train_ds = datasets.Dataset.from_pandas(df=df[:80000])
    dev_ds = datasets.Dataset.from_pandas(df=df[80000:100000])
    test_ds = datasets.Dataset.from_pandas(df=df[100000:])
    """example:dataset
    print(train_ds[:3])
    {
        'label': [1, 1, 1], 
        'review':  ['\ufeff更博了，爆照了，帅的呀，就是越来越爱你！生快傻缺[爱你][爱你][爱你]',
                    '@张晓鹏jonathan 土耳其的事要认真对待[哈哈]，否则直接开除。@丁丁看世界 很是细心，酒店都全部OK啦。',
                    '姑娘都羡慕你呢…还有招财猫高兴……//@爱在蔓延-JC:[哈哈]小学徒一枚，等着明天见您呢//@李欣芸SharonLee:大佬范儿[书呆子]',]
    }

    """
    # preprocess data
    train_ds.rename_column('label', 'labels')
    dev_ds.rename_column('label', 'labels')
    dataset = datasets.DatasetDict({
        "train": train_ds,
        "dev": dev_ds,
        "test": test_ds
    })
    dataset = dataset.map(transform_dataset, batched=True)
    """Map Output
    print(dataset)
    DatasetDict({
        train: Dataset({
            features: ['label', 'review', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 80000
        })
        dev: Dataset({
            features: ['label', 'review', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 20000
        })
        test: Dataset({
            features: ['label', 'review', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 19988
        })
    })
    """

    """first train example
    print(dataset["train"][0])
    {
        'label': 1, 
        'review': '\ufeff更博了，爆照了，帅的呀，就是越来越爱你！生快傻缺[爱你][爱你][爱你]', 
        'input_ids': [101, 3291, 1300, 749, 8024, 4255, 4212, 749, 8024, 2358, 4638, 1435, 8024, 2218, 3221, 6632, 3341, 
                      6632, 4263, 872, 8013, 4495, 2571, 1004, 5375, 138, 4263, 872, 140, 138, 4263, 872, 140, 138, 4263, 
                      872, 140, 102], 
        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0], 
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1]
    }
    """
    dataset = dataset.with_format('torch')

    """tensor example
    print(dataset['train'][0])
    {
        'label': tensor(1), 
        'review': '\ufeff更博了，爆照了，帅的呀，就是越来越爱你！生快傻缺[爱你][爱你][爱你]', 
        'input_ids': tensor([101, 3291, 1300,  749, 8024, 4255, 4212,  749, 8024, 2358, 4638, 1435,
                             8024, 2218, 3221, 6632, 3341, 6632, 4263,  872, 8013, 4495, 2571, 1004,
                             5375,  138, 4263,  872,  140,  138, 4263,  872,  140,  138, 4263,  872,
                             140,  102]), 
        'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 
        'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}

    """

    # create a smaller subset of the full dataset to fine-tune on to reduce the time it takes

    # small_train_dataset = pt_ds["train"].shuffle(seed=42).select(range(1000))
    # small_eval_dataset = pt_ds["test"].shuffle(seed=42).select(range(1000))

    # prepare fine-tuning

    metric = evaluate.load("accuracy")

    if not os.path.exists(output_checkpoints):
        os.mkdir(output_checkpoints)

    # accelerate
    accelerator = Accelerator()

    model = accelerator.prepare_model(model)
    dataset = accelerator.prepare_data_loader(dataset)

    training_args = TrainingArguments(
        warmup_ratio=warmup_ratio,
        output_dir=output_checkpoints,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    task_evaluator = evaluate.evaluator("text-classification")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # train
    trainer.train()


if __name__ == '__main__':
    Fire(main)
