# 基于bert微调的中文情感分析
## 简介
*使用huggingface的库，对预先训练的BERT模型对中文文本进行情感分析。
## Change Logs
* [2023/7/27] The script **train.py** supports **parse arguments**. Now you can run the train.py in CLI with the arguments you set
## Requirements
~~~shell
pip install -r requirements.txt
~~~
## Dataset
使用WeiboSenti100k数据集，该数据集包含在中/数据集文件夹。该数据集包含10万条中国微博帖子，每条帖子都被标记为正面或负面。
## Usage
* ### train
    **run the CLI** 
    ~~~shell
        python train.py <Args>
    ~~~
    **or**
    ```shell
        accelerate launch train.py <Args>
    ```
    **Args**
    * -o, --output_checkpoints=OUTPUT_CHECKPOINTS
        Default: './checkpoints'
        "the dir where you want to save checkpoints ."
    * -m, --model_path=MODEL_PATH
        Default: 'bert-base-chinese'
        "name of huggingface repo or a local model dir"
    * -d, --dataset_path=DATASET_PATH
        Default: './dataset/weibo_se...
        "the dataset dir"
    * -l, --learning_rate=LEARNING_RATE
        Default: 2e-05
    * -b, --batch_size=BATCH_SIZE
        Default: 80
    * -e, --epoch=EPOCH
        Default: 5
    * --weight_decay=WEIGHT_DECAY
        Default: 0.02
    * --warmup_ratio=WARMUP_RATIO
        Default: 0.2
    * -u, --use_gpu=USE_GPU
        Default: '0'
        "the indexes of gpus  you want to use. such as "0,1,2","0","1,3,6" etc."  

* ### inference
  ~~~shell
      python inference.py -s "There input your sentence." # Inference for single sentence.
  ~~~
  ~~~shell
      python inference.py -i True # Continuous inference.
  ~~~

## Related Docs
* *<a href="https://huggingface.co/docs">HuggingFace Docs</a>*
* *<a href="https://pytorch.org/docs/stable/index.html">Pytorch Docs</a>*
