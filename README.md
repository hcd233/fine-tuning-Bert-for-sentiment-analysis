# 基于bert微调的中文情感分析
## 简介
*使用huggingface的库，对预先训练的BERT模型对中文文本进行情感分析。*
## Requirements
python version : python == 3.9.13
~~~shell
pip install -r requirements.txt
~~~
## Dataset
使用WeiboSenti100k数据集，该数据集包含在中/数据集文件夹。该数据集包含10万条中国微博帖子，每条帖子都被标记为正面或负面。
## Usage
* ### train
    ~~~shell
        python train-hf.py
    ~~~
    The code will train the model using the first 80,000 samples of the dataset and evaluate it on the next 20,000 samples. Finally, it will test the model on the remaining samples. The trained model will be saved in the ./checkpoints folder.
* ### inference
    ~~~shell
        python inference.py -s "There input your sentence." # Inference for single sentence.
    ~~~
    ~~~shell
        python inference.py -i True # Continuous inference.
    ~~~
## Train Hyperparameter
* #### OUTPUT_PATH: path to save the trained model
* #### MODEL_PATH: path to the pre-trained BERT model
* #### DATASET_PATH: path to the dataset
* #### LEARNING_RATE: learning rate of the optimizer
* #### BATCH_SIZE: number of samples per batch
* #### EPOCH: number of epochs to train the model
* #### WEIGHT_DECAY: weight decay of the optimizer
## Related Docs
* *<a href="https://huggingface.co/docs">HuggingFace Docs</a>*
* *<a href="https://pytorch.org/docs/stable/index.html">Pytorch Docs</a>*