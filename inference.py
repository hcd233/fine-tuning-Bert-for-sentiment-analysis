from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import argparse

# hyperparameter

OUTPUT_PATH = "./checkpoints"
CHECKPOINT_PATH = "/checkpoint"
MODEL_PATH = "./bert-base-chinese"

# load model

model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_PATH + CHECKPOINT_PATH)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_PATH + CHECKPOINT_PATH)


# receive input function
def Infer(input_review):
    encoded_input = tokenizer.encode(input_review, return_tensors='pt')
    result = model(encoded_input)['logits']
    label = torch.argmax(result, dim=-1)
    return int(label)


def InferMode():
    print("InferMode.Press Q to exit.")
    while True:
        input_review = input("Input Review: ")
        if input_review == 'Q':
            print("Exit.")
            break
        label = Infer(input_review)
        print("Positive." if label else "Negative.")


# use argparse
parser = argparse.ArgumentParser(description='Quick Inference')
parser.add_argument("-s", "--sentence", metavar=None, type=str,
                    default="今天天气真好，适合睡觉~")
parser.add_argument("-i", "--infer_mode", metavar=None, type=bool, default=False)
args = parser.parse_args()

if __name__ == '__main__':
    if args.infer_mode:
        InferMode()
    else:
        infer_label = Infer(args.sentence)
        print("Your Input: " + args.sentence)
        print("Infer Result: " + ("Positive." if infer_label else "Negative."))
