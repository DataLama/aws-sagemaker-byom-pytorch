import os
import json
import torch
import numpy as np

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, BatchEncoding

def model_fn(model_dir):
    config = AutoConfig.from_pretrained(os.path.join(model_dir, 'config.json'))
    model = AutoModelForTokenClassification.from_pretrained(os.path.join(model_dir, 'pytorch_model.bin'), config=config)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, device


def input_fn(request_body, request_content_type):
    if request_content_type=="application/json":
        data = json.loads(request_body)['text']
        return data
    else:
        raise TypeError('This API supports following types - "application/json".')

def predict_fn(input_data, models):
    model, tokenizer, device = models
    inputs = BatchEncoding({k:v.to(device) for k, v in tokenizer(input_data, return_tensors='pt').items()})
    with torch.no_grad():
        return model(**inputs).logits.cpu(), model.config.id2label
    
    
def output_fn(prediction, content_type):
    pred, ID2LABEL = prediction
    assert content_type == 'application/json'
    results = []
    for sequence in pred.argmax(axis=-1).numpy().tolist():
        results.append([ID2LABEL[tag] for tag in sequence])
    
    return json.dumps({
        "predicts": results
    })