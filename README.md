# Multi-Modal Emotion Detection

This project focuses on detecting human emotions by analyzing multiple data sources. It goes beyond traditional methods that only look at one type of data, such as facial expressions or voice tone, by combining information from several modalities.

## Repository Structure
```
.
├── config.py
├── data_loader.py
├── inference.py
├── models
│   ├── attention_pooling.py
│   ├── early_fusion.py
│   ├── fine_grained_fusion.py
│   ├── __init__.py
│   ├── memocmt.py
├── README.md
├── saved_models
├── train.py
```



## Installing the dependencies


```bash
pip install datasets==3.6.0
```

## Datasets

- [MELD](https://affective-meld.github.io/)

## Training quick start

1. Modify [`config.py`](configs.py)
   if needed, in order to set training configurations.

3. If you want to train a model from scratch with the configs set in step 2, run the following terminal command from the project directory:

```bash
python train.py
```

4. If you want to continue model training from a saved model (only weights), you can download an example
   from [here](https://drive.google.com/drive/folders/1zMsgI35nwuV7eNUIpt4NDGALaYJShE1S?usp=sharing) and
   place the content inside a `saved_model` folder



## Inference

In order to perform inference on  sample text+audio located in [`data`](data/) folder you can run this command:

 ```bash
python inference.py --checkpoint=saved_model --n_samples=3 
```
## Notes
 - Use pooling heads or attention to fuse modalities instead of simple mean+concat
 - Try LoRA or adapter tuning to speed up experiments (not very good idea)
 - Consider freezing encoder layers and fine-tuning only classifier at first
 - Test approach described in this [article](https://www.nature.com/articles/s41598-025-89202-x.pdf)
 - Implement  gradient accumulation and train with larger batch sizes
 - [x] Read the following survey [paper](https://www.mdpi.com/1099-4300/25/10/1440) 


## Results
method: Fine-Grained Interaction Fusion, batch size 4, epochs 2 all, 1 epochs head only, RoBerta    

Train loss: 0.9392 Val loss: 1.1096 val_acc: 0.6318 train_acc:0.6842 val_f1: 0.6051

method: atention polling, without LoRA, batch size 4, epochs 2, bert-base-uncased + wav2vec2-base  
train_loss=0.8340 val_loss=1.1157 val_acc=0.6300 val_f1=0.6083

method: Fine-Grained Interaction Fusion, without LoRA, batch size 4, epochs 2, bert-base-uncased + wav2vec2-base  
train_loss=0.8617 val_loss=1.1186 val_acc=0.6264 val_f1=0.6016

method: Early Fusion, without LoRA, batch size 4, epochs 2, bert-base-uncased + wav2vec2-base  
Train loss: 0.8942 Val loss: 1.1382 val_acc: 0.6218 train_acc:0.7097 val_macro_f1: 0.4019

method: atention polling, batch size 4, epochs 2, RoBerta   
Train loss: 0.9419 Val loss: 1.1029 val_acc: 0.6273 train_acc:0.6889 val_f1: 0.6043

method: Fine-Grained Interaction Fusion, batch size 4, epochs 3, RoBerta  
train_loss=0.9782 val_loss=1.1240 val_acc=0.6218 val_f1=0.592

