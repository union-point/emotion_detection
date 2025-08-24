# Multi-Modal Emotion Detection

This project focuses on detecting human emotions by analyzing multiple data sources. It goes beyond traditional methods that only look at one type of data, such as facial expressions or voice tone, by combining information from several modalities.

## Repository Structure
```
.
├── config.py
├── data_loader.py
├── inference.py
├── train.py
├── models
│   ├── attention_pooling.py
│   ├── early_fusion.py
│   ├── fine_grained_fusion.py
│   ├── __init__.py
│   ├── memocmt.py
├── data
├── saved_models
├── README.md
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
   from [here](https://drive.google.com/) and
   place the content inside a `saved_model` folder



## Inference

In order to perform inference on  sample text+audio located in [`data`](data/) folder you can run this command:

 ```bash
python inference.py --checkpoint=saved_model --n_samples=3 
```


## Results
### RoBERTa + Wav2Vec2 (Batch size = 4, LR = 2e-4)

| Method                         | Training Setup                   | Train Loss | Val Loss | Train Acc | Val Acc | Val F1  |
|--------------------------------|----------------------------------|------------|----------|-----------|---------|---------|
| **memoCMT**                    | 2 epochs (all), 1 epoch (head)  | 0.9392     | 1.1096   | 0.6842    | 0.6518  | 0.6251  |
| **Attention Pooling**          | 2 epochs                        | 0.8340     | 1.1157   | –         | 0.6300  | 0.6183  |
| **Fine-Grained Interaction**   | 2 epochs                        | 0.8617     | 1.1186   | –         | 0.6264  | 0.6016  |
| **Early Fusion**               | 2 epochs                        | 0.8942     | 1.1382   | 0.7097    | 0.6218  | 0.6042  |


