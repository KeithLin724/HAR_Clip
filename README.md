# Human Action Recognition using Clip

> Written By KYLiN

![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

## Download dataset

```sh
./download_har.sh
```

## Env setup

```sh
conda env create -f environment.yml
```

## Train Model

```sh
# pre-train clip script
python ./train_model_clip.py
```

## Test Model

```sh
# pre-train clip script
python ./test_model_clip.py
```

## Develop

Here is Develop Document : [English](./DEV.md), [Traditional Chinese](./Dev-zh.md)

## Test Result

About different model performance in [here](./Test_Result.md)

---

## Reference

Hugging face CLIP : [here](https://huggingface.co/docs/transformers/en/model_doc/clip)

HAR dataset : [here](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset)

pytorch-lightning : [here](https://lightning.ai/docs/pytorch/stable/)
