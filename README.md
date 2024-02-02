# Quantifying Stereotypes in Language

> Code for paper ***[Quantifying Stereotypes in Language](https://arxiv.org/abs/2401.15535)*** (accepted by **[EACL 2024](https://2024.eacl.org/)**)

## Dataset Details

| train | test | val |
|-------|------|-----|
| 2,260 | 358  | 358 |

## Training Models

You can run the following script to train the models:

``` 
python train.py  \
--mode train \
--pre_trained_model_name_or_path [bert-base-uncased, distilbert-base-uncased, roberta-base] \
--train_path data/train.csv \
--val_path data/val.csv \
--test_path data/test.csv \
--lr 0.00001 \
--max_len 50 \
--max_epochs 30 \
--batch_size 128 \
--model_saving_path models
```

The weights are saved to the [models](models) folder.


# Predict Stereotype Scores
You can run the following script to predict stereotype scores:
```
python train.py \
--mode=predict \
--pre_trained_model_name_or_path=models/bert-base-uncased \
--predict_data_path=data/predict/cp_sentence.csv \
--test_saving_path=results/bert_cp_sentence_results.csv
```
The prediction results are saved to the [results](results) folder.

## The Trained Weights of the Models

You can download the pre-trained weights for use directly from [huggingface](https://huggingface.co/):

For the BERT model
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lauyon/quantifying-stereotype-bert")
model = AutoModelForSequenceClassification.from_pretrained("lauyon/quantifying-stereotype-bert")
```

For the DistilBERT model
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lauyon/quantifying-stereotype-distilbert")
model = AutoModelForSequenceClassification.from_pretrained("lauyon/quantifying-stereotype-distilbert")
```

For the RoBERTa model
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lauyon/quantifying-stereotype-roberta")
model = AutoModelForSequenceClassification.from_pretrained("lauyon/quantifying-stereotype-roberta")
```

If this work has helped you in any way, please cite it by the following:
```bibtex
@article{liu2024quantifying,
  title={Quantifying Stereotypes in Language},
  author={Liu, Yang},
  journal={arXiv preprint arXiv:2401.15535},
  year={2024}
}
```
