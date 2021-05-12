
"""
Put all the Stanford Sentiment Treebank phrase data into test, training, and dev CSVs.

Socher, R., Perelygin, A., Wu, J. Y., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). Recursive Deep Models
for Semantic Compositionality Over a Sentiment Treebank. Presented at the Conference on Empirical Methods in Natural
Language Processing EMNLP.

https://nlp.stanford.edu/sentiment/

CREDITS: https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4
"""
import os
import pandas as pd
import argparse
import  pytreebank #library for preprocessing this dataset

parser = argparse.ArgumentParser(description="Preprocessing Sentiment Dataset")
parser.add_argument("--out", type=str, default="", help="Output Sentiment Format")
ARGS = parser.parse_args()

# Make output paths and directories
os.makedirs(ARGS.out, exist_ok=True)
out_path = os.path.join(ARGS.out, 'sst_{}.txt')

# Load dataset from library
dataset = pytreebank.load_sst(os.path.join(ARGS.out, 'raw_data'))

# Create train, dev and test in txt files
for category in ['train', 'test', 'dev']:
    with open(out_path.format(category), 'w') as outfile:
        for item in dataset[category]:
            outfile.write("__label__{}\t{}\n".format(
                item.to_labeled_lines()[0][0] + 1,
                item.to_labeled_lines()[0][1]
            ))

# Load txt files and turn them into csvs
train = pd.read_csv(os.path.join(ARGS.out, 'sst_train.txt'), sep='\t', header=None, names=['truth', 'text'])
train.to_csv(os.path.join(ARGS.out, 'train.csv'))

val = pd.read_csv(os.path.join(ARGS.out, 'sst_dev.txt'), sep='\t', header=None, names=['truth', 'text'])
val.to_csv(os.path.join(ARGS.out, 'val.csv'))

test = pd.read_csv(os.path.join(ARGS.out, 'sst_test.txt'), sep='\t', header=None, names=['truth', 'text'])
test.to_csv(os.path.join(ARGS.out, 'test.csv'))
