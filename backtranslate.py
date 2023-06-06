
import pickle
import argparse
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import MarianMTModel, MarianTokenizer

from dataloader import load_zeroshot_data_allmentions

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Backtranslate to English through a language.')
parser.add_argument('--language', help='desired backtranslation language')
parser.add_argument('--output_path', default='backtranslations/', help='output folder')

args = parser.parse_args()

TARGET_LANGUAGE = args.language
OUTPUT_PATH = args.output_path

if TARGET_LANGUAGE == 'german':
    source_model_name = 'Helsinki-NLP/opus-mt-en-de'
    target_model_name = 'Helsinki-NLP/opus-mt-de-en'
    BATCH_SIZE = 64
elif TARGET_LANGUAGE == 'french':
    source_model_name = 'Helsinki-NLP/opus-mt-en-fr'
    target_model_name = 'Helsinki-NLP/opus-mt-fr-en'
    BATCH_SIZE = 64
elif TARGET_LANGUAGE == 'spanish':
    source_model_name = 'Helsinki-NLP/opus-mt-en-es'
    target_model_name = 'Helsinki-NLP/opus-mt-es-en'
    BATCH_SIZE = 64
elif TARGET_LANGUAGE == 'chinese':
    source_model_name = 'Helsinki-NLP/opus-mt-en-zh'
    target_model_name = 'Helsinki-NLP/opus-mt-zh-en'
    BATCH_SIZE = 32

source_tokenizer = MarianTokenizer.from_pretrained(source_model_name)
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
source_model = MarianMTModel.from_pretrained(source_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

tmp = source_model.to(device)
tmp = target_model.to(device)

dataset = load_zeroshot_data_allmentions()

dataloader_params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 0}

generator = DataLoader(dataset, **dataloader_params)


translations = {}
backtranslations = {}

#for i, d in enumerate(generator):
for d in tqdm(generator):
    text, labels, raw_text = d

    subtype_repres, subtype_oh, roles_oh, seen, EID, subtypes, all_roles_oh = labels

    translated = source_model.generate(**source_tokenizer(raw_text, return_tensors="pt", padding=True).to(device))
    translated = [source_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    backtranslated = target_model.generate(**target_tokenizer(translated, return_tensors="pt", padding=True).to(device))
    backtranslated = [target_tokenizer.decode(bt, skip_special_tokens=True) for bt in backtranslated]

    for e, t, bt in zip(EID, translated, backtranslated):
        translations[e] = t
        backtranslations[e] = bt

    #if (i+1)%10 == 0: print(i+1, "batches translated.")

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

with open(os.path.join(OUTPUT_PATH, TARGET_LANGUAGE + '_translations.allmentions.pkl'), 'wb') as f:
    pickle.dump(translations, f)

with open(os.path.join(OUTPUT_PATH, TARGET_LANGUAGE + '_backtranslations.allmentions.pkl'), 'wb') as f:
    pickle.dump(backtranslations, f)

