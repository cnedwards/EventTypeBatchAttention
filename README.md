# Semi-supervised New Event Type Induction and Description via Contrastive Loss-Enforced Batch Attention


This repository contains code for the paper "[Semi-supervised New Event Type Induction and Description via Contrastive Loss-Enforced Batch Attention](https://aclanthology.org/2023.eacl-main.275/)" (EACL23).


The following setup commands are required. 
```
conda env create -f environment.yml
conda activate ZS_IE

python backtranslate.py --language german
python backtranslate.py --language spanish
python backtranslate.py --language french
python backtranslate.py --language chinese

python generate_SBERT_embeddings.py
```

To run the primary method:

Note that these files have variables which need to be updated. 
```
python main.py
get_stopping_point.py
python store_embeddings.py
python clustering_test.py
```

To run name prediction:
```
python main_name.py
get_stopping_point.py
python test_finetuned_name.py
```

To run FrameNet linking:
```
python main_framenet.py
get_stopping_point.py
python test_finetuned_FN.py
```


### External Resources
Download FrameNet data to './framenet/', e.g. 'fndata-1.7.zip', and unzip it. Run `python create_heirarchy.py`. 

The code assumes that ACE05 data files are located at ``../../../ACE05EN/source``. Please change this to your local ACE05 path. 

### Citation
If you found our work useful, please cite:
```bibtex
@inproceedings{edwards-ji-2023-semi,
    title = "Semi-supervised New Event Type Induction and Description via Contrastive Loss-Enforced Batch Attention",
    author = "Edwards, Carl  and
      Ji, Heng",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.275",
    pages = "3805--3827",
}
```

