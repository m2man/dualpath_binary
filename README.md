# Dualpath Binary
This is the tensorflow2 reimplementation of dualpath embedding model for image-text matching for Flickr30k dataset. But differ from the original [paper](https://arxiv.org/abs/1711.05535), now it is changed to binary classification if image and text match each other or not.

You will need the GoogleNews word2vec model ([Dowload](https://drive.google.com/a/mail.dcu.ie/uc?id=1lX6iq6_TfngYZKUhJoppEWhqzkS30Dhc&export=download)) and Flickr30k Dataset ([Download](https://drive.google.com/a/mail.dcu.ie/uc?id=12KSjtMLt5gL23aNlqZLigf6jYkjo3Svt&export=download)) to run.

Change parameter in ***config.py*** file, then run ***run_train.py*** file.

### Evalutaion
#### Loss and Accuracy
run ***run_eval.py*** for evaluating loss and binary accuracy on validate set.

Syntax: ```run_eval.py -se [start_epoch] -ee[end_epoch]```

#### Recall@k
run ***run_recall.py*** for evaluating recall@k on validate set

Syntax: ```run_recall.py -se [start_epoch] -ee[end_epoch] -k [top_k_value]```

### Requirements
- tensorflow 2.0.0 (or tensorflow-gpu 2.0.0)
- tqdm
- nltk
- sklearn
- scipy
- cv2
- PIL
- gensim

### Download files from Drive with gdown
To download file from Drive, you can use gdown (install by ```pip install gdown```)

Syntax: ```gdown https://drive.google.com/uc?id=[FILEID]```

#### FILEID
- word2vec id: 1lX6iq6_TfngYZKUhJoppEWhqzkS30Dhc 
- flickr30k id: 12KSjtMLt5gL23aNlqZLigf6jYkjo3Svt
