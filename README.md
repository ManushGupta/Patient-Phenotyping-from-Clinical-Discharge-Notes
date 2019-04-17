
# Patient Phenotyping

This repository contains the code and data for the paper 
"Comparing deep learning and concept extraction based methods for patient phenotyping".


## Data 

In data/annotations.csv, you can find our annotations as well 
as unique identifiers for patient visits in MIMIC-III, namely 
the hospital admission ID, subject ID, and chart time. 
Due to HIPAA requirements, we cannot provide the text of the patients' 
discharge summary in this repository. With the information 
named above and access to MIMIC-III, it is easy to extract the text 
from the identifiers. 
We are in the process of submitting the annotations as a direct
add-on to MIMIC-III to physionet which will make the linking-step 
obsolete. If you experience difficulties with the data, please 
contact us and we are happy to help! 

In the following sections, we assume that annotations.csv is extended
by an additional column named "text" that contains the discharge summary.

## Code

Here is how to run the code for baselines and deep learning components. 

### Preprocessing

To run all the code on the same training and test splits, we provide preprocessing code in
preprocessing.py. We assume that you ran word2vec on the extracted texts first and saved 
the resulting vectors in a file named "w2v.txt". If you need assistance or want to use
our vectors, please contact us (as the file size is too large for this repository).

run it with the following command (with python 2.7): 

```
python preprocess.py data/annotations.csv w2v.txt 
```

This will create one file data.h5 and one file data-nobatch.h5 in your main directory. 
Use the batched file for a speedup in the lua code, and the non-batched file for the baselines. 




### Baselines

The code for baselines can be found in basic_models.py. It is compatible with both 
python 2 and 3. To run it, simply enter

```
python basic_models.py --data data-nobatch.h5 --ngram 5
```

### CNN

The CNN code can be run on PyTorch

