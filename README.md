# IndoPOSTagger-SSL
Implementation of the conference paper "Minimal Data for Maximum Impact: An Indonesian Part-of-Speech Tagging Case Study"

# Instruction
## Dataset

### Annotated Data
You can find a sample of Annotated Data in **sample/UD_Indonesian-GSD_520_tokens.train** 


## Raw Data
You can find a sample of Raw Data in **sample/input_419055_records_from_cc100_Indonesian.txt**

*Note: Must put <?> in the end of the txt file*

## Brown Clustering
Please refer to this Github repository https://github.com/percyliang/brown-cluster

## Semi-Supervised
Install the necessarry libraries in order to make sure everything run successfully
```
!pip install -r requirements.txt
```

Download the necessary punkt tokenizer data
```
import nltk
nltk.download('punkt')
```

Must run the following cell in order to import the library
```
from SemiScikit import Semi_Scikit
```
*Note:you need to download the **SeniScikit.py** in order to run the code above.*