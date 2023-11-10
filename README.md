# IndoPOSTagger-SSL
Implementation of the conference paper "Minimal Data for Maximum Impact: An Indonesian Part-of-Speech Tagging Case Study"

# Instruction
## Dataset

### Annotated Data
You can find a sample of Annotated Data in **sample/UD_Indonesian-GSD_520_tokens.train** 


### Raw Data
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
*Note: you need to download the **SemiScikit.py** in order to run the code above.*

For subsequent steps, you can refer to **sample/Semi-Indonesian.ipynb**

## Baseline
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
from SupervisedScikit import Supervised_Scikit
```
*Note: you need to download the **SupervisedScikit.py** in order to run the code above.*

For subsequent steps, you can refer to **sample/Base-Indonesian.ipynb**

# Other Functions (SemiScikit & SupervisedScikit)
1. SemiScikit
   1. Predict
      1. ***predict()*** is a method for predicting POS tags, takes one parameter, and the input data type can be a string or a list of strings.
        ```
        semi_clf.predict("I love looking at you.")
        ```
        OR
        ```
        semi_clf.predict(["I love looking at you.","Are you talk to me?"])
        ```
   2. Export
      1.  **export()** is a method for exporting trained model, takes one parameter which is the folder path, and the input data type is string.
        ```
        semi_clf.export("./semi_svm")
        ```

2. SupervisedScikit
   1. Predict
      1. ***predict()*** is a method for predicting POS tags, takes one parameter, and the input data type can be a string or a list of strings.
        ```
        base_clf.predict("I love looking at you.")
        ```
        OR
        ```
        base_clf.predict(["I love looking at you.","Are you talk to me?"])
        ```
   2. Export
      1.  **export()** is a method for exporting trained model, takes one parameter which is the folder path, and the input data type is string.
        ```
        base_clf.export("./base_svm")
        ```