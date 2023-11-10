import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report

class Semi_Scikit:

    UNKNOWN_SYMBOL = "<?>" 
    model = None
    word_bitstring = None
    full_bitstring_feature = None
    one_hot_encoder = None
    label_encoder = None

    def __init__(self,folder_path=None):
        if folder_path:
            model_data_path = f"{folder_path}/model.pkl"
            word_bitstring_data_path = f"{folder_path}/word_bitstring.pkl"
            full_bitstring_data_path = f"{folder_path}/full_bitstring.pkl"
            one_hot_encoder_data_path = f"{folder_path}/one_hot_encoder.pkl"
            label_encoder_data_path = f"{folder_path}/label_encoder.pkl"

            self.model = joblib.load(model_data_path)
            self.word_bitstring = joblib.load(word_bitstring_data_path)
            self.full_bitstring_feature = joblib.load(full_bitstring_data_path)
            self.one_hot_encoder = joblib.load(one_hot_encoder_data_path)
            self.label_encoder = joblib.load(label_encoder_data_path)


    def bitstring_features(self,sentence,position):
        features = {}
        word = sentence[position]  # current word
        if word in self.word_bitstring:
            word_bitstring = self.word_bitstring[word]
        else:
            word_bitstring = self.word_bitstring[self.UNKNOWN_SYMBOL]
        for i in range(1, len(word_bitstring) + 1):
            features["bitstring(0)_prefix({0})".format(
                    i)] = word_bitstring[:i]
        features["bitstring(0)_all"] = word_bitstring

        if position > 0:
            word = sentence[position - 1]  # word to the left
            if word in self.word_bitstring:
                word_bitstring = self.word_bitstring[word]
            else:
                word_bitstring = self.word_bitstring[self.UNKNOWN_SYMBOL]
            for i in range(1, len(word_bitstring) + 1):
                features["bitstring(-1)_prefix({0})".format(
                        i)] = word_bitstring[:i]
            features["bitstring(-1)_all"] = word_bitstring

        if position < len(sentence) - 1:
            word = sentence[position + 1]  # word to the right
            if word in self.word_bitstring:
                word_bitstring = self.word_bitstring[word]
            else:
                word_bitstring = self.word_bitstring[self.UNKNOWN_SYMBOL]
            for i in range(1, len(word_bitstring) + 1):
                features["bitstring(+1)_prefix({0})".format(
                        i)] = word_bitstring[:i]
            features["bitstring(+1)_all"] = word_bitstring    
        
        return features    

    def get_full_bitstring_feature(self,tagged_sentences):
        full_bitstring__feature = {}
        for sentence, tags in tagged_sentences:
            for index in range(len(sentence)):
                feature = self.bitstring_features(sentence,index)
                temp_dict = {f"{item}={feature[item]}": 1 for item in feature}
                full_bitstring__feature.update(temp_dict)
        
        self.full_bitstring_feature = full_bitstring__feature

    def datasets(self,train_data_path,test_data_path,bitstring_data_path):
        train_data_list = []
        word_list = []
        tag_list = []
        word_bitstring = {}

        if bitstring_data_path:
            with open(bitstring_data_path, "r", encoding="utf-8", errors="replace") as infile:
                for line in infile:
                    toks = line.split()
                    if len(toks) == 0:
                        continue

                    word_bitstring[toks[1]] = toks[0]

                assert self.UNKNOWN_SYMBOL in word_bitstring

                if "(" in word_bitstring:
                    word_bitstring["-LCB-"] = word_bitstring["("]
                    word_bitstring["-LRB-"] = word_bitstring["("]
                    word_bitstring["*LCB*"] = word_bitstring["("]
                    word_bitstring["*LRB*"] = word_bitstring["("]
                if ")" in word_bitstring:
                    word_bitstring["-RCB-"] = word_bitstring[")"]
                    word_bitstring["-RRB-"] = word_bitstring[")"]
                    word_bitstring["*RCB*"] = word_bitstring[")"]
                    word_bitstring["*RRB*"] = word_bitstring[")"]
                if "\"" in word_bitstring:
                    word_bitstring["``"] = word_bitstring["\""]
                    word_bitstring["''"] = word_bitstring["\""]
                    word_bitstring["`"] = word_bitstring["\""]
                    word_bitstring["'"] = word_bitstring["\""]
                    
                self.word_bitstring = word_bitstring

        if train_data_path:
            with open(train_data_path,"r",encoding="utf-8", errors="replace") as infile:
                for line in infile:
                    item = line.split()
                    if(len(item) > 0):
                        word_list.append(item[0])
                        tag_list.append(item[1])
                    else:
                        train_data_list.append((word_list,tag_list))
                        word_list, tag_list = [], []

        test_data_list = []            
        word_list = []
        tag_list = []

        if test_data_path:
            with open(test_data_path,"r",encoding="utf-8",errors="replace") as infile:
                for line in infile:
                    item = line.split()
                    if(len(item) > 0):
                        word_list.append(item[0])
                        tag_list.append(item[1])
                    else:
                        test_data_list.append((word_list,tag_list))
                        word_list, tag_list = [], []
                
        return train_data_list, test_data_list
    
    def extract_features(self, sentence, index):

        def isfloat(num):
            try:
                float(num)
                return True
            except ValueError:
                return False

        def is_all_nonalphanumeric(word):
            """Is the word all nonalphanumeric?"""
            for char in word:
                if char.isalnum():
                    return False
            return True

        def get_prefix(word, length):
            """Gets a padded prefix of the word up to the given length."""
            prefix = ""
            for i in range(length):
                if i < len(word):
                    prefix += word[i]
                else:
                    prefix += "*"
            return prefix

        def get_suffix(word, length):
            """Gets a padded suffix of the word up to the given length."""
            suffix = ""
            for i in range(length):
                if i < len(word):
                    suffix = word[-i-1] + suffix
                else:
                    suffix = "*" + suffix
            return suffix

        base = [
        str(sentence[index]),
        str(sentence[index][0].upper() == sentence[index][0]),
        str(is_all_nonalphanumeric(sentence[index])),
        get_prefix(sentence[index],1),
        get_prefix(sentence[index],2),
        get_prefix(sentence[index],3),
        get_prefix(sentence[index],4),
        get_suffix(sentence[index],1),
        get_suffix(sentence[index],2),
        get_suffix(sentence[index],3),
        get_suffix(sentence[index],4),
        '' if index == 0 else sentence[index-1],
        '' if (index-1) <= 0 else sentence[index-2],
        '' if (index+1) == len(sentence) else sentence[index+1],
        '' if (index+2) >= len(sentence) else sentence[index+2],
        str(isfloat(sentence[index]))
        ]
        
        temp_feature = self.bitstring_features(sentence,index)
        bitstring_feature = {f"{item}={temp_feature[item]}": temp_feature[item] for item in temp_feature}
        for key in self.full_bitstring_feature:
            if key in bitstring_feature:
                base.append(bitstring_feature[key])
            else:
                base.append('')
            
        return base
    
    def transform_to_dataset(self, data_list,train,test):
        X, y = [], []
        if train or test:
            for sentence, tags in data_list:
                for index in range(len(sentence)):
                    X.append(self.extract_features(sentence, index)),
                    y.append(tags[index])
        else:
            for sentence in data_list:
                for index in range(len(sentence)):
                    X.append(self.extract_features(sentence, index))

        return X, y
    
    def oneHotGenerator(self,data_list,train=True,test=False):

        x,y = self.transform_to_dataset(data_list,train,test)

        if train:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(x)
            self.one_hot_encoder = enc
        else:
            enc = self.one_hot_encoder

        one_hot_x = []
        batch_size = 1000
        num_samples = len(x)
        num_batches = int(np.ceil(num_samples / batch_size))

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i+1)*batch_size, num_samples)
            batch_data = x[start_index:end_index]
            encoded_batch = enc.transform(batch_data)
            one_hot_x.append(encoded_batch.toarray())

        if len(one_hot_x) > 0:
            one_hot_x = np.concatenate(one_hot_x,axis=0)
        else:
            one_hot_x = []

        
        if train:
            le = LabelEncoder()
            le.fit(y)
            self.label_encoder = le
            le_y = le.transform(y)
        elif test:
            le_y = y
        else:
            le_y = None

        

        return one_hot_x, le_y


    
    def fit(self,train_data_path,bitstring_data_path,classifier):
        train_data_list, _ = self.datasets(train_data_path=train_data_path,test_data_path=None,bitstring_data_path=bitstring_data_path)

        self.get_full_bitstring_feature(train_data_list)

        one_hot_x, le_y = self.oneHotGenerator(train_data_list,True,False)
        classifier.fit(one_hot_x,le_y)

        self.model = classifier

    def predict(self,data):
        x = []

        if type(data) == type([]):
            list_sentences = [word_tokenize(sentence) for sentence in data]
            
        elif type(data) == type(""):
            list_sentences = [word_tokenize(data)]
        else:
            print("you should provide a string or list of string")
            return
        
        x, _ = self.oneHotGenerator(list_sentences,False,False)
        le_y_pred = self.model.predict(x)
        y_pred = self.label_encoder.inverse_transform(le_y_pred)

        y_index = 0
        predict_result = []
        for sentence in list_sentences:
            sentence_result = []
            for word in sentence:
                sentence_result.append((word,y_pred[y_index]))
                y_index = y_index + 1
            predict_result.append(sentence_result)

        return predict_result
    

    def evaluation(self, test_data_path):
        _, test_data_list = self.datasets(train_data_path=None,test_data_path=test_data_path,bitstring_data_path=None)

        one_hot_test_x, test_y = self.oneHotGenerator(test_data_list,False,True)

        le_y_pred = self.model.predict(one_hot_test_x)
        y_pred = self.label_encoder.inverse_transform(le_y_pred)

        accuracy = accuracy_score(test_y,y_pred)
        precision = precision_score(test_y,y_pred,average='weighted',zero_division=1.0)
        recall = recall_score(test_y,y_pred,average='weighted', zero_division=1.0)
        f1 = f1_score(test_y,y_pred,average='weighted', zero_division=1.0)

        report = classification_report(test_y,y_pred)

        return accuracy,precision,recall,f1,report
    

    def export(self,folder_path):
        model_data_path = f"{folder_path}/model.pkl"
        word_bitstring_data_path = f"{folder_path}/word_bitstring.pkl"
        full_bitstring_data_path = f"{folder_path}/full_bitstring.pkl"
        one_hot_encoder_data_path = f"{folder_path}/one_hot_encoder.pkl"
        label_encoder_data_path = f"{folder_path}/label_encoder.pkl"

        joblib.dump(self.model,model_data_path)
        joblib.dump(self.word_bitstring,word_bitstring_data_path)
        joblib.dump(self.full_bitstring_feature,full_bitstring_data_path)
        joblib.dump(self.one_hot_encoder,one_hot_encoder_data_path)
        joblib.dump(self.label_encoder,label_encoder_data_path)






