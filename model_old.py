import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.under_sampling import RandomUnderSampler


train_df = pd.read_csv("data/train.csv")

# split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=2018)
train_X = train_df["question_text"].values.astype('U')
val_X = val_df["question_text"].values.astype('U')
train_y = train_df['target'].values
val_y = val_df['target'].values

# Word Count
print("model : logistic regression on word count")
ctv = CountVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1, 2), stop_words='english', min_df=2)
ctv.fit(train_X)
train_X_ctv = ctv.transform(train_X)
val_X_ctv = ctv.transform(val_X)

# random undersampling because of 15:1 imbalance
n_min = sum(train_y == 1)
rus = RandomUnderSampler(sampling_strategy={0: 5 * n_min, 1: n_min})
(train_X_res, train_y_res) = rus.fit_sample(train_X_ctv, train_y)

model_logistic_word_count = LogisticRegression(verbose=1)
model_logistic_word_count.fit(train_X_res, train_y_res)
predictions = model_logistic_word_count.predict_proba(val_X_ctv)
predictions = predictions[:, 1] > 0.5

print("")
print("recall score : %s" % recall_score(val_y, predictions))
print("precision score: %s" % precision_score(val_y, predictions))
print("F1 score: %s" % f1_score(val_y, predictions))
print("Confusion Matrix : ")
print(confusion_matrix(val_y, predictions))
