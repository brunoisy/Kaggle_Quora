import ktrain
import pandas as pd
from sklearn.model_selection import train_test_split

MODEL_NAME = 'distilbert-base-uncased'
TRAINING_DATA_FILE = "data/train.csv"

max_qst_length = 100  # max number of words in a question to use

###
# data preparation
train = pd.read_csv(TRAINING_DATA_FILE)[:1000]
ids = train['qid'].values
X = train['question_text'].values
y = train['target'].values

print("accuracy baseline : ", 1 - round(sum(y) / len(y), 3), "% of questions are sincere")

ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(ids, X, y, test_size=0.2, random_state=2020)
del X, y  # save RAM

transformer = ktrain.text.Transformer(MODEL_NAME, maxlen=max_qst_length, class_names=[0, 1])
data_train = transformer.preprocess_train(X_train, y_train)
data_test = transformer.preprocess_test(X_test, y_test)

model = transformer.get_classifier()
learner = ktrain.get_learner(model, train_data=data_train, val_data=data_test, batch_size=6)
learner.fit_onecycle(5e-5, 4)
