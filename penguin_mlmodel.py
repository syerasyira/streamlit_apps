import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

penguin_df = pd.read_csv("penguins.csv")
penguin_df.dropna(inplace=True)

# this is the target output that we want to predict
output =  penguin_df['species']
# this is the variables/parameters that will be used to predicts
features = penguin_df[['island', 'bill_length_mm',
'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

# original after cleaning NaN values
# print(features.tail())
# after encoding the categorical variable to numbers
features = pd.get_dummies(features)
# print(" ")
# print(features.tail())

output, uniques = pd.factorize(output)

# train test split
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.2, random_state=123)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

# nak predict output from features
y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred, y_test)
print("Our accuracy for this model is {}".format(score))
print("OK!")

# save the penguin RF classifier
rfc_pickle = open("random_forest_penguin.pickle", 'wb')
pickle.dump(rfc, rfc_pickle)
rfc_pickle.close()

output_pickle = open('output_penguin.pickle', 'wb') #write bytes
pickle.dump(uniques, output_pickle)
output_pickle.close()
