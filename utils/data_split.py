import argparse
from sklearn.model_selection import train_test_split
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_dir')
args = parser.parse_args()

columns = ["age", 
        "workclass", 
        "fnlwgt", 
        "education", 
        "education-num", 
        "marital-status", 
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "target"]

df = pd.read_csv(args.input_file, names=columns)

X = df.drop("target", axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
train_df = pd.DataFrame(X_train, columns=columns[:-1])
train_df["target"] = y_train
test_df = pd.DataFrame(X_test, columns=columns[:-1])
test_df["target"] = y_test

train_df.to_csv(args.output_dir + "train.data", index=False, header=False)
test_df.to_csv(args.output_dir + "test.data", index=False, header=False)