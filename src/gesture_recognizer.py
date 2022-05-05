import pandas as pd
import numpy as np
import os
import random
import wandb

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


RANDOM_SEED = 1

label_map = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "forward": 4,
    "backward": 5,
    "clockwise": 6,
    "counterclockwise": 7,
}

sweep_config = {
  "name": "Hand Gesture - MLP Classifier",
  "method": "bayes",
  "metric": {
      "goal": "minimize",
      "name": "val_loss",
  },
  "parameters": {
    "solver": {
      "values": ["lbfgs", "sgd", "adam"],
    },
    "activation": {
      "values": ["identity", "logistic", "tanh", "relu"],
    },
    "max_iter": {
      "values": [10, 20, 50, 100, 200, 500, 1000],
    },
    "batch_size": {
      "values": [4, 8, 16, 32, 64],
    },
    "learning_rate": {
        "values": ['constant', 'invscaling', 'adaptive']
    }
  }
}

df = pd.read_csv(os.path.join(os.getcwd(),
                              "..",
                              "data",
                              "gestures",
                              "hand-annotations.csv"))

y = df['label'].apply(lambda x: label_map[x]).values
x = df[[f"{i}-{p}" for p in ['x', 'y', 'z'] for i in range(0, 21)]].values

x_tr, x_te, y_tr, y_te = train_test_split(x, y,
                                          test_size=0.2,
                                          random_state=RANDOM_SEED)

def train():
    with wandb.init(project="Gesture-Recognizer", entity="briancsavage") as run:
        config = wandb.config
        clf = MLPClassifier(learning_rate=config.learning_rate,
                              solver=config.solver,
                              activation=config.activation,
                              batch_size=config.batch_size,
                              max_iter=config.max_iter)
        clf.fit(x_tr, y_tr)
        run.log({"val_loss": clf.score(x_te, y_te)})

        y_hat = clf.predict(x_te)
        y_proba = clf.predict_proba(x_te)

        labels = list(label_map.keys())
        wandb.sklearn.plot_learning_curve(clf, x_tr, y_tr)
        wandb.sklearn.plot_confusion_matrix(y_te, y_hat, labels)
        wandb.sklearn.plot_class_proportions(y_tr, y_te, labels)
        wandb.sklearn.plot_precision_recall(y_te, y_proba, labels)
        wandb.sklearn.plot_roc(y_te, y_proba, labels)
        wandb.sklearn.plot_summary_metrics(clf, x_tr, y_tr, x_te, y_te)



def main():
    wandb.init(project="Gesture-Recognizer", entity="briancsavage")
    sweep_id = wandb.sweep(sweep_config)

    df = pd.read_csv(os.path.join(os.getcwd(),
                                  "..",
                                  "data",
                                  "gestures",
                                  "hand-annotations.csv"))
    table = wandb.Table(columns=list(df.columns), data=df)
    wandb.log({"table": table})

    y = df['label'].apply(lambda x: label_map[x]).values
    x = df[[f"{i}-{p}" for p in ['x', 'y', 'z'] for i in range(0, 21)]].values

    x_tr, x_te, y_tr, y_te = train_test_split(x, y,
                                              test_size=0.2,
                                              random_state=RANDOM_SEED)

    labels = list(label_map.keys())
    clf = MLPClassifier(random_state=1, max_iter=1000)
    clf.fit(x_tr, y_tr)

    y_hat = clf.predict(x_te)
    y_proba = clf.predict_proba(x_te)

    wandb.sklearn.plot_learning_curve(clf, x_tr, y_tr)
    wandb.sklearn.plot_confusion_matrix(y_te, y_hat, labels)
    wandb.sklearn.plot_class_proportions(y_tr, y_te, labels)
    wandb.sklearn.plot_precision_recall(y_te, y_proba, labels)
    wandb.sklearn.plot_roc(y_te, y_proba, labels)
    wandb.sklearn.plot_summary_metrics(clf, x_tr, y_tr, x_te, y_te)
    wandb.finish()



if __name__ == "__main__":
    # main()
    count = 10  # number of runs to execute
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train, count=count)