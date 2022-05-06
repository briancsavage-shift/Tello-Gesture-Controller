import pandas as pd
import numpy as np
import os
import wandb
import pickle
import cv2
import time
from detectors import HandDetector
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 12345

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


emojis = [
    "☝️",
    "👇",
    "👈",
    "👉",
    "🔴",
    "🟢",
    "⏩",
    "⏪",
]

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

    df = pd.read_csv(os.path.join(os.getcwd(),
                                  "..",
                                  "data",
                                  "gestures",
                                  "hand-annotations.csv"))
    testing = pd.read_csv(os.path.join(os.getcwd(),
                                       "..",
                                       "data",
                                       "gestures",
                                       "hand-annotations-old.csv"))

    table = wandb.Table(columns=list(df.columns), data=df)
    tests = wandb.Table(columns=list(testing.columns), data=testing)
    wandb.log({"Training Table": table})
    wandb.log({"Testing Table": tests})

    y = df['label'].apply(lambda x: label_map[x]).values
    x = df[[f"{i}-{p}" for p in ['x', 'y', 'z'] for i in range(0, 21)]].values

    y_test = testing['label'].apply(lambda x: int(label_map[x])).values
    x_test = testing[[f"{i}-{p}" for p in ['x', 'y', 'z'] for i in range(0, 21)]].values

    print("Number of Training Samples:", len(x), len(y))
    print("Number of Test Samples:", len(x_test), len(y_test))

    x_tr, x_te, y_tr, y_te = train_test_split(x, y,
                                              test_size=0.2,
                                              random_state=RANDOM_SEED)
    scaler = StandardScaler()
    x_tr = pd.DataFrame(scaler.fit_transform(x_tr))
    x_te = pd.DataFrame(scaler.transform(x_te))

    x_test = pd.DataFrame(scaler.transform(x_test))

    # print(y_te)
    # print(y_tr)
    print(x_te.head(5))
    print(x_test.head(5))

    labels = list(label_map.keys())
    # clf = MLPClassifier(random_state=RANDOM_SEED,
    #                     learning_rate="constant",
    #                     hidden_layer_sizes=(64, 8),
    #                     solver="sgd",
    #                     activation="tanh",
    #                     batch_size=32,
    #                     max_iter=5000)

    clf = RandomForestClassifier(n_estimators=100,
                                 max_features='sqrt',
                                 max_depth=10,
                                 random_state=42)

    clf.fit(x_tr, y_tr)

    y_hat = clf.predict(x_te)
    y_hat_test = clf.predict(x_test)
    # print(y_hat_test)

    for te, hat in zip(y_te, y_hat):
        print(f"Validation: y_te: {emojis[te]}, y_hat: {emojis[hat]}")

    for te, hat in zip(y_test, y_hat_test):
        print(f"Testing: y_te: {emojis[te]}, y_hat: {emojis[hat]}")

    #
    # print("Pred: ", ' '.join([emojis[i] for i in y_hat]))
    # print("True: ", ' '.join([emojis[i] for i in y_te]))

    print(" Test Accuracy: ", accuracy_score(y_te, y_hat,
                                            normalize=True,
                                            sample_weight=None))
    print("Train Accuracy: ", accuracy_score(y_tr, clf.predict(x_tr),
                                            normalize=True,
                                            sample_weight=None))




    hand_detector = HandDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        hands = hand_detector.detect(frame)
        image = hand_detector.visualize(hands, frame)

        hand = hands.multi_hand_landmarks[0] if \
               hands.multi_hand_landmarks else None

        if hand is not None:
            values = []
            row = {}
            for i, p in enumerate(hand.landmark):
                row.update({f"{i}-x": p.x,
                            f"{i}-y": p.y,
                            f"{i}-z": p.z})
            inference = pd.DataFrame(columns=list(range(0, 63)))
            inference.loc[0] = list(row.values())

            print("YEYYSYSYSS", inference)


            label = clf.predict(inference)
            print(label)
            print(emojis[label[0]])
        else:
            print("")

        cv2.imshow('frame', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    y_proba = clf.predict_proba(x_te)

    print('Classification Report: \n', classification_report(y_te, y_hat))

    wandb.sklearn.plot_learning_curve(clf, x_tr, y_tr)
    wandb.sklearn.plot_confusion_matrix(y_te, y_hat, labels)
    wandb.sklearn.plot_class_proportions(y_tr, y_te, labels)
    wandb.sklearn.plot_precision_recall(y_te, y_proba, labels)
    wandb.sklearn.plot_roc(y_te, y_proba, labels)
    wandb.sklearn.plot_summary_metrics(clf, x_tr, y_tr, x_te, y_te)
    wandb.finish()

    with open(os.path.join(os.getcwd(),
                           "..",
                           "models",
                           "gesture-detection",
                           "mlp_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)


class GestureRecognizer:
    def __init__(self):
        with open(os.path.join(os.getcwd(),
                               "..",
                               "models",
                               "gesture-detection",
                               "mlp_classifier.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def predict(self, hand):
        values = []
        for i, p in enumerate(hand.landmark):
            values += [p.x, p.y, p.z]
        return self.model.predict_proba([values])




if __name__ == "__main__":
    # count = 10  # number of runs to execute
    # sweep_id = wandb.sweep(sweep_config)
    # wandb.agent(sweep_id, function=train, count=count)
    main()