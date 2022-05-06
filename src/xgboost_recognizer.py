import pandas as pd
import numpy as np
import os
import wandb
import pickle
import cv2
from detectors import HandDetector
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import xgboost from xgb

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

def main():
    wandb.init(project="Gesture-Recognizer", entity="briancsavage")

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
    scaler = StandardScaler()
    x_tr = pd.DataFrame(scaler.fit_transform(x_tr))
    x_te = pd.DataFrame(scaler.transform(x_te))


    print(y_te)
    print(y_tr)

    labels = list(label_map.keys())
    # clf = MLPClassifier(random_state=RANDOM_SEED,
    #                     learning_rate="constant",
    #                     hidden_layer_sizes=(64, 8),
    #                     solver="sgd",
    #                     activation="tanh",
    #                     batch_size=32,
    #                     max_iter=5000)

    # clf = RandomForestClassifier(n_estimators=100,
    #                              max_features='sqrt',
    #                              random_state=42)

    clf = xgboost.XGBClassifier(n_estimators=100,
                                max_depth=3,
                                verbose=1)

    clf.fit(x_tr, y_tr)

    y_hat = clf.predict(x_te)

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
            for i, p in enumerate(hand.landmark):
                values += [p.x, p.y, p.z]
            label = clf.predict([values])
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