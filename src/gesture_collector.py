import os
import cv2
import pandas as pd
from detectors import HandDetector

commands = {
    49: 'up',
    50: 'down',
    51: 'left',
    52: 'right',
    53: 'forward',
    54: 'backward',
    55: 'clockwise',
    56: 'counterclockwise',
}


def main():
    hand_detector = HandDetector()
    cap = cv2.VideoCapture(0)
    entries = pd.DataFrame(columns=['label',
                                    '0-x', '0-y', '0-z',
                                    '1-x', '1-y', '1-z',
                                    '2-x', '2-y', '2-z',
                                    '3-x', '3-y', '3-z',
                                    '4-x', '4-y', '4-z',
                                    '5-x', '5-y', '5-z',
                                    '6-x', '6-y', '6-z',
                                    '7-x', '7-y', '7-z',
                                    '8-x', '8-y', '8-z',
                                    '9-x', '9-y', '9-z',
                                    '10-x', '10-y', '10-z',
                                    '11-x', '11-y', '11-z',
                                    '12-x', '12-y', '12-z',
                                    '13-x', '13-y', '13-z',
                                    '14-x', '14-y', '14-z',
                                    '15-x', '15-y', '15-z',
                                    '16-x', '16-y', '16-z',
                                    '17-x', '17-y', '17-z',
                                    '18-x', '18-y', '18-z',
                                    '19-x', '19-y', '19-z',
                                    '20-x', '20-y', '20-z'])






    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hands = hand_detector.detect(frame)
        image = hand_detector.visualize(hands, frame)

        cv2.imshow('frame', cv2.flip(image, 1))

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            entries.to_csv(os.path.join(os.getcwd(),
                                        '..',
                                        'data',
                                        'gestures',
                                        'hand-annotations.csv'))
            break

        if key == 48:  # '0' == DELETE PREVIOUSLY SAVED
            entries = entries.iloc[:-1, :]
            print('Deleted previous entry')

        if key in commands:
            hand = hands.multi_hand_landmarks[0] if \
                   hands.multi_hand_landmarks else []

            row = {"label": commands[key]}

            # print(hand)
            if hand:
                for i, p in enumerate(hand.landmark):
                    print(p)
                    row.update({f"{i}-x": p.x,
                                f"{i}-y": p.y,
                                f"{i}-z": p.z})

                entries = entries.append(row, ignore_index=True)
                print(entries.tail(5))
            else:
                print('No hand detected')


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

