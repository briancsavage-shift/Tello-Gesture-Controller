from djitellopy import Tello
import cv2


def main():
    drone = Tello()
    drone.connect()
    # drone.takeoff()
    # drone.land()
    print(drone.get_battery())
    drone.streamon()
    while True:
        img = drone.get_frame_read().frame
        img = cv2.resize(img, (360, 240))
        cv2.imshow('Image', img)
        cv2.waitKey(1)
    return


if __name__ == '__main__':
    main()
