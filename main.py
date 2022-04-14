from djitellopy import Tello


def main():
    drone = Tello()
    drone.connect()
    # drone.takeoff()
    # drone.land()
    print(drone.get_battery())
    return


if __name__ == '__main__':
    main()
