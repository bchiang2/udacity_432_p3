import csv
import cv2
import numpy as np

lines = []
correction = 0.5

def path_to_img(filename):
    clean_file_name = filename.strip().split("IMG")[1]
    return cv2.imread(r'data/sets/1/IMG' + clean_file_name)


def log_to_data(row):
    steering_center = float(row[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    steering = [steering_center, steering_left, steering_right]
    [img_center, img_left, img_right] = [path_to_img(row[i]) for i in range(3)]
    images = [img_center, img_left, img_right]

    return (images, steering)


def load_data():
    x_train = []
    y_train = []

    with open(r'data/sets/1/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            images, steering = log_to_data(line)
            x_train += images
            y_train += steering

    return np.array(x_train), np.array(y_train)
