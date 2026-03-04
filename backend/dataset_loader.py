import os

dataset_path = "../dataset/fer2013/train"

emotions = os.listdir(dataset_path)

print("Available emotion classes:")
for emotion in emotions:
    print(emotion)
