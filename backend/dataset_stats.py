import os

dataset_path = "../dataset/fer2013/train"

total_images = 0

print("Emotion Distribution:")
print("----------------------")

for emotion in os.listdir(dataset_path):
    emotion_path = os.path.join(dataset_path, emotion)

    if os.path.isdir(emotion_path):
        count = len(os.listdir(emotion_path))
        total_images += count
        print(f"{emotion}: {count} images")

print("----------------------")
print("Total Images:", total_images)
