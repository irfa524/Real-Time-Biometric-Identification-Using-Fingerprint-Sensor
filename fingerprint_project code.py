#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Fingerprint Recognition Project
End-to-End Script:
1️- Balance dataset using augmentation
2️- Train a ResNet18 model (with train/validation split)
3️- Predict fingerprints in realtime from sensor (or test image)

Requirements:
    pip install tensorflow pillow torch torchvision pyfingerprint

"""
# Step 01. Imports libraries
import os
import time
from datetime import datetime
from PIL import Image
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Try to import sensor library
try:
    from pyfingerprint.pyfingerprint import PyFingerprint
    HAVE_SENSOR_LIB = True
except Exception:
    HAVE_SENSOR_LIB = False

# Step 2. Data Augmented Function
def augment_dataset(source_dir=r"C:\Users\HP\OneDrive\Documents\fingerprint_simple_knn\fingerprint_simple_knn\New Dataset",
                    target_dir="New Dataset Augmented",
                    target_count=100):
    print("\n Starting dataset augmentation ...")
    os.makedirs(target_dir, exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )

    for person in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person)
        if not os.path.isdir(person_path):
            continue

        target_person_dir = os.path.join(target_dir, person)
        os.makedirs(target_person_dir, exist_ok=True)

        images = [f for f in os.listdir(person_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

        # Copy original images
        for img_name in images:
            src = os.path.join(person_path, img_name)
            dst = os.path.join(target_person_dir, img_name)
            if not os.path.exists(dst):
                Image.open(src).convert('RGB').save(dst)

        current_count = len(images)
        print(f"\n{person}: {current_count} original images found.")

        if current_count >= target_count:
            print(" Already balanced, skipping augmentation.")
            continue

        print(f"  Augmenting {person} → to reach {target_count} images ...")
        img_index = 0
        while current_count < target_count:
            img_path = os.path.join(person_path, images[img_index % len(images)])
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            for _ in datagen.flow(x, batch_size=1,
                                  save_to_dir=target_person_dir,
                                  save_prefix='aug',
                                  save_format='jpg'):
                current_count += 1
                if current_count >= target_count:
                    break
            img_index += 1

        print(f" {person} folder now has {current_count} images.")

    print("\n Dataset augmentation completed successfully!")
    return target_dir

# Step 3. Train Model Function (with train/vald split)
def train_model(data_dir="New Dataset Augmented", epochs=5, save_path="best_model.pth", val_split=0.2):
    """
    Trains a ResNet18 model with an 80/20 train-validation split.
    """
    print("\n Starting model training with validation split ...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Split into training and validation sets
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
              f"Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\n Model saved as {save_path}")
    print(f" Final Training Accuracy: {train_acc:.2f}%")
    print(f" Final Validation Accuracy: {val_acc:.2f}%")

    return model, dataset.classes

#  Step 4. Real-Time Sensor prediction function
def capture_from_sensor(port='COM3', baudrate=57600, address=int('FFFFFFFF', 16),
                        password=int('00000000', 16), out_dir='captured_images'):
    if not HAVE_SENSOR_LIB:
        raise RuntimeError("pyfingerprint library not installed. Install it or use test image mode.")

    os.makedirs(out_dir, exist_ok=True)
    print("\n Connecting to sensor ...")

    try:
        f = PyFingerprint(port, baudrate, address, password)
        if not f.verifyPassword():
            raise ValueError("Sensor password incorrect!")
    except Exception as e:
        raise RuntimeError(f"Sensor initialization failed: {e}")

    print(" Sensor connected successfully!")
    print("Place your finger on the sensor ...")

    while not f.readImage():
        time.sleep(0.05)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path = os.path.join(out_dir, f"finger_{ts}.bmp")

    print("Downloading image ...")
    f.downloadImage(image_path)
    print(f" Image saved: {image_path}")
    return image_path

# Step 5.Prediction Function
def predict_image(model, class_names, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    pred_class = class_names[idx.item()]
    return pred_class, float(conf.item())

# Step 6. Main execution menu
def main():
    print("\n Fingerprint Recognition Project")
    print("------------------------------------")
    print("1️-  Augment dataset")
    print("2️- Train model")
    print("3️- Realtime prediction (sensor)")
    print("4️- Test prediction (using image)")
    print("------------------------------------")

    choice = input("Select option (1/2/3/4): ").strip()

    if choice == "1":
        augment_dataset()

    elif choice == "2":
        train_model()

    elif choice == "3":
        model_path = "best_model.pth"
        data_dir = "New Dataset Augmented"

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        class_names = dataset.classes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        img_path = capture_from_sensor()
        pred, conf = predict_image(model, class_names, img_path)
        print(f"\n Prediction: {pred}  (confidence: {conf:.3f})")

    elif choice == "4":
        img_path = input("Enter path to test image: ").strip()
        model_path = "best_model.pth"
        data_dir = "New Dataset Augmented"

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        class_names = dataset.classes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        pred, conf = predict_image(model, class_names, img_path)
        print(f"\n Prediction: {pred}  (confidence: {conf:.3f})")

    else:
        print(" Invalid choice. Exiting.")

# Step 7. RUN
if __name__ == "__main__":
    main()
