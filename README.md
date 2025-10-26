# Real-Time-Biometric-Identification-Using-Fingerprint-Sensor
This project is a "Realtime Biometric Identification System" that integrates "hardware-based fingerprint scanning" with "deep learning (ResNet18)"for accurate and efficient user identification.  
It demonstrates a hybrid approach combining embedded hardware (sensor interfacing) and machine learning (image classification) to recognize individuals based on their fingerprints.

 ##Overview

The system captures fingerprint images through an "AS608 fingerprint sensor" connected to a laptop via a "USB-to-TTL converter".  
The collected images are processed and stored as a dataset. A "ResNet18-based deep learning model" is then trained to classify fingerprints and identify individuals in real time.
The project supports:
1-Dataset creation from live fingerprint captures  
2-Model training with data augmentation  
3-Realtime prediction and class-based identification  

##Features

1-Real-time fingerprint acquisition using AS608 sensor  
2-Automatic dataset creation and labeling  
3-Deep learning–based classification (ResNet18)  
4-Augmentation for improved accuracy and generalization  
5-Realtime identification with live predictions  
6-Modular code for easy retraining and updates  

##  "Hardware Components"

     "Component"                                                  "Description" 
AS608 Fingerprint Sensor                     |       Captures high-resolution fingerprint images 
USB to TTL Converter                         |       Interfaces the sensor with laptop (serial communication) 
Jumper Wires                                 |       Used for hardware connections 
Laptop / PC                                  |       For running data processing and training scripts 

## "Software & Tools"

    "Tool"                                   |            "Purpose"
Python (3.9.1)                               |     Main programming language 
Google Colab / Jupyter Notebook              |     Model training and testing 
VS Code                                      |     Code development and debugging 
PyTorch / TorchVision                        |     Implementation of ResNet18 model 
OpenCV                                       |     Image preprocessing and data handling 
TensorFlow / Keras                           | Dataset augmentation 
Pillow                                       | Image input/output 
pyfingerprint                                | Sensor interfacing (for real-time capture) 

## "Installation & How to Run"
pip install torch torchvision
pip install tensorflow
pip install pillow
pip install pyfingerprint

##  "System Workflow"

1. "Fingerprint Capture": 
   The AS608 sensor captures fingerprint images via serial interface.

2. "Dataset Creation":  
   Each user’s fingerprint images are saved in a separate folder named after the person (e.g., `User_1/`, `User_2/`, etc.).

3. "Model Training":  
   The dataset is trained using "ResNet18" (a CNN model) with "augmentation techniques" like rotation, flip, and normalization to improve accuracy.

4. "Realtime Prediction": 
   A new fingerprint is captured and passed through the trained model for classification.  
   The system identifies the user by predicting the corresponding class label.

5. "Output":  
   The predicted name or class is displayed in real time, confirming the identity.

##  "Model Performance"

 "Metric"                           |     "Accuracy" 

Training Accuracy                   |      99.88% 
Validation Accuracy                 |      98.28% 

 The model achieved "high reliability" and generalization performance on unseen data.

##  "Applications"

- Secure access control systems  
- Biometric attendance and identity verification  
- AI-based biometric authentication  
- IoT and embedded security systems  

## "Future Enhancements"

- Integration with cloud storage for centralized data access  
- Multi-modal biometrics (face + fingerprint)  
- Lightweight model deployment on embedded devices (e.g., Raspberry Pi)  
- Real-time GUI dashboard for monitoring  

## "Authors"

"Nawal Qamar"  and  "Ounzila" 
Email:  nawalqamar2022@gmail.com
Email:  ounzila12345@gmail.com

##  "License"

This project is developed for "educational and research purposes".  
You are free to modify and enhance it with proper credits to the original authors.
