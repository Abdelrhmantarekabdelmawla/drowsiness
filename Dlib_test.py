import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import mediapipe as mp
import time
import numpy as np
from PIL import Image
from collections import deque
from scipy.spatial import distance

# التحقق مما إذا كان الـ GPU متاح
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# تعريف نفس نموذج PyTorch المحفوظ
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 30 * 30, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])  # Flatten dynamically
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# تحميل النموذج المدرب
model = DrowsinessCNN().to(device)
model.load_state_dict(torch.load("F:\machine_learning_studying\jupyter\graduation_project\Drowsiness\drowsiness-dlib\src\models\drowsiness_model.pth", map_location=device))
model.eval()

# إعداد تحويل الصورة للنموذج
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# تحميل MediaPipe لاكتشاف الوجوه
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# دالة لحساب نسبة فتح العين (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# دالة لحساب نسبة فتح الفم (MOR)
def calculate_mor(mouth):
    A = distance.euclidean(mouth[3], mouth[9])  # Vertical distance
    B = distance.euclidean(mouth[0], mouth[6])  # Horizontal distance
    return A / B

# فتح الكاميرا
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# قائمة لتخزين التوقعات (Voting System)
votes = deque(maxlen=25)  # تخزين آخر 25 إطارًا (تقريبًا 5 ثوانٍ عند 5 FPS)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading camera")
        break

    # تحويل الصورة إلى RGB لاستخدامها مع MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # اكتشاف الوجه
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # تحويل الوجه إلى Tensor
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            input_face = transform(face_pil).unsqueeze(0).to(device)

            # طباعة بعض المعلومات المفيدة
            print(f"Face shape: {face.shape}")
            print(f"Input tensor shape: {input_face.shape}")

            # إجراء التنبؤ
            # with torch.no_grad():
            #     output = model(input_face).item()
            #     print(f"Model output: {output}")
            #     pred = 1 if output > 0.5 else 0
            #     print(f"Prediction: {pred}")
            #     votes.append(pred)

            # # رسم المستطيل حول الوجه
            # color = (0, 0, 255) if pred == 1 else (0, 255, 0)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # استخراج النقاط الرئيسية للوجه باستخدام MediaPipe Face Mesh
            face_mesh_results = face_mesh.process(rgb_frame)
            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    # استخراج نقاط العين والفم
                    left_eye = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in range(33, 42)]
                    right_eye = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in range(42, 51)]
                    mouth = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in range(61, 68)]

                    # حساب نسبة فتح العين (EAR) ونسبة فتح الفم (MOR)
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    mor = calculate_mor(mouth)

                    # طباعة القيم المحسوبة
                    print(f"EAR: {ear}, MOR: {mor}")

                    # رسم النقاط على العين والفم
                    for point in left_eye + right_eye + mouth:
                        cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # **التصويت كل 5 ثواني**
    if time.time() - start_time >= 5:
        if votes:
            majority_vote = 1 if sum(votes) > len(votes) / 2 else 0
            label = "DROWSINESS ALERT!" if majority_vote == 1 else "Alert"
            color = (0, 0, 255) if majority_vote == 1 else (0, 255, 0)
            cv2.putText(frame, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            print(f"Majority Vote Result: {label}")

        votes.clear()
        start_time = time.time()

    # عرض الفيديو
    cv2.imshow("Drowsiness Detection", frame)

    # الخروج عند الضغط على "ESC"
    if cv2.waitKey(1) & 0xFF == 27:
        print("Video stopped")
        break

cap.release()
cv2.destroyAllWindows()