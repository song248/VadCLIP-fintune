import torch
import clip
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

# 논문 설정에 따라 ViT-B/16 CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/16", device=device)

# 논문에서 제시한 파라미터
segment_length = 16  # 일반적인 segment 길이
window_length = 8    # UCF-Crime 논문 설정
default_embedding_dim = 512

# 비디오에서 올바르게 CLIP 특징 추출하는 함수 정의
def extract_clip_features(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    segment_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
        frames.append(frame_tensor)

        if len(frames) == segment_length:
            frames_tensor = torch.cat(frames, dim=0)
            with torch.no_grad():
                frame_features = clip_model.encode_image(frames_tensor).cpu().numpy()
            segment_feature = np.mean(frame_features, axis=0)
            segment_features.append(segment_feature)
            frames = []

    if frames:
        frames_tensor = torch.cat(frames, dim=0)
        with torch.no_grad():
            frame_features = clip_model.encode_image(frames_tensor).cpu().numpy()
        segment_feature = np.mean(frame_features, axis=0)
        segment_features.append(segment_feature)

    cap.release()

    segment_features = np.array(segment_features)
    np.save(output_path, segment_features)

# 경로 설정
video_dir = './violence_data/'
output_dir = './myClipFeatures/'
categories = ['assault', 'fighting', 'normal']

for category in categories:
    category_input_dir = os.path.join(video_dir, category)
    category_output_dir = os.path.join(output_dir, category)
    os.makedirs(category_output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(category_input_dir) if f.endswith('.mp4')]

    for video_file in tqdm(video_files, desc=f"Extracting {category} features"):
        video_path = os.path.join(category_input_dir, video_file)
        output_path = os.path.join(category_output_dir, video_file.replace('.mp4', '.npy'))

        extract_clip_features(video_path, output_path)

print('All videos processed successfully.')
