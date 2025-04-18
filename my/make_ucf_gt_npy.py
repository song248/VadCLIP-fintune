import numpy as np
import json

class_map = {
    "Normal": 0, "Abuse": 1, "Arrest": 2, "Arson": 3, "Assault": 4, "Burglary": 5,
    "Explosion": 6, "Fighting": 7, "RoadAccidents": 8, "Robbery": 9,
    "Shooting": 10, "Shoplifting": 11, "Stealing": 12, "Vandalism": 13
}

with open("my_test.json", "r") as f:
    data = json.load(f)

gt_ucf = []
gt_segment_ucf = []
gt_label_ucf = []

# 비디오 별로 루프
for video in data["videos"]:
    num_frames = video["num_frames"]
    label_name = video["label"]
    anomalous_intervals = video["anomalous"]

    # 비디오 단위 라벨 생성 (정상/이상)
    is_anomaly_video = 0 if label_name == "Normal" else 1
    gt_segment_ucf.append(is_anomaly_video)

    # 클래스 단위 라벨 생성
    gt_label_ucf.append(class_map[label_name])

    # 프레임 단위 라벨 생성
    gt_frames = np.zeros(num_frames, dtype=int)
    for interval in anomalous_intervals:
        start, end = interval["start_frame"], interval["end_frame"]
        gt_frames[start:end+1] = 1

    gt_ucf.extend(gt_frames.tolist())

# 최종 NPY 파일 저장
np.save("gt_ucf.npy", np.array(gt_ucf))
np.save("gt_segment_ucf.npy", np.array(gt_segment_ucf))
np.save("gt_label_ucf.npy", np.array(gt_label_ucf))
