import pandas as pd
import os

def update_csv_paths(csv_path, new_base_path, output_csv_path):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    # 'path' 컬럼의 경로 수정
    df['path'] = df['path'].apply(lambda x: os.path.join(new_base_path, *x.split('/')[-3:]))

    # 수정된 데이터를 새로운 CSV로 저장
    df.to_csv(output_csv_path, index=False)

csv_path = '/home/song/Desktop/VadCLIP/list/ucf_CLIP_rgb.csv'            # 원본 CSV 파일 경로
new_base_path = '/home/song/Desktop/VadCLIP/'                # 새롭게 옮긴 폴더의 상위 경로
output_csv_path = 'list/new_ucf_CLIP_rgb.csv'     # 수정된 경로를 저장할 새로운 CSV 파일 경로

update_csv_paths(csv_path, new_base_path, output_csv_path)
