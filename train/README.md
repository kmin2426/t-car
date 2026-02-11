# Traffic Sign & Traffic Light Training (Stage 1)
## 1. Overview
This directory contains:
- convert_to_yolo.py: Converts original Jetson-format annotations to YOLO format
- data.yaml: Dataset configuration for Ultralytics YOLO training.

The actual dataset is stored only on the GPU server and is not included in this repository.

## 2. Dataset
### 2-1. Dataset Source
Download from the following AIHub datasets:

승용 자율주행차 주간 도심도로 데이터
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EC%8A%B9%EC%9A%A9%20%EC%9E%90%EC%9C%A8%EC%A3%BC%ED%96%89%EC%B0%A8%20%EC%A3%BC%EA%B0%84%20%EB%8F%84%EC%8B%AC%EB%8F%84%EB%A1%9C%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71621

승용 자율주행차 주간 자동차 전용도로 데이터
https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%EC%8A%B9%EC%9A%A9+%EC%9E%90%EC%9C%A8%EC%A3%BC%ED%96%89%EC%B0%A8+%EC%A3%BC%EA%B0%84+%EC%9E%90%EB%8F%99%EC%B0%A8+%EC%A0%84%EC%9A%A9%EB%8F%84%EB%A1%9C+%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71623

신호등/도로표지판 인지 영상(수도권)
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EC%8B%A0%ED%98%B8%EB%93%B1/%EB%8F%84%EB%A1%9C%ED%91%9C%EC%A7%80%ED%8C%90%20%EC%9D%B8%EC%A7%80%20%EC%98%81%EC%83%81(%EC%88%98%EB%8F%84%EA%B6%8C)&aihubDataSe=data&dataSetSn=188


### 2-2. Raw Data Location
YOLO-formatted dataset must exist at:
```bash
/data4/dongmin/t-car/data/yolo
```

Expected structure:
```bash
yolo/
  ├── images/
  │     ├── train/
  │     └── val/
  └── labels/
        ├── train/
        └── val/
```

## 3. How to Run?

Activate the virtual environment on the GPU server:
```bash
# Ultralytics YOLO must already be installed inside this environment.
source ~/.venv/bin/activate
```

Run training using 4 GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 yolo detect train \
  data=/data4/dongmin/t-car/data/yaml/traffic_stage1.yaml \
  model=/data4/dongmin/t-car/tools/yolo11s.pt \
  imgsz=640 \
  epochs=200 \
  batch=256 \
  device=0,1,2,3 \
  workers=20 \
  cache=disk \
  amp=True \
  name=traffic_stage1
```

## 4. Result

Training results will be saved under:
```bash
runs/detect/traffic_stage1/
```

Best model:
```bash
runs/detect/traffic_stage1/weights/best.pt
```