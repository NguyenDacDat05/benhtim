# Đề 2: Dự đoán Bệnh Tim (Heart Disease Prediction)

**Học phần:** Khai phá Dữ liệu | **HK II – 2025–2026**  
**Giảng viên:** ThS. Lê Thị Thùy Trang  
**Dataset:** [UCI/Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## Mục tiêu

Xây dựng pipeline khai phá dữ liệu y tế để:
1. Phát hiện tổ hợp triệu chứng bệnh tim bằng **Association Rules (Apriori)**
2. Phân nhóm bệnh nhân theo mức độ nguy cơ bằng **Clustering (KMeans/HAC)**
3. Phân lớp nguy cơ bệnh tim bằng **SVM / Random Forest / XGBoost**
4. Thực nghiệm **bán giám sát** (self-training / label spreading) với ít nhãn
5. Hồi quy chỉ số sức khỏe (huyết áp, cholesterol) theo yếu tố nguy cơ

---

## Cấu trúc thư mục

```
de2_heart_disease/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── params.yaml              # Toàn bộ tham số pipeline
├── data/
│   ├── raw/                     # Dữ liệu gốc (heart.csv)
│   └── processed/               # Dữ liệu đã xử lý (.parquet)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py            # Đọc dữ liệu, kiểm tra schema
│   │   └── cleaner.py           # Xử lý missing, outlier, encoding
│   ├── features/
│   │   └── builder.py           # Feature engineering, rời rạc hoá
│   ├── mining/
│   │   ├── association.py       # Apriori / FP-Growth + luật kết hợp
│   │   └── clustering.py        # KMeans / HAC / DBSCAN + profiling
│   ├── models/
│   │   ├── supervised.py        # Train/predict classification + regression
│   │   └── semi_supervised.py   # Self-training + Label Spreading
│   ├── evaluation/
│   │   ├── metrics.py           # Accuracy, F1, PR-AUC, ROC-AUC, RMSE, MAE
│   │   └── report.py            # Tổng hợp bảng / biểu đồ kết quả
│   └── visualization/
│       └── plots.py             # Hàm vẽ dùng chung
├── scripts/
│   ├── run_pipeline.py          # Chạy toàn bộ pipeline
│   └── run_papermill.py         # Chạy notebook bằng papermill
└── outputs/
    ├── figures/
    ├── tables/
    ├── models/
    └── reports/
```

---

## Hướng dẫn cài đặt & chạy

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Tải dataset

Tải dataset từ Kaggle: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

Đặt file `heart.csv` vào `data/raw/heart.csv`

Hoặc dùng Kaggle API:
```bash
kaggle datasets download -d johnsmith88/heart-disease-dataset -p data/raw/ --unzip
```

### 3. Cập nhật cấu hình (nếu cần)

Chỉnh đường dẫn dữ liệu trong `configs/params.yaml`.

### 4. Chạy toàn bộ pipeline

```bash
python scripts/run_pipeline.py
```

### 5. Chạy notebook bằng papermill (tái lập đầy đủ)

```bash
python scripts/run_papermill.py
```


### 6. Mở notebook thủ công (theo thứ tự)

```
01_eda.ipynb → 02_preprocess_feature.ipynb → 03_mining_clustering.ipynb
→ 04_modeling.ipynb → 04b_semi_supervised.ipynb → 05_evaluation_report.ipynb
```

---

## Data Dictionary

| Cột | Kiểu | Ý nghĩa |
|-----|------|---------|
| age | int | Tuổi bệnh nhân |
| sex | int | Giới tính (0=Nữ, 1=Nam) |
| cp | int | Loại đau ngực (0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic) |
| trestbps | int | Huyết áp nghỉ ngơi (mmHg) |
| chol | int | Cholesterol huyết thanh (mg/dl) |
| fbs | int | Đường huyết lúc đói > 120mg/dl (0=Không, 1=Có) |
| restecg | int | Kết quả ECG nghỉ ngơi (0=Normal, 1=ST-T wave abnormality, 2=LV hypertrophy) |
| thalach | int | Nhịp tim tối đa đạt được |
| exang | int | Đau ngực khi tập thể dục (0=Không, 1=Có) |
| oldpeak | float | ST depression khi tập so với nghỉ |
| slope | int | Độ dốc của đỉnh ST khi tập (0=Upsloping, 1=Flat, 2=Downsloping) |
| ca | int | Số mạch máu lớn được tô màu bằng fluoroscopy (0–3) |
| thal | int | Thalassemia (0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Unknown) |
| **target** | **int** | **Bệnh tim (0=Không, 1=Có)** ← biến mục tiêu |

**Rủi ro dữ liệu:**
- Mất cân bằng lớp nhẹ (cần kiểm tra)
- Một số cột có giá trị 0 bất thường (ca, thal)
- Không có data leakage rõ ràng (tất cả đặc trưng là đầu vào lâm sàng)

---

## Kết quả nổi bật

Xem `outputs/reports/` sau khi chạy pipeline.
