# 🏭 Manufacturing Inspection Dashboard

<p align="center">
  <img src="assets/demo.gif" width="700"/>
</p>

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

반도체 검사 이미지를 대상으로 분류, 추론, 액티브 러닝 기반 샘플 선택, 파인튜닝을 수행하는 Streamlit 대시보드 프로젝트입니다.

---



## 🚀 Features

* Streamlit 기반 검사 대시보드
* MobileViT 이미지 분류 학습 / 추론
* Active Learning 기반 샘플링 
* 선택 이미지 기반 인터랙티브 파인튜닝
* Gemma 기반 보조 응답

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

필요한 설정:

* `data/semicondotor_seg_data_path.json`
* `.streamlit/secrets.toml`
* 선택 사항: `model/google__gemma-4-E2B-it/` 로컬 모델

---

## ▶️ Usage

### Dashboard

```bash
streamlit run streamlit_dashboard.py
```

---

## 📚 Documentation

- [Quick Start](docs/quickstart.md)


---

## 📂 Project Structure

```text
.
├── .streamlit/
├── assets/
├── data/
├── log/
├── model/
├── output/
├── pages/
├── scripts/
├── detail_finetune_mcp.py
├── local_gemma_model.py
├── streamlit_dashboard.py
├── requirements.txt
└── README.md
```

---

## 📌 Notes

### Pages

* Dashboard : Data Distribution, Recent Runs, Latest Logs, Current model configuration
* Summary : Data Distribution, Normal, Defect, Monthly, Weekly, Daily graph , Comment by LLM
* Detail : Classification model inference results
* Fine tuning : Model selection, Fine tuning, Active learning sampling, Data Labeling
* Setting : DB Setting, LLM Setting
* Log : Log history

### Version Update

* [Change Log](CHANGELOG.md)