# 🏭 Manufacturing Inspection Application

<p align="center">
  <a href="https://youtu.be/LEap37YUmlg">
    <img src="assets/manufacturing_inspection.png" width="700"/><br>
    <sub> Demo Video </sub>
  </a>
  
</p>

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

A Streamlit-based application for semiconductor inspection, featuring image classification, inference, active learning-based sample selection, and model fine-tuning.

---


## 🚀 Features

* Streamlit-based inspection Application
* MobileViT image classification training and inference
* PatchCore-based Anomaly Detection
* Active Learning-based sampling
* Interactive fine-tuning with selected images
* Gemma-based assistant responses

---

## 🛠️ Installation

```bash
git clone https://github.com/GritGlass/Manufacturing_Inspection.git
cd Manufacturing_Inspection

git lfs install
git lfs pull


pip install -r requirements.txt
```

Required setup:

* Training/runtime config is no longer loaded from the legacy local `data` folder
* `.streamlit/secrets.toml`
* Optional: local model files in `model/google__gemma-4-E2B-it/`

---

## ▶️ Usage

### Dashboard

```bash
streamlit run Dashboard.py
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
│   ├── detail_finetune_mcp.py
│   └── local_gemma_model.py
├── streamlit_dashboard.py
├── requirements.txt
└── README.md
```

---

## 📌 Notes

### Pages

* Dashboard : Data distribution, recent runs, latest logs, current model configuration
* Summary : Data distribution, normal/defect status, monthly/weekly/daily graphs, LLM comments
* Detail : Classification model inference results
* Fine tuning : Model selection, fine-tuning, Active Learning sampling, data labeling
* Setting : Database settings, LLM settings
* Log : Log history

### Version Update

* [Change Log](CHANGELOG.md)




#### Reference
- Data reference : [Semiconductor](https://www.kaggle.com/datasets/drtawfikrrahman/multi-class-semiconductor-wafer-image-dataset)
- Model : [Mobilevit_small](https://huggingface.co/apple/mobilevit-small)
