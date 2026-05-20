# 🏭 Production-Ready Industrial Predictive Maintenance ML Pipeline

> Production-grade modular ML system for industrial equipment failure prediction —
> modelled on real ML work delivered at Ferrari S.p.A (Maranello, Italy)

---

## 🎯 Results

| Model | Precision | Recall | F1 Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Random Forest | 100% | 97.06% | 98.51% | 98.53% |
| Isolation Forest | 86.44% | 75.00% | 80.31% | 87.29% |

✅ Random Forest selected as best model — Zero false alarms (100% Precision)

---

## 🏗️ Production Architecture
main.py                          # Pipeline manager — runs everything
│
├── config/config.yaml           # All settings — nothing hardcoded
│
├── src/
│   ├── logger.py                # Logging to terminal + file
│   ├── exception.py             # Custom exception handling
│   ├── data_ingestion.py        # Data loading + validation
│   ├── data_preprocessing.py   # Cleaning + feature engineering
│   ├── model_trainer.py         # Training + evaluation + saving
│   └── model_predictor.py      # Production inference pipeline
│
└── artifacts/
├── model.pkl                # Saved best model
├── scaler.pkl               # Saved scaler
├── confusion_matrix.png     # Evaluation chart
---

## 📊 Dataset

AI4I 2020 Predictive Maintenance Dataset — 10,000 industrial sensor readings:

| Feature | Description |
|---------|-------------|
| Air temperature | Ambient temperature (K) |
| Process temperature | Machine working temperature (K) |
| Rotational speed | Machine speed (RPM) |
| Torque | Rotational force (Nm) |
| Tool wear | Tool usage duration (min) |
| Machine failure | Target — 0: Normal, 1: Failure |

**Class distribution:** 96.61% Normal / 3.39% Failure — highly imbalanced

---

## ⚙️ Feature Engineering

Three domain-knowledge features engineered from raw sensors:

- **temp_difference** — Process vs Air temperature gap (heat dissipation indicator)
- **power** — Torque × RPM (actual mechanical power — physics-based feature)
- **wear_rate** — Tool wear × Torque (combined degradation signal)

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/UzmaLatif/production-ready-industrial-predictive-maintenance.git
cd production-ready-industrial-predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Run full pipeline — training + prediction
python main.py
```

---

## 🔍 Make a Prediction

```python
from src.model_predictor import ModelPredictor

predictor = ModelPredictor()

result = predictor.predict({
    "Type": 1,
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
    "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0
})

print(result)
# {
#   "status": "✅ NORMAL OPERATION",
#   "confidence": "99.95%",
#   "action": "Continue monitoring"
# }
```

---

## 🛠️ Tech Stack

- **Python** — NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **ML Models** — Random Forest, Isolation Forest
- **MLOps** — Modular pipeline, config management, custom logging, artifact versioning
- **Deployment Ready** — Inference pipeline separated from training

---

## 🔗 Related Experience

This pipeline is modelled on production ML work delivered at
**Ferrari S.p.A (Maranello, Italy)** — where anomaly detection systems
were built on real engine assembly line sensor data and deployed via
Flask REST APIs to Ferrari engineering teams.

---

## 👩‍💻 Author

**Uzma Latif** — Data Scientist | ML Engineer | AI Researcher  
📧 uzmalatif777@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/uzma-latif-4b2604130)  
🐙 [GitHub](https://github.com/UzmaLatif)
└── feature_importance.png   # Feature importance chart
