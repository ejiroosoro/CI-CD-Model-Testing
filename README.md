# 🤖 ML CI/CD Pipeline

![ML Model CI](https://github.com/YOUR_USERNAME/ml-cicd-pipeline/actions/workflows/ci.yml/badge.svg)

A machine learning portfolio project that trains an Iris flower classifier and runs automated tests on every push using GitHub Actions — with a live green ✓ badge on the repo.

---

## 📦 Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| `scikit-learn` | ML model training |
| `pytest` | Automated testing |
| GitHub Actions | CI/CD automation |

---

## 🗂️ Project Structure

```
ml-cicd-pipeline/
├── train.py          # Trains Iris classifier, saves model.pkl
├── test_model.py     # Pytest tests — asserts accuracy > 90%
├── requirements.txt  # Dependencies
├── .github/
│   └── workflows/
│       └── ci.yml    # GitHub Actions — runs tests on every push
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/ml-cicd-pipeline.git
cd ml-cicd-pipeline
pip install -r requirements.txt

# 2. Train the model
python train.py

# 3. Run tests
pytest test_model.py -v
```

---

## 🧪 What Gets Tested

| Test | What it checks |
|------|---------------|
| `test_model_file_exists` | model.pkl is created after training |
| `test_model_loads` | model.pkl loads as a valid sklearn model |
| `test_model_accuracy_above_90` | Accuracy must be > 90% |
| `test_model_predicts_correct_classes` | Only valid class labels (0, 1, 2) |
| `test_model_prediction_shape` | Output shape matches input |
| `test_model_single_prediction` | Handles a single sample |
| `test_model_probability_output` | Probabilities sum to 1.0 |

---

## 🤖 GitHub Actions

Every push to `main` automatically:
1. Sets up Python
2. Installs dependencies
3. Trains the model
4. Runs all pytest tests
5. Shows ✅ or ❌ on the repo

To get the green badge replace `YOUR_USERNAME` in the badge URL at the top of this README.

---

## 🧠 What This Project Demonstrates

- End-to-end ML workflow (train → test → validate)
- Professional pytest structure with fixtures
- CI/CD automation with GitHub Actions
- Model persistence with pickle
- Writing tested, production-quality ML code

---

## 📄 License

MIT
