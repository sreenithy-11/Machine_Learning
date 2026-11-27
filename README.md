# Machine Learning Learning Hub

A beginner-friendly interactive application for undergraduate students to learn core Machine Learning concepts, explore algorithms, and practice basic experiments.

## What is included

- **Home:** learning roadmap and shortcuts to the main sections.
- **ML Fundamentals:** concise lessons on AI/ML, learning settings, data, workflow, and evaluation.
- **ML Algorithms:** explanations of 12 supervised and unsupervised algorithms, plus reinforcement learning and Q-learning.
- **Interactive Visualizations:** adjustable regression, K-Means, decision-tree, and classification demonstrations.
- **Practical ML Lab:** built-in housing and Iris data, CSV upload, basic preprocessing, model training, metrics, and prediction.
- **Quiz & Assessment:** 26 questions with answer review and explanations.

The sample data is for learning only; results should not be used for consequential decisions.

## Technologies

Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, and Plotly.

## Installation

Use Python 3.10 or later. From the project folder:

```powershell
python -m pip install -r requirements.txt
```

## Run locally

```powershell
python -m streamlit run app.py
```

Then open the local address printed in the terminal (usually `http://localhost:8501`).

## Project structure

```text
ML_Learning_Hub/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── style.css
├── modules/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── sample_datasets.py
└── pages/
    ├── algorithms.py
    ├── fundamentals.py
    ├── practical_lab.py
    ├── quiz.py
    └── visualizations.py
```
