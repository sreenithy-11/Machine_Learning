import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.data_preprocessing import basic_summary
from modules.model_evaluation import evaluate_classification, evaluate_clustering, evaluate_regression
from modules.model_training import train_clustering, train_supervised
from modules.sample_datasets import get_housing_dataframe, get_iris_dataframe


def _load_dataset(source, uploaded_file):
    if source == "Built-in housing data":
        signature = source
        data = get_housing_dataframe()
    elif source == "Built-in iris data":
        signature = source
        data = get_iris_dataframe()
    else:
        if uploaded_file is None:
            return None
        content = uploaded_file.getvalue()
        signature = hashlib.sha256(content).hexdigest()
        try:
            data = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not read this CSV. Check that it is a valid, comma-separated text file. Details: {exc}")
            return None

    if data.empty or len(data.columns) == 0:
        st.error("The selected dataset has no rows or columns.")
        return None
    if st.session_state.get("lab_dataset_signature") != signature:
        st.session_state.lab_dataset_signature = signature
        st.session_state.lab_data = data.copy()
        st.session_state.lab_model_result = None
    return st.session_state.lab_data


def render():
    st.title("Practical ML Lab")
    st.write("Explore a sample dataset or upload a CSV. Prepare the data, train a model, review its test results, and try a prediction.")

    source = st.radio(
        "Dataset source",
        ["Built-in housing data", "Built-in iris data", "Upload CSV"],
        horizontal=True,
        key="lab_source",
    )
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="lab_upload") if source == "Upload CSV" else None
    df = _load_dataset(source, uploaded_file)
    if df is None:
        if source == "Upload CSV" and uploaded_file is None:
            st.info("Upload a CSV file, or select one of the built-in datasets to explore the lab.")
        return

    st.subheader("Dataset overview")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Rows", len(df))
    metric2.metric("Columns", len(df.columns))
    metric3.metric("Missing cells", int(df.isna().sum().sum()))
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Inspect and prepare")
    st.dataframe(basic_summary(df).rename_axis("Column"), use_container_width=True)
    action1, action2, action3 = st.columns(3)
    if action1.button("Remove rows with missing values", use_container_width=True):
        st.session_state.lab_data = df.dropna().reset_index(drop=True)
        st.session_state.lab_model_result = None
        st.rerun()
    if action2.button("Remove duplicate rows", use_container_width=True):
        st.session_state.lab_data = df.drop_duplicates().reset_index(drop=True)
        st.session_state.lab_model_result = None
        st.rerun()
    if action3.button("Restore original dataset", use_container_width=True):
        if source == "Built-in housing data":
            st.session_state.lab_data = get_housing_dataframe()
        elif source == "Built-in iris data":
            st.session_state.lab_data = get_iris_dataframe()
        else:
            if uploaded_file is None:
                st.warning("Upload the CSV again to restore it.")
                return
            st.session_state.lab_data = pd.read_csv(uploaded_file)
        st.session_state.lab_model_result = None
        st.rerun()

    st.subheader("Choose a learning task")
    task = st.radio(
        "Learning task",
        ["Regression", "Classification", "Clustering"],
        horizontal=True,
        key="lab_task",
    )
    if task == "Clustering":
        _render_clustering(df)
    else:
        _render_supervised(df, task)


def _render_clustering(df):
    numeric_df = df.select_dtypes(include=np.number).dropna()
    if len(numeric_df) < 3:
        st.warning("Clustering needs at least three complete rows with numeric features.")
        return

    features = st.multiselect(
        "Numeric features to cluster",
        numeric_df.columns.tolist(),
        default=numeric_df.columns.tolist()[: min(3, len(numeric_df.columns))],
        key="lab_cluster_features",
    )
    maximum_k = min(8, len(numeric_df) - 1)
    clusters = st.slider("Number of clusters", 2, maximum_k, min(3, maximum_k), key="lab_cluster_count")
    if st.button("Train clustering model", type="primary", key="train_cluster"):
        if not features:
            st.error("Select at least one numeric feature.")
            return
        try:
            X = numeric_df[features]
            model = train_clustering(clusters, X)
            labels = model.predict(X)
            scaled_X = model.named_steps["standardscaler"].transform(X)
            score = evaluate_clustering(scaled_X, labels)["Silhouette score"]
            st.session_state.lab_model_result = {
                "kind": "clustering",
                "features": features,
                "labels": labels,
                "score": score,
                "clusters": clusters,
            }
        except (ValueError, TypeError) as exc:
            st.error(f"Clustering could not be completed: {exc}")
            return

    result = st.session_state.get("lab_model_result")
    if result and result["kind"] == "clustering":
        if result["features"] != features or result["clusters"] != clusters:
            st.info("Clustering settings changed. Train the model again to update the results.")
            return
        st.success("Clustering model trained.")
        score_text = f"{result['score']:.3f}" if result["score"] is not None else "Not defined for these assignments"
        col1, col2 = st.columns(2)
        col1.metric("Requested clusters", result["clusters"])
        col2.metric("Silhouette score", score_text)
        clustered = numeric_df[result["features"]].copy()
        clustered["Cluster"] = result["labels"]
        st.dataframe(clustered.head(30), use_container_width=True)
        st.caption("Silhouette scores range from -1 to 1; larger values generally indicate better-separated clusters. A score may be undefined if all observations receive one cluster.")


def _render_supervised(df, task):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    default_target = df.columns[-1]
    if task == "Regression" and not pd.api.types.is_numeric_dtype(df[default_target]):
        default_target = numeric_columns[0] if numeric_columns else df.columns[0]
    target_options = df.columns.tolist()
    target = st.selectbox(
        "Target column",
        target_options,
        index=target_options.index(default_target),
        key=f"lab_target_{task.lower()}",
    )
    features = st.multiselect(
        "Numeric feature columns",
        [column for column in numeric_columns if column != target],
        default=[column for column in numeric_columns if column != target][: min(3, max(0, len(numeric_columns) - 1))],
        key=f"lab_features_{task.lower()}",
        help="This starter lab uses numeric inputs. Classification targets may be text or numeric.",
    )

    if task == "Regression":
        if not pd.api.types.is_numeric_dtype(df[target]):
            st.warning("Regression requires a numeric target. Choose a numeric column or switch to Classification.")
        models = ["Linear Regression"]
    else:
        models = ["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors"]
    model_name = st.selectbox("Model", models, key=f"lab_model_{task.lower()}")
    test_size = st.slider("Test data fraction", 0.1, 0.4, 0.2, 0.05, key=f"lab_test_size_{task.lower()}")

    if st.button("Train model", type="primary", key="train_supervised"):
        if not features:
            st.error("Choose at least one numeric feature.")
            return
        if task == "Regression" and not pd.api.types.is_numeric_dtype(df[target]):
            st.error("Regression requires a numeric target.")
            return

        work = df[features + [target]].dropna().copy()
        if len(work) < 5:
            st.error("At least five complete rows are needed for a train/test split.")
            return
        X = work[features]
        y = work[target]
        if task == "Classification" and y.nunique() < 2:
            st.error("Classification requires at least two target classes.")
            return

        stratify = None
        if task == "Classification":
            class_counts = y.value_counts()
            test_rows = int(np.ceil(len(y) * test_size))
            train_rows = len(y) - test_rows
            if class_counts.min() >= 2 and test_rows >= y.nunique() and train_rows >= y.nunique():
                stratify = y

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
            if task == "Classification" and y_train.nunique() < 2:
                raise ValueError("The training split contains only one class. Use more rows or adjust the test fraction.")
            model = train_supervised(model_name, X_train, y_train)
            predictions = model.predict(X_test)
            metrics = (
                evaluate_regression(y_test, predictions)
                if task == "Regression"
                else evaluate_classification(y_test, predictions, labels=model.classes_)
            )
        except (ValueError, TypeError) as exc:
            st.error(f"Model training failed: {exc}")
            return

        st.session_state.lab_model_result = {
            "kind": "supervised",
            "task": task,
            "model_name": model_name,
            "features": features,
            "target": target,
            "test_size": test_size,
            "model": model,
            "metrics": metrics,
        }

    result = st.session_state.get("lab_model_result")
    if not result or result["kind"] != "supervised":
        return
    if (
        result["task"] != task
        or result["model_name"] != model_name
        or result["features"] != features
        or result["target"] != target
        or result["test_size"] != test_size
    ):
        st.info("Model settings changed. Train again to refresh the evaluation and prediction.")
        return

    st.success(f"{result['model_name']} trained and evaluated on held-out data.")
    metrics = result["metrics"]
    if result["task"] == "Regression":
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error", f"{metrics['MAE']:.3f}")
        col2.metric("Mean Squared Error", f"{metrics['MSE']:.3f}")
        col3.metric("R-squared", f"{metrics['R-squared']:.3f}")
        st.caption("MAE and MSE measure prediction error (lower is better); R-squared compares performance with a mean-only baseline.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['Precision']:.3f}")
        col3.metric("Recall", f"{metrics['Recall']:.3f}")
        col4.metric("F1-score", f"{metrics['F1-score']:.3f}")
        st.write("**Confusion matrix**")
        class_labels = result["model"].classes_
        st.dataframe(
            pd.DataFrame(metrics["Confusion matrix"], index=class_labels, columns=class_labels),
            use_container_width=True,
        )
        st.caption("Rows are actual classes and columns are predicted classes.")

    st.subheader("Try a prediction")
    st.write("Enter new numeric feature values. The model and these controls stay available after training.")
    input_values = {}
    model = result["model"]
    columns = st.columns(min(3, len(result["features"])))
    for index, feature in enumerate(result["features"]):
        with columns[index % len(columns)]:
            input_values[feature] = st.number_input(
                feature,
                value=float(df[feature].mean()),
                key=f"lab_predict_{feature}",
            )
    if st.button("Predict", key="lab_predict_button"):
        prediction = model.predict(pd.DataFrame([input_values], columns=result["features"]))[0]
        st.info(f"Predicted **{result['target']}**: **{prediction}**")
