import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs, make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans


def render():
    st.title("Interactive Visualizations")
    st.write("Change a setting and watch how a model's predictions or groups respond.")
    regression_tab, clustering_tab, tree_tab, classification_tab = st.tabs(
        ["Linear Regression", "K-Means", "Decision Tree", "Classification"]
    )

    with regression_tab:
        render_regression()
    with clustering_tab:
        render_clustering()
    with tree_tab:
        render_tree()
    with classification_tab:
        render_classification()


def render_regression():
    st.subheader("Fit a line to sample data")
    st.write("Adjust the blue line yourself, then compare it with the least-squares line fitted to the same points.")
    st.caption("Graph guide: each dot is one example. The horizontal axis is the input x and the vertical axis is the observed output y. The two lines show your chosen relationship and the model's best-fit relationship.")
    col1, col2 = st.columns(2)
    slope = col1.slider("Your line: slope", -3.0, 5.0, 2.0, 0.1, key="reg_slope")
    intercept = col2.slider("Your line: intercept", -20.0, 20.0, 5.0, 1.0, key="reg_intercept")
    noise = st.slider("Data noise", 0.0, 30.0, 10.0, key="reg_noise")

    x = np.linspace(0, 20, 30)
    y = 2.5 * x + 8 + np.random.default_rng(42).normal(0, noise, len(x))
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    manual = slope * x + intercept
    prediction_x = st.slider("Try a prediction at x =", 0.0, 20.0, 10.0, 0.5, key="reg_prediction_x")

    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", name="Sample data")
    fig.add_scatter(x=x, y=manual, mode="lines", name="Your line")
    fig.add_scatter(x=x, y=model.predict(x.reshape(-1, 1)), mode="lines", name="Fitted line")
    fig.add_scatter(
        x=[prediction_x],
        y=[model.predict([[prediction_x]])[0]],
        mode="markers",
        marker={"size": 12, "symbol": "diamond"},
        name="Fitted prediction",
    )
    fig.update_layout(xaxis_title="Input x", yaxis_title="Output y", height=430)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("How to read this graph"):
        st.markdown(
            "- **Dots:** sample observations; they do not have to fall exactly on a line.\n"
            "- **Your line:** the slope controls how quickly predictions rise or fall; the intercept is its predicted value when x is zero.\n"
            "- **Fitted line:** the line learned from all displayed points by minimizing squared vertical errors.\n"
            "- **Diamond:** the fitted model's prediction for the selected x. The model's prediction is an estimate, not a guarantee."
        )

    fitted_prediction = model.predict([[prediction_x]])[0]
    manual_prediction = slope * prediction_x + intercept
    left, right = st.columns(2)
    left.metric("Your line predicts", f"{manual_prediction:.2f}", help=f"y = {slope:.1f}x + {intercept:.1f}")
    right.metric("Fitted line predicts", f"{fitted_prediction:.2f}", help=f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
    st.caption("The fitted line minimizes the sum of squared residuals on these sample points.")


def render_clustering():
    st.subheader("Group points with K-Means")
    st.write("K-Means assigns each point to its nearest center and repeatedly moves centers to the mean of their assigned points.")
    st.caption("The axes are two generated numeric features with no real-world units. The clustering calculation scales both features first, so neither axis dominates just because of its numeric range.")
    k = st.slider("Number of clusters (K)", 2, 6, 3, key="cluster_k")
    points, _ = make_blobs(n_samples=180, centers=4, cluster_std=1.2, random_state=42)
    scaler = StandardScaler().fit(points)
    scaled_points = scaler.transform(points)
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_points)
    labels = model.labels_
    centers = scaler.inverse_transform(model.cluster_centers_)

    df = pd.DataFrame({"X": points[:, 0], "Y": points[:, 1], "Cluster": [f"Cluster {i + 1}" for i in labels]})
    fig = px.scatter(df, x="X", y="Y", color="Cluster", title="K-Means assignments")
    fig.add_scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode="markers",
        marker={"symbol": "x", "size": 16, "color": "black"},
        name="Cluster centers",
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("How to read this graph and score"):
        st.markdown(
            "- **Each dot:** one generated observation; nearby dots have similar feature values.\n"
            "- **Color:** the cluster assigned by K-Means. Cluster numbers/colors are identifiers, not rankings or known categories.\n"
            "- **Black X:** the center (centroid) of a cluster, transformed back to the original plot coordinates.\n"
            "- **Within-cluster squared distance:** the sum of squared distances to cluster centers in scaled feature space. It usually falls as K increases, so compare it alongside the plot rather than choosing K from this number alone."
        )
    st.metric("Within-cluster squared distance", f"{model.inertia_:.1f}")
    st.info("The metric above is the total squared distance to cluster centers in scaled feature space. Compare different K values, but also inspect whether the groups make sense.")


def render_tree():
    st.subheader("How a decision tree splits data")
    st.write("Every internal node asks a feature question; the branches carry answers to later questions, and a leaf provides the predicted class.")
    st.caption("This demo uses a generated two-feature dataset. Feature 1 and Feature 2 are coordinates created for the illustration, not measurements with a real-world meaning.")
    x, y = make_moons(n_samples=200, noise=0.12, random_state=42)
    depth = st.slider("Maximum tree depth", 1, 5, 3, key="tree_depth")
    model = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(x, y)

    fig, ax = plt.subplots(figsize=(13, 6))
    plot_tree(
        model,
        filled=True,
        rounded=True,
        feature_names=["Feature 1", "Feature 2"],
        class_names=["Class 0", "Class 1"],
        ax=ax,
        fontsize=8,
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("How to read a tree node"):
        st.markdown(
            "- **Feature ≤ threshold:** the question asked at that node. Follow the left branch when true and the right branch when false.\n"
            "- **Gini:** how mixed the classes are at that node; 0 means the node contains only one class.\n"
            "- **Samples:** how many training examples reached the node.\n"
            "- **Value:** the count of examples from each class, in the order shown by the legend.\n"
            "- **Color:** the majority predicted class; darker shading indicates a purer node.\n"
            "- **Leaf:** the final prediction after following the questions from the root."
        )
    st.metric("Training accuracy", f"{accuracy_score(y, model.predict(x)):.1%}")
    st.caption("A deeper tree can represent more detailed rules, but it can also memorize training examples. This training score alone does not measure performance on new data.")


def render_classification():
    st.subheader("Compare classification boundaries")
    st.write("The colored background shows each model's predicted class over the feature plane; points are the example observations.")
    st.caption("This is a generated two-feature dataset. Its coordinates are for demonstrating boundaries and do not represent named real-world properties.")
    model_name = st.selectbox(
        "Classifier",
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Support Vector Machine"],
        key="boundary_model",
    )
    noise = st.slider("Dataset noise", 0.05, 0.40, 0.18, 0.01, key="boundary_noise")
    x, y = make_moons(n_samples=250, noise=noise, random_state=42)

    if model_name == "Logistic Regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif model_name == "Decision Tree":
        depth = st.slider("Tree depth", 1, 10, 4, key="boundary_depth")
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    elif model_name == "K-Nearest Neighbors":
        neighbors = st.slider("Number of neighbors", 1, 25, 7, key="boundary_neighbors")
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=neighbors))
    else:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

    model.fit(x, y)
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    regions = model.predict(grid).reshape(grid_x.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=regions,
            showscale=False,
            opacity=0.32,
            contours={"start": 0, "end": 1, "size": 1, "coloring": "fill"},
            line={"width": 0},
            name="Predicted regions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode="markers",
            marker={"color": y, "colorscale": [[0, "#4f72d8"], [1, "#e07b45"]], "line": {"width": 0.5, "color": "white"}},
            name="Observed examples",
        )
    )
    fig.update_layout(title="Learned class regions", xaxis_title="Feature 1", yaxis_title="Feature 2", height=470)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("How to read the boundary graph"):
        st.markdown(
            "- **Axes:** the two generated input features used by the classifier.\n"
            "- **Background region:** the class the model predicts for locations in that part of the feature plane.\n"
            "- **Dots:** generated training examples; their color is their known class. Dots in a region of the other color are examples the fitted model gets wrong.\n"
            "- **Boundary:** where the predicted class changes. Its shape depends on the selected algorithm and settings.\n"
            "- **Training accuracy:** the share of these same training examples classified correctly. It is not a test score and can look high when a model overfits."
        )
    st.metric("Training accuracy", f"{accuracy_score(y, model.predict(x)):.1%}")
    st.caption("Training accuracy is shown only to illustrate the fitted model. Use held-out data to compare real generalization performance.")
