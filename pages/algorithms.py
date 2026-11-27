import streamlit as st


ALGORITHMS = {
    "Linear Regression": {
        "group": "Regression",
        "summary": "A supervised method for estimating a continuous numeric outcome using a linear relationship between input features and the target.",
        "intuition": "Find the line (or, with multiple features, the plane) that keeps prediction errors small across the training examples.",
        "steps": [
            "Choose numeric features and a numeric target.",
            "Fit an intercept and coefficient for each feature.",
            "Predict by multiplying each feature by its coefficient and adding the results.",
        ],
        "math": r"\hat y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p",
        "example": "Estimate rent from apartment size. A positive size coefficient means that, in the fitted model, larger apartments tend to have higher rent, holding other included features fixed.",
        "uses": "Price and demand estimation, trend analysis, and interpretable numeric baselines.",
        "pros": "Fast, easy to explain, and useful as a baseline.",
        "cons": "A straight-line relationship may be too simple; outliers and strongly correlated features can affect fitted coefficients.",
        "when": "Use when the target is numeric and a simple, interpretable relationship is useful. Check residuals rather than assuming linearity is adequate.",
    },
    "Multiple Linear Regression": {
        "group": "Regression",
        "summary": "Linear regression with several input features. Each coefficient estimates the feature's association with the target while the other included features are held fixed.",
        "intuition": "Combine several pieces of evidence—such as size, age, and location—to estimate one numeric outcome.",
        "steps": [
            "Select a numeric target and relevant input features.",
            "Estimate one coefficient per feature plus an intercept.",
            "Combine weighted feature values to produce a prediction.",
        ],
        "math": r"\hat y = \beta_0 + \sum_{j=1}^{p}\beta_j x_j",
        "example": "A home-value baseline can use floor area, age, and number of bedrooms. The coefficients are associations in the data, not proof that changing a feature causes a price change.",
        "uses": "Sales, energy consumption, demand, and other numeric outcomes with multiple useful predictors.",
        "pros": "Uses several predictors and remains relatively interpretable.",
        "cons": "Correlated predictors make individual coefficients unstable; missing nonlinear effects can reduce accuracy.",
        "when": "Use when several numeric features plausibly relate to a continuous target and you need an understandable starting point.",
    },
    "Logistic Regression": {
        "group": "Classification",
        "summary": "A supervised classification method that converts a linear score into class probabilities. The name includes “regression,” but it is commonly used to classify.",
        "intuition": "Estimate how likely an example belongs to a class, then apply a decision threshold to choose a label.",
        "steps": [
            "Calculate a weighted sum of the input features.",
            "Transform the score into a probability (sigmoid for two classes; softmax-style handling for multiple classes).",
            "Choose the class with the highest probability or apply a threshold selected for the task.",
        ],
        "math": r"P(y=1\mid x)=\frac{1}{1+e^{-(\beta_0+\beta^T x)}}",
        "example": "For a spam filter, estimate the probability an email is spam. A threshold of 0.5 is a default, not necessarily the best choice when false positives and missed spam have different costs.",
        "uses": "Risk scoring, spam filtering, and interpretable binary or multiclass classification.",
        "pros": "Fast, provides probabilities, and often works well as a baseline.",
        "cons": "A linear decision boundary may not capture complex patterns; probabilities can need calibration.",
        "when": "Use for classification when speed and a useful baseline matter. Scale numeric features and choose the threshold using validation data.",
    },
    "K-Nearest Neighbors (KNN)": {
        "group": "Classification",
        "summary": "A distance-based method that predicts using the labels of the closest training examples. It can also be adapted to regression by averaging nearby target values.",
        "intuition": "Examples that are close under a meaningful distance measure are assumed to have similar outcomes.",
        "steps": [
            "Scale numeric features so large-unit features do not dominate distance.",
            "Choose the number of neighbors, K, and a distance measure.",
            "Find the nearest training examples and vote (or average) their outcomes.",
        ],
        "math": r"d(x,z)=\sqrt{\sum_j(x_j-z_j)^2}",
        "example": "Classify a flower by finding the five most similar measured flowers in the training set and taking their majority class.",
        "uses": "Small or medium datasets, educational demonstrations, and local pattern classification.",
        "pros": "Simple intuition; little model fitting; can represent nonlinear boundaries.",
        "cons": "Prediction can be slow on large datasets, sensitive to scale and irrelevant features, and affected by the choice of K.",
        "when": "Use when a meaningful distance exists and the dataset is manageable. Scale features and compare several K values on validation data.",
    },
    "Decision Tree": {
        "group": "Classification",
        "summary": "A model made of feature-based questions. Each split sends records down a branch until a leaf provides a predicted class or number.",
        "intuition": "Build a sequence of questions that separates examples with different outcomes.",
        "steps": [
            "At each node, test candidate feature splits.",
            "Choose a split that improves class purity or reduces target variation.",
            "Repeat until a stopping rule is met; predict using the leaf's training outcomes.",
        ],
        "math": r"\text{Gini}=1-\sum_{k=1}^{K}p_k^2",
        "example": "A simple loan-risk tree might first ask whether existing debt is high, then whether income is stable, and use the final leaf to estimate a risk class.",
        "uses": "Readable decision rules, tabular classification, and simple nonlinear relationships.",
        "pros": "Easy to visualize; handles nonlinear feature interactions; little scaling needed.",
        "cons": "A deep tree can memorize training data and small data changes can produce a different tree.",
        "when": "Use when rule-like explanations are helpful. Limit depth or leaf size and evaluate on held-out data.",
    },
    "Random Forest": {
        "group": "Classification",
        "summary": "An ensemble of decision trees trained on varied samples and feature subsets. Their predictions are combined by voting for classification or averaging for regression.",
        "intuition": "A group of diverse trees is often more reliable than trusting one tree's decisions.",
        "steps": [
            "Draw bootstrap samples from the training set.",
            "Train each tree while considering random feature subsets at splits.",
            "Aggregate tree outputs into one prediction.",
        ],
        "math": r"\hat y_{\mathrm{class}}=\operatorname{mode}(T_1(x),\ldots,T_B(x))",
        "example": "For disease screening, many trees learn different patterns from training records; the forest returns the majority class or an average probability.",
        "uses": "General-purpose tabular classification and regression when a strong baseline is needed.",
        "pros": "Usually robust, captures nonlinear patterns, and needs little feature scaling.",
        "cons": "Less transparent than one small tree, can use more memory, and feature importance should not be mistaken for causation.",
        "when": "Use when predictive performance is more important than a short list of rules. Tune tree depth and number of trees using validation.",
    },
    "Support Vector Machine (SVM)": {
        "group": "Classification",
        "summary": "A classification method that seeks a separating boundary with a wide margin. Kernel functions can represent more complex boundaries.",
        "intuition": "Place a boundary between classes while keeping it as far as possible from the closest training examples.",
        "steps": [
            "Scale numeric features.",
            "Choose a linear boundary or a kernel and tune regularization.",
            "Fit the maximum-margin separator; nearby support vectors determine its position.",
        ],
        "math": r"\text{margin width}\propto \frac{1}{\|w\|}",
        "example": "A linear SVM can separate two groups of measurements with a straight boundary; an RBF kernel can model a curved separation.",
        "uses": "Classification with clear margins and small-to-medium datasets, including high-dimensional feature spaces.",
        "pros": "Can work well in high-dimensional spaces; kernels model nonlinear patterns.",
        "cons": "Scaling and parameter tuning matter; kernel models can be slow on large datasets and are not very transparent.",
        "when": "Use when feature count is substantial and dataset size is manageable. Compare kernels and parameters on validation data.",
    },
    "Naive Bayes": {
        "group": "Classification",
        "summary": "A family of probabilistic classifiers based on Bayes' theorem, using a simplifying assumption that features are conditionally independent given the class.",
        "intuition": "Estimate how likely each feature value is under each class, then combine the evidence to compare class probabilities.",
        "steps": [
            "Estimate the prior probability of each class.",
            "Estimate feature likelihoods for each class.",
            "Combine prior and likelihoods to select the most probable class.",
        ],
        "math": r"P(c\mid x)\propto P(c)\prod_j P(x_j\mid c)",
        "example": "A text classifier counts how frequently words appear in spam and non-spam messages, then estimates which class best explains a new message.",
        "uses": "Text categorization, spam filters, and fast classification baselines.",
        "pros": "Fast, works with high-dimensional text features, and can perform well with limited training data.",
        "cons": "The independence assumption is often unrealistic; probability estimates can be poorly calibrated.",
        "when": "Use as a fast baseline, particularly for text. Select a variant appropriate to count, binary, or continuous features.",
    },
    "K-Means Clustering": {
        "group": "Clustering",
        "summary": "An unsupervised algorithm that divides numeric observations into a chosen number K of clusters around their centroids.",
        "intuition": "Each group is represented by its center; points are assigned to their nearest center.",
        "steps": [
            "Choose K and initialize K centers.",
            "Assign each observation to the nearest center.",
            "Move each center to the mean of its assigned observations and repeat until assignments stabilize.",
        ],
        "math": r"\min_{\mu_1,\ldots,\mu_K}\sum_i\min_k\|x_i-\mu_k\|^2",
        "example": "Group shoppers by scaled spending and visit frequency. Examine cluster profiles afterward; cluster numbers have no built-in business meaning.",
        "uses": "Exploratory grouping, image-color compression, and segmentation of numeric data.",
        "pros": "Simple, relatively fast, and easy to visualize in two dimensions.",
        "cons": "Must choose K; sensitive to scaling and outliers; works best for roughly compact, similarly sized clusters.",
        "when": "Use when numeric data and distance-based groups make sense. Scale features and compare plausible K values.",
    },
    "Hierarchical Clustering": {
        "group": "Clustering",
        "summary": "A family of methods that builds a nested hierarchy of groups. Agglomerative clustering starts with individual observations and repeatedly merges groups.",
        "intuition": "Show which observations or groups join together at different levels of similarity.",
        "steps": [
            "Choose a distance measure and a linkage rule for distance between groups.",
            "Merge the closest groups repeatedly.",
            "Inspect a dendrogram and cut it at a useful level to get clusters.",
        ],
        "math": "Linkage rules include single (nearest pair), complete (farthest pair), average, and Ward's variance-based criterion.",
        "example": "Group related documents and inspect a dendrogram to see broad topics and smaller subtopics.",
        "uses": "Exploratory analysis when a hierarchy is informative and the dataset is not too large.",
        "pros": "Does not require setting the final number of clusters at the start; dendrogram provides a useful overview.",
        "cons": "Can be computationally expensive; results depend on distance and linkage; early merge decisions are not usually undone.",
        "when": "Use for relatively small datasets where nested group structure is useful.",
    },
    "DBSCAN": {
        "group": "Clustering",
        "summary": "A density-based clustering method that finds dense regions and marks isolated observations as noise. It does not require the number of clusters in advance.",
        "intuition": "Points in crowded neighborhoods can grow a cluster; sparse points may be noise or connect nearby dense regions.",
        "steps": [
            "Choose neighborhood radius (epsilon) and minimum neighbors (min_samples).",
            "Mark points with enough neighbors as core points.",
            "Grow clusters through neighboring core points; label remaining points as border points or noise.",
        ],
        "math": "A core point has at least min_samples observations within distance epsilon.",
        "example": "Find dense geographic activity zones while flagging isolated GPS records as possible noise.",
        "uses": "Spatial data, anomaly exploration, and clusters with irregular shapes.",
        "pros": "Can identify arbitrarily shaped clusters and noise without specifying K.",
        "cons": "Parameter choice is difficult when densities vary; feature scaling and distance choice matter.",
        "when": "Use when clusters are dense regions separated by sparse areas and outlier marking is useful.",
    },
    "Principal Component Analysis (PCA)": {
        "group": "Dimensionality Reduction",
        "summary": "A linear method that transforms correlated numeric features into fewer uncorrelated directions, ordered by how much variance they explain.",
        "intuition": "Rotate the coordinate system so the first new axis captures as much variation as possible, then keep a selected number of axes.",
        "steps": [
            "Center the numeric features and usually scale them when their units differ.",
            "Find principal directions from the data's covariance structure.",
            "Project observations onto the first components and inspect explained variance.",
        ],
        "math": r"\text{Explained variance ratio}_j=\frac{\lambda_j}{\sum_k\lambda_k}",
        "example": "Reduce many correlated sensor readings to two components for a scatter plot, while checking how much information the projection retains.",
        "uses": "Visualization, compression, noise reduction, and preprocessing before some models.",
        "pros": "Can reduce redundancy and make high-dimensional data easier to inspect.",
        "cons": "Components are combinations of original features and can be hard to interpret; PCA captures linear variance, not necessarily task relevance.",
        "when": "Use for numeric features when compact representation is valuable. Scale as appropriate and inspect explained variance.",
    },
}


def render():
    st.title("Machine Learning Algorithms")
    st.write(
        "Choose a learning family and algorithm. Each guide covers its intuition, "
        "main steps, a practical example, trade-offs, and when it is a reasonable choice."
    )
    family = st.radio(
        "Learning family",
        ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
        horizontal=True,
        key="algorithm_family",
    )

    if family == "Reinforcement Learning":
        render_reinforcement_learning()
        return

    if family == "Supervised Learning":
        names = [name for name, details in ALGORITHMS.items() if details["group"] in {"Regression", "Classification"}]
        st.caption("Supervised models learn from examples with a known target.")
    else:
        names = [name for name, details in ALGORITHMS.items() if details["group"] in {"Clustering", "Dimensionality Reduction"}]
        st.caption("Unsupervised methods explore inputs without a target label.")

    selected = st.selectbox("Algorithm", names, key="selected_algorithm")
    details = ALGORITHMS[selected]
    st.subheader(selected)
    st.caption(f"Category: {details['group']}")
    st.write(details["summary"])

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("#### Intuition")
            st.write(details["intuition"])
            st.markdown("#### How it works")
            for step in details["steps"]:
                st.markdown(f"- {step}")
    with right:
        with st.container(border=True):
            st.markdown("#### Example")
            st.write(details["example"])
            st.markdown("#### Common uses")
            st.write(details["uses"])
            st.markdown("#### Core idea")
            st.latex(details["math"])

    tradeoffs, choice = st.columns(2)
    with tradeoffs:
        with st.container(border=True):
            st.markdown("#### Strengths")
            st.write(details["pros"])
            st.markdown("#### Limitations")
            st.write(details["cons"])
    with choice:
        with st.container(border=True):
            st.markdown("#### When is it a reasonable choice?")
            st.write(details["when"])


def render_reinforcement_learning():
    st.caption("Reinforcement learning learns a sequence of actions through feedback from an environment.")
    st.subheader("The interaction loop")
    st.markdown(
        "**Agent** observes a **state** → chooses an **action** → the **environment** "
        "returns a new state and a **reward** → the agent updates its strategy."
    )

    terms = {
        "Agent": "The learner or decision-maker.",
        "Environment": "The system or world the agent interacts with.",
        "State": "A representation of the current situation.",
        "Action": "A choice available to the agent in a state.",
        "Reward": "Feedback indicating the immediate value of an action.",
        "Policy": "The agent's strategy for choosing actions.",
    }
    st.markdown("#### Key parts")
    for term, definition in terms.items():
        st.markdown(f"- **{term}:** {definition}")

    st.subheader("Q-Learning: a basic value-learning method")
    st.write(
        "Q-learning estimates how useful an action is in a state. After taking an action, "
        "it updates the estimate using the reward just received and the best estimated "
        "future value in the next state."
    )
    st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]")
    st.markdown(
        "- **α (learning rate):** how much a new experience changes the estimate.\n"
        "- **γ (discount factor):** how much future rewards matter.\n"
        "- **Exploration:** trying actions to learn their outcomes; **exploitation:** choosing actions currently believed to be best."
    )
    st.info("Example: a maze agent can receive a positive reward for reaching the goal and a small step cost. It gradually learns routes with higher total reward.")

    with st.expander("Strengths and limitations"):
        st.markdown(
            "**Useful for:** sequential decisions where actions change future states and rewards provide feedback.\n\n"
            "**Challenges:** learning can require many interactions, reward design is important, and a learned policy should be tested safely before real-world use."
        )
