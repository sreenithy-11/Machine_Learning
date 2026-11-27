from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


def train_supervised(model_name, X_train, y_train):
    if model_name == "Linear Regression":
        return make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    if model_name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    if model_name == "K-Nearest Neighbors":
        neighbors = min(5, len(X_train))
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=neighbors)).fit(X_train, y_train)
    raise ValueError("Unsupported supervised model")


def train_clustering(n_clusters, X):
    return make_pipeline(StandardScaler(), KMeans(n_clusters=n_clusters, random_state=42, n_init=10)).fit(X)
