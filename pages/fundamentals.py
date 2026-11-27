import streamlit as st


TOPICS = {
    "1. Machine Learning: the idea": {
        "summary": "Machine Learning (ML) is a way to build computer programs that learn useful patterns from examples. Instead of writing a separate rule for every case, we provide data and a learning method; the resulting model uses the patterns it learned to make predictions or decisions.",
        "example": "Rather than hand-writing every clue that makes an email suspicious, a spam filter learns from examples previously marked “spam” or “not spam.”",
        "terms": {
            "Data": "Recorded examples used for learning.",
            "Feature": "An input measurement or property, such as a message's word count.",
            "Label / target": "The answer we want the model to learn to predict.",
            "Model": "The learned relationship between input features and an output.",
            "Prediction": "The model's output for new input data.",
        },
        "takeaway": "A model learns from examples; it does not automatically understand the problem like a person.",
    },
    "2. Artificial Intelligence, ML, and Deep Learning": {
        "summary": "Artificial Intelligence (AI) is the broad field of making machines perform tasks associated with intelligent behavior. Machine Learning is one way to build AI systems by learning from data. Deep Learning is a family of ML methods that uses neural networks with many layers.",
        "example": "A voice assistant is an AI product. It may use ML to recognize speech, and a deep neural network may be part of that speech-recognition system.",
        "terms": {
            "AI": "The broad goal or field of intelligent computer behavior.",
            "ML": "Methods that learn patterns from data.",
            "Deep Learning": "ML based on multi-layer neural networks, often requiring substantial data and computing resources.",
        },
        "takeaway": "Deep Learning is part of ML, and ML is part of the wider AI field.",
    },
    "3. The three main learning settings": {
        "summary": "The learning setting depends on what feedback is available. Supervised learning has known answers for training examples. Unsupervised learning has no answer column and looks for structure. Reinforcement learning learns a sequence of actions using feedback from an environment.",
        "example": "Predicting a known house price is supervised regression; grouping shoppers is unsupervised clustering; learning moves in a game using scores is reinforcement learning.",
        "terms": {
            "Supervised": "Learn a mapping from features to known labels or numeric targets.",
            "Unsupervised": "Describe structure in inputs without target labels.",
            "Reinforcement": "Choose actions over time to gain long-term reward.",
        },
        "takeaway": "First ask what kind of feedback the problem provides; that usually identifies the learning setting.",
    },
    "4. Supervised learning: regression and classification": {
        "summary": "In supervised learning, training rows contain input features and a known target. Regression predicts a number, while classification predicts a category. The model learns from training data and is evaluated on examples it did not train on.",
        "example": "Use area, location, and age to estimate a home's price (regression), or use message features to classify an email as spam or not spam (classification).",
        "terms": {
            "Training set": "Examples used to fit model parameters.",
            "Test set": "Held-back examples used to estimate performance on unseen data.",
            "Regression": "Prediction of a continuous numeric value.",
            "Classification": "Prediction of a discrete class or category.",
        },
        "takeaway": "Choose regression for a numeric quantity and classification for a category.",
    },
    "5. Unsupervised learning": {
        "summary": "Unsupervised learning works with inputs that do not have a target label. It can group similar records, find unusual observations, or represent many correlated measurements with fewer dimensions. A discovered group is a pattern, not automatically a meaningful real-world category.",
        "example": "A retailer can cluster customers by purchasing behavior, then inspect the groups to decide whether they represent useful shopping patterns.",
        "terms": {
            "Clustering": "Grouping observations that are similar under a chosen distance or similarity measure.",
            "Dimensionality reduction": "Representing data with fewer variables while retaining useful structure.",
            "Anomaly": "An observation that differs substantially from the majority of data.",
        },
        "takeaway": "The algorithm discovers structure; a person still needs to interpret whether that structure is useful.",
    },
    "6. Reinforcement learning": {
        "summary": "Reinforcement learning (RL) studies how an agent can learn to act through repeated interaction. At each step, the agent observes a state, chooses an action, receives a reward, and moves to another state. Good actions may have delayed benefits, so the agent aims to maximize total future reward.",
        "example": "A game-playing agent may receive a reward for winning and a penalty for losing. It tries different moves and gradually favors actions that lead to better outcomes.",
        "terms": {
            "Agent": "The learner or decision-maker.",
            "Environment": "The world with which the agent interacts.",
            "State and action": "The situation observed and a choice available in that situation.",
            "Reward / policy": "Feedback after acting, and the strategy for choosing actions.",
        },
        "takeaway": "RL feedback is about consequences of actions, often across multiple steps.",
    },
    "7. A practical ML workflow": {
        "summary": "A reliable project starts with a clear question, not with picking an algorithm. Define the outcome, inspect and prepare representative data, build a simple baseline, train using only training data, evaluate on held-out data, then improve and monitor the model.",
        "example": "For predicting house prices: define the price and intended use; inspect historical sales; prepare features; split data; train a regression model; check its errors on held-out homes; then decide whether the accuracy is useful.",
        "terms": {
            "Baseline": "A simple reference prediction used to judge whether a more complex model helps.",
            "Data leakage": "Information unavailable at prediction time accidentally entering training or evaluation.",
            "Generalization": "How well a model performs on new examples from the intended population.",
            "Deployment and monitoring": "Using a model in practice and checking whether its data and performance change.",
        },
        "takeaway": "A good test set must stay separate from the model-fitting process; repeated tuning against it makes its score optimistic.",
    },
    "8. Data preparation and features": {
        "summary": "Real data can contain missing values, duplicates, inconsistent formats, and features measured on very different scales. Inspect these issues before fitting a model. Feature selection should use information that would genuinely be available when predictions are made.",
        "example": "For customer churn prediction, use information collected before the customer leaves. A column recording the cancellation date would reveal the answer and cause data leakage.",
        "terms": {
            "Missing value": "An absent measurement that may need investigation, removal, or imputation.",
            "Scaling": "Changing numeric feature ranges; often important for distance-based models.",
            "Encoding": "Converting categorical values into a representation a model can use.",
            "Data leakage": "Accidentally giving a model target or future information.",
        },
        "takeaway": "Preprocessing choices should be learned from training data and then applied consistently to test and future data.",
    },
    "9. Evaluating model quality": {
        "summary": "A model is not good merely because it fits its training examples. Evaluation uses held-out data and metrics that match the problem. Regression metrics describe numeric errors. Classification metrics summarize different kinds of correct and incorrect decisions; accuracy alone can be misleading when classes are imbalanced.",
        "example": "If only 1 in 100 transactions is fraudulent, a model that always predicts “not fraud” is 99% accurate but detects no fraud. Recall and precision help expose that failure.",
        "terms": {
            "MAE": "Average absolute difference between actual and predicted numeric values; lower is generally better.",
            "R-squared": "A measure of variance explained relative to a mean-only baseline; it can be negative on test data.",
            "Precision": "Of predicted positives, the proportion that are truly positive.",
            "Recall": "Of actual positives, the proportion the model finds.",
            "F1-score": "A combined measure of precision and recall.",
        },
        "takeaway": "Choose metrics based on the cost of different errors and the real goal of the project.",
    },
    "10. Applications and responsible use": {
        "summary": "ML can support recommendations, medical screening, forecasting, image recognition, speech tools, and fraud review. A model's prediction is not automatically a fact or a fair decision. Data quality, privacy, bias, explainability, human oversight, and the impact of errors should all be considered.",
        "example": "A model used to prioritize applications should be checked for systematic errors across relevant groups, and people affected by its decisions may need a way to request human review.",
        "terms": {
            "Bias": "Systematic error that can arise from data, measurement, or model design.",
            "Privacy": "Responsible collection, use, and protection of personal information.",
            "Human oversight": "Appropriate review and accountability for consequential decisions.",
        },
        "takeaway": "Use models as tools within a responsible process, especially where mistakes can affect people.",
    },
}


def render():
    st.title("Machine Learning Fundamentals")
    st.write(
        "A concise learning path from the basic idea of ML to data, workflows, and evaluation. "
        "Work through the topics in order, or choose one you want to review."
    )

    st.info("**Learning goal:** Be able to identify the learning setting, describe the ML workflow, and explain how a model should be evaluated.")
    topic = st.selectbox("Learning path", list(TOPICS.keys()), key="fundamentals_topic")
    lesson = TOPICS[topic]

    st.subheader(topic.split(". ", 1)[1])
    st.write(lesson["summary"])

    with st.container(border=True):
        st.markdown("#### A practical example")
        st.write(lesson["example"])

    st.markdown("#### Key terms")
    for term, definition in lesson["terms"].items():
        st.markdown(f"- **{term}:** {definition}")

    st.success(f"**Remember:** {lesson['takeaway']}")

    st.divider()
    st.subheader("At a glance: choosing a learning setting")
    comparison = {
        "Setting": ["Supervised", "Unsupervised", "Reinforcement"],
        "Training feedback": ["Known target for each example", "No target column", "Reward after actions"],
        "Typical task": ["Predict a value or class", "Find groups or structure", "Learn a strategy over time"],
        "Example": ["Estimate house prices", "Group similar customers", "Learn to play a game"],
    }
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.subheader("The project workflow")
    steps = [
        ("1. Define", "Specify the question, target, intended users, and what a useful result means."),
        ("2. Inspect", "Check data coverage, units, missing values, duplicates, and possible bias."),
        ("3. Prepare", "Create usable features and keep preprocessing consistent."),
        ("4. Split", "Reserve test data; use training data for fitting and validation/tuning."),
        ("5. Train", "Start with a simple baseline, then fit a suitable algorithm."),
        ("6. Evaluate", "Use held-out data and metrics that reflect the cost of errors."),
        ("7. Use and monitor", "Deploy only when appropriate; monitor changes and reassess performance."),
    ]
    for heading, description in steps:
        st.markdown(f"**{heading}** — {description}")
