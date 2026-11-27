import streamlit as st

QUESTIONS = [
    {"topic":"ML Fundamentals","q":"What is Machine Learning?","options":["Learning patterns from data","A database","A programming language","A type of computer only"],"answer":0,"explanation":"ML enables systems to learn patterns and improve at tasks using data."},
    {"topic":"ML Fundamentals","q":"Which is broader: AI or ML?","options":["ML","AI","Both identical","Neither"],"answer":1,"explanation":"AI is the broad field; ML is one approach within AI."},
    {"topic":"ML Fundamentals","q":"Which step usually comes before model training?","options":["Deployment","Data preprocessing","Selling data","Guessing results"],"answer":1,"explanation":"Data preparation prepares raw data for reliable model training."},
    {"topic":"ML Fundamentals","q":"What is supervised learning?","options":["Learning with labels","Learning without labels","Learning from rewards only","Deleting data"],"answer":0,"explanation":"Supervised learning uses input-output pairs or labeled examples."},
    {"topic":"ML Fundamentals","q":"What is unsupervised learning?","options":["Learning from a teacher","Finding patterns without labels","Only classification","Only reinforcement"],"answer":1,"explanation":"Unsupervised algorithms discover structure in unlabeled data."},
    {"topic":"Supervised Learning","q":"Linear Regression mainly predicts what?","options":["A continuous value","A cluster","A reward","An image"],"answer":0,"explanation":"Linear regression is commonly used for continuous target values."},
    {"topic":"Supervised Learning","q":"Logistic Regression is mainly used for:","options":["Clustering","Classification","Sorting files","Dimensionality reduction"],"answer":1,"explanation":"Despite its name, logistic regression predicts class probabilities."},
    {"topic":"Supervised Learning","q":"A decision tree makes decisions using:","options":["Textures","Feature-based splits","Rewards only","Pixels only"],"answer":1,"explanation":"Decision trees recursively split on feature values."},
    {"topic":"Supervised Learning","q":"Random Forest combines:","options":["Many decision trees","One linear line","Only K-Means","SQL queries"],"answer":0,"explanation":"Random Forest aggregates predictions from many decision trees."},
    {"topic":"Supervised Learning","q":"KNN classifies a sample using:","options":["Nearest labeled neighbors","Random labels","Cloud storage","Reinforcement"],"answer":0,"explanation":"KNN uses nearby training observations to vote on a class."},
    {"topic":"Supervised Learning","q":"SVM tries to maximize:","options":["Margin","Dataset size","File size","Missing values"],"answer":0,"explanation":"SVM seeks a boundary with a maximal separation margin."},
    {"topic":"Supervised Learning","q":"Naive Bayes is based on:","options":["Bayes' theorem","Gradient descent only","Clustering","Sorting"],"answer":0,"explanation":"Naive Bayes applies probability theory with a feature independence assumption."},
    {"topic":"Algorithms","q":"PCA is mainly used for:","options":["Dimensionality reduction","Image loading","Data backup","Rule writing"],"answer":0,"explanation":"PCA creates fewer components that retain data variance."},
    {"topic":"Algorithms","q":"K-Means requires you to choose:","options":["Number of clusters","One pixel","Learning reward","Text format"],"answer":0,"explanation":"The K in K-Means determines the number of clusters."},
    {"topic":"Algorithms","q":"DBSCAN identifies dense regions and:","options":["Noise points","All points as labels","Only linear lines","Only prices"],"answer":0,"explanation":"DBSCAN can mark low-density isolated observations as noise."},
    {"topic":"Algorithms","q":"Hierarchical clustering produces:","options":["A cluster hierarchy","Only a reward","A single constant","A regression equation"],"answer":0,"explanation":"It creates a tree-like nested structure of clusters."},
    {"topic":"Evaluation","q":"Accuracy measures:","options":["Correct predictions divided by total predictions","Tree height","File size","Feature count"],"answer":0,"explanation":"Accuracy is the proportion of all predictions that are correct."},
    {"topic":"Evaluation","q":"R-squared is used for:","options":["Regression evaluation","Clustering only","Text parsing","Reinforcement rewards"],"answer":0,"explanation":"R-squared indicates how much variance a regression model explains."},
    {"topic":"Evaluation","q":"A confusion matrix summarizes:","options":["Correct and incorrect class predictions","Missing numbers","Feature scales","Cluster centers"],"answer":0,"explanation":"It compares predicted and actual classes in table form."},
    {"topic":"Evaluation","q":"Silhouette score is used for:","options":["Clustering quality","Linear regression","Bayesian probability","Neural activation"],"answer":0,"explanation":"It measures how similar points are to their cluster compared with others."},
    {"topic":"Reinforcement Learning","q":"In reinforcement learning, an agent learns from:","options":["Rewards and penalties","A fixed table only","Only unlabeled CSV files","PCA components"],"answer":0,"explanation":"Rewards provide feedback for the agent's actions."},
    {"topic":"Reinforcement Learning","q":"What represents the world in which an agent acts?","options":["Environment","Feature column","Confusion matrix","Dataset row"],"answer":0,"explanation":"The environment provides states and responses to agent actions."},
    {"topic":"Reinforcement Learning","q":"Q-learning learns:","options":["Action values","Customer names","House colors","CSV headers"],"answer":0,"explanation":"Q-learning estimates the value of taking actions in states."},
    {"topic":"Data Preprocessing","q":"Missing values should be handled:","options":["Before useful model training","Only after deployment","Never","By renaming files"],"answer":0,"explanation":"Handling missing values before training helps prevent invalid or biased results."},
    {"topic":"Data Preprocessing","q":"Feature scaling is useful for:","options":["Distance-based algorithms","Changing image format","Deleting the target","Creating a README"],"answer":0,"explanation":"Scaling keeps features on comparable ranges for methods based on distance or gradients."},
    {"topic":"Data Preprocessing","q":"Duplicate records should generally be:","options":["Reviewed and possibly removed","Always deployed","Converted to targets","Ignored without checking"],"answer":0,"explanation":"Duplicates can distort statistics and training depending on their meaning."},
]


def render():
    st.title("Quiz & Assessment")
    st.caption(f"{len(QUESTIONS)} questions · Fundamentals, algorithms, evaluation, and data preparation")
    st.write("Choose an answer for each question. Nothing is selected for you by default.")
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    def submit_quiz():
        st.session_state.quiz_submitted = True

    def reset_quiz():
        st.session_state.quiz_submitted = False
        for question_index in range(len(QUESTIONS)):
            st.session_state[f"quiz_choice_{question_index}"] = "Select an answer"

    current_topic = None
    for index, question in enumerate(QUESTIONS):
        if question["topic"] != current_topic:
            current_topic = question["topic"]
            st.subheader(current_topic)
        st.markdown(f"**{index + 1}. {question['q']}**")
        key = f"quiz_choice_{index}"
        if key not in st.session_state:
            st.session_state[key] = "Select an answer"
        st.radio(
            f"Answer to question {index + 1}",
            ["Select an answer"] + question["options"],
            key=key,
            label_visibility="collapsed",
        )

    submit_col, reset_col = st.columns(2)
    submit_col.button("Submit Quiz", type="primary", on_click=submit_quiz, use_container_width=True)
    reset_col.button("Reset Quiz", on_click=reset_quiz, use_container_width=True)

    answers = []
    unanswered = []
    for index, question in enumerate(QUESTIONS):
        choice = st.session_state[f"quiz_choice_{index}"]
        if choice == "Select an answer":
            unanswered.append(index)
            answers.append(None)
        else:
            answers.append(question["options"].index(choice))

    if st.session_state.quiz_submitted:
        if unanswered:
            st.warning(f"Answer all questions before submitting. {len(unanswered)} question(s) remain unanswered.")
            return
        score = sum(answers[index] == question["answer"] for index, question in enumerate(QUESTIONS))
        percentage = score / len(QUESTIONS) * 100
        st.success(f"Final score: **{score}/{len(QUESTIONS)}** ({percentage:.1f}%)")
        st.progress(percentage / 100)
        st.subheader("Review your answers")
        for index, question in enumerate(QUESTIONS):
            correct = answers[index] == question["answer"]
            status = "Correct" if correct else "Review"
            with st.expander(f"{index + 1}. {question['q']} — {status}"):
                st.write("**Your answer:**", question["options"][answers[index]])
                st.write("**Correct answer:**", question["options"][question["answer"]])
                st.info(question["explanation"])
