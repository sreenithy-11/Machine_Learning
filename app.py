import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Machine Learning Learning Hub", page_icon="🧠", layout="wide")

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("🧠 ML Learning Hub")
st.sidebar.caption("Learn • Practice • Assess")

pages = ["Home", "ML Fundamentals", "ML Algorithms", "Interactive Visualizations", "Practical ML Lab", "Quiz & Assessment"]
if st.session_state.page not in pages:
    st.session_state.page = "Home"
if st.session_state.get("navigation_selection") not in pages:
    st.session_state.navigation_selection = st.session_state.page


def navigate_from_sidebar():
    st.session_state.page = st.session_state.navigation_selection


st.sidebar.radio(
    "Navigation",
    pages,
    key="navigation_selection",
    on_change=navigate_from_sidebar,
)

from pages import fundamentals, algorithms, visualizations, practical_lab, quiz


def navigate_to(page):
    st.session_state.page = page
    st.session_state.navigation_selection = page


if st.session_state.page == "Home":
    st.markdown(
        '<div class="home-hero"><h1>Machine Learning Learning Hub</h1>'
        '<p><b>Welcome, students!</b> Build a solid foundation in machine learning, '
        'see how algorithms behave, and practice with real datasets.</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Learning Categories", "3")
    c2.metric("Algorithms Explained", "12")
    c3.metric("Quiz Questions", "26")

    st.subheader("What will you learn?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-card"><h3>🔵 Supervised Learning</h3><p>Learn from labeled examples to predict a number or a category.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><h3>🟣 Unsupervised Learning</h3><p>Find groups and useful structure when examples have no answer labels.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card"><h3>🟢 Reinforcement Learning</h3><p>Understand how an agent learns actions from rewards and feedback.</p></div>', unsafe_allow_html=True)

    st.subheader("Important Topics")
    st.write("ML workflow · Regression · Classification · Clustering · Dimensionality reduction · Model evaluation · Data preparation")

    st.subheader("Learning Roadmap")
    st.markdown(
        "1. Learn the basic ideas and ML workflow.\n"
        "2. Compare algorithms and understand when they are useful.\n"
        "3. Explore visual demos, then train a model in the lab.\n"
        "4. Finish with the quiz and review any missed answers."
    )

    st.subheader("Quick Navigation")
    b1, b2, b3, b4 = st.columns(4)
    b1.button("Learn Fundamentals", use_container_width=True, on_click=navigate_to, args=("ML Fundamentals",))
    b2.button("Explore Algorithms", use_container_width=True, on_click=navigate_to, args=("ML Algorithms",))
    b3.button("Open ML Lab", use_container_width=True, on_click=navigate_to, args=("Practical ML Lab",))
    b4.button("Take Quiz", use_container_width=True, on_click=navigate_to, args=("Quiz & Assessment",))

elif st.session_state.page == "ML Fundamentals": fundamentals.render()
elif st.session_state.page == "ML Algorithms": algorithms.render()
elif st.session_state.page == "Interactive Visualizations": visualizations.render()
elif st.session_state.page == "Practical ML Lab": practical_lab.render()
elif st.session_state.page == "Quiz & Assessment": quiz.render()
