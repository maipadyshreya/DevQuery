import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("questions.csv", encoding="latin1")
df = df.dropna()

# -----------------------------
# App title
# -----------------------------
st.title("AI Software Engineering Interviewer")
st.write("This app asks software interview questions and scores your answer.")

# -----------------------------
# Select category
# -----------------------------
category = st.selectbox("Choose a topic:", df["Category"].unique())

filtered_df = df[df["Category"] == category]

# FIX 1: Reset question when category changes
if "current_category" not in st.session_state or st.session_state.current_category != category:
    st.session_state.current_category = category
    st.session_state.question_row = filtered_df.sample(1).iloc[0]

# FIX 2: New Question button also respects current category
if st.button("New Question"):
    st.session_state.question_row = filtered_df.sample(1).iloc[0]

question_row = st.session_state.question_row
question = question_row["Question"]
correct_answer = question_row["Answer"]

# -----------------------------
# Display question
# -----------------------------
st.subheader("Interview Question")
st.caption(f"Difficulty: {question_row['Difficulty']} | Category: {question_row['Category']}")
st.write(question)

# -----------------------------
# User answer
# -----------------------------
user_answer = st.text_area("Type your answer here:")

# -----------------------------
# Score function
# -----------------------------
def calculate_score(correct_answer, user_answer):
    correct_embedding = model.encode(correct_answer, convert_to_tensor=True)
    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    similarity = util.cos_sim(correct_embedding, user_embedding).item()
    # FIX 3: Clamp score to 0-100 (cosine similarity can be slightly negative)
    score = round(max(0, similarity) * 100, 2)
    return score

# -----------------------------
# Submit answer
# -----------------------------
if st.button("Submit Answer"):
    if user_answer.strip() == "":
        st.warning("Please type an answer first.")
    else:
        score = calculate_score(correct_answer, user_answer)

        st.subheader("Your Score")
        st.metric(label="Similarity Score", value=f"{score}/100")  # FIX 4: nicer display

        if score >= 75:
            st.success("Good answer! You explained the concept well.")
        elif score >= 50:
            st.warning("Partially correct. Add more detail.")
        else:
            st.error("Needs improvement. Review the expected answer.")

        st.subheader("Expected Answer")
        st.write(correct_answer)

        st.subheader("Your Answer")
        st.write(user_answer)