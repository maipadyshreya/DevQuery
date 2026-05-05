import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# cache model so only loads once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

#load cached model
model = load_model()

#read the file with question and answers
df = pd.read_csv("questions.csv", encoding="latin1")
#remove rows with missing values
df = df.dropna()

#App title and description
st.title("AI Software Engineering Interviewer")
st.write("This app asks software interview questions and scores your answer.")

#Dropdown to seelct category in dataset
category = st.selectbox("Choose a topic:", df["Category"].unique())

#filter the dataset based on the selected category
filtered_df = df[df["Category"] == category]

#button to generate a new random question from the filtered dataset
if "current_category" not in st.session_state or st.session_state.current_category != category:
    st.session_state.current_category = category
    st.session_state.question_row = filtered_df.sample(1).iloc[0]

#button to generate a new random question from the filtered dataset
if st.button("New Question"):
    st.session_state.question_row = filtered_df.sample(1).iloc[0]

##take row, question and the corrct answer
question_row = st.session_state.question_row
question = question_row["Question"]
correct_answer = question_row["Answer"]

#Display question and difficult category 
st.subheader("Interview Question")
st.caption(f"Difficulty: {question_row['Difficulty']} | Category: {question_row['Category']}")

#display the question
st.write(question)

#text area for user to type their answer
user_answer = st.text_area("Type your answer here:")

#
def calculate_score(correct_answer, user_answer):
    correct_embedding = model.encode(correct_answer, convert_to_tensor=True)
    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    similarity = util.cos_sim(correct_embedding, user_embedding).item()
    # FIX 3: Clamp score to 0-100 (cosine similarity can be slightly negative)
    score = round(max(0, similarity) * 100, 2)
    return score

#submit answer button
if st.button("Submit Answer"):
    #Check if the suer left answer empty 
    if user_answer.strip() == "":
        st.warning("Please type an answer first.")
    else:
        #Calculate the similarity score between the correct answer and the user's answer
        score = calculate_score(correct_answer, user_answer)

        #Display the score and feedback
        st.subheader("Your Score")
        st.metric(label="Similarity Score", value=f"{score}/100")  # FIX 4: nicer display

        #Provide feedback based on the score
        if score >= 75:
            st.success("Good answer! You explained the concept well.")
        elif score >= 50:
            st.warning("Partially correct. Add more detail.")
        else:
            st.error("Needs improvement. Review the expected answer.")

        #Display the expected answer and the user's answer for comparison
        st.subheader("Expected Answer")
        st.write(correct_answer)

        st.subheader("Your Answer")
        st.write(user_answer)