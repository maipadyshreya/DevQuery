from re import match

import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from sympy import re
import re

# Load environment variables
load_dotenv()

# OpenRouter API
client = OpenAI(
     base_url = "https://integrate.api.nvidia.com/v1",
     #remove for final 
     api_key=os.getenv("NVIDIA_API_KEY")
     )


#read the file with question and answers
df = pd.read_csv("questions.csv", encoding="latin1")
#remove rows with missing values
df = df.dropna()

#App title and description
st.title("AI Software Engineering Interviewer")
st.write("This app asks software interview questions and scores your answer.")

#Dropdown to select category in dataset
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

# LLM grading function
def calculate_score(question, user_answer):

    prompt = f"""
You are a very lenient software engineering interviewer.

Grade the candidate's answer from 0 to 100.

Rules:
- Give 100 if the answer is factually correct.
- Do not require the answer to match any expected wording.
- Do not remove points for being brief.
- Only give below 100 if the answer is incorrect, unclear, or missing the main concept.

Interview Question:
{question}

Candidate Answer:
{user_answer}

Return ONLY one number.
"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    score_text = response.choices[0].message.content.strip()

    match = re.search(r"\d+(\.\d+)?", score_text)

    if match:
         score = float(match.group())
    else:
     score = 0

    return round(score, 2)



#submit answer button
if st.button("Submit Answer"):
    #Check if the suer left answer empty 
    if user_answer.strip() == "":
        st.warning("Please type an answer first.")
    else:
        #Calculate the similarity score between the correct answer and the user's answer
        score = calculate_score(question, user_answer)

        #Display the score and feedback
        st.subheader("Your Score")
        st.metric(label="LLM Score", value=f"{score}/100")
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