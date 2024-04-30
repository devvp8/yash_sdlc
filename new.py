import streamlit as st
import google.generativeai as genai
import os
import pickle


generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 5000
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

api_key = os.environ.get('GOOGLE_API_KEY')

if api_key:
    genai.configure(api_key=api_key)
else:
    print("Error: GENAI_API_KEY not found.")

model = genai.GenerativeModel(
        model_name="gemini-1.0-pro",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

def generate_analysis(promp):
    
    convo = model.start_chat(history=[])
    res = convo.send_message(
        f"You are a code optimizer tool. Your task is to provide the time and space complexity for the given {promp} and also provide the more optimized code with a test case including in code itself it if possible. Also provide how new code is better than previous.",
        stream=True,
    )
    response = ""
    for chunk in res:
        response += chunk.text
    return response

def main():

    st.title("OPTIMIZATION TOOL")
    promp = st.text_area("Enter your code or prompt here:", height=400)
    if st.button("Generate"):
        response = generate_analysis(promp)
        st.subheader("Result:")
        st.text(response)

if __name__ == '__main__':
    main()