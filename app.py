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

def generate_analysis(code,promp):
    
    res=model.generate_content(promp+code)
    response = ""
    for chunk in res:
        response += chunk.text
    return response

def gen_code(code,promp):
    
    res=model.generate_content(promp+code)
    response = ""
    for chunk in res:
        response += chunk.text
    return response

def generate_test_cases(code,promp):
    
    res=model.generate_content(promp+code)
    response = ""
    for chunk in res:
        response += chunk.text
    return response

def gen_req(code,promp):
    
    res=model.generate_content(promp+code)
    response = ""
    for chunk in res:
        response += chunk.text
    return response

def generate_data(code,promp):
    
    res=model.generate_content(promp+code)
    response = ""
    for chunk in res:
        response += chunk.text
    return response

def main():
    st.sidebar.title("Options")

    st.title("SDLC LIFE CYCLE PROCESS")

    analysis_type = st.sidebar.selectbox("Select Analysis Type:", ["Data Generation","Requirements Gathering","Code Generation","Code Analysis", "Test Case Generation"])

    promp = st.text_area("Enter the task which you want to perform:", height=80)
    code = st.text_area("Enter your code or prompt here:", height=400)
    
    if st.button("Analyze"):
        if analysis_type == "Data Generation":
            response = generate_data(code,promp)
        elif analysis_type == "Requirements Gathering":
            response = gen_req(code,promp)
        elif analysis_type == "Code Generation":
            response = gen_code(code,promp)
        elif analysis_type == "Code Analysis":
            response = generate_analysis(code,promp)
        else:
            response = generate_test_cases(code,promp)
        st.subheader("Result:")
        st.text(response)

if __name__ == '__main__':
    main()