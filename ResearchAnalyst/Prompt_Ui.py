from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

import streamlit as st

st.header("Research Papers Summarizer")

Paper_Input = st.selectbox("Select Research Paper Name", ["A Mathematical Theory of Communication",
"Computing Machinery and Intelligence", "Attention Is All You Need", "Language Models are Few-Shot Learners",
"The First Direct Observation of Gravitational Waves", "Bounding the Gaps Between Prime Numbers"])

Length_Input = st.selectbox("Select Explanation Length", ["Short (50-100 words)", "Medium (100-200 words)", "Detailed (200-500 words)"])

AudienceLevel_Input = st.selectbox("Select the Audience Type", ["Beginner", "Business Professional", "Researcher", "Student"])

#template
template = PromptTemplate(
    template = """
    You are an expert Research Analyst and Academic Reviewer.
Analyze the research paper titled "{paper_input}" provided and generate a structured summary based on the following user selections:

Explanation Length:{length_input}
Explanation Level : {audiencelevel_input}


Include the following sections:

Research Objective – What problem or question does the paper address?
Approach – What methodology, model, experiment, or dataset was used?
Key Findings – What are the most important results or conclusions?
Significance – Why do these findings matter? What is their practical or academic impact?
Limitations – What constraints, assumptions, or weaknesses should readers be aware of?

Guidelines:
If certain information is not available in the Paper respond with "Insufficient Information" instead of guessing.
Ensure the summary is clear, accurate and aligned with the provided User selections.
Focus on the most important information.
Avoid unnecessary technical details unless they are critical to understanding the findings.

Research Paper's: Title, Authors, Publication Year

Research Paper: Provide the link of Research Paper if available.
Sources : provide the links of the resources you found this information.""",
input_variables = ["paper_input", "length_input", "audiencelevel_input"]
)
prompt = template.invoke({
    'paper_input':Paper_Input,
    'length_input':Length_Input,
    'audiencelevel_input':AudienceLevel_Input
})

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)