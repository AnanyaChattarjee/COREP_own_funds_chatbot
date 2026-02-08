import streamlit as st
from src import corep_assistant, load_retrieval_system, init_gemini_client
API_KEY = st.secrets["API_KEY"]


st.title("📌 COREP Reporting Assistant")

scenario = st.text_area("Enter Scenario", height=200)
question = st.text_input("Enter Question")

if st.button("Generate COREP Output"):
    model, index, metadata = load_retrieval_system()

    client = init_gemini_client(API_KEY)

    output = corep_assistant(
        client=client,
        question=question,
        scenario=scenario,
        model=model,
        index=index,
        metadata=metadata
    )

    st.json(output["structured_json"])
