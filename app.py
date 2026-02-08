import streamlit as st
from src import corep_assistant, load_retrieval_system, init_gemini_client, json_to_corep_table, definition_print
from api_key import API_KEY

st.title("📌 COREP Reporting Assistant")

scenario = st.text_area("Enter Scenario", height=200)
question = st.text_input("Enter Question")

if st.button("Generate COREP Output"):

    model, index, metadata = load_retrieval_system()
    client = init_gemini_client(API_KEY)

    query = f"Scenario:\n{scenario}\n\nQuestion:\n{question}"

    # Show retrieved chunks output
    
    

    retrieved_text = definition_print(query, model, index, metadata, top_k=5)
    with st.expander("📚 Retrieved Regulatory Context"):
        st.code(retrieved_text)


    output = corep_assistant(
        client=client,
        question=question,
        scenario=scenario,
        model=model,
        index=index,
        metadata=metadata
    )

    result_json = output["structured_json"]

    st.subheader("📌 COREP JSON Output")
    st.json(result_json)

    st.subheader("📊 COREP Human Readable Table")
    corep_df = json_to_corep_table(result_json)
    st.dataframe(corep_df)
