import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")

st.subheader("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
    }
    response = requests.post(f"{API_URL}/upload", files=files)

    if response.status_code == 200:
        st.success("PDF uploaded and indexed successfully.")
    else:
        st.error("Failed to upload PDF.")

st.subheader("Ask a Question")
question = st.text_input("Enter your question")

if st.button("Ask"):
    if question.strip():
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question}
        )

        if response.status_code == 200:
            data = response.json()

            st.markdown("### Answer")
            st.write(data["answer"])

            st.markdown("### LLM Used")
            st.write(data["llm_used"])

            st.markdown("### Source Pages")
            st.write(data.get("source_pages", []))

            with st.expander("Retrieved Chunks"):
                for i, r in enumerate(data["results"], start=1):
                    st.markdown(f"**Result {i}**")
                    st.write(f"Page: {r['page_number']}")
                    st.write(r["text"])
                    st.markdown("---")
        else:
            st.error("Failed to get answer.")