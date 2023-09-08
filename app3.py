import yaml
import streamlit as st
from langchain import OpenAI
from pdf_loaders import PdfToTextLoader
from dataset_vectorizers import DatasetVectorizer
from langchain import OpenAI, VectorDBQA, LLMChain
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

OPENAI_API_KEY = config['OPENAI_KEY']
PDFS, NAMES, TXTS = [], [], []
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

# ----- Header of the app -----
st.title("Insurance Plan Summary")
st.write("This app summarizes insurance plan documents using the OpenAI API. It is a prototype and not intended for actual use.")

# ----- Select and upload the file -----
st.header("Select the file to summarize")
st.write("The file should be in PDF format.")
file_1 = st.file_uploader("File 1")
name_1 = st.text_input("Name of the file", value="Plan 1")

# ----- Load the file -----
if file_1:

    with open("./data/" + file_1.name, "wb") as f:
        f.write(file_1.getbuffer())

    PDFS = ["./data/" + file_1.name]
    NAMES = [name_1]

    for pdf_path in PDFS:
        txt_path = pdf_path.replace(".pdf", ".txt")
        pdf_loader = PdfToTextLoader(pdf_path, txt_path)
        text = pdf_loader.load_pdf()
        TXTS.append(txt_path)
    st.write("File loaded successfully.")

    dataset_vectorizer = DatasetVectorizer()
    documents_1, texts_1, docsearch_1 = dataset_vectorizer.vectorize([TXTS[0]], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, openai_key=OPENAI_API_KEY)
    llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=OPENAI_API_KEY)
    qa_chain_1 = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch_1)
    st.write("File vectorized successfully.")

    # ----- Generate the summary for the document -----
    st.header("Summary of the document")
    summary = ""
    
    # Example: Generate a summary by asking multiple questions and concatenate answers
    questions = [
        "What are the key benefits of this insurance plan?",
        "How does the deductible work?",
        "Tell me about the coverage for pre-existing conditions."
        
       
    ]
 
    
    for question in questions:
        answer = qa_chain_1.run(question)
        summary += f"{answer}\n"

    # Print the summary
    st.write(summary)

    # Create an input element for user questions
    user_question = st.text_input("Ask a question about the document")

    # Generate answers when the user clicks a button
    if st.button("Generate Answer"):
        if user_question:
            # Generate an answer to the user's question
            answer = qa_chain_1.run(user_question)
            st.write("Answer: ", answer)
        else:
            st.write("Please enter a question.")
