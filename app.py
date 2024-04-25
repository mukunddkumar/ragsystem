import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="Hiding because of publishing it on github",model="models/embedding-001")

# Set up a connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Define a function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) 

# Convert CHROMA db_connection to a Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""Welcome to Contextual Q&A Assistant. 
    I'm here to provide assistance based on the given context. 
    Please make sure your questions are clear and relevant."""),
    HumanMessagePromptTemplate.from_template("Answer the question based on the provided context:\n{context}\nQuestion:\n{question}\nAnswer:")
])

# Initialize the chat model
chat_model = ChatGoogleGenerativeAI(google_api_key="Hiding because of publishing it on github", 
                                    model="gemini-1.5-pro-latest")

# Initialize the output parser
output_parser = StrOutputParser()

# Define RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
st.title("✨ Contextual Q&A Assistant ✨")
st.subheader("An Advanced AI System for Contextual Question Answering")

question = st.text_input("Ask your question:")

if st.button("Generate Answer"):
    if question:
        response = rag_chain.invoke(question)
        st.write("📝 Answer:")
        st.write(response)
    else:
        st.warning("Please enter a question.")
