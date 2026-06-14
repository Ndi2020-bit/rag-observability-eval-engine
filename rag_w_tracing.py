import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 1. IMPORT OPENTELEMETRY TRACING POWER TOOLS
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

load_dotenv()

# UPDATE THIS BLOCK IN YOUR SCRIPT:
# Force OpenTelemetry to route directly to your active background port
register(
    project_name="enterprise-rag-evaluation",
    endpoint="http://localhost:6006/v1/traces"
)
LangChainInstrumentor().instrument()

print("🔍 OpenTelemetry active. Tracing all downstream LangChain operations...\n")

# 3. RUN AN END-TO-END WORKFLOW (Your Data Sandbox Engine)
with open("lab_notes.txt", "w") as f:
    f.write("Lab Experiment PVT-01: Target pressure achieved at 145.6 PSI. Temperature normalized at 22.4 degrees Celsius.")

loader = TextLoader("lab_notes.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=15)
chunks = text_splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings()
vector_db = Chroma.from_documents(chunks, embedding_model)

user_question = "What was the target pressure achieved in the PVT-01 lab experiment?"
retrieved_docs = vector_db.similarity_search(user_question, k=1) 
context = retrieved_docs[0].page_content

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the following context.
    Context: {context}
    Question: {question}
    """
)
final_prompt = prompt_template.format(context=context, question=user_question)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke(final_prompt)

print("--- AI ANSWER ---")
print(response.content)
