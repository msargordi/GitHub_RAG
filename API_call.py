import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, PythonCodeTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_fireworks import FireworksEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_fireworks import ChatFireworks
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
import subprocess
from pprint import pprint
import argparse
import time
import os

# Set your Fireworks AI API key
os.environ["FIREWORKS_API_KEY"] = "XrpkqpVwX8Z7bo4Au4vYAmAyABR4NMwwG4Kt2uoa0v4zUJ7h"

# Set up argument parser
parser = argparse.ArgumentParser(description="Process repository with custom parameters.")
parser.add_argument("--file_types", nargs='+', default=['.py', '.md', '.ipynb', '.sh'], 
                    help="List of file extensions to process (default: ['.py', '.md', '.ipynb', '.sh'])")
parser.add_argument("--chunk_size", type=int, default=250, 
                    help="Chunk size for text splitting (default: 250)")
parser.add_argument("--num_docs", type=int, default=4, 
                    help="Number of best retrieved documents (default: 4)")
parser.add_argument("--model_name", type=str, default="accounts/fireworks/models/llama-v2-7b-chat",
                    help="Name of the Fireworks AI model to use")
args = parser.parse_args()

# LLM
fireworks_model = args.model_name

# Ask for the repository URL from the user
repo_url = input("Please enter the repository URL (e.g., https://github.com/matsilv/knowledge-injection-dnn): ")

# Extract the repository path from the URL
if "https://github.com/" in repo_url:
    repo_path = "repos/" + repo_url.split("https://github.com/")[1]
else:
    print("Invalid URL. Please make sure it starts with 'https://github.com/'")
    exit(1)

# Function to clone the repository
def clone_repository(repo_url, repo_path):
    if not os.path.exists(repo_path):
        print(f"Cloning repository from {repo_url} into {repo_path}...")
        subprocess.run(['git', 'clone', repo_url, repo_path], check=True)
    else:
        print(f"Repository already exists at {repo_path}.")

# Clone the repository using a system call
clone_repository(repo_url, repo_path)

# Function to load documents from files and add line numbers and file name
def load_documents_from_repo(repo_path, file_extensions):
    docs = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read lines and number them
                    lines = [f"{idx+1}: {line.strip()}" for idx, line in enumerate(f)]
                    # Prepend the file name to the first line
                    if lines:
                        lines[0] = f"File Name: {file}\n{lines[0]}"
                    content_with_filename = "\n".join(lines)
                    docs.append(content_with_filename)
    return docs

docs_lists = load_documents_from_repo(repo_path, args.file_types)

# Convert list of strings to list of Document objects
documents = [Document(page_content=doc) for doc in docs_lists]

# Text splitting with character-based text splitter
text_splitter = PythonCodeTextSplitter.from_tiktoken_encoder(
    chunk_size=args.chunk_size, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(documents)
print("Example of a chunk:")
print(doc_splits[0])

# embedding_instance = FireworksEmbeddings(model="accounts/fireworks/models/text-embedding-3-small")
embedding_instance = GPT4AllEmbeddings()
# Adding to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedding_instance,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": args.num_docs})
print("Retriever is ready!")

# LLM
llm = ChatFireworks(model=fireworks_model, temperature=0)

# Retrieval Grader
prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

    Here is the retrieved document:

    {document}

    Here is the user question: {question}
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

# Generate
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:""",
    input_variables=["question", "context"],
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Hallucination Grader
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

    Here are the facts:
    
    {documents}
    
    Here is the answer: {generation}""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()

# Answer Grader
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

    Here is the answer:
    
    {generation}
    
    Here is the question: {question}""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()

# State
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    attempt_count: int

# Nodes
def retrieve(state):
    print("---Retrieve")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    print("---Generate")
    question = state["question"]
    documents = state["documents"]
    attempt_count = state.get("attempt_count", 0)
    attempt_count += 1

    if attempt_count > 3:
        generation = "Based on the repo, I don't have enough information to answer this question accurately."
    else:
        generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    
    return {"documents": documents, "question": question, "generation": generation, "attempt_count": attempt_count}

def grade_documents(state):
    print("---Check doc relevance")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade.lower() == "yes":
            print("---Grade: document is relevant")
            filtered_docs.append(d)
        else:
            print("---Grade: document is NOT relevant")
    return {"documents": filtered_docs, "question": question}

def decide_to_generate(state):
    print("---Looking into graded docs")
    if not state["documents"]:
        print("---Decision: document is NOT relevant, can not answer")
    else:
        print("---Decision: generate")
    return "generate"

def grade_generation_v_documents_and_question(state):
    print("---Hallucination check")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    attempt_count = state.get("attempt_count", 0)

    if attempt_count > 3:
        print("---Decision: max attempt reached, can not answer")
        return "cannot_answer"

    score = hallucination_grader.invoke(
        {"documents": format_docs(documents), "generation": generation}
    )
    print("score: ", score)
    grade = score["score"]

    if grade == "yes":
        print("---Decision: Generation is grounded in docs")
        print("---Grade docs vs questions")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---Decision: generation addresses question")
            return "useful"
        else:
            print("---Decision: generation does NOT address question")
            return "not useful"
    else:
        print("---Decision: generation is NOT grounded in docs, retry")
        return "not supported"

# Workflow
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": END,
        "cannot_answer": END,
    },
)

app = workflow.compile()

# Main loop
while True:
    question = input("Write your question (type 'stop' to end): ")
    start_time = time.time()
    if question.lower() == 'stop':
        break
    inputs = {"question": question, "attempt_count": 0}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    print("_____________________________________________________________________________")
    pprint(value["generation"])
    print("_____________________________________________________________________________")
    end_time = time.process_time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")