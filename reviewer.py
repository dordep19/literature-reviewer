import os
import sys
import argparse
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


if __name__ == "__main__":
    # Parse command-line arguments (e.g. python reviewer.py -p )
    parser = argparse.ArgumentParser(description="AI assistant for conducting literature reviews")
    parser.add_argument("-p", "--project", metavar="project", type=str, required=True, help="project title")
    args = parser.parse_args()
    load_dotenv()

    # Initialize project storage
    project_path = os.path.join("projects", args.project)    
    if not os.path.exists(project_path):
        os.mkdir(project_path)

    papers_path = os.path.join(project_path, "papers")
    if not os.path.exists(papers_path):
        os.mkdir(papers_path)

    indexes_path = os.path.join(project_path, "indexes")
    if not os.path.exists(indexes_path):
        os.mkdir(indexes_path)

    
    papers = [fname.split(".")[0] for fname in os.listdir(papers_path)]
    new_papers = [paper for paper in papers if paper not in os.listdir(indexes_path)]

    # Create vector store when running project for the first time
    embeddings = OpenAIEmbeddings()
    for paper in  new_papers:
        print(f"Creating vector store for {paper}...")
        loader = PyPDFLoader(file_path=os.path.join(papers_path, paper+".pdf"))
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=docs)

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(os.path.join(indexes_path, paper))
