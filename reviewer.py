import os
import sys
import argparse
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader


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

    processed_path = os.path.join(project_path, "processed.txt")
    if not os.path.exists(processed_path):
        open(processed_path, "x")

    with open(processed_path, "r") as f:
        processed_papers = f.read().splitlines()    
    all_papers = os.listdir(papers_path)
    new_papers = [paper for paper in all_papers if paper not in processed_papers]

    if all_papers == []:
        raise AssertionError("no papers found inside project directory")
    
    # Create vectorstore when running project for the first time
    embeddings = OpenAIEmbeddings()
    index_path = os.path.join(project_path, "index")
    if not os.path.exists(index_path):
        print(f"Creating vector store and inserting all PDFs inside {args.project}/papers/...")
        loader = PyPDFDirectoryLoader(papers_path)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=docs)

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_path)
        
        with open(processed_path, "w") as f:
            f.write("/n".join(all_papers))

    # Append to vectorstore on consequent runs
    else:
        if len(new_papers) == 0:
            print("No new papers found, loading vector store...")
            vectorstore = FAISS.load_local(index_path, embeddings)

        else:
            print(f"Adding new papers ({new_papers}) to the vectore store...")
            vectorstore = FAISS.load_local(index_path, embeddings)
            texts = []

            for paper in new_papers:
                paper_path = os.path.join(papers_path, paper)
                loader = PyPDFLoader(file_path=paper_path)
                docs = loader.load()
                texts.extend([doc.page_content for doc in docs])
            vectorstore.add_texts(texts)
            
            with open(processed_path, "a") as f:
                f.write("\n"+"\n".join(new_papers))
