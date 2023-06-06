import os
import sys
import argparse
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


if __name__ == "__main__":
    # Parse command-line arguments (e.g. python reviewer.py -p )
    parser = argparse.ArgumentParser(description="AI assistant for conducting literature reviews")
    parser.add_argument("-p", "--project", metavar="project", type=str, required=True, help="project title")
    parser.add_argument("-r", "--revise", action='store_true', help="revise existing reviews")
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

    reviews_path = os.path.join(project_path, "reviews")
    if not os.path.exists(reviews_path):
        os.mkdir(reviews_path)
    
    papers = [fname.split(".")[0] for fname in os.listdir(papers_path)]
    new_papers = [paper for paper in papers if paper not in os.listdir(indexes_path)]
    embeddings = OpenAIEmbeddings()

    # Embed new papers into vector stores
    for paper in  new_papers:
        print(f"Creating vector store for {paper}...")
        loader = PyPDFLoader(file_path=os.path.join(papers_path, paper+".pdf"))
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=docs)

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(os.path.join(indexes_path, paper))

    # Create reviews for 
    if not args.revise:
        papers = [paper for paper in papers if paper if paper not in os.listdir(reviews_path)]

    for paper in papers[0:1]:
        vectorstore = FAISS.load_local(os.path.join(indexes_path, paper), embeddings)
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever())
        prompt = f"""
        Summarize the {paper} paper in no more than 5 sentences.
        Then, on a new line, provide a numbered list of 5 key points and ideas of the paper, each seperated by a new line.
        Then, on a new line, write 2 sentences discussing limitations of the paper.

        For example:
        The Transcend paper introduces a statistical framework for assessing decisions made by a classifier to identify concept drift. It translates the decision assessment problem to a constraint optimization problem which enables Transcend to be parametric with diverse operational goals. 

        Key points and ideas include:
        1) the proposal of both meaningful and sufficient abstract assessment metrics
        2) the translation of the decision assessment problem to a constraint optimization problem
        3) the bootstrapping of the framework with pre-specified parameters
        4) the evaluation of algorithm performances within a conformal evaluator framework
        5) the application of the framework to machine learning-based security research and deployments
        
        One limitation of the paper is the limited number of datasets used for experimentation. It is also highly computationally expensive to run.
        """
        res = qa.run(prompt)
        
        with open(os.path.join(reviews_path, paper+".txt"), "w") as f:
            f.write(res)
