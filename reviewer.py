import os
import argparse
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


if __name__ == "__main__":
    # Parse command-line arguments (e.g. python reviewer.py -p )
    parser = argparse.ArgumentParser(description="AI assistant for conducting literature reviews")
    parser.add_argument("-p", "--project", metavar="project", type=str, required=True, help="project title")
    parser.add_argument("-r", "--revise", action='store_true', help="revise all reviews")
    args = parser.parse_args()
    load_dotenv()

    # Verify and initialize project storage
    project_path = os.path.join("projects", args.project)
    papers_path = os.path.join(project_path, "papers")

    if not os.path.exists(papers_path):
        print(f"No papers directory found under project - created {papers_path}")
        os.makedirs(papers_path)

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

    # Generate reviews only for new papers, unless asked to revise all
    if not args.revise:
        papers = [paper for paper in papers if paper if paper not in os.listdir(reviews_path)]

    for paper in papers:
        print(f"Generating review for {paper}...")
        # Load and pre-process paper content
        loader = PyPDFLoader(file_path=os.path.join(papers_path, paper+".pdf"))
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=docs)
        docs = [Document(page_content=doc.page_content) for doc in docs]
        
        # Prepare prompts
        map_template_string = """
        Given the segment {text} of a paper, summarize all of the technical information found in this segment
        """
        reduce_template_string = """
        Given the summaries {text} of an academic paper, I want you to create:
        1. A technical summary of the paper in a few sentences
        2. A list of key points and ideas
        3. Any limitations and future work listed
        """

        map_template = PromptTemplate(input_variables=["text"], template=map_template_string)
        reduce_template = PromptTemplate(input_variables=["text"], template=reduce_template_string)

        # Generate reviews
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_template, combine_prompt=reduce_template)
        res = chain({"input_documents": docs}, return_only_outputs=True)

        review_path = os.path.join(reviews_path, paper+".txt")
        with open(review_path, "w") as f:
            f.write(res["output_text"])

        print(f"Stored review in {review_path}")
