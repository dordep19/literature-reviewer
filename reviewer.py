import os
import argparse
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
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
    parser.add_argument("-r", "--revise", action="store_true", help="revise all reviews")
    parser.add_argument("-i", "--interactive", action="store_true", help="interactively ask questions about paper")
    args = parser.parse_args()
    load_dotenv()

    # Verify and initialize project storage
    project_path = os.path.join("projects", args.project)
    papers_path = os.path.join(project_path, "papers")

    if not os.path.exists(papers_path):
        print(f"No papers directory found under project - created {papers_path}. Please populate directory with relevant papers")
        os.makedirs(papers_path)
        exit(0)

    indexes_path = os.path.join(project_path, "indexes")
    if not os.path.exists(indexes_path):
        os.mkdir(indexes_path)

    reviews_path = os.path.join(project_path, "reviews")
    if not os.path.exists(reviews_path):
        os.mkdir(reviews_path)
    
    papers = [fname.split(".")[0] for fname in os.listdir(papers_path)]
    new_papers = [paper for paper in papers if paper not in os.listdir(indexes_path)]
    if len(papers) == 0:
        print(f"{papers_path} directory is empty. Please populate directory with relevant papers")
        exit(0)

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

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
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_template, combine_prompt=reduce_template)
        res = chain({"input_documents": docs}, return_only_outputs=True)

        review_path = os.path.join(reviews_path, paper+".txt")
        with open(review_path, "w") as f:
            f.write(res["output_text"])

        print(f"Stored review in {review_path}")

    if args.interactive:
        available_papers = [fname.split(".")[0] for fname in os.listdir(indexes_path)]
        print("\nThe following papers are available for discussion:")
        for i, paper in enumerate(available_papers):
            print(f"{i+1}. {paper}")
        paper_id = input(f"\nEnter the number of the paper you want to discuss: ")
        

        # Validate paper selection
        valid = False
        while not valid:
            try:
                paper_id = int(paper_id)
            except:
                paper_id = input("Invalid response. Select a number correspdonding to your paper of interest from the above list: ")
                continue
            if paper_id >= 1 and paper_id <= len(available_papers):
                valid = True
            else:
                paper_id = input("Invalid response. Select a number that appears in the above list: ")
        paper = available_papers[paper_id-1]

        # Launch Q/A chatbot using paper vector store
        prompt_template = """
        Use the following pieces of context to answer the question at the end. All questions will be about the academic paper. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        vectorstore = FAISS.load_local(os.path.join(indexes_path, paper), embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs)
        finished = False
        
        print(f"\nAssistant: I am an AI, here to assist you in reviewing the literature for the {args.project} project. Let's discuss the {paper} paper! Once you are done, please reply with \"Bye\"")
        while not finished:
            question = input("User: ")

            if question == "Bye":
                print("Assistant: Goodbye!")
                exit(0)
            else:
                response = qa.run(question)
                print(f"Assistant: {response}")
