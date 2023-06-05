import os
import argparse


if __name__ == "__main__":
    # Parse command-line arguments (e.g. python reviewer.py -p )
    parser = argparse.ArgumentParser(description="AI assistant for conducting literature reviews")
    parser.add_argument("-p", "--project", metavar="project", type=str, required=True, help="project title")
    args = parser.parse_args()

    # Initialize project storage
    project_path = os.path.join('projects', args.project)
    if not os.path.exists(project_path):
        os.mkdir(project_path)

    papers_path = os.path.join(project_path, 'papers')
    if not os.path.exists(papers_path):
        os.mkdir(papers_path)

    index_path = os.path.join(project_path, 'index')
    if not os.path.exists(index_path):
        os.mkdir(index_path)

    processed_path = os.path.join(project_path, 'processed.txt')
    if not os.path.exists(processed_path):
        open(processed_path, "x")
