# Literature Reviewer

This project uses LangChain to implement an AI assistant for conducting academic literature reviews.

## Requirements

To create virtual environment:

```
python -m venv env
```

To activate virtual environment:
```
source env/bin/activate
```

To install requirements:
```
pip install -r requirements.txt
```

## Review

To build your literature review:
1. Create your project directory under [projects](projects) (e.g., federated-learning)
2. Place PDFs of relevant papers inside of the [papers](projects/federated-learning/papers) directory
3. Run the reviewer. Reviews are automatically generated for all new PDFs found in [papers](projects/federated-learning/papers). Revision flag (-r) re-generates reviews for all papers. Interactive flag (-i) allows you to discuss the paper with a chatbot
```
python reviewer.py -p <project_title> [-r] [-i]
```