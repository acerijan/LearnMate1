## LearnMateAI

LearnMateAI is a minimal Streamlit-based study assistant that lets you upload PDFs, get a TF-IDF summary of the key sentences, generate smart flashcards using Sentence-BERT, and chat with your document using TF-IDF retrieval.

### Features
- **PDF upload**: Simple drag-and-drop using Streamlit
- **TF-IDF Summarization**: Extracts top scoring sentences as bullet points
- **Smart Flashcards**: Uses Sentence-BERT embeddings and cosine similarity to create cloze-style Q&A
- **Simple Chatbot**: Answers questions by retrieving the most relevant sentences from the PDF
- **Clean UI**: Organized into tabs for Upload, Summary, Flashcards, and Chat

### Tech Stack
- **Python**: Streamlit UI
- **NLP**: scikit-learn (TF-IDF), sentence-transformers (Sentence-BERT)
- **PDF**: PyMuPDF for extraction

---

## Getting Started

### Prerequisites
- Python 3.10+

### Setup
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

---

## Usage
1. Go to the "Upload PDF" tab and drop a PDF
2. Check the "Summary" tab for TF-IDF key points
3. Visit "Flashcards" to generate cloze-style cards
4. Use "Chat" to ask questions about the document

