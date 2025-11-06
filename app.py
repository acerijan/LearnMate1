import io
import os
import re
import sys
from contextlib import contextmanager
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import transformers, but make it optional
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

from utils import DocumentReader
from tfidf_from_scratch import TFIDFVectorizer, load_default_stopwords


def split_into_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter avoiding heavy dependencies
    # Splits on ., !, ? while keeping abbreviations moderately safe
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sentences = [s.strip() for s in sentences if s and len(s.strip()) > 1]
    return sentences


@st.cache_resource(show_spinner=False)
def load_sbert_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading summarization model (first time only, ~1.6GB download)...")
def load_summarization_model():
    """Load abstractive summarization model"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Try T5-small first (lighter, faster)
        summarizer = pipeline(
            "summarization",
            model="t5-small",
            device=-1,  # Use CPU (-1), set to 0 for GPU if available
            max_length=512,
            min_length=50,
        )
        return summarizer
    except Exception as e:
        # Fallback to BART if T5 fails
        try:
            model_name = "facebook/bart-large-cnn"
            summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=-1,
                max_length=512,
                min_length=50,
            )
            return summarizer
        except Exception as e2:
            return None


def chunk_text(text: str, max_chunk_size: int = 1024, overlap: int = 100) -> List[str]:
    """
    Split text into chunks that fit within model's context window
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-min(overlap // 10, len(current_chunk)):]
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def generate_abstractive_summary(
    text: str,
    max_length: int = 150,
    min_length: int = 50,
    summary_type: str = "paragraph"
) -> str:
    """
    Generate abstractive summary like ChatGPT/Gemini - full paragraphs synthesizing information
    
    Args:
        text: Input text to summarize
        max_length: Maximum length of summary (in tokens)
        min_length: Minimum length of summary (in tokens)
        summary_type: "paragraph" for full paragraphs, "bullet" for bullet points
    """
    if not text or len(text.strip()) < 100:
        return "Text is too short to generate a meaningful summary."
    
    if not TRANSFORMERS_AVAILABLE:
        # Fallback: Use extractive summarization
        st.warning("⚠️ Transformers library not available. Using extractive summarization instead.")
        sentences = split_into_sentences(text)
        if len(sentences) < 5:
            return " ".join(sentences)
        ranked = enhanced_tfidf_rank_sentences(sentences, top_k=min(10, len(sentences)))
        summary_sentences = [sentences[idx] for idx, _ in ranked if 0 <= idx < len(sentences)]
        return " ".join(summary_sentences)
    
    summarizer = load_summarization_model()
    if summarizer is None:
        # Fallback: Use extractive summarization
        st.warning("⚠️ Summarization model not available. Using extractive summarization instead.")
        sentences = split_into_sentences(text)
        if len(sentences) < 5:
            return " ".join(sentences)
        ranked = enhanced_tfidf_rank_sentences(sentences, top_k=min(10, len(sentences)))
        summary_sentences = [sentences[idx] for idx, _ in ranked if 0 <= idx < len(sentences)]
        return " ".join(summary_sentences)
    
    try:
        # Clean and prepare text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle long documents by chunking
        if len(text) > 1500:  # Rough estimate for 1024 tokens
            chunks = chunk_text(text, max_chunk_size=1024, overlap=100)
            summaries = []
            
            with st.spinner(f"Summarizing {len(chunks)} sections..."):
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:
                        try:
                            chunk_summary = summarizer(
                                chunk,
                                max_length=max_length // len(chunks) + 50,
                                min_length=min_length // len(chunks),
                                do_sample=False,
                                truncation=True
                            )
                            if chunk_summary and len(chunk_summary) > 0:
                                summaries.append(chunk_summary[0]['summary_text'])
                        except Exception as e:
                            st.warning(f"Error summarizing chunk {i+1}: {str(e)}")
                            continue
            
            # Combine chunk summaries
            if summaries:
                combined_text = " ".join(summaries)
                # Summarize the combined summaries for final output
                if len(combined_text) > 500:
                    final_summary = summarizer(
                        combined_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    if final_summary and len(final_summary) > 0:
                        return final_summary[0]['summary_text']
                return combined_text
            else:
                return "Could not generate summary from chunks."
        else:
            # Single pass for shorter documents
            try:
                result = summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                
                if result and len(result) > 0 and isinstance(result, list):
                    summary_text = result[0].get('summary_text', '') if isinstance(result[0], dict) else str(result[0])
                    
                    if summary_text:
                        # Format based on type
                        if summary_type == "paragraph":
                            # Ensure it's formatted as paragraphs
                            summary_text = re.sub(r'\.\s+([A-Z])', r'.\n\n\1', summary_text)
                            summary_text = re.sub(r'\n\s*\n\s*\n', '\n\n', summary_text)
                            return summary_text.strip()
                        else:
                            return summary_text.strip()
                
                # If we get here, fallback to extractive
                st.warning("⚠️ Abstractive summarization failed. Using extractive fallback.")
                sentences = split_into_sentences(text)
                if len(sentences) < 5:
                    return " ".join(sentences)
                ranked = enhanced_tfidf_rank_sentences(sentences, top_k=min(10, len(sentences)))
                summary_sentences = [sentences[idx] for idx, _ in ranked if 0 <= idx < len(sentences)]
                return " ".join(summary_sentences)
            except Exception as chunk_error:
                # Fallback to extractive on any error
                st.warning(f"⚠️ Error in summarization: {str(chunk_error)}. Using extractive fallback.")
                sentences = split_into_sentences(text)
                if len(sentences) < 5:
                    return " ".join(sentences)
                ranked = enhanced_tfidf_rank_sentences(sentences, top_k=min(10, len(sentences)))
                summary_sentences = [sentences[idx] for idx, _ in ranked if 0 <= idx < len(sentences)]
                return " ".join(summary_sentences)
    
    except Exception as e:
        # Final fallback to extractive
        st.warning(f"⚠️ Error generating summary: {str(e)}. Using extractive fallback.")
        try:
            sentences = split_into_sentences(text)
            if len(sentences) < 5:
                return " ".join(sentences)
            ranked = enhanced_tfidf_rank_sentences(sentences, top_k=min(10, len(sentences)))
            summary_sentences = [sentences[idx] for idx, _ in ranked if 0 <= idx < len(sentences)]
            return " ".join(summary_sentences)
        except Exception as fallback_error:
            return f"Error: Could not generate summary. {str(fallback_error)}"


@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout to avoid verbose TF-IDF output in Streamlit"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def enhanced_tfidf_rank_sentences(
    sentences: List[str], 
    top_k: int = 5,
    use_redundancy_removal: bool = True,
    similarity_threshold: float = 0.7,
    position_weight: float = 0.2,
    min_sentence_length: int = 5,
    max_sentence_length: int = 300
) -> List[Tuple[int, float]]:
    """
    Enhanced TF-IDF summarization with:
    - Custom TF-IDF from scratch implementation
    - Position weighting (first and last sentences are more important)
    - Length normalization (fair comparison across sentence lengths)
    - Redundancy removal (avoids selecting similar sentences)
    - Topic diversity (ensures coverage across document sections)
    
    Args:
        sentences: List of sentences to rank (must be valid strings)
        top_k: Number of top sentences to return
        use_redundancy_removal: Whether to remove redundant sentences
        similarity_threshold: Threshold for considering sentences redundant (0-1)
        position_weight: Weight bonus for first/last sentences (0-1)
        min_sentence_length: Minimum words per sentence
        max_sentence_length: Maximum words per sentence
    """
    # Validate input
    if not sentences:
        return []
    
    # Filter out invalid sentences and ensure they're strings
    valid_sentences = []
    valid_indices = []
    for i, sent in enumerate(sentences):
        if not isinstance(sent, str):
            continue
        sent = sent.strip()
        word_count = len(sent.split())
        if sent and min_sentence_length <= word_count <= max_sentence_length:
            valid_sentences.append(sent)
            valid_indices.append(i)
    
    if not valid_sentences:
        return []
    
    # If we have fewer valid sentences than requested, return all
    if len(valid_sentences) <= top_k:
        return [(valid_indices[i], 1.0) for i in range(len(valid_sentences))]
    
    try:
        # 1. Use custom TF-IDF from scratch (suppress verbose output)
        stop_words = load_default_stopwords()
        vectorizer = TFIDFVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.95,
            stop_words=stop_words,
            lowercase=True
        )
        
        # Suppress verbose output from TF-IDF
        with suppress_stdout():
            tfidf_matrix = vectorizer.fit_transform(valid_sentences)
        
        # Validate TF-IDF matrix
        if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
            # Fallback: return first sentences
            return [(valid_indices[i], 1.0) for i in range(min(top_k, len(valid_sentences)))]
        
        # 2. Calculate base TF-IDF scores (sum of all word scores in sentence)
        base_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
        
        # Check if all scores are zero (empty vocabulary issue)
        if np.all(base_scores == 0):
            # Fallback: return sentences with position weighting only
            base_scores = np.ones(len(valid_sentences))
        
        # 3. Position weighting - first and last sentences get bonus
        position_weights = np.ones(len(valid_sentences))
        first_portion = max(1, int(len(valid_sentences) * position_weight))
        last_portion = max(1, int(len(valid_sentences) * position_weight))
        
        # Weight first sentences (introduction importance)
        position_weights[:first_portion] *= 1.3
        
        # Weight last sentences (conclusion importance)
        position_weights[-last_portion:] *= 1.2
        
        # 4. Length normalization - normalize by sqrt of sentence length
        sentence_lengths = np.array([max(1, len(s.split())) for s in valid_sentences])
        length_normalization = np.sqrt(sentence_lengths)
        
        # 5. Combined enhanced score
        enhanced_scores = (base_scores * position_weights) / (length_normalization + 1e-6)
        
        # 6. Rank by enhanced scores
        ranked_indices = np.argsort(enhanced_scores)[::-1]
        
        # 7. Topic diversity: divide document into sections and ensure coverage
        selected = []
        num_sections = min(5, len(valid_sentences) // 10)  # Create 5 sections or less
        section_size = len(valid_sentences) // max(1, num_sections)
        
        if use_redundancy_removal and len(valid_sentences) > 3:
            model = load_sbert_model()
            embeddings = model.encode(valid_sentences, convert_to_numpy=True, normalize_embeddings=True)
            
            # Try to get at least one sentence from each section
            sections_covered = set()
            
            for idx in ranked_indices:
                if len(selected) >= top_k:
                    break
                
                # Determine which section this sentence belongs to
                section = idx // max(1, section_size)
                
                # Check if we need sentences from this section
                if section not in sections_covered or len(sections_covered) >= num_sections:
                    if not selected:
                        selected.append(idx)
                        sections_covered.add(section)
                    else:
                        # Check similarity with already selected sentences
                        current_embedding = embeddings[idx:idx+1]
                        selected_embeddings = embeddings[selected]
                        
                        similarities = cosine_similarity(current_embedding, selected_embeddings)[0]
                        
                        # Only add if not too similar to existing selections
                        if similarities.max() < similarity_threshold:
                            selected.append(idx)
                            sections_covered.add(section)
                elif not selected:
                    # Always add first sentence
                    selected.append(idx)
                    sections_covered.add(section)
                else:
                    # Check similarity before adding
                    current_embedding = embeddings[idx:idx+1]
                    selected_embeddings = embeddings[selected]
                    similarities = cosine_similarity(current_embedding, selected_embeddings)[0]
                    if similarities.max() < similarity_threshold:
                        selected.append(idx)
        else:
            # No redundancy removal, but still try for topic diversity
            sections_covered = set()
            for idx in ranked_indices:
                if len(selected) >= top_k:
                    break
                section = idx // max(1, section_size)
                if section not in sections_covered or len(sections_covered) >= num_sections:
                    selected.append(idx)
                    sections_covered.add(section)
                elif len(selected) < top_k:
                    selected.append(idx)
        
        # Sort by original order for readability
        selected.sort()
        
        # Map back to original indices and return with scores
        return [(valid_indices[idx], float(enhanced_scores[idx])) for idx in selected]
    
    except Exception as e:
        # Fallback: return first sentences if anything fails
        return [(valid_indices[i], 1.0) for i in range(min(top_k, len(valid_sentences)))]


def tfidf_rank_sentences(sentences: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Legacy function for backward compatibility.
    Now uses enhanced summarization.
    """
    return enhanced_tfidf_rank_sentences(sentences, top_k=top_k)


def make_flashcards(sentences: List[str], num_cards: int = 5) -> List[Tuple[str, str]]:
    if not sentences:
        return []
    model = load_sbert_model()
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, -1.0)

    # Pick diverse anchors using TF-IDF ranking
    ranked = tfidf_rank_sentences(sentences, top_k=min(num_cards * 2, len(sentences)))
    anchors = [i for i, _ in ranked]

    flashcards: List[Tuple[str, str]] = []
    used = set()
    for anchor in anchors:
        if len(flashcards) >= num_cards:
            break
        partner = int(np.argmax(sim[anchor]))
        if partner in used or anchor in used or partner < 0:
            continue
        q, a = cloze_from_sentence_pair(sentences[anchor], sentences[partner])
        if q and a:
            flashcards.append((q, a))
            used.add(anchor)
            used.add(partner)
    return flashcards


def top_keyword_for_sentence(sentence: str, vectorizer: TFIDFVectorizer) -> str:
    """Get top keyword from sentence using TF-IDF vectorizer"""
    matrix = vectorizer.transform([sentence])
    if matrix.shape[1] == 0 or matrix.sum() == 0:
        return ""
    idx = np.argmax(matrix[0])
    if idx < len(vectorizer.feature_names_):
        return vectorizer.feature_names_[idx]
    return ""


def cloze_from_sentence_pair(s1: str, s2: str) -> Tuple[str, str]:
    # Build a TF-IDF over the two sentences to choose a keyword to blank
    stop_words = load_default_stopwords()
    vec = TFIDFVectorizer(stop_words=stop_words, max_features=100)
    with suppress_stdout():
        _ = vec.fit_transform([s1, s2])
    keyword = top_keyword_for_sentence(s1, vec)
    if not keyword:
        # fallback to s2
        keyword = top_keyword_for_sentence(s2, vec)
    if not keyword or len(keyword) < 3:
        return "", ""
    pattern = re.compile(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE)
    question = pattern.sub("____", s1, count=1)
    answer = keyword
    return question, answer


def chatbot_answer(query: str, sentences: List[str], k: int = 3) -> str:
    if not query.strip() or not sentences:
        return ""
    
    # Use custom TF-IDF for consistency
    stop_words = load_default_stopwords()
    vectorizer = TFIDFVectorizer(stop_words=stop_words, max_features=1000)
    
    with suppress_stdout():
        doc_matrix = vectorizer.fit_transform(sentences)
        q_vec = vectorizer.transform([query])
    
    sims = cosine_similarity(q_vec, doc_matrix).ravel()
    if sims.max(initial=0.0) <= 0:
        return "I couldn't find an exact answer. Try rephrasing your question."
    top_idx = np.argsort(sims)[::-1][: max(1, k)]
    chosen = [sentences[i] for i in top_idx]
    return "\n".join(f"- {c}" for c in chosen)


def main():
    st.set_page_config(page_title="LearnMateAI", page_icon="📚", layout="wide")
    st.title("LearnMateAI: AI Study Assistant")
    st.caption("Summarize PDFs, generate flashcards, and chat with your document.")

    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    if "sentences" not in st.session_state:
        st.session_state.sentences = []

    tab_upload, tab_summary, tab_flashcards, tab_chat = st.tabs(
        ["Upload PDF", "Summary", "Flashcards", "Chat"]
    )

    with tab_upload:
        st.subheader("Upload a PDF")
        uploaded = st.file_uploader("Drag and drop your PDF here", type=["pdf"], accept_multiple_files=False)
        if uploaded is not None:
            file_bytes = uploaded.read()
            reader = DocumentReader(io.BytesIO(file_bytes))
            text = reader.extract_text()
            st.session_state.pdf_text = text
            st.session_state.sentences = split_into_sentences(text)
            st.success("PDF loaded successfully. Switch tabs to explore.")

        if st.session_state.pdf_text:
            with st.expander("Preview extracted text", expanded=False):
                st.text_area("Extracted Text", st.session_state.pdf_text[:5000], height=200)

    with tab_summary:
        st.subheader("Document Summary")
        
        
        if not st.session_state.pdf_text:
            st.info("Upload a PDF first in the Upload tab.")
        else:
            # Summary type selection
            summary_type = st.radio(
                "Summary Type",
                ["Abstractive (AI-Generated Paragraphs)", "Extractive (Key Sentences)"],
                index=0,
                help="Abstractive: AI generates new paragraphs synthesizing information. Extractive: Selects important sentences from the document."
            )
            
            if summary_type == "Abstractive (AI-Generated Paragraphs)":
                if not TRANSFORMERS_AVAILABLE:
                    st.warning("⚠️ **Transformers library not installed.** Install with: `pip install transformers torch`")
                    st.info("💡 For now, you can use the Extractive summary option below, which works without transformers.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        summary_length = st.slider(
                            "Summary Length",
                            min_value=50,
                            max_value=300,
                            value=150,
                            step=25,
                            help="Target length of summary in tokens"
                        )
                    with col2:
                        min_length = st.slider(
                            "Minimum Length",
                            min_value=30,
                            max_value=100,
                            value=50,
                            step=10
                        )
                
                if st.button("Generate Summary", type="primary", use_container_width=True):
                    if not TRANSFORMERS_AVAILABLE:
                        st.error("Please install transformers: `pip install transformers torch`")
                    else:
                        with st.spinner("Generating AI summary... This may take a moment."):
                            try:
                                summary = generate_abstractive_summary(
                                    st.session_state.pdf_text,
                                    max_length=summary_length if TRANSFORMERS_AVAILABLE else 150,
                                    min_length=min_length if TRANSFORMERS_AVAILABLE else 50,
                                    summary_type="paragraph"
                                )
                                
                                if summary and summary.strip():
                                    st.markdown("### Summary")
                                    st.markdown("---")
                                    st.write(summary)
                                    st.markdown("---")
                                    st.success(f"✓ Generated summary from {len(st.session_state.pdf_text)} characters")
                                else:
                                    st.error("Could not generate summary. Please try again or use extractive summary.")
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
                                st.info("💡 Try using the Extractive summary option, which is more reliable.")
            
            else:
                # Extractive summary (original TF-IDF method)
                st.markdown("*Using custom TF-IDF from scratch with position weighting, length normalization, and redundancy removal*")
                col1, col2 = st.columns(2)
                with col1:
                    top_n = st.slider("Number of key sentences", 5, 25, 10)
                with col2:
                    use_redundancy = st.checkbox("Remove redundant sentences", value=True, help="Avoids selecting similar sentences")
                
                if st.button("Generate Extractive Summary", type="primary", use_container_width=True):
                    with st.spinner("Generating extractive summary..."):
                        ranked = enhanced_tfidf_rank_sentences(
                            st.session_state.sentences, 
                            top_k=top_n,
                            use_redundancy_removal=use_redundancy
                        )
                    
                    if ranked:
                        st.markdown("### Key Points")
                        st.markdown("---")
                        
                        for i, (idx, score) in enumerate(ranked, 1):
                            # Validate index before accessing
                            if 0 <= idx < len(st.session_state.sentences):
                                sentence = st.session_state.sentences[idx]
                                if isinstance(sentence, str) and sentence.strip():
                                    st.markdown(f"**{i}. {sentence.strip()}**")
                                    # Show score only in expander for cleaner look
                                    with st.expander("Details", expanded=False):
                                        st.caption(f"Relevance Score: {score:.4f} | Position: {idx + 1}/{len(st.session_state.sentences)}")
                                else:
                                    st.warning(f"Invalid sentence at index {idx}")
                            else:
                                st.error(f"Index {idx} out of range (total sentences: {len(st.session_state.sentences)})")
                        
                        st.markdown("---")
                        st.info(f"✓ Summary generated from {len(st.session_state.sentences)} sentences")
                    else:
                        st.warning("Could not generate summary. Please check your document.")

    with tab_flashcards:
        st.subheader("Smart Flashcards (Sentence-BERT)")
        if not st.session_state.sentences:
            st.info("Upload a PDF first in the Upload tab.")
        else:
            num_cards = st.slider("Number of flashcards", 3, 15, 6)
            with st.spinner("Generating flashcards..."):
                cards = make_flashcards(st.session_state.sentences, num_cards=num_cards)
            if not cards:
                st.warning("Could not generate flashcards. Try a different document.")
            else:
                for i, (q, a) in enumerate(cards, start=1):
                    with st.container(border=True):
                        st.markdown(f"**Q{i}.** {q}")
                        st.markdown(f"<details><summary>Show Answer</summary><p>{a}</p></details>", unsafe_allow_html=True)

    with tab_chat:
        st.subheader("Chat with your PDF")
        if not st.session_state.sentences:
            st.info("Upload a PDF first in the Upload tab.")
        else:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.chat_message("user").markdown(msg)
                else:
                    st.chat_message("assistant").markdown(msg)

            prompt = st.chat_input("Ask a question about the PDF...")
            if prompt:
                st.session_state.chat_history.append(("user", prompt))
                answer = chatbot_answer(prompt, st.session_state.sentences, k=3)
                st.session_state.chat_history.append(("assistant", answer))
                st.chat_message("assistant").markdown(answer)


if __name__ == "__main__":
    main()


