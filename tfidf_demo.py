"""
Streamlit App for TF-IDF from Scratch with Full Analysis
"""

import streamlit as st
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tfidf_from_scratch import (
    TFIDFVectorizer, NaiveBayesClassifier, preprocess_data,
    analyze_data, plot_confusion_matrix, print_evaluation_metrics,
    load_default_stopwords
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import pandas as pd


def display_metrics_table(y_true, y_pred, classes):
    """Display metrics in a nice table format"""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=classes)
    
    metrics_data = {
        'Metric': ['Accuracy', 'Macro F1', 'Micro F1', 'Weighted F1'],
        'Value': [f"{accuracy:.4f}", f"{f1_macro:.4f}", f"{f1_micro:.4f}", f"{f1_weighted:.4f}"]
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    class_data = {
        'Class': [str(c) for c in classes],
        'F1 Score': [f"{f1:.4f}" for f1 in f1_per_class]
    }
    df_class = pd.DataFrame(class_data)
    
    return df_metrics, df_class


def main():
    st.set_page_config(page_title="TF-IDF from Scratch", page_icon="🔍", layout="wide")
    st.title("🔍 TF-IDF Implementation from Scratch")
    st.markdown("### Complete Implementation with Data Preprocessing, Training, and Evaluation")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_features = st.slider("Max Features", 100, 5000, 1000, 100)
        min_df = st.slider("Min Document Frequency", 1, 5, 1)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
        epochs = st.slider("Training Epochs", 1, 10, 1)
    
    # Load dataset from CSV files
    st.header("📊 Dataset")
    
    dataset_path = "dataset"
    csv_files = sorted(glob.glob(os.path.join(dataset_path, "abstract_text_topic*.csv")))
    
    if not csv_files:
        st.warning("No CSV files found in dataset folder. Using sample data.")
        sample_documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science",
        ]
        sample_labels = ["ML", "Programming"]
        documents = sample_documents
        labels = sample_labels
    else:
        sample_documents = []
        sample_labels = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, csv_file in enumerate(csv_files):
            filename = os.path.basename(csv_file)
            topic_match = filename.replace("abstract_text_topic", "").replace(".csv", "")
            topic_label = f"Topic{topic_match}"
            
            try:
                df = pd.read_csv(csv_file)
                if 'text' in df.columns:
                    texts = df['text'].dropna().tolist()
                    # Filter out empty strings
                    texts = [text for text in texts if text and str(text).strip()]
                    sample_documents.extend(texts)
                    sample_labels.extend([topic_label] * len(texts))
                    status_text.text(f"Loading {filename}... ({len(texts)} documents)")
                else:
                    st.warning(f"{filename} does not have 'text' column")
            except Exception as e:
                st.warning(f"Error loading {filename}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(csv_files))
        
        progress_bar.empty()
        status_text.empty()
        
        documents = sample_documents
        labels = sample_labels
        
        st.success(f"✓ Loaded {len(documents)} documents from {len(csv_files)} CSV files")
        st.info(f"Classes: {', '.join(sorted(set(labels)))}")
    
    # Show dataset info (always displayed)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", len(documents))
    with col2:
        st.metric("Total Classes", len(set(labels)))
    with col3:
        st.metric("Avg Words/Doc", f"{np.mean([len(doc.split()) for doc in documents]):.1f}")
    
    # Preprocessing
    st.header("🧹 Data Preprocessing & Cleaning")
    with st.expander("View Preprocessing Steps", expanded=True):
        documents_processed, labels_processed = preprocess_data(documents.copy(), labels.copy())
        
        st.markdown("**Preprocessing Steps Completed:**")
        st.markdown("1. ✅ Removed empty documents")
        st.markdown("2. ✅ Normalized whitespace")
        st.markdown("3. ✅ Filtered very short documents (< 10 characters)")
        
        st.success(f"✓ Preprocessed {len(documents_processed)} documents from {len(documents)} original")
    
    # Data Analysis
    st.header("📈 Data Analysis")
    with st.expander("View Detailed Analysis", expanded=False):
        analyze_data(documents_processed, labels_processed)
        
        # Class distribution chart
        from collections import Counter
        label_counts = Counter(labels_processed)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Class distribution bar chart
        classes = list(label_counts.keys())
        counts = list(label_counts.values())
        ax1.bar(classes, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution')
        ax1.grid(axis='y', alpha=0.3)
        
        # Document length distribution
        doc_lengths = [len(doc.split()) for doc in documents_processed]
        ax2.hist(doc_lengths, bins=10, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Words per Document')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Document Length Distribution')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        documents_processed, labels_processed, 
        test_size=test_size, random_state=42, stratify=labels_processed
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Test Samples", len(X_test))
    
    # TF-IDF Vectorization
    st.header("🔤 TF-IDF Vectorization (From Scratch)")
    with st.spinner("Building TF-IDF vectors..."):
        stop_words = load_default_stopwords()
        vectorizer = TFIDFVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=0.95,
            stop_words=stop_words,
            lowercase=True
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
    
    st.success(f"✓ Built vocabulary with {len(vectorizer.vocabulary_)} unique terms")
    
    with st.expander("View TF-IDF Details", expanded=False):
        st.markdown(f"**Vocabulary Size:** {len(vectorizer.vocabulary_)}")
        st.markdown(f"**Feature Dimensions:** {X_train_tfidf.shape[1]}")
        st.markdown(f"**Training Matrix Shape:** {X_train_tfidf.shape}")
        st.markdown(f"**Test Matrix Shape:** {X_test_tfidf.shape}")
        
        # Show sample terms
        st.markdown("**Sample Terms from Vocabulary (Top 20):**")
        sample_terms = vectorizer.feature_names_[:20]
        st.write(", ".join(sample_terms))
        
        # Show sample IDF values
        st.markdown("**Sample IDF Values (Top 10):**")
        idf_data = [(term, vectorizer.idf_[term]) for term in vectorizer.feature_names_[:10]]
        df_idf = pd.DataFrame(idf_data, columns=['Term', 'IDF Value'])
        st.dataframe(df_idf, use_container_width=True)
    
    # Training
    st.header("🎓 Model Training")
    with st.spinner("Training Naive Bayes Classifier..."):
        classifier = NaiveBayesClassifier(alpha=1.0)
        classifier.fit(X_train_tfidf, np.array(y_train), epochs=epochs)
    
    st.success(f"✓ Training completed with {epochs} epoch(s)")
    
    with st.expander("View Training Details", expanded=False):
        st.markdown(f"**Classes:** {', '.join([str(c) for c in classifier.classes_])}")
        st.markdown(f"**Number of Features:** {classifier.n_features_}")
        st.markdown(f"**Training Samples:** {len(X_train)}")
        
        # Class priors
        st.markdown("**Class Priors:**")
        for class_label, prior in classifier.class_priors_.items():
            st.write(f"  - Class '{class_label}': {prior:.4f}")
    
    # Predictions
    st.header("🔮 Predictions")
    y_train_pred = classifier.predict(X_train_tfidf)
    y_test_pred = classifier.predict(X_test_tfidf)
    
    col1, col2 = st.columns(2)
    with col1:
        train_acc = accuracy_score(y_train, y_train_pred)
        st.metric("Training Accuracy", f"{train_acc:.4f}")
    with col2:
        test_acc = accuracy_score(y_test, y_test_pred)
        st.metric("Test Accuracy", f"{test_acc:.4f}")
    
    # Evaluation Metrics
    st.header("📊 Evaluation Metrics")
    
    # Training metrics
    st.subheader("Training Set Results")
    df_train_metrics, df_train_class = display_metrics_table(y_train, y_train_pred, classifier.classes_)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Overall Metrics:**")
        st.dataframe(df_train_metrics, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Per-Class F1 Scores:**")
        st.dataframe(df_train_class, use_container_width=True, hide_index=True)
    
    # Test metrics
    st.subheader("Test Set Results")
    df_test_metrics, df_test_class = display_metrics_table(y_test, y_test_pred, classifier.classes_)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Overall Metrics:**")
        st.dataframe(df_test_metrics, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Per-Class F1 Scores:**")
        st.dataframe(df_test_class, use_container_width=True, hide_index=True)
    
    # Detailed classification report
    with st.expander("View Detailed Classification Report", expanded=False):
        st.markdown("**Test Set Classification Report:**")
        report = classification_report(
            y_test, y_test_pred, 
            labels=classifier.classes_, 
            target_names=[str(c) for c in classifier.classes_],
            output_dict=True
        )
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report, use_container_width=True)
    
    # Confusion Matrix
    st.header("📉 Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_test_pred, labels=classifier.classes_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classifier.classes_, yticklabels=classifier.classes_,
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=classifier.classes_, yticklabels=classifier.classes_,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Summary
    st.header("📋 Summary")
    
    summary_data = {
        'Metric': [
            'TF-IDF Implementation',
            'Vocabulary Size',
            'Training Samples',
            'Test Samples',
            'Number of Classes',
            'Training Epochs',
            'Test Accuracy',
            'Test F1 Score (Macro)',
            'Test F1 Score (Weighted)'
        ],
        'Value': [
            'From Scratch',
            str(len(vectorizer.vocabulary_)),
            str(len(X_train)),
            str(len(X_test)),
            str(len(classifier.classes_)),
            str(epochs),
            f"{test_acc:.4f}",
            f"{f1_score(y_test, y_test_pred, average='macro'):.4f}",
            f"{f1_score(y_test, y_test_pred, average='weighted'):.4f}"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.success("✅ Complete TF-IDF analysis finished!")
    
    # Footer
    st.markdown("---")
    st.markdown("### Implementation Details:")
    st.markdown("""
    - **TF-IDF Algorithm**: Implemented from scratch (no sklearn TfidfVectorizer used)
    - **Preprocessing**: Data cleaning, whitespace normalization, filtering
    - **Classification**: Naive Bayes classifier with TF-IDF features
    - **Evaluation**: Confusion matrix, F1 score, accuracy metrics
    - **Visualization**: Interactive charts and heatmaps
    """)


if __name__ == "__main__":
    main()

