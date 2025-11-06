"""
TF-IDF Implementation from Scratch
Complete implementation with data preprocessing, training, and evaluation
"""

import re
import math
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class TFIDFVectorizer:
    """
    TF-IDF Vectorizer implemented from scratch
    """
    
    def __init__(self, max_features: int = None, min_df: int = 1, max_df: float = 1.0, 
                 stop_words: Set[str] = None, lowercase: bool = True):
        """
        Initialize TF-IDF Vectorizer
        
        Args:
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency (int) or ratio (float)
            max_df: Maximum document frequency (int) or ratio (float)
            stop_words: Set of stop words to remove
            lowercase: Whether to convert to lowercase
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words or set()
        self.lowercase = lowercase
        self.vocabulary_ = {}
        self.idf_ = {}
        self.feature_names_ = []
        self.n_documents_ = 0
        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess a single text: lowercase, tokenize, remove stop words
        """
        if self.lowercase:
            text = text.lower()
        
        # Tokenize: split on non-word characters
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def _build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from documents
        """
        # Count document frequency for each term
        doc_freq = defaultdict(int)
        all_terms = []
        
        for doc in documents:
            tokens = self._preprocess_text(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
            all_terms.extend(tokens)
        
        self.n_documents_ = len(documents)
        
        # Filter by min_df and max_df
        min_doc_freq = self.min_df if isinstance(self.min_df, int) else int(self.min_df * self.n_documents_)
        max_doc_freq = self.max_df if isinstance(self.max_df, int) else int(self.max_df * self.n_documents_)
        
        # Filter vocabulary
        filtered_vocab = {}
        for term, df in doc_freq.items():
            if min_doc_freq <= df <= max_doc_freq:
                filtered_vocab[term] = df
        
        # Sort by frequency and limit to max_features
        sorted_vocab = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
        if self.max_features:
            sorted_vocab = sorted_vocab[:self.max_features]
        
        # Create vocabulary mapping
        vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_vocab)}
        self.feature_names_ = [term for term, _ in sorted_vocab]
        
        return vocabulary
    
    def _calculate_idf(self, documents: List[str]) -> Dict[str, float]:
        """
        Calculate Inverse Document Frequency (IDF) for each term
        IDF(t) = log(N / df(t)) where N is total documents, df(t) is document frequency
        """
        doc_freq = defaultdict(int)
        
        for doc in documents:
            tokens = self._preprocess_text(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.vocabulary_:
                    doc_freq[token] += 1
        
        idf = {}
        for term in self.vocabulary_:
            df = doc_freq.get(term, 1)  # Avoid division by zero
            idf[term] = math.log(self.n_documents_ / (df + 1))  # Add 1 for smoothing
        
        return idf
    
    def fit(self, documents: List[str]):
        """
        Fit the vectorizer on documents
        """
        print("=" * 80)
        print("STEP 1: BUILDING VOCABULARY")
        print("=" * 80)
        
        # Build vocabulary
        self.vocabulary_ = self._build_vocabulary(documents)
        print(f"✓ Vocabulary built: {len(self.vocabulary_)} unique terms")
        print(f"✓ Total documents processed: {self.n_documents_}")
        
        print("\n" + "=" * 80)
        print("STEP 2: CALCULATING IDF VALUES")
        print("=" * 80)
        
        # Calculate IDF
        self.idf_ = self._calculate_idf(documents)
        print(f"✓ IDF values calculated for {len(self.idf_)} terms")
        
        # Show sample IDF values
        sample_terms = list(self.idf_.items())[:10]
        print("\nSample IDF values (top 10):")
        for term, idf_val in sample_terms:
            print(f"  '{term}': {idf_val:.4f}")
        
        return self
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate Term Frequency (TF) for a document
        TF(t,d) = count(t,d) / total_terms_in_d
        """
        term_counts = Counter(tokens)
        total_terms = len(tokens)
        
        if total_terms == 0:
            return {}
        
        tf = {}
        for term, count in term_counts.items():
            if term in self.vocabulary_:
                tf[term] = count / total_terms
        
        return tf
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors
        TF-IDF(t,d) = TF(t,d) * IDF(t)
        """
        print("\n" + "=" * 80)
        print("STEP 3: TRANSFORMING DOCUMENTS TO TF-IDF VECTORS")
        print("=" * 80)
        
        vectors = []
        for idx, doc in enumerate(documents):
            tokens = self._preprocess_text(doc)
            tf = self._calculate_tf(tokens)
            
            # Create TF-IDF vector
            vector = np.zeros(len(self.vocabulary_))
            for term, tf_val in tf.items():
                term_idx = self.vocabulary_[term]
                idf_val = self.idf_[term]
                vector[term_idx] = tf_val * idf_val
            
            vectors.append(vector)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(documents)} documents")
        
        print(f"✓ Transformed {len(documents)} documents to {len(self.vocabulary_)}-dimensional vectors")
        
        return np.array(vectors)
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit and transform in one step
        """
        self.fit(documents)
        return self.transform(documents)


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier for text classification using TF-IDF features
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Naive Bayes Classifier
        
        Args:
            alpha: Smoothing parameter (Laplace smoothing)
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = {}
        self.feature_probs_ = {}  # {class: {feature_idx: probability}}
        self.n_features_ = 0
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1):
        """
        Train the Naive Bayes classifier
        
        Args:
            X: TF-IDF feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            epochs: Number of training epochs (for iterative algorithms)
        """
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING NAIVE BAYES CLASSIFIER")
        print("=" * 80)
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        print(f"✓ Training samples: {n_samples}")
        print(f"✓ Features per sample: {self.n_features_}")
        print(f"✓ Classes: {self.classes_}")
        print(f"✓ Training epochs: {epochs}")
        
        # Calculate class priors
        print("\n" + "-" * 80)
        print("Calculating class priors...")
        print("-" * 80)
        
        for class_label in self.classes_:
            class_mask = y == class_label
            self.class_priors_[class_label] = np.sum(class_mask) / n_samples
            print(f"  Class '{class_label}': {np.sum(class_mask)} samples ({self.class_priors_[class_label]:.4f} prior)")
        
        # Calculate feature probabilities for each class
        print("\n" + "-" * 80)
        print("Calculating feature probabilities (P(feature|class))...")
        print("-" * 80)
        
        self.feature_probs_ = {class_label: {} for class_label in self.classes_}
        
        for epoch in range(epochs):
            if epoch > 0:
                print(f"\nEpoch {epoch + 1}/{epochs}:")
            
            for class_label in self.classes_:
                class_mask = y == class_label
                class_samples = X[class_mask]
                
                # Sum of TF-IDF values for each feature in this class
                feature_sums = np.sum(class_samples, axis=0)
                total_sum = np.sum(feature_sums) + self.alpha * self.n_features_
                
                # Calculate probabilities with smoothing
                for feature_idx in range(self.n_features_):
                    feature_sum = feature_sums[feature_idx]
                    prob = (feature_sum + self.alpha) / total_sum
                    self.feature_probs_[class_label][feature_idx] = prob
                
                if epoch == 0:
                    print(f"  Class '{class_label}': Calculated probabilities for {self.n_features_} features")
        
        print("\n✓ Training completed!")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, sample in enumerate(X):
            class_probs = []
            for class_label in self.classes_:
                # Log probability to avoid underflow
                log_prob = math.log(self.class_priors_[class_label])
                
                for feature_idx in range(self.n_features_):
                    feature_value = sample[feature_idx]
                    if feature_value > 0:
                        feature_prob = self.feature_probs_[class_label].get(feature_idx, self.alpha)
                        log_prob += feature_value * math.log(feature_prob + 1e-10)
                
                class_probs.append(log_prob)
            
            # Convert log probabilities to probabilities
            class_probs = np.array(class_probs)
            class_probs = class_probs - np.max(class_probs)  # Normalize
            class_probs = np.exp(class_probs)
            probabilities[i] = class_probs / np.sum(class_probs)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        """
        probabilities = self.predict_proba(X)
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        return predictions


def load_default_stopwords() -> Set[str]:
    """
    Load default English stop words
    """
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'said', 'each', 'which', 'their', 'if',
        'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
        'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
        'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
        'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
        'come', 'made', 'may', 'part'
    }
    return stop_words


def preprocess_data(documents: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """
    Preprocess and clean the data
    
    Returns:
        cleaned_documents, labels
    """
    print("=" * 80)
    print("DATA PREPROCESSING AND CLEANING")
    print("=" * 80)
    
    print(f"\nInitial data:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Total labels: {len(labels)}")
    
    # Data cleaning steps
    cleaned_documents = []
    cleaned_labels = []
    
    print("\n" + "-" * 80)
    print("Cleaning steps:")
    print("-" * 80)
    
    # Step 1: Remove empty documents
    print("\n1. Removing empty documents...")
    initial_count = len(documents)
    for doc, label in zip(documents, labels):
        if doc and doc.strip():
            cleaned_documents.append(doc.strip())
            cleaned_labels.append(label)
    removed = initial_count - len(cleaned_documents)
    print(f"   ✓ Removed {removed} empty documents")
    print(f"   ✓ Remaining: {len(cleaned_documents)} documents")
    
    # Step 2: Remove extra whitespace
    print("\n2. Normalizing whitespace...")
    for i in range(len(cleaned_documents)):
        cleaned_documents[i] = re.sub(r'\s+', ' ', cleaned_documents[i])
    print("   ✓ Normalized whitespace in all documents")
    
    # Step 3: Remove very short documents
    print("\n3. Filtering very short documents (< 10 characters)...")
    final_documents = []
    final_labels = []
    for doc, label in zip(cleaned_documents, cleaned_labels):
        if len(doc) >= 10:
            final_documents.append(doc)
            final_labels.append(label)
    removed_short = len(cleaned_documents) - len(final_documents)
    print(f"   ✓ Removed {removed_short} very short documents")
    print(f"   ✓ Final count: {len(final_documents)} documents")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    
    return final_documents, final_labels


def analyze_data(documents: List[str], labels: List[str]):
    """
    Analyze the dataset
    """
    print("\n" + "=" * 80)
    print("DATA ANALYSIS")
    print("=" * 80)
    
    # Class distribution
    from collections import Counter
    label_counts = Counter(labels)
    
    print("\nClass Distribution:")
    print("-" * 80)
    for label, count in label_counts.most_common():
        percentage = (count / len(labels)) * 100
        print(f"  Class '{label}': {count} samples ({percentage:.2f}%)")
    
    # Document length statistics
    doc_lengths = [len(doc.split()) for doc in documents]
    
    print("\nDocument Statistics:")
    print("-" * 80)
    print(f"  Average words per document: {np.mean(doc_lengths):.2f}")
    print(f"  Median words per document: {np.median(doc_lengths):.2f}")
    print(f"  Min words per document: {np.min(doc_lengths)}")
    print(f"  Max words per document: {np.max(doc_lengths)}")
    print(f"  Standard deviation: {np.std(doc_lengths):.2f}")
    
    # Vocabulary size estimate
    all_words = set()
    for doc in documents:
        words = re.findall(r'\b\w+\b', doc.lower())
        all_words.update(words)
    
    print(f"\n  Estimated vocabulary size: {len(all_words)} unique words")
    
    print("\n" + "=" * 80)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (Counts)")
    print("=" * 80)
    print(cm)
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (Percentages)")
    print("=" * 80)
    for i, class_label in enumerate(classes):
        print(f"\nClass '{class_label}':")
        for j, pred_label in enumerate(classes):
            print(f"  Predicted as '{pred_label}': {cm_percent[i, j]:.2f}%")
    
    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        return plt
    else:
        print("\nNote: Matplotlib not available. Skipping plot generation.")
        return None


def print_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray):
    """
    Print detailed evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=classes)
    
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nF1 Scores:")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Micro F1: {f1_micro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for class_label, f1_val in zip(classes, f1_per_class):
        print(f"  Class '{class_label}': {f1_val:.4f}")
    
    # Classification report
    print("\n" + "-" * 80)
    print("Detailed Classification Report:")
    print("-" * 80)
    print(classification_report(y_true, y_pred, labels=classes, target_names=[str(c) for c in classes]))
    
    print("\n" + "=" * 80)


def main():
    """
    Main function demonstrating TF-IDF from scratch with classification
    """
    print("\n" + "=" * 80)
    print("TF-IDF FROM SCRATCH - TEXT CLASSIFICATION")
    print("=" * 80)
    
    # Load dataset from CSV files in dataset folder
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    dataset_path = "dataset"
    csv_files = sorted(glob.glob(os.path.join(dataset_path, "abstract_text_topic*.csv")))
    
    if not csv_files:
        print("⚠ Warning: No CSV files found in dataset folder. Using sample data.")
        sample_documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science",
        ]
        sample_labels = ["ML", "Programming"]
    else:
        sample_documents = []
        sample_labels = []
        
        for csv_file in csv_files:
            # Extract topic number from filename (e.g., "abstract_text_topic1.csv" -> "Topic1")
            filename = os.path.basename(csv_file)
            # Extract topic number
            topic_match = filename.replace("abstract_text_topic", "").replace(".csv", "")
            topic_label = f"Topic{topic_match}"
            
            # Read CSV file
            try:
                df = pd.read_csv(csv_file)
                if 'text' in df.columns:
                    texts = df['text'].dropna().tolist()
                    # Filter out empty strings
                    texts = [text for text in texts if text and str(text).strip()]
                    sample_documents.extend(texts)
                    sample_labels.extend([topic_label] * len(texts))
                    print(f"✓ Loaded {len(texts)} documents from {filename} (Label: {topic_label})")
                else:
                    print(f"⚠ Warning: {filename} does not have 'text' column")
            except Exception as e:
                print(f"⚠ Error loading {filename}: {str(e)}")
        
        print(f"\n✓ Total documents loaded: {len(sample_documents)}")
        print(f"✓ Total labels: {len(sample_labels)}")
        print(f"✓ Unique classes: {sorted(set(sample_labels))}")
    
    # Preprocess data
    documents, labels = preprocess_data(sample_documents, sample_labels)
    
    # Analyze data
    analyze_data(documents, labels)
    
    # Split data
    print("\n" + "=" * 80)
    print("SPLITTING DATA INTO TRAIN AND TEST SETS")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        documents, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Initialize and fit TF-IDF vectorizer
    stop_words = load_default_stopwords()
    vectorizer = TFIDFVectorizer(
        max_features=1000,
        min_df=1,
        max_df=0.95,
        stop_words=stop_words,
        lowercase=True
    )
    
    # Fit and transform
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train classifier
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.fit(X_train_tfidf, np.array(y_train), epochs=1)
    
    # Predictions
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)
    y_train_pred = classifier.predict(X_train_tfidf)
    y_test_pred = classifier.predict(X_test_tfidf)
    
    print(f"✓ Training predictions completed")
    print(f"✓ Test predictions completed")
    
    # Evaluation
    print("\n" + "=" * 80)
    print("TRAINING SET RESULTS")
    print("=" * 80)
    print_evaluation_metrics(np.array(y_train), y_train_pred, classifier.classes_)
    
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print_evaluation_metrics(np.array(y_test), y_test_pred, classifier.classes_)
    
    # Plot confusion matrix
    fig = plot_confusion_matrix(np.array(y_test), y_test_pred, classifier.classes_)
    if fig is not None:
        fig.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved to 'confusion_matrix.png'")
        plt.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - TF-IDF implemented from scratch")
    print(f"  - Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  - Classes: {classifier.classes_}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  - Test F1 Score (Macro): {f1_score(y_test, y_test_pred, average='macro'):.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

