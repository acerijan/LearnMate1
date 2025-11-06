# TF-IDF Implementation from Scratch

Complete TF-IDF (Term Frequency-Inverse Document Frequency) implementation from scratch with full data preprocessing, training, and evaluation pipeline.

## Features

✅ **TF-IDF Algorithm from Scratch** - No sklearn TfidfVectorizer used  
✅ **Complete Data Preprocessing** - Data cleaning, normalization, filtering  
✅ **Naive Bayes Classifier** - Text classification using TF-IDF features  
✅ **Comprehensive Evaluation** - Confusion matrix, F1 score, accuracy  
✅ **Detailed Analysis** - Class distribution, document statistics, training metrics  
✅ **Visualization** - Interactive charts and heatmaps (Streamlit interface)

## Files

1. **`tfidf_from_scratch.py`** - Standalone script with complete implementation
2. **`tfidf_demo.py`** - Streamlit web interface for interactive visualization

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy scikit-learn matplotlib seaborn pandas streamlit
```

## Usage

### Standalone Script

Run the standalone script to see complete TF-IDF analysis:

```bash
python tfidf_from_scratch.py
```

This will:
- Load and preprocess the dataset
- Build TF-IDF vocabulary from scratch
- Calculate IDF values
- Train Naive Bayes classifier
- Evaluate on test set
- Print confusion matrix, F1 scores, and accuracy
- Generate confusion matrix visualization (if matplotlib available)

### Streamlit Interface

Run the interactive Streamlit app:

```bash
streamlit run tfidf_demo.py
```

This provides:
- Interactive configuration (max features, epochs, etc.)
- Real-time data analysis
- Visualizations (charts, confusion matrices)
- Detailed metrics tables
- Complete training pipeline visualization

## Implementation Details

### TF-IDF Formula

**Term Frequency (TF):**
```
TF(t,d) = count(t,d) / total_terms_in_d
```

**Inverse Document Frequency (IDF):**
```
IDF(t) = log(N / df(t))
```

**TF-IDF:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

Where:
- `t` = term
- `d` = document
- `N` = total number of documents
- `df(t)` = document frequency of term t

### Processing Steps

1. **Data Preprocessing**
   - Remove empty documents
   - Normalize whitespace
   - Filter very short documents

2. **Vocabulary Building**
   - Tokenize documents
   - Remove stop words
   - Filter by min/max document frequency
   - Limit vocabulary size

3. **TF-IDF Calculation**
   - Calculate term frequency for each document
   - Calculate inverse document frequency
   - Compute TF-IDF vectors

4. **Model Training**
   - Train Naive Bayes classifier
   - Calculate class priors
   - Estimate feature probabilities

5. **Evaluation**
   - Predict on test set
   - Calculate accuracy
   - Calculate F1 scores (macro, micro, weighted, per-class)
   - Generate confusion matrix

## Output Metrics

The implementation provides:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: 
  - Macro F1 (unweighted mean)
  - Micro F1 (global metrics)
  - Weighted F1 (weighted by support)
  - Per-class F1 scores
- **Confusion Matrix**: 
  - Counts matrix
  - Percentage matrix
  - Visualization (heatmap)

## Customization

### Modify TF-IDF Parameters

```python
vectorizer = TFIDFVectorizer(
    max_features=1000,    # Maximum vocabulary size
    min_df=1,            # Minimum document frequency
    max_df=0.95,         # Maximum document frequency (ratio)
    stop_words=set(),    # Custom stop words
    lowercase=True       # Convert to lowercase
)
```

### Modify Classifier Parameters

```python
classifier = NaiveBayesClassifier(
    alpha=1.0  # Laplace smoothing parameter
)
```

### Training Epochs

The Naive Bayes classifier supports multiple epochs (though for this algorithm, multiple epochs have limited effect):

```python
classifier.fit(X_train, y_train, epochs=1)
```

## Dataset

The default implementation includes a sample dataset. To use your own data:

1. Prepare your documents as a list of strings
2. Prepare your labels as a list of strings/integers
3. Replace the sample data in the `main()` function

Example:
```python
documents = ["document 1 text", "document 2 text", ...]
labels = ["class1", "class2", ...]
```

## Class Information

The implementation tracks and displays:
- Number of classes
- Class distribution
- Class priors (for Naive Bayes)
- Per-class F1 scores
- Confusion matrix breakdown by class

## Training Process

The training process shows:
- Number of training samples
- Number of features
- Vocabulary size
- Class distribution
- Training epochs
- Feature probability calculations

## Notes

- The TF-IDF implementation is completely from scratch (no sklearn TfidfVectorizer)
- Stop words can be customized
- The classifier uses Laplace smoothing to handle unseen features
- All metrics are calculated using scikit-learn for standard evaluation
- Visualization requires matplotlib and seaborn

## Example Output

```
================================================================================
TF-IDF FROM SCRATCH - TEXT CLASSIFICATION
================================================================================

STEP 1: BUILDING VOCABULARY
================================================================================
✓ Vocabulary built: 1000 unique terms
✓ Total documents processed: 30

STEP 2: CALCULATING IDF VALUES
================================================================================
✓ IDF values calculated for 1000 terms

STEP 3: TRANSFORMING DOCUMENTS TO TF-IDF VECTORS
================================================================================
✓ Transformed 21 documents to 1000-dimensional vectors

STEP 4: TRAINING NAIVE BAYES CLASSIFIER
================================================================================
✓ Training samples: 21
✓ Features per sample: 1000
✓ Classes: ['AI' 'Data Science' 'ML' 'Programming']
✓ Training epochs: 1

EVALUATION METRICS
================================================================================
Overall Accuracy: 0.8889 (88.89%)
F1 Scores:
  Macro F1: 0.8889
  Micro F1: 0.8889
  Weighted F1: 0.8889
```

## License

This implementation is provided as-is for educational purposes.

