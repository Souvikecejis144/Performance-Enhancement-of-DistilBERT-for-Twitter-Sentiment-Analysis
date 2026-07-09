# Performance Enhancement of DistilBERT for Twitter Sentiment Analysis Using Metaheuristic Hyperparameter Optimization
**Author**: Souvik Dinda  
**Date**: 2026-07-09  

---

## Abstract
This study investigates the performance enhancement of a DistilBERT model on Twitter sentiment analysis using a custom-built Genetic Algorithm (GA) metaheuristic optimizer. By optimization of hyperparameters, we show a direct performance boost in sentiment classification. Due to CPU constraints, training is performed on a subset of the Sentiment140 dataset. The results indicate that GA-discovered hyperparameters improve macro F1 score relative to default manual settings, demonstrating the efficacy of metaheuristics for model tuning.

---

## 1. Dataset Collection
We utilize the **Sentiment140** dataset, containing 1.6 million tweets annotated with polarity (0 = negative, 4 = positive).
- **Source**: Kaggle/Stanford NLP.
- **Labels Mapping**: Polarity labels mapped from {0, 4} to binary classes {0, 1}.
- **Full Dataset Size**: 1,600,000 tweets.
- **Sampled Subset**: A stratified subset of **10,000** tweets was extracted to preserve class distribution while maintaining computational viability.
- **Sampled Class Balance**:
  - Negative: 4,988 tweets
  - Positive: 4,994 tweets

---

## 2. Exploratory Data Analysis (EDA)
EDA was conducted on the sampled dataset:
1. **Class Distribution**: Perfect 50/50 balance is preserved in the stratified subset (visualized in `plots/eda_class_distribution.png`).
2. **Token Length Analysis**: We tokenized the cleaned tweets using the Fast DistilBERT Tokenizer. The 95th percentile token length is **37** tokens. Therefore, we select `max_length = 48` for padding and truncation to prevent data loss while optimizing performance.
3. **Word Frequency**: The most common words are plotted per class in `plots/eda_top_words.png`.

*EDA Plots:*
- **Class Distribution**: ![Class Distribution](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\eda_class_distribution.png)
- **Token Length**: ![Token Length Distribution](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\eda_token_length_histogram.png)
- **Top Words**: ![Top Words per Class](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\eda_top_words.png)

---

## 3. Data Preprocessing
Tweets were cleaned using the following steps:
1. **HTML Unescaping**: Resolved character references like `&amp;` -> `&`.
2. **Mentions & URLs**: Removed using regular expressions (`@username` and `http://...`).
3. **Hashtags**: Stripped the `#` symbol but kept the word token to preserve semantic meaning (e.g. `#happy` -> `happy`).
4. **Lowercasing**: All characters converted to lowercase.
5. **Whitespace**: Replaced duplicate spacing.
*Stopwords Choice*: Stopwords were **kept** because pre-trained Transformer architectures rely heavily on the context, syntax, and order of words, which is preserved through subword tokenization.

---

## 4. Train / Validation / Test Split
The data was split using stratified selection:
- **Train Set**: 70% (6,987 tweets)
- **Validation Set**: 15% (1,497 tweets)
- **Test Set**: 15% (1,498 tweets)

To facilitate quick execution under CPU constraints, model training uses a sub-sample:
- **Training Subset size**: 2000 tweets.
- **Validation Subset size**: 500 tweets.
- **Evaluation Subset size**: 500 tweets.

---

## 5. DistilBERT Architecture Study
DistilBERT is a distilled version of the BERT-base model:
- **Knowledge Distillation**: Replicates the output probabilities and hidden states of BERT-base using a student-teacher training framework.
- **Layer Reduction**: Removing token-type embeddings and pooler, reducing the number of transformer layers from 12 (BERT-base) to 6.
- **Parameter Count**: Features approximately 66M parameters compared to BERT's 110M.
- **Performance**: Retains over 97% of BERT's performance while being 40% smaller and 60% faster, making it ideal for low-resource environments.

```
+-------------------------------------------------------------+
|                     Input Text Embeddings                   |
| (Token Embeddings + Position Embeddings, 66M parameters)    |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|             6x Transformer Encoder Blocks                   |
|      (Multi-Head Self-Attention + Feed-Forward Networks)    |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                    Classification Head                      |
|            (Linear Classifier -> Dropout -> Softmax)        |
+-------------------------------------------------------------+
```

---

## 6 & 7. Train and Evaluate Baseline Model
The baseline DistilBERT model was trained with the following manual default hyperparameters:
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 1
- **Weight Decay**: 0.01
- **Dropout**: 0.1

### Baseline Test Set Results:
- **Accuracy**: 0.7200
- **Precision (Macro)**: 0.7231
- **Recall (Macro)**: 0.7236
- **F1-Score (Macro)**: 0.7200
- **F1-Score (Weighted)**: 0.7202
- **Training Time**: 216.97 seconds
- **Inference Latency**: 23.12 ms/tweet

---

## 8. Baseline Performance Weakness Analysis
A selection of misclassified tweets from the baseline model:

| Text | Cleaned Text | True Label | Predicted Label |
|---|---|---|---|
| 4 days until Paris! Can't wait.  | 4 days until paris! can't wait. | 1 | 0 |
| GOOD NIGHT TWITTERERS !!!! ill see ya'll tomorrow when i rise ( SO EARLY!!!!)  | good night twitterers !!!! ill see ya'll tomorrow when i rise ( so early!!!!) | 0 | 1 |
| @vanwas Good Morning buddy, I can't wait to see the new place  | good morning buddy, i can't wait to see the new place | 1 | 0 |
| ughhhh!!! BOOOOO for morning classes!!!  | ughhhh!!! booooo for morning classes!!! | 0 | 1 |
| @JasonBradbury Good luck, have fun... I'm off to the job centre   | good luck, have fun... i'm off to the job centre | 0 | 1 |
| Swine flu...i kissed a pig and i liked it...  | swine flu...i kissed a pig and i liked it... | 1 | 0 |
| studying for my exams  then going to danielles house | studying for my exams then going to danielles house | 0 | 1 |
| aaarrgghh in car &amp; looks like nasty traffic  #kwinanafreeway | aaarrgghh in car & looks like nasty traffic kwinanafreeway | 0 | 1 |

### Key Error Patterns:
1. **Sarcasm/Negation**: Tweets that express sarcasm or contain subtle double negatives are often misclassified because the contextual representation requires deeper modeling.
2. **Short Tweets**: Very short tweets containing heavy slang (e.g. "bummer", "cry") are sometimes misclassified as negative or positive based on singular tokens without surrounding context.
3. **Class Overfitting**: Given only 1 epoch of training, the model's loss curves show the loss is still relatively high, indicating minor underfitting.

---

## 9. Metaheuristic Optimization Study
We selected a custom **Genetic Algorithm (GA)** to perform hyperparameter optimization:
- **Justification**: Unlike Grid Search (computationally slow) or Random Search (stochastic and lacking direction), GAs balance exploration and exploitation by mimicking evolutionary biology. Candidates are selected based on fitness, combined through crossover to propagate good hyperparameter blocks, and mutated to escape local optima.
- **Search Space**:
  - **Learning rate**: 1e-5 to 5e-5 (log-scale)
  - **Batch size**: {8, 16, 32}
  - **Epochs**: {1, 2}
  - **Warmup ratio**: 0.0 to 0.2
  - **Weight decay**: 0.0 to 0.1
  - **Dropout probability**: 0.1 to 0.3

---

## 10, 11 & 12. Genetic Algorithm Optimization
The optimization loop ran for **3** generations with a population size of **4**:
- **Candidate Evaluation**: Trained on **500** samples for candidate's epochs, evaluated on **200** validation samples.
- **Fitness Function**: Macro F1 Score on the validation subset.
- **Discovery**: The GA successfully found the best hyperparameter configuration:
  - **Learning Rate**: 4.05e-05
  - **Batch Size**: 8
  - **Epochs**: 2
  - **Warmup Ratio**: 0.0490
  - **Weight Decay**: 0.0314
  - **Dropout**: 0.1109
  - **Validation Fitness**: 0.7694

*GA Convergence Plot:*
![GA Convergence](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\ga_convergence.png)

---

## 13 & 14. Train and Evaluate Optimized Model
The optimized model was retrained on the full training subset of 2000 tweets using the GA-discovered hyperparameters.

### Optimized Test Set Results:
- **Accuracy**: 0.7920
- **Precision (Macro)**: 0.7910
- **Recall (Macro)**: 0.7923
- **F1-Score (Macro)**: 0.7913
- **F1-Score (Weighted)**: 0.7922
- **Training Time**: 680.39 seconds
- **Inference Latency**: 23.00 ms/tweet

---

## 15 & 16. Comparative Analysis & Visualizations

The comparison table below details the performance change:

| Metric | Baseline | Optimized | Relative Improvement (%) |
|---|---|---|---|
| Accuracy | 0.7200 | 0.7920 | 10.00% |
| Precision (Macro) | 0.7231 | 0.7910 | 9.39% |
| Recall (Macro) | 0.7236 | 0.7923 | 9.50% |
| F1 (Macro) | 0.7200 | 0.7913 | 9.91% |
| F1 (Weighted) | 0.7202 | 0.7922 | 10.01% |
| Training Time (s) | 216.9725 | 680.3872 | 213.58% |
| Inference Latency (ms/tweet) | 23.1201 | 23.0002 | -0.52% |

### Statistical Significance (McNemar's Test)
- **Chi-Square Statistic**: 13.6111
- **P-Value**: 0.000225
- **Interpretation**: A p-value of less than 0.05 indicates statistical significance. The test results show a **statistically significant** difference between the baseline and optimized predictions.

### Visualizations
- **Metrics Comparison**: ![Metrics Comparison](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\metrics_comparison.png)
- **Confusion Matrices**: ![Confusion Matrices](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\confusion_matrices.png)
- **Loss Curves**: ![Loss Curves](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\loss_curves.png)

---

## 17. Research Conclusion
In this study, a custom Genetic Algorithm successfully optimized the hyperparameters of a DistilBERT classifier for Twitter sentiment analysis. Under CPU-constrained settings, the optimized configuration yielded a **9.91%** change in macro F1-score over the manual baseline model. Although metaheuristic optimization introduces a search overhead (running multiple candidate evaluations), the resulting model achieves superior prediction metrics. This demonstrates that bio-inspired search is an effective alternative to standard grid or random search.

---

## 18. Future Work
1. **Comparative Metaheuristic Study**: Implement Particle Swarm Optimization (PSO) or Grey Wolf Optimizer (GWO) to compare optimization efficiency against the Genetic Algorithm.
2. **Multi-Objective Optimization**: Optimize simultaneously for classification Accuracy, training efficiency, and inference latency.
3. **Quantization and Pruning**: Apply post-training quantization to the optimized model to reduce inference latency on edge devices.
4. **Cross-Lingual Evaluation**: Validate the pipeline on multi-lingual sentiment datasets (e.g. Spanish or French tweets).
5. **Scale Up Environment**: Run the pipeline on larger datasets (e.g. 100,000+ tweets) using GPU acceleration to examine search scalability.
