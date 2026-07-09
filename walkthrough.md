# Walkthrough - Performance Enhancement of DistilBERT for Twitter Sentiment Analysis Using Metaheuristic Hyperparameter Optimization

We have completed the implementation and execution of the entire 18-stage sentiment classification research pipeline. Under CPU-only constraints, the pipeline successfully completed baseline training, Genetic Algorithm optimization, retraining, and full comparison.

---

## 1. Key Results

The table below summarizes the final performance metrics on the test set (1,500 samples):

| Metric | Baseline | Optimized | Relative Improvement (%) |
|---|---|---|---|
| **Accuracy** | 0.7200 | 0.7920 | **+10.00%** |
| **Precision (Macro)** | 0.7231 | 0.7910 | **+9.39%** |
| **Recall (Macro)** | 0.7236 | 0.7923 | **+9.50%** |
| **F1-Score (Macro)** | 0.7200 | 0.7913 | **+9.91%** |
| **F1-Score (Weighted)** | 0.7202 | 0.7922 | **+10.01%** |
| **Training Time (s)** | 216.97s | 680.39s | +213.58% |
| **Inference Latency** | 23.12 ms/tweet | 23.00 ms/tweet | -0.52% |

### Statistical Significance (McNemar's Test)
- **Chi-Square Statistic**: 13.6111
- **P-Value**: 0.000225 (p < 0.001)
- **Conclusion**: The improvement achieved by the optimized model is **statistically significant**, confirming that metaheuristic hyperparameter optimization yields a reliable performance boost on Twitter sentiment classification.

---

## 2. Optimal Hyperparameters Discovered
Our custom Genetic Algorithm search (3 generations, population of 4) discovered the following optimal hyperparameters:
- **Learning Rate**: $4.05 \times 10^{-5}$
- **Batch Size**: 8
- **Epochs**: 2
- **Warmup Ratio**: 0.0490
- **Weight Decay**: 0.0314
- **Dropout Probability**: 0.1109
- **Validation Fitness (Val F1 Macro)**: 0.7694

---

## 3. Visualizations

Here are the visual artifacts generated during the pipeline run:

### Exploratory Data Analysis
- **Class Distribution**: Shows a balanced stratified subset of 10,000 tweets.
  ![Class Distribution](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\eda_class_distribution.png)
- **Token Length Histogram**: Displays the token distribution after text cleaning. This justified choosing `max_length = 48` (covering the 95th percentile).
  ![Token Length Histogram](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\eda_token_length_histogram.png)
- **Top 10 Words per Class**: The distribution of frequent tokens after stripping HTML, URLs, and mentions.
  ![Top Words](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\eda_top_words.png)

### Optimization and Convergence
- **GA Optimization Convergence**: Tracks the best F1-score across generations.
  ![GA Convergence](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\ga_convergence.png)

### Model Performance Comparison
- **Performance Metrics Comparison**: Bar chart comparing baseline and optimized model scores.
  ![Metrics Comparison](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\metrics_comparison.png)
- **Confusion Matrices Comparison**:
  ![Confusion Matrices](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\confusion_matrices.png)
- **Loss Curves (Baseline vs. Optimized)**: Overlaid training and validation cross-entropy loss curves.
  ![Loss Curves](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\loss_curves.png)

---

## 4. Model Quantization & Explainability

We implemented post-training dynamic quantization and attention explainability visualization in `run_inference_opt.py`.

### 4.1 PyTorch Dynamic Quantization (Option 2)
By converting the model's linear layers from float32 to int8, we significantly improved inference speed on the CPU and minimized the storage footprint:
- **Standard Model Size**: 255.43 MB
- **Quantized Model Size**: 132.29 MB (a **1.93x footprint reduction**)
- **Standard Model Latency**: 18.14 ms/tweet
- **Quantized Model Latency**: 11.33 ms/tweet (a **1.60x CPU speedup**)

### 4.2 Attention-Based Explainability (Option 4)
We extracted the attention weights from the `[CLS]` token in the final layer to identify which words the model attended to most for its predictions. Below is the generated visual attribution chart for the sample tweet:
*\"I absolutely love this new update! It runs incredibly fast and makes everything so much easier.\"* (Predicted: **POSITIVE** with **98.89%** confidence)

![Attention Explainability](C:\Users\souvi\.gemini\antigravity-ide\brain\240cb6e3-c7fe-4061-93f7-d44d9583d6f0\plots\attention_explainability.png)

---

## 5. Deliverables Checklists
- [x] Clean, commented, runnable Python code (`pipeline.py` and `run_inference_opt.py`)
- [x] All plots generated in `plots/`
- [x] Comprehensive research paper/report (`research_report.md`)
- [x] Baseline vs. optimized comparison table (`model_comparison.csv`)
- [x] Quantization and speedup benchmarks (`quantization_benchmarks.csv`)
- [x] Requirements.txt file
