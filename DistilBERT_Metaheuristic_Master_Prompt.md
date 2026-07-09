# MASTER PROMPT
## Performance Enhancement of DistilBERT for Twitter Sentiment Analysis Using Metaheuristic Hyperparameter Optimization

---

### HOW TO USE THIS PROMPT
Paste this entire document into an AI coding assistant (Claude Code, ChatGPT, etc.) one phase at a time, or all at once if the assistant supports long-context multi-step execution. Each phase produces working code + written analysis, matching the pipeline below. Ask the assistant to complete one phase fully, show outputs, then move to the next.

---

## ROLE & OBJECTIVE

You are an expert NLP research engineer and technical writer. Build a complete, reproducible research project titled:

**"Performance Enhancement of DistilBERT for Twitter Sentiment Analysis Using Metaheuristic Hyperparameter Optimization"**

The goal is to demonstrate that a metaheuristic optimization algorithm (Genetic Algorithm, Particle Swarm Optimization, or Grey Wolf Optimizer — you choose one and justify it) can find a hyperparameter configuration that measurably outperforms a manually/default-tuned baseline DistilBERT model on Twitter sentiment classification, across Accuracy, F1-score, Precision, Recall, and training efficiency.

Follow the exact pipeline below. Do not skip or reorder stages. For every stage, output: (a) working Python code, (b) the result/artifact it produces, (c) a short written explanation suitable for a research paper's Methodology or Results section.

**Tech stack:** Python, PyTorch, HuggingFace `transformers` + `datasets`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`/`seaborn`, and `DEAP` or `PySwarms` (or a hand-rolled implementation) for the metaheuristic algorithm. Assume a single-GPU (Colab/Kaggle T4 class) environment and keep runtime pragmatic (use a sampled subset of the dataset and a small population/iteration count if full-scale training is too slow).

---

## PIPELINE — EXECUTE IN THIS ORDER

### 1. Dataset Collection
- Use the **Sentiment140** dataset (1.6M tweets, binary labels: 0 = negative, 4 = positive) from Kaggle/Stanford, or an equivalent public Twitter sentiment dataset if unavailable.
- Write code to download/load it, rename columns clearly (`text`, `label`), and map labels to {0, 1}.
- Report dataset size, class balance, and source citation.

### 2. Exploratory Data Analysis (EDA)
- Class distribution plot (bar chart).
- Tweet length distribution (characters and tokens) — histogram.
- Most frequent words/hashtags per class (after basic cleaning) — bar chart or word cloud.
- Check for duplicates, empty tweets, non-English text, class imbalance ratio.
- Summarize 3–5 key EDA insights that will justify preprocessing decisions.

### 3. Data Preprocessing
- Lowercase, remove URLs, mentions (@user), hashtags handling (strip `#` but keep word), emojis (convert to text or strip — decide and justify), extra whitespace, HTML entities.
- Remove or keep stopwords — justify given DistilBERT uses subword tokenization (usually keep them).
- Tokenize using `DistilBertTokenizerFast` (`distilbert-base-uncased`), decide and justify `max_length` (based on EDA token-length distribution), padding, and truncation strategy.
- Handle class imbalance if found in EDA (class weights or stratified sampling — justify choice).

### 4. Train / Validation / Test Split
- Stratified split: 70% train / 15% validation / 15% test (or justify a different ratio).
- Fix a random seed for reproducibility.
- Print class distribution per split to confirm stratification worked.
- Wrap into PyTorch `Dataset`/`DataLoader` objects.

### 5. Study DistilBERT Architecture
- Write a concise (300–400 word) technical explanation covering: knowledge distillation from BERT-base, removal of token-type embeddings and pooler, 6 transformer layers vs BERT's 12, parameter count (~66M vs ~110M), and why it's suited to a resource-constrained sentiment classification task.
- Include a simple architecture diagram description (layers: embedding → 6× transformer encoder blocks → classification head).

### 6. Train Baseline DistilBERT
- Load `DistilBertForSequenceClassification` (2 labels) with HuggingFace `Trainer` or a manual PyTorch training loop.
- Use **default/reasonable manual hyperparameters** as the baseline (e.g., learning rate 2e-5, batch size 16, epochs 3, AdamW, linear warmup) — these represent "before optimization."
- Log training/validation loss curves per epoch.
- Save the baseline model checkpoint.

### 7. Evaluate Baseline Model
- Compute Accuracy, Precision, Recall, F1-score (macro and weighted), and confusion matrix on the test set.
- Report total training time and inference latency.

### 8. Analyze Performance
- Identify baseline weaknesses: misclassified examples (show 5–10), error patterns (e.g., sarcasm, negation, short tweets), and whether validation loss suggests under/overfitting.
- This analysis motivates *why* hyperparameter tuning is worth doing.

### 9. Study Metaheuristic Optimization
- Write a concise explanation of metaheuristic optimization generally, then detail your chosen algorithm specifically:
  - **Genetic Algorithm (GA):** population, chromosome encoding of hyperparameters, selection, crossover, mutation, fitness function.
  - *(or)* **Particle Swarm Optimization (PSO):** particles, velocity/position update, personal/global best.
  - *(or)* **Grey Wolf Optimizer (GWO):** alpha/beta/delta wolves, encircling/hunting behavior.
- Justify your choice versus grid search / random search / Bayesian optimization (e.g., better exploration-exploitation tradeoff, fewer evaluations needed, no gradient/surrogate model assumption).

### 10. Select Hyperparameters
Define the search space clearly, e.g.:
| Hyperparameter | Range/Options |
|---|---|
| Learning rate | 1e-5 to 5e-5 (log scale) |
| Batch size | {8, 16, 32} |
| Number of epochs | 2–5 |
| Warmup ratio | 0.0–0.2 |
| Weight decay | 0.0–0.1 |
| Dropout probability | 0.1–0.3 |

### 11. Apply Optimization Algorithm
- Implement the fitness/objective function: train DistilBERT for a given hyperparameter set (on a reduced subset/fewer epochs for search efficiency) and return validation F1-score (or a weighted combination of F1 + inverse training time).
- Implement the chosen metaheuristic loop (population size 8–15, iterations 8–15 — keep small for compute budget) using `DEAP`/`PySwarms` or custom code.
- Log every candidate's hyperparameters and fitness score to a table/CSV for transparency.

### 12. Find Best Hyperparameters
- Report the best hyperparameter configuration found and its validation fitness score.
- Plot the convergence curve (best fitness per generation/iteration) to show the algorithm improving over time.

### 13. Retrain DistilBERT
- Retrain a fresh DistilBERT model on the **full training set** using the best hyperparameters found in Step 12.
- Use the same random seed and training infrastructure as the baseline for a fair comparison.

### 14. Evaluate Optimized Model
- Compute the same metrics as Step 7 (Accuracy, Precision, Recall, F1, confusion matrix, training time, inference latency) on the identical test set.

### 15. Compare Baseline vs Optimized
- Build a comparison table (baseline vs optimized) across all metrics with % improvement.
- Run a statistical significance check if feasible (e.g., McNemar's test on paired predictions).

### 16. Visualize Results
- Bar chart: baseline vs optimized metrics side by side.
- Side-by-side confusion matrices.
- Overlaid training/validation loss curves (baseline vs optimized).
- Convergence plot of the metaheuristic search (from Step 12).

### 17. Research Conclusion
- Summarize whether metaheuristic optimization measurably improved DistilBERT's sentiment classification performance, by how much, and at what computational cost (search overhead vs performance gain).

### 18. Future Work
- Suggest 3–5 extensions: trying alternate metaheuristics (comparative study), multi-objective optimization (accuracy vs latency vs model size), applying to multi-class/emotion datasets, combining with quantization/pruning for deployment, or testing on cross-lingual tweets.

---

## DELIVERABLES CHECKLIST
- [ ] Clean, commented, runnable Python code for all 18 stages (notebook or script form)
- [ ] All plots/tables generated as described
- [ ] A written report section per stage (methodology-style prose)
- [ ] Final comparison table + conclusion paragraph
- [ ] requirements.txt / environment notes

## CONSTRAINTS
- Keep the metaheuristic search computationally light (small population, few generations, subset of data, fewer epochs per candidate) so the full pipeline can run in a Colab/Kaggle session.
- Use fixed random seeds throughout for reproducibility.
- Clearly separate "search-time" training (cheap, for hyperparameter evaluation) from "final" training (full data, best hyperparameters) so the comparison in Step 15 is fair.
