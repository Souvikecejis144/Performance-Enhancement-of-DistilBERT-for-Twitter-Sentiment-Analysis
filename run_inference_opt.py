import os
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Define directories
MODEL_DIR = './optimized_model'
QUANT_MODEL_PATH = './quantized_model.pt'
PLOTS_DIR = './plots'
BRAIN_PLOTS_DIR = 'C:/Users/souvi/.gemini/antigravity-ide/brain/240cb6e3-c7fe-4061-93f7-d44d9583d6f0/plots'

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(BRAIN_PLOTS_DIR, exist_ok=True)

def benchmark_inference(model, text_list, tokenizer, num_runs=5):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=48)
    
    # Warmup runs
    for _ in range(3):
        with torch.no_grad():
            _ = model(**inputs)
            
    # Measure latency
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**inputs)
    avg_time = (time.time() - start) / num_runs / len(text_list)
    return avg_time

def get_attention_weights(model, text, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=48)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # attentions is a tuple of length 6 (one per transformer layer)
    # Each layer's attention tensor has shape: (batch_size, num_heads, sequence_length, sequence_length)
    attentions = outputs.attentions
    
    # Extract attention weights from the final layer (layer 5, which is the 6th layer)
    last_layer_attention = attentions[-1][0]  # shape: (num_heads, seq_len, seq_len)
    
    # Average across all attention heads
    avg_attention = torch.mean(last_layer_attention, dim=0)  # shape: (seq_len, seq_len)
    
    # Get the attention weights from the [CLS] token (index 0) to all other tokens
    cls_attention = avg_attention[0]  # shape: (seq_len,)
    attention_scores = cls_attention.cpu().tolist()
    
    # Extract input IDs and map back to tokens
    input_ids = inputs['input_ids'][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Filter out [PAD] tokens for cleaner visualization
    clean_tokens = []
    clean_scores = []
    for tok, score in zip(tokens, attention_scores):
        if tok not in ['[PAD]']:
            clean_tokens.append(tok)
            clean_scores.append(score)
            
    # Normalize scores to sum to 1 for relative visual clarity
    sum_scores = sum(clean_scores) if sum(clean_scores) > 0 else 1.0
    norm_scores = [s / sum_scores for s in clean_scores]
            
    return clean_tokens, norm_scores

def plot_attention_importance(tokens, scores, text, prediction, confidence, filename='plots/attention_explainability.png'):
    # Clean tokens for plotting (remove [CLS], [SEP] but keep them visible as boundaries)
    plt.figure(figsize=(10, len(tokens) * 0.4 + 2))
    
    # Use color map based on importance
    colors = sns.color_palette("viridis", len(tokens))
    sns.barplot(x=scores, y=tokens, palette='viridis', hue=tokens, legend=False)
    
    plt.xlabel('Normalized Self-Attention Importance')
    plt.ylabel('Tokens')
    plt.title(f"Self-Attention Explainability Attribution\nTweet: \"{text[:60]}...\"\nPrediction: {prediction} ({confidence*100:.1f}% Confidence)")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # Copy to brain folder for artifact linking
    shutil.copy(filename, os.path.join(BRAIN_PLOTS_DIR, 'attention_explainability.png'))
    print(f"Explainability plot successfully saved to {filename} and copied to brain artifact directory.")

def main():
    print("====================================================")
    print("Loading optimized DistilBERT and Tokenizer...")
    print("====================================================")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, attn_implementation="eager")
    
    # ----------------------------------------------------
    # Stage 1: PyTorch Dynamic Quantization
    # ----------------------------------------------------
    print("\n[Option 2] Applying PyTorch Dynamic Quantization...")
    # Quantize nn.Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Save the quantized model state dict
    torch.save(quantized_model.state_dict(), QUANT_MODEL_PATH)
    print(f"Quantized model state saved to {QUANT_MODEL_PATH}")
    
    # Compare model file sizes
    std_model_path = os.path.join(MODEL_DIR, 'model.safetensors')
    std_size_mb = os.path.getsize(std_model_path) / (1024 * 1024)
    quant_size_mb = os.path.getsize(QUANT_MODEL_PATH) / (1024 * 1024)
    reduction_ratio = std_size_mb / quant_size_mb
    print(f"Standard Model size:  {std_size_mb:.2f} MB")
    print(f"Quantized Model size: {quant_size_mb:.2f} MB")
    print(f"Size Reduction:       {reduction_ratio:.2f}x smaller")
    
    # ----------------------------------------------------
    # Stage 2: Inference Speed Benchmarking
    # ----------------------------------------------------
    print("\nBenchmarking speed performance on 50 sample tweets...")
    csv_file = 'training.1600000.processed.noemoticon.csv'
    df = pd.read_csv(csv_file, encoding='latin-1', header=None)
    df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    
    # Draw sample
    sample_tweets = df['text'].sample(50, random_state=RANDOM_SEED).tolist()
    
    # Measure latency on standard model
    print("Benchmarking standard model...")
    std_latency = benchmark_inference(model, sample_tweets, tokenizer)
    print(f"Standard Model Avg Latency:  {std_latency * 1000:.2f} ms/tweet")
    
    # Measure latency on quantized model
    print("Benchmarking quantized model...")
    quant_latency = benchmark_inference(quantized_model, sample_tweets, tokenizer)
    print(f"Quantized Model Avg Latency: {quant_latency * 1000:.2f} ms/tweet")
    
    speedup = std_latency / quant_latency
    print(f"Quantization CPU Speedup:    {speedup:.2f}x faster inference")
    
    # Save benchmark metrics to CSV
    benchmark_df = pd.DataFrame({
        'Model Type': ['Standard', 'Quantized'],
        'Model Size (MB)': [std_size_mb, quant_size_mb],
        'Avg Latency (ms/tweet)': [std_latency * 1000, quant_latency * 1000],
        'Relative Footprint': ['1.0x', f'1/{reduction_ratio:.1f}x'],
        'Speedup Factor': ['1.0x', f'{speedup:.2f}x']
    })
    benchmark_df.to_csv('quantization_benchmarks.csv', index=False)
    print("Saved benchmarks to quantization_benchmarks.csv")

    # ----------------------------------------------------
    # Stage 3: Explainability and Attention Visualization (Option 4)
    # ----------------------------------------------------
    print("\n[Option 4] Generating Attention Explainability for a sample tweet...")
    sample_tweet = "I absolutely love this new update! It runs incredibly fast and makes everything so much easier."
    print(f"Test Tweet: \"{sample_tweet}\"")
    
    # Predict with standard model (set config to output attentions)
    inputs = tokenizer(sample_tweet, return_tensors='pt', padding=True, truncation=True, max_length=48)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        conf = torch.max(probs).item()
        pred_label = torch.argmax(logits, dim=1).item()
        prediction_text = "POSITIVE" if pred_label == 1 else "NEGATIVE"
        
    print(f"Predicted Sentiment: {prediction_text} ({conf*100:.2f}% confidence)")
    
    # Get attentions
    tokens, scores = get_attention_weights(model, sample_tweet, tokenizer)
    
    # Plot self-attention weights
    plot_attention_importance(tokens, scores, sample_tweet, prediction_text, conf)

if __name__ == '__main__':
    RANDOM_SEED = 42
    main()
