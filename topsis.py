import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import pandas as pd
import matplotlib.pyplot as plt

# Disable symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# List of pre-trained models
models = ["gpt2", "distilgpt2", "gpt2-medium", "microsoft/DialoGPT-small"]

# Retry logic for model loading
def retry_model_load(model_name, retries=3, delay=5):
    for i in range(retries):
        try:
            print(f"Evaluating model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print(f"Retrying... ({i+1}/{retries})")
            time.sleep(delay)
    raise RuntimeError(f"Failed to load model {model_name} after {retries} attempts.")

# Function to evaluate each model (mock example)
def evaluate_model(model_name):
    tokenizer, model = retry_model_load(model_name)
    
    # Example of mock evaluation criteria (replace with actual evaluation logic)
    fluency = np.random.uniform(0.7, 1.0)  # Random score between 0.7 and 1.0
    coherence = np.random.uniform(0.6, 1.0)  # Random score between 0.6 and 1.0
    speed = np.random.uniform(0.8, 1.0)  # Random score between 0.8 and 1.0
    
    return [fluency, coherence, speed]

# Evaluate all models and collect criteria scores
criteria_matrix = []
for model_name in models:
    criteria_scores = evaluate_model(model_name)
    criteria_matrix.append(criteria_scores)

# Convert criteria matrix to a numpy array
criteria_matrix = np.array(criteria_matrix)
print("Criteria Matrix:\n", criteria_matrix)

# Weights for criteria
weights = np.array([0.4, 0.4, 0.2])  # Weights for fluency, coherence, and speed

# Step 1: Normalize the decision matrix
def normalize_matrix(matrix):
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    return norm_matrix

norm_matrix = normalize_matrix(criteria_matrix)
print("\nNormalized Matrix:\n", norm_matrix)

# Step 2: Weighted Normalization
def weighted_normalization(matrix, weights):
    weighted_matrix = matrix * weights
    return weighted_matrix

weighted_matrix = weighted_normalization(norm_matrix, weights)
print("\nWeighted Normalized Matrix:\n", weighted_matrix)

# Step 3: Determine Ideal and Negative-Ideal Solutions
def ideal_solutions(matrix):
    ideal_solution = np.max(matrix, axis=0)  # Positive ideal solution
    negative_ideal_solution = np.min(matrix, axis=0)  # Negative ideal solution
    return ideal_solution, negative_ideal_solution

ideal_solution, negative_ideal_solution = ideal_solutions(weighted_matrix)
print("\nIdeal Solution:\n", ideal_solution)
print("\nNegative Ideal Solution:\n", negative_ideal_solution)

# Step 4: Calculate Separation from Ideal and Negative-Ideal Solutions
def separation_from_ideal(matrix, ideal_solution):
    return np.sqrt(((matrix - ideal_solution)**2).sum(axis=1))

def separation_from_negative(matrix, negative_ideal_solution):
    return np.sqrt(((matrix - negative_ideal_solution)**2).sum(axis=1))

separation_ideal = separation_from_ideal(weighted_matrix, ideal_solution)
separation_negative = separation_from_negative(weighted_matrix, negative_ideal_solution)
print("\nSeparation from Ideal:\n", separation_ideal)
print("\nSeparation from Negative Ideal:\n", separation_negative)

# Step 5: Calculate the Relative Closeness to the Ideal Solution
def relative_closeness(separation_ideal, separation_negative):
    return separation_negative / (separation_ideal + separation_negative)

relative_closeness_values = relative_closeness(separation_ideal, separation_negative)
print("\nRelative Closeness to Ideal Solution:\n", relative_closeness_values)

# Step 6: Rank the models based on relative closeness
rankings = np.argsort(relative_closeness_values)[::-1]  # Sort in descending order
print("\nRankings (best to worst):\n", rankings)

# Print out the final rankings with model names
for rank, idx in enumerate(rankings):
    print(f"Rank {rank + 1}: Model {models[idx]} with Closeness Value {relative_closeness_values[idx]}")

# Convert results to a DataFrame for easier visualization
results_df = pd.DataFrame({
    'Model': models,
    'Fluency': criteria_matrix[:, 0],
    'Coherence': criteria_matrix[:, 1],
    'Speed': criteria_matrix[:, 2],
    'TOPSIS Score': relative_closeness_values
})

# Display the results DataFrame
print("\nResults DataFrame:\n", results_df)

# Create a bar plot for the TOPSIS scores
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['TOPSIS Score'], color='skyblue')
plt.xlabel('Models')
plt.ylabel('TOPSIS Score')
plt.title('TOPSIS Scores of Different Models')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()

# Show plot
plt.show()

# Save results to a CSV file
output_file = 'model_evaluation_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")
