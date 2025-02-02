# TOPSIS for Pretrained Models in Text Generation

## Overview
This repository implements **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** to evaluate and rank pretrained models for text generation. The repository contains scripts, results, and visualizations for assessing multiple models.

## File Structure
### Files Included:
- **`Figure_1.png`**: Graphical representation of model rankings.
- **`Figure_2.png`**: Metric-wise performance comparison.
- **`README.md`**: Documentation for the repository.
- **`model_evaluation_results.csv`**: Evaluation metrics for different models.
- **`output.txt`**: Final TOPSIS scores and rankings.
- **`topsis.py`**: Python script implementing the TOPSIS methodology.

## Motivation
Pretrained models for text generation, such as **GPT-3, GPT-2, BART, and T5**, offer varying performance in terms of fluency, coherence, and diversity. Traditional evaluation methods focus on individual scores like BLEU or ROUGE, but **TOPSIS provides a multi-criteria decision-making approach** to rank models more comprehensively.

## Dataset
- **Benchmark datasets used:** CNN/DailyMail, WikiText-103, and OpenWebText
- **Metrics considered:**
  - Perplexity (PPL)
  - BLEU Score
  - ROUGE Score
  - Diversity (Unique n-grams)
  - Inference Time (Latency in ms)

## Methodology
1. **Feature Normalization:** Convert metric scores into a comparable scale.
2. **Weight Assignment:** Assign weights to each metric based on its importance.
3. **Ideal and Negative Ideal Solutions:** Identify best and worst possible scores.
4. **Distance Calculation:** Compute closeness of each model to ideal solution.
5. **Ranking:** Rank models based on computed scores.

## Results
### Performance Comparison
Below is a summary table showing the scores of each model on different metrics:

| Model  | Perplexity ↓ | BLEU ↑ | ROUGE ↑ | Diversity ↑ | Inference Time ↓ | TOPSIS Score ↑ | Rank |
|--------|-------------|--------|---------|------------|------------------|---------------|------|
| GPT-3  | 12.4        | 0.42   | 0.38    | 0.65       | 320              | 0.81          | 1    |
| GPT-2  | 28.6        | 0.34   | 0.31    | 0.60       | 280              | 0.65          | 3    |
| BART   | 20.1        | 0.39   | 0.36    | 0.63       | 290              | 0.74          | 2    |
| T5     | 25.8        | 0.37   | 0.33    | 0.61       | 270              | 0.69          | 4    |

### Graphical Representation
- **Figure 1:** Model rankings (see `Figure_1.png`)
- **Figure 2:** Metric-wise performance comparison (see `Figure_2.png`)

## Installation
To run the TOPSIS evaluation, install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage
Run the evaluation script:
```bash
python topsis.py --data model_evaluation_results.csv
```
Output will be saved in `output.txt` and visualized in `Figure_1.png` and `Figure_2.png`.

## Future Work
- Expanding evaluation to **newer models like LLaMA and Falcon**.
- Integrating **human evaluation** alongside automatic metrics.
- Optimizing weight selection **using AHP (Analytic Hierarchy Process)**.

## Contributors
- **Archie Bajaj** ([GitHub](https://github.com/abajaj15))

## License
This project is licensed under the **MIT License**.

---
**For more details, check the full results in** `model_evaluation_results.csv`.

