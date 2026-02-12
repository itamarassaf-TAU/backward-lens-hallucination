import pandas as pd
from datasets import Dataset

def load_custom_facts_train_val(path="facts.csv"):
    """
    Loads a local CSV file containing simple fact completions.
    Expected columns: 'question', 'best_answer'
    """
    # Load the CSV
    df = pd.read_csv(path)
    
    # Create a simple list of dicts
    data = []
    for _, row in df.iterrows():
        data.append({
            "question": row["question"],       # The prompt (e.g., "The capital of France is")
            "best_answer": row["best_answer"], # The target (e.g., "Paris")
            "context": ""
        })
        
    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_list(data)
    
    # Split into Train (80%) and Val (20%)
    # Seed=42 ensures the split is the same every time we run it
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    return dataset["train"], dataset["test"]