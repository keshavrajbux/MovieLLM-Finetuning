#!/usr/bin/env python3
"""
Model Fine-tuning Script
-----------------------
This script fine-tunes the Mistral-7B model on movie data using QLoRA.
It handles:
1. Loading and preparing the model with quantization
2. Setting up LoRA adapters
3. Preparing and tokenizing the training data
4. Training the model with proper configuration
5. Saving and testing the fine-tuned model
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
from tqdm import tqdm
import os
from pathlib import Path
import logging
import time
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
Path('models').mkdir(parents=True, exist_ok=True)

def setup_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """
    Set up the model and tokenizer with proper quantization.
    
    Args:
        model_name (str): Name or path of the base model
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer from {model_name}")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def apply_lora(model):
    """
    Apply LoRA adapters to the model.
    
    Args:
        model: The base model
        
    Returns:
        model: The model with LoRA adapters
    """
    logger.info("Applying LoRA adapters")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Alpha scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model

def prepare_training_data(tokenizer, max_length: int = 512):
    """
    Prepare and tokenize training data.
    
    Args:
        tokenizer: The tokenizer to use
        max_length (int): Maximum sequence length
        
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    logger.info("Preparing training data")
    
    # Load training examples
    with open('data/processed/training_examples.json', 'r') as f:
        examples = json.load(f)
    
    # Format data for instruction fine-tuning
    formatted_data = []
    for example in tqdm(examples, desc="Formatting data"):
        text = f"""### Instruction: {example['instruction']}

### Input: {example['input']}

### Response: {example['output']}

"""
        formatted_data.append({"text": text})
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Tokenization function
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    # Tokenize datasets
    train_dataset = dataset["train"].map(
        tokenize,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count()
    )
    eval_dataset = dataset["test"].map(
        tokenize,
        remove_columns=dataset["test"].column_names,
        num_proc=os.cpu_count()
    )
    
    return train_dataset, eval_dataset

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """
    Train the model using the prepared datasets.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
    """
    logger.info("Starting model training")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/movie-llm-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        evaluation_strategy="steps",
        eval_steps=100
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Training started")
    trainer.train()
    
    # Save the model
    logger.info("Saving model")
    trainer.save_model("models/movie-llm-final")

def test_model(model, tokenizer):
    """
    Test the fine-tuned model with comprehensive evaluation metrics.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
    """
    logger.info("Testing fine-tuned model")
    
    # Define test categories
    test_categories = {
        "basic_info": [
            {
                "instruction": "Describe the plot of this movie.",
                "input": "Movie: The Matrix"
            },
            {
                "instruction": "Who directed this movie?",
                "input": "Movie: Inception"
            },
            {
                "instruction": "What are the genres of this movie?",
                "input": "Movie: The Dark Knight"
            }
        ],
        "complex_analysis": [
            {
                "instruction": "Compare the themes and symbolism in these movies.",
                "input": "Movies: The Matrix and Inception"
            },
            {
                "instruction": "Analyze the character development in this movie.",
                "input": "Movie: The Godfather"
            }
        ],
        "creative_tasks": [
            {
                "instruction": "Write a brief review of this movie.",
                "input": "Movie: Pulp Fiction"
            },
            {
                "instruction": "Suggest similar movies based on this one.",
                "input": "Movie: The Shawshank Redemption"
            }
        ]
    }
    
    # Initialize metrics
    metrics = {
        "response_times": [],
        "response_lengths": [],
        "category_performance": defaultdict(list)
    }
    
    # Test each category
    for category, queries in test_categories.items():
        logger.info(f"\nTesting {category} queries...")
        for query in queries:
            prompt = f"""### Instruction: {query['instruction']}

### Input: {query['input']}

### Response:"""
            
            try:
                # Time the generation
                start_time = time.time()
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                generation_time = time.time() - start_time
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_text = response.split('### Response:')[-1].strip()
                
                # Record metrics
                metrics["response_times"].append(generation_time)
                metrics["response_lengths"].append(len(response_text.split()))
                metrics["category_performance"][category].append({
                    "query": query["input"],
                    "response": response_text,
                    "time": generation_time
                })
                
                # Log results
                logger.info(f"\nQuery: {query['input']}")
                logger.info(f"Response: {response_text}")
                logger.info(f"Generation time: {generation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error testing query '{query['input']}': {str(e)}")
                continue
    
    # Calculate and log summary metrics
    logger.info("\nModel Evaluation Summary:")
    logger.info("=" * 50)
    
    avg_response_time = np.mean(metrics["response_times"])
    avg_response_length = np.mean(metrics["response_lengths"])
    
    logger.info(f"Average response time: {avg_response_time:.2f}s")
    logger.info(f"Average response length: {avg_response_length:.1f} words")
    
    # Category-specific metrics
    for category, results in metrics["category_performance"].items():
        category_times = [r["time"] for r in results]
        category_lengths = [len(r["response"].split()) for r in results]
        
        logger.info(f"\n{category.title()} Category:")
        logger.info(f"Average time: {np.mean(category_times):.2f}s")
        logger.info(f"Average length: {np.mean(category_lengths):.1f} words")
        logger.info(f"Success rate: {len(results)}/{len(test_categories[category])} queries")
    
    # Save detailed results
    results_path = Path("models/evaluation_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nDetailed results saved to {results_path}")

def main():
    """Main execution function."""
    try:
        # Set up model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Apply LoRA
        model = apply_lora(model)
        
        # Prepare training data
        train_dataset, eval_dataset = prepare_training_data(tokenizer)
        
        # Train model
        train_model(model, tokenizer, train_dataset, eval_dataset)
        
        # Test model
        test_model(model, tokenizer)
        
        logger.info("Model fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main() 