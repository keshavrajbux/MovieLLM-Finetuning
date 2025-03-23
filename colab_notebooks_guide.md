# Guide to Setting Up MovieLLM-Finetuning in Google Colab

Since the Jupyter notebook files in this repository may have issues, follow this guide to set them up directly in Google Colab.

## 1. Fine-Tuning Notebook

Create a new notebook in [Google Colab](https://colab.research.google.com/) and copy the code below into separate cells:

### Movie LLM Fine-Tuning with qLoRA

```markdown
# Movie LLM Fine-Tuning with qLoRA

This notebook demonstrates how to fine-tune an open-source LLM (Mistral-7B) on movie data using qLoRA, making it possible to run on Google Colab.
```

### Setup Environment

```python
# Check for GPU
!nvidia-smi

# Install required libraries
!pip install -q bitsandbytes transformers datasets peft trl torch accelerate scipy sentencepiece huggingface_hub
```

### Load and Prepare the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Mistral-7B model with quantization
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

### Prepare for LoRA Fine-Tuning

```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=16,               # Rank
    lora_alpha=32,      # Alpha parameter for LoRA scaling
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
```

### Load and Process Movie Dataset

```python
import pandas as pd
from datasets import Dataset
import os
from google.colab import files

# Upload movie_conversations.csv or clone from GitHub if available
try:
    uploaded = files.upload()  # Upload your movie_conversations.csv
    file_path = list(uploaded.keys())[0]
except:
    # If file upload doesn't work, create a simple dataset
    data = {
        'instruction': [
            "Recommend movies similar to The Godfather.",
            "Explain the plot of Inception.",
            "List the main cast of Pulp Fiction."
        ],
        'input': [
            "I enjoyed watching The Godfather directed by Francis Ford Coppola. I particularly like Crime, Drama movies. Can you suggest similar films?",
            "What happens in the movie Inception? Please avoid major spoilers.",
            "Who starred in Pulp Fiction? What roles did they play?"
        ],
        'response': [
            "Based on your enjoyment of The Godfather, I recommend checking out these similar Crime, Drama films:\n\n1. Goodfellas - Like The Godfather, this film features a deep exploration of organized crime and family dynamics.\n2. Casino - Directed by Martin Scorsese, who has a style comparable to Francis Ford Coppola.\n3. Heat - Stars Al Pacino from The Godfather in a similar complex criminal role.",
            "Dom Cobb is a skilled thief who specializes in extracting secrets from dreams. He's offered a chance to have his criminal history erased if he can plant an idea in someone's mind rather than steal one. Cobb and his team navigate multiple layers of dreams, battling projections and personal demons.",
            "The main cast of Pulp Fiction (1994) includes:\n\nJohn Travolta, Samuel L. Jackson, Uma Thurman, Harvey Keitel, Tim Roth, Amanda Plummer, Bruce Willis\n\nDirected by Quentin Tarantino, this Crime, Drama film features these actors in their respective iconic roles."
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv('movie_conversations.csv', index=False)
    file_path = 'movie_conversations.csv'

df = pd.read_csv(file_path)
df.head()
```

### Format and Tokenize Data

```python
# Format data for instruction fine-tuning
def format_instruction(row):
    return f"""### Instruction: {row['instruction']}

### Input: {row['input']}

### Response: {row['response']}
"""

df['text'] = df.apply(format_instruction, axis=1)

# Create HF dataset
dataset = Dataset.from_pandas(df[['text']])

# Split into train and validation
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```

### Fine-Tune the Model

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Define training arguments
training_args = TrainingArguments(
    output_dir="./movie-llm-model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=25,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    learning_rate=2e-4,
    save_steps=50,
    fp16=True,
    save_total_limit=3,
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Start training
trainer.train()
```

### Save the Model

```python
# Save model
model.save_pretrained("./movie-llm-model-final")
tokenizer.save_pretrained("./movie-llm-model-final")

# Download the model
from google.colab import files
!zip -r movie-llm-model-final.zip ./movie-llm-model-final
files.download("movie-llm-model-final.zip")
```

### Test the Model

```python
def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test with some movie-related queries
prompts = [
    "### Instruction: Recommend some sci-fi movies similar to Blade Runner.\n\n### Input: I enjoy dystopian futures and philosophical themes.\n\n### Response:",
    "### Instruction: Explain the ending of Inception.\n\n### Input: Was it all a dream?\n\n### Response:",
    "### Instruction: Write dialogue in the style of Quentin Tarantino.\n\n### Input: Two hitmen discussing fast food.\n\n### Response:"
]

for prompt in prompts:
    response = generate_response(prompt)
    print(f"Prompt: {prompt}\n\nResponse: {response}\n\n{'='*50}\n")
```

## 2. RAG Implementation Notebook

Create another notebook for the RAG implementation with these cells:

### Movie RAG System Implementation

```markdown
# Movie RAG System Implementation

This notebook implements a Retrieval-Augmented Generation (RAG) system for our movie-themed LLM.
```

### Install Dependencies

```python
!pip install -q langchain transformers faiss-gpu sentence-transformers nltk pypdf tqdm
```

### Create Sample Movie Data

```python
import pandas as pd
import numpy as np
import os
import json
from tqdm.auto import tqdm

# Sample movie data for the demo
sample_movies = [
    {
        'title': 'The Shawshank Redemption',
        'year': '1994',
        'director': 'Frank Darabont',
        'actors': 'Tim Robbins, Morgan Freeman, Bob Gunton',
        'plot': 'Over the course of several years, two convicts form a friendship, seeking consolation and, eventually, redemption through basic compassion.',
        'genres': 'Drama',
        'reviews': 'A masterpiece of storytelling. One of the greatest films ever made.'
    },
    {
        'title': 'The Godfather',
        'year': '1972',
        'director': 'Francis Ford Coppola',
        'actors': 'Marlon Brando, Al Pacino, James Caan',
        'plot': 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
        'genres': 'Crime, Drama',
        'reviews': 'A defining example of the gangster film. Brando gives an iconic performance.'
    },
    {
        'title': 'Pulp Fiction',
        'year': '1994',
        'director': 'Quentin Tarantino',
        'actors': 'John Travolta, Uma Thurman, Samuel L. Jackson',
        'plot': 'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
        'genres': 'Crime, Drama',
        'reviews': 'Revolutionary narrative structure. Tarantino's masterpiece.'
    },
    {
        'title': 'Inception',
        'year': '2010',
        'director': 'Christopher Nolan',
        'actors': 'Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page',
        'plot': 'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
        'genres': 'Action, Adventure, Sci-Fi',
        'reviews': 'Mind-bending and visually stunning. A complex narrative executed brilliantly.'
    },
    {
        'title': 'The Dark Knight',
        'year': '2008',
        'director': 'Christopher Nolan',
        'actors': 'Christian Bale, Heath Ledger, Aaron Eckhart',
        'plot': 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
        'genres': 'Action, Crime, Drama',
        'reviews': 'Heath Ledger's Joker is one of the greatest villain performances of all time.'
    }
]

# Create dataframe
movies_df = pd.DataFrame(sample_movies)
movies_df.head()
```

### Build Vector Database

```python
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# Create documents for RAG
documents = []
for _, movie in movies_df.iterrows():
    doc = f"Title: {movie['title']}\n"
    doc += f"Year: {movie['year']}\n"
    doc += f"Director: {movie['director']}\n"
    doc += f"Cast: {movie['actors']}\n"
    doc += f"Genre: {movie['genres']}\n"
    doc += f"Plot: {movie['plot']}\n"
    doc += f"Reviews: {movie['reviews']}"
    
    documents.append({
        'content': doc,
        'title': movie['title'],
        'id': movie['title'].replace(' ', '_').lower()
    })

# Initialize sentence encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = []
for doc in tqdm(documents):
    embedding = encoder.encode(doc['content'])
    embeddings.append(embedding)

# Create FAISS index
embeddings = np.array(embeddings).astype('float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Create output directory
os.makedirs('vector_db', exist_ok=True)

# Save the index and documents
faiss.write_index(index, 'vector_db/movie_faiss_index')
with open('vector_db/movie_documents.pkl', 'wb') as f:
    pickle.dump(documents, f)
```

### Load the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model - for demo we'll use base model
# In real scenario, you'd load your fine-tuned model
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

### Implement RAG Functions

```python
def retrieve_context(query, k=2):
    """Retrieve relevant documents for the query"""
    # Load the index and documents
    index = faiss.read_index('vector_db/movie_faiss_index')
    with open('vector_db/movie_documents.pkl', 'rb') as f:
        documents = pickle.load(f)
    
    # Encode the query
    query_vector = encoder.encode([query])[0].reshape(1, -1).astype('float32')
    
    # Search for similar documents
    distances, indices = index.search(query_vector, k)
    
    # Retrieve the top k documents
    retrieved_docs = [documents[idx] for idx in indices[0]]
    
    # Format context
    context = "\n\n".join([doc['content'] for doc in retrieved_docs])
    
    return context

def rag_generate_response(query, max_length=500):
    """Generate a response with RAG"""
    # Retrieve relevant context
    context = retrieve_context(query)
    
    # Format prompt with retrieved context
    prompt = f"""### Instruction: Answer the following question about movies using the provided context.

### Context:
{context}

### Input:
{query}

### Response:
"""
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=len(inputs.input_ids[0]) + max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    response = response.split("### Response:")[-1].strip()
    
    return response
```

### Test the RAG System

```python
# Test with some movie queries
test_queries = [
    "What are Christopher Nolan's best movies?",
    "Can you recommend some classic crime films?",
    "Tell me about the plot of The Godfather",
    "Who played the main character in Pulp Fiction?"
]

for query in test_queries:
    print(f"Query: {query}")
    response = rag_generate_response(query)
    print(f"Response: {response}\n")
    print("-" * 80)
```

### Interactive Movie Chatbot

```python
def movie_chatbot():
    print("🎬 Welcome to MovieBot! Ask me anything about movies. Type 'exit' to quit.")
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nMovieBot: Thanks for chatting about movies! Goodbye!")
            break
            
        response = rag_generate_response(query)
        print(f"\nMovieBot: {response}")

# Run the chatbot
movie_chatbot()
```

## 3. Movie Chatbot Demo with Gradio

Create a third notebook for the interactive demo:

### Movie Chatbot Demo

```markdown
# Movie Chatbot Demo

This notebook demonstrates the movie chatbot with RAG system through a Gradio interface.
```

### Install Dependencies

```python
!pip install -q gradio transformers torch bitsandbytes peft faiss-gpu sentence-transformers pandas
```

### Load Model and Vector Database

```python
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import random

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model
model_name = "mistralai/Mistral-7B-v0.1"  # Replace with your fine-tuned model path
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the encoder model
encoder = SentenceTransformer('all-MiniLM-L6-v2')
```

### RAG Functions

```python
def retrieve_context(query, k=2):
    """Retrieve relevant documents for the query"""
    try:
        # Load the index and documents
        index = faiss.read_index('vector_db/movie_faiss_index')
        with open('vector_db/movie_documents.pkl', 'rb') as f:
            documents = pickle.load(f)
        
        # Encode the query
        query_vector = encoder.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search for similar documents
        distances, indices = index.search(query_vector, k)
        
        # Retrieve the top k documents
        retrieved_docs = [documents[idx] for idx in indices[0]]
        
        # Format context
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        
        return context
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return ""

def generate_response(query, use_rag=True, max_length=500):
    """Generate a response to the query using the LLM and RAG"""
    try:
        if use_rag:
            # Retrieve relevant context
            context = retrieve_context(query)
            
            # Format prompt with retrieved context
            prompt = f"""### Instruction: Answer the following question about movies using the provided context.

### Context:
{context}

### Input:
{query}

### Response:
"""
        else:
            # No RAG - direct prompt
            prompt = f"""### Instruction: Answer the following question about movies.

### Input:
{query}

### Response:
"""
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=len(inputs.input_ids[0]) + max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        response = response.split("### Response:")[-1].strip()
        
        return response
    except Exception as e:
        print(f"Error in generation: {e}")
        return "I'm sorry, I encountered an error while trying to answer your question."
```

### Gradio Interface

```python
def respond_to_user(message, history, use_rag):
    """Process user message and return a response"""
    return generate_response(message, use_rag=use_rag)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎬 Movie Chatbot with Fine-tuned LLM and RAG")
    gr.Markdown("Ask me anything about movies, directors, actors, plots, or recommendations!")
    
    with gr.Row():
        use_rag = gr.Checkbox(label="Use RAG (Retrieval-Augmented Generation)", value=True,
                             info="Enable to improve factual accuracy with knowledge retrieval")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your question")
    clear = gr.Button("Clear")
    
    msg.submit(respond_to_user, [msg, chatbot, use_rag], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)
    
    gr.Markdown("### Sample Questions")
    sample_questions = [
        "Who directed Pulp Fiction and what other films has he made?",
        "What's the plot of Inception?",
        "Recommend some movies similar to The Dark Knight",
        "Who are the main actors in The Godfather?",
        "What are some classic films from the 1970s?"
    ]
    
    for question in sample_questions:
        gr.Button(question).click(lambda q: q, [gr.Textbox(value=question, visible=False)], msg)

# Launch the demo
demo.launch(share=True)
```

## Using This Guide

1. For each notebook section:
   - Create a new Colab notebook
   - Copy and paste the code into separate cells
   - Run the cells in order

2. Adapt paths and configurations as needed:
   - Update the model name if you want to use a different base model
   - Adjust batch sizes based on your Colab GPU memory
   - Change input/output paths if needed

3. Be patient during model loading and training:
   - Loading large models takes time
   - Training with qLoRA is faster than full fine-tuning but still takes time

This guide should allow you to replicate the functionality of the MovieLLM-Finetuning project, even if the original notebook files have issues. 