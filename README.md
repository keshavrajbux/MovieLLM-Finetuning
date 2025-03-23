# MovieLLM-Finetuning

A project for fine-tuning an open-source LLM (Mistral-7B) on movie data using qLoRA and implementing a RAG system for an interactive movie quiz chatbot.

## Project Overview

This project demonstrates how to:
1. Fine-tune a large language model (Mistral-7B) on movie data using qLoRA
2. Implement a Retrieval-Augmented Generation (RAG) system for movie knowledge
3. Create an interactive movie quiz chatbot

The project uses:
- Mistral-7B as the base model
- qLoRA for efficient fine-tuning
- FAISS for vector similarity search
- OMDB and TMDB APIs for movie data collection
- Gradio for the interactive interface

## Requirements

- Python 3.8+
- Google Colab (for fine-tuning)
- API keys for:
  - OMDB (http://www.omdbapi.com/)
  - TMDB (https://www.themoviedb.org/)

## Project Structure

```
MovieLLM-Finetuning/
├── data/
│   ├── raw/           # Raw movie data from APIs
│   └── processed/     # Processed data for fine-tuning
├── data_collection/
│   └── scrape_movie_data.py  # Script to collect movie data
├── vector_db/         # Vector database for RAG
└── colab_notebooks_guide.md  # Guide for setting up notebooks
```

## How to Use

1. **Data Collection**
   ```bash
   # Install requirements
   pip install -r requirements.txt
   
   # Set up your API keys in scrape_movie_data.py
   # Run the data collection script
   python data_collection/scrape_movie_data.py
   ```

2. **Fine-tuning**
   - Follow the instructions in `colab_notebooks_guide.md` to set up the fine-tuning notebook
   - The guide includes all necessary code for:
     - Loading and quantizing the model
     - Applying qLoRA
     - Processing the dataset
     - Training the model
     - Saving the fine-tuned model

3. **RAG Implementation**
   - Follow the RAG implementation section in the guide
   - This includes:
     - Building the vector database
     - Creating retrieval functions
     - Implementing the chatbot

4. **Interactive Demo**
   - Use the Gradio interface section to create an interactive movie quiz chatbot
   - The demo includes:
     - Movie recommendations
     - Plot explanations
     - Cast information
     - Director details
     - Genre-based queries

## Model Details

- **Base Model**: Mistral-7B
- **Fine-tuning Method**: qLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit with double quantization
- **Training Data**: Movie information, reviews, and generated conversations

## RAG System

The RAG system enhances the LLM's capabilities by:
1. Retrieving relevant movie information from the vector database
2. Using this context to generate more accurate and detailed responses
3. Providing factual information about movies, directors, and actors

## Example Applications

1. **Movie Chatbot**
   - Answer questions about movies
   - Provide recommendations
   - Explain plots and themes
   - Share cast and crew information

2. **Movie Quiz System**
   - Generate questions about movies
   - Check answers
   - Provide explanations
   - Track scores

## Extending the Project

You can enhance this project by:
1. Adding more movie data sources
2. Implementing additional quiz types
3. Creating a web interface
4. Adding user authentication
5. Implementing conversation history

## License and Acknowledgments

This project is open source and available under the MIT License. It builds upon:
- Mistral AI's Mistral-7B model
- Hugging Face's transformers library
- Google Colab's free GPU resources
- OMDB and TMDB APIs for movie data
