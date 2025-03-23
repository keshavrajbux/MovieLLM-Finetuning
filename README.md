# MovieLLM-Finetuning

This project demonstrates how to fine-tune an open-source Large Language Model (LLM) on movie data using quantized Low-Rank Adaptation (qLoRA) and implement a Retrieval-Augmented Generation (RAG) system, all running on Google Colab.

## Project Overview

This repository contains code to:

1. Collect and process movie data from online sources (OMDB and TMDB APIs)
2. Fine-tune an open-source LLM (Mistral-7B) on movie-related conversations using qLoRA
3. Build a vector database of movie information for RAG
4. Create an interactive movie chatbot and quiz system

## Requirements

- Google Colab with GPU (T4 is sufficient)
- API keys for:
  - [OMDB API](http://www.omdbapi.com/)
  - [TMDB API](https://www.themoviedb.org/documentation/api)

## Project Structure

```
MovieLLM-Finetuning/
│
├── movie_llm_finetuning.ipynb      # Main fine-tuning notebook
├── movie_rag_implementation.ipynb  # RAG implementation notebook
├── movie_chatbot_demo.ipynb        # Interactive demo with Gradio
│
├── data_collection/
│   ├── scrape_movie_data.py        # Script to gather movie data
│   └── process_movie_data.py       # Script to process data for RAG
│
└── data/                           # Generated during execution
    ├── raw/                        # Raw scraped data
    └── processed/                  # Processed data for training and RAG
        └── rag/                    # Vector database files
```

## How to Use

### 1. Data Collection

First, update your API keys in `data_collection/scrape_movie_data.py`, then run:

```bash
# Make sure the directories exist
mkdir -p data/raw data/processed

# Collect movie data
python data_collection/scrape_movie_data.py

# Process the data for RAG and fine-tuning
python data_collection/process_movie_data.py
```

### 2. Fine-Tuning

Upload `movie_llm_finetuning.ipynb` to Google Colab and run through the notebook. This will:

- Set up the environment
- Load and quantize the Mistral-7B model
- Apply qLoRA for parameter-efficient fine-tuning
- Train the model on movie data
- Save the fine-tuned model

### 3. RAG Implementation

Upload `movie_rag_implementation.ipynb` to Google Colab and run through the notebook. This will:

- Build a vector database from the movie data
- Implement retrieval functions
- Create an integrated movie chatbot with RAG
- Implement a movie quiz system

### 4. Interactive Demo

Upload `movie_chatbot_demo.ipynb` to Google Colab and run it to interact with your fine-tuned model through a Gradio interface.

## Model Details

- Base model: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- Fine-tuning method: qLoRA (quantized Low-Rank Adaptation)
- Quantization: 4-bit quantization with double quantization
- Training data: Movie conversations, descriptions, reviews, and quiz questions

## Important Note

Please check the `how_to_use.md` file for detailed instructions on getting the notebooks properly, as there may be issues with the notebook files in this repository.

## RAG System

The RAG system enhances the LLM by retrieving relevant movie information before generating responses. This improves accuracy for factual questions and provides more detailed information about movies, directors, actors, and plots.

## Example Applications

- **Movie Chatbot**: Ask questions about movies, directors, plots, recommendations
- **Movie Quiz**: Interactive multiple-choice quiz to test movie knowledge
- **Recommendation System**: Get personalized movie recommendations based on preferences

## Extending the Project

- Add more data sources (Wikipedia, IMDb, Rotten Tomatoes)
- Enhance the RAG system with more context and better retrieval
- Create a web interface for the chatbot
- Expand to TV shows or other entertainment categories

## License

This project is open source, intended for educational purposes.

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for the Mistral-7B model
- [HuggingFace](https://huggingface.co/) for the Transformers library
- [OMDB API](http://www.omdbapi.com/) and [TMDB API](https://www.themoviedb.org/) for movie data
