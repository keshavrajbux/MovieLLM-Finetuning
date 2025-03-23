# 🎬 MovieLLM: Your AI Movie Companion

> *"Movies are like an expensive form of therapy for me." - Tim Burton*

Welcome to MovieLLM, where we've trained an AI to be your ultimate movie companion! This isn't just another chatbot - it's a fine-tuned Mistral-7B model that's been specifically trained to understand and discuss movies with the depth of a film critic and the enthusiasm of a movie buff.

## 🎯 What Makes This Special?

- **Movie-Savvy AI**: Fine-tuned on a curated dataset of movie information, our model understands cinema like a true connoisseur
- **Smart Memory**: Uses RAG (Retrieval-Augmented Generation) to provide accurate, up-to-date movie information
- **Interactive Learning**: Test your movie knowledge with our engaging quiz system
- **Beautiful Interface**: A sleek Gradio interface that makes movie discussions feel natural and fun

## 🚀 Quick Start

```bash
# 1. Set up your environment
python -m venv movie_env
source movie_env/bin/activate  # On Windows: movie_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python scripts/01_data_preparation.py
python scripts/02_model_finetuning.py
python scripts/03_rag_implementation.py
python scripts/04_movie_quiz_chatbot.py
```

## 🎮 Features

### 🎯 Movie Knowledge Base
- Deep understanding of movie plots, characters, and themes
- Accurate information about directors, actors, and production details
- Context-aware responses that consider movie history and connections

### 🧠 Smart Retrieval System
- Lightning-fast access to movie information
- Contextual understanding of movie-related queries
- Ability to make connections between different movies and genres

### 🎲 Interactive Quiz Mode
- Test your movie knowledge
- Get instant feedback and explanations
- Learn interesting facts about your favorite films

### 💬 Natural Conversation
- Chat naturally about movies
- Get personalized recommendations
- Discuss plot points and character arcs

## 🛠️ Technical Stack

- **Base Model**: Mistral-7B
- **Fine-tuning**: QLoRA (4-bit quantization)
- **Vector Database**: FAISS
- **Frontend**: Gradio
- **Embeddings**: Sentence Transformers

## 🎬 Example Interactions

```
You: "What's the connection between Inception and The Matrix?"
MovieLLM: "Both films explore the nature of reality and consciousness..."

You: "Who would win in a fight: John Wick or James Bond?"
MovieLLM: "While both are skilled, John Wick's specialized training..."

You: "Explain the ending of Shutter Island"
MovieLLM: "The ending reveals that Teddy Daniels is actually..."
```

## 🎯 Use Cases

- **Movie Research**: Get detailed information about films
- **Film Analysis**: Understand themes and symbolism
- **Recommendations**: Discover new movies based on your preferences
- **Trivia**: Test your movie knowledge
- **Discussion**: Engage in meaningful conversations about cinema

## 🎨 Project Structure

```
MovieLLM-Finetuning/
├── data/
│   ├── movies.csv           # Our movie knowledge base
│   └── processed/           # Processed training data
├── models/
│   └── movie-llm-final/     # Your movie-savvy AI
├── scripts/
│   ├── 01_data_preparation.py      # Data processing
│   ├── 02_model_finetuning.py      # Model training
│   ├── 03_rag_implementation.py    # Smart retrieval
│   └── 04_movie_quiz_chatbot.py    # Interactive interface
├── vector_db/               # Fast movie information retrieval
└── requirements.txt         # Dependencies
```

## 🎯 Requirements

- Python 3.8+
- CUDA-capable GPU (for optimal performance)
- 16GB+ RAM
- 20GB+ disk space
- A love for movies! 🎬

## 🤝 Contributing

We welcome contributions! Whether you're a movie buff, AI enthusiast, or developer, there's a place for you here. Check out our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai/) for the incredible base model
- The movie community for their passion and knowledge
- All the filmmakers who inspire us daily

## 📧 Get in Touch

Have questions? Found a cool movie fact we should know about? Want to discuss the latest blockbuster? Reach out to us through GitHub issues or join our community discussions!

---

*"In the end, we all become stories." - Margaret Atwood*

Let's make your movie stories more interesting with MovieLLM! 🎬✨ 