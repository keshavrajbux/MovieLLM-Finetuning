#!/usr/bin/env python3
"""
Movie Quiz Chatbot Script
------------------------
This script creates an interactive movie quiz chatbot using Gradio.
It combines the fine-tuned model with RAG to provide:
1. Movie knowledge Q&A
2. Interactive quiz game
3. Movie recommendations
4. Beautiful web interface
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import faiss
from sentence_transformers import SentenceTransformer, util
import pickle
import random
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Union
import time
from pathlib import Path
from difflib import SequenceMatcher
import numpy as np
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieQuizBot:
    def __init__(
        self,
        model_path: str = "models/movie-llm-final",
        encoder_name: str = "all-MiniLM-L6-v2",
        vector_db_path: str = "vector_db",
        movie_data_path: str = "data/movies.csv",
        similarity_threshold: float = 0.85
    ):
        """
        Initialize the movie quiz chatbot.
        
        Args:
            model_path (str): Path to the fine-tuned model
            encoder_name (str): Name of the sentence transformer model
            vector_db_path (str): Path to the vector database
            movie_data_path (str): Path to the movie dataset
            similarity_threshold (float): Threshold for semantic similarity matching
        """
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.similarity_threshold = similarity_threshold
            
            # Load the fine-tuned model
            logger.info("Loading fine-tuned model...")
            self.model, self.tokenizer = self._load_model(model_path)
            
            # Load the encoder for semantic similarity
            logger.info("Loading encoder model...")
            self.encoder = SentenceTransformer(encoder_name)
            self.encoder.to(self.device)
            
            # Load vector database
            logger.info("Loading vector database...")
            self.index, self.documents = self._load_vector_db(vector_db_path)
            
            # Load movie data efficiently
            logger.info("Loading movie data...")
            self._load_movie_data(movie_data_path)
            
            # Initialize quiz state
            self.current_quiz = None
            self.quiz_score = 0
            self.total_questions = 0
            self.quiz_history = []
            
            logger.info("Movie quiz chatbot initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize quiz bot: {str(e)}")
            raise RuntimeError("Quiz bot initialization failed") from e

    def _load_model(self, model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the fine-tuned model and tokenizer."""
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def _load_vector_db(self, vector_db_path: str) -> Tuple[faiss.Index, List[Dict]]:
        """Load the FAISS index and documents."""
        index = faiss.read_index(f"{vector_db_path}/movie_faiss_index")
        with open(f"{vector_db_path}/movie_documents.pkl", 'rb') as f:
            documents = pickle.load(f)
        return index, documents
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant documents for the query."""
        # Encode the query
        query_vector = self.encoder.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search for similar documents
        distances, indices = self.index.search(query_vector, top_k)
        
        # Format context
        contexts = []
        for idx in indices[0]:
            contexts.append(self.documents[idx]['content'])
        
        return "\n\n---\n\n".join(contexts)
    
    def generate_response(
        self,
        instruction: str,
        input_text: str,
        max_length: int = 200
    ) -> str:
        """Generate a response using RAG."""
        # Retrieve context
        context = self.retrieve_context(input_text)
        
        # Format prompt
        prompt = f"""### Instruction: {instruction}
Use the following context to help answer the question:

### Context:
{context}

### Input: {input_text}

### Response:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=len(inputs.input_ids[0]) + max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("### Response:")[-1].strip()
    
    def _load_movie_data(self, movie_data_path: str):
        """Load movie data with efficient chunking."""
        try:
            # Read data in chunks to handle large datasets
            chunks = []
            for chunk in pd.read_csv(movie_data_path, chunksize=1000):
                chunks.append(chunk)
            self.movies_df = pd.concat(chunks)
            
            # Create indices for faster lookups
            self.movies_df.set_index('Title', inplace=True)
            
            # Preprocess text columns
            text_columns = ['Plot', 'Director', 'Actors', 'Genre']
            for col in text_columns:
                if col in self.movies_df.columns:
                    self.movies_df[col] = self.movies_df[col].fillna('')
                    
        except Exception as e:
            logger.error(f"Error loading movie data: {str(e)}")
            raise

    def check_answer(self, user_answer: str, correct_answer: str, answer_type: str) -> Tuple[bool, float, str]:
        """
        Check if the user's answer is correct using multiple validation methods.
        
        Args:
            user_answer (str): The user's answer
            correct_answer (str): The correct answer
            answer_type (str): Type of answer (Plot, Director, Actors, Genre, Year)
            
        Returns:
            Tuple[bool, float, str]: (is_correct, confidence_score, feedback)
        """
        try:
            user_answer = user_answer.lower().strip()
            correct_answer = correct_answer.lower().strip()
            
            if answer_type == "Year":
                # Extract years from answers
                user_year = self._extract_year(user_answer)
                correct_year = self._extract_year(correct_answer)
                if user_year and correct_year:
                    return (user_year == correct_year, 1.0, f"The correct year was {correct_year}")
            
            # Calculate different similarity scores
            exact_match = user_answer == correct_answer
            sequence_similarity = SequenceMatcher(None, user_answer, correct_answer).ratio()
            
            # Calculate semantic similarity
            semantic_similarity = self._calculate_semantic_similarity(user_answer, correct_answer)
            
            # Different thresholds for different answer types
            thresholds = {
                "Plot": 0.8,
                "Director": 0.9,
                "Actors": 0.85,
                "Genre": 0.9
            }
            threshold = thresholds.get(answer_type, self.similarity_threshold)
            
            # Combine similarity scores
            combined_score = np.mean([
                1.0 if exact_match else 0.0,
                sequence_similarity,
                semantic_similarity
            ])
            
            is_correct = combined_score >= threshold
            
            # Generate feedback
            feedback = self._generate_feedback(
                is_correct,
                combined_score,
                correct_answer,
                answer_type
            )
            
            return is_correct, combined_score, feedback
            
        except Exception as e:
            logger.error(f"Error checking answer: {str(e)}")
            return False, 0.0, "Error validating answer"

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Encode texts
            embeddings = self.encoder.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1)
            )
            
            return float(similarity[0][0])
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract year from text."""
        try:
            # Find four-digit numbers between 1900 and current year
            years = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', text)
            return int(years[0]) if years else None
        except Exception:
            return None

    def _generate_feedback(
        self,
        is_correct: bool,
        confidence: float,
        correct_answer: str,
        answer_type: str
    ) -> str:
        """Generate detailed feedback for the answer."""
        if is_correct:
            if confidence > 0.95:
                return "Perfect answer! You really know your movies!"
            elif confidence > 0.85:
                return "Correct! Your answer was very close to the mark."
            else:
                return "Correct! Your answer captured the main points."
        else:
            if confidence > 0.7:
                return f"Close, but not quite. The correct answer was: {correct_answer}\nYou were on the right track!"
            elif confidence > 0.5:
                return f"Your answer had some similarities, but wasn't quite right. The answer was: {correct_answer}\nKeep trying!"
            else:
                return f"That's not correct. The answer was: {correct_answer}\nDon't worry, this was a tough one!"

    def generate_quiz_question(self) -> Dict:
        """Generate a random movie quiz question with difficulty scaling."""
        try:
            # Scale difficulty based on performance
            if self.total_questions > 0:
                success_rate = self.quiz_score / self.total_questions
                
                # Adjust question selection based on performance
                if success_rate > 0.8:
                    # Generate harder questions
                    return self._generate_hard_question()
                elif success_rate < 0.4:
                    # Generate easier questions
                    return self._generate_easy_question()
            
            # Default to medium difficulty
            return self._generate_medium_question()
            
        except Exception as e:
            logger.error(f"Error generating quiz question: {str(e)}")
            # Fallback to basic question
            movie = self.movies_df.sample(n=1).iloc[0]
            return {
                "movie": movie.name,  # Using index as movie title
                "question": "Who directed this movie?",
                "answer": str(movie["Director"]),
                "type": "Director"
            }

    def _generate_easy_question(self) -> Dict:
        """Generate an easy question."""
        try:
            movie = self.movies_df.sample(n=1).iloc[0]
            
            # Simple questions with clear answers
            questions = [
                ("What is the title of this movie?", "Title"),
                ("Is this a comedy?", "Genre"),
                ("Is this a recent movie?", "Year"),
                ("Is this a well-known director?", "Director"),
                ("Is this a popular actor?", "Actors")
            ]
            
            question, answer_key = random.choice(questions)
            return {
                "movie": movie.name,
                "question": question,
                "answer": str(movie[answer_key]),
                "type": answer_key,
                "difficulty": "easy",
                "hints": self._generate_hints(movie, answer_key)
            }
            
        except Exception as e:
            logger.error(f"Error generating easy question: {str(e)}")
            return self._generate_medium_question()

    def _generate_hard_question(self) -> Dict:
        """Generate a hard question (e.g., plot details, multiple aspects)."""
        try:
            movie = self.movies_df.sample(n=1).iloc[0]
            
            # Complex questions requiring detailed knowledge
            questions = [
                ("What is the complete plot of this movie?", "Plot"),
                ("List all the main actors and their roles in this movie.", "Actors"),
                (f"Compare this movie with other {movie['Genre'].split(',')[0]} movies.", "Genre"),
                ("What are the main themes and symbolism in this movie?", "Plot"),
                ("How does this movie's cinematography contribute to its storytelling?", "Plot")
            ]
            
            question, answer_key = random.choice(questions)
            return {
                "movie": movie.name,
                "question": question,
                "answer": str(movie[answer_key]),
                "type": answer_key,
                "difficulty": "hard",
                "hints": self._generate_hints(movie, answer_key)
            }
            
        except Exception as e:
            logger.error(f"Error generating hard question: {str(e)}")
            return self._generate_medium_question()

    def _generate_medium_question(self) -> Dict:
        """Generate a medium difficulty question."""
        try:
            movie = self.movies_df.sample(n=1).iloc[0]
            
            # Medium difficulty questions
            questions = [
                ("Who directed this movie?", "Director"),
                ("What year was this movie released?", "Year"),
                ("What are the main genres of this movie?", "Genre"),
                ("Who are the lead actors in this movie?", "Actors"),
                ("Give a brief summary of the plot.", "Plot")
            ]
            
            question, answer_key = random.choice(questions)
            return {
                "movie": movie.name,
                "question": question,
                "answer": str(movie[answer_key]),
                "type": answer_key,
                "difficulty": "medium",
                "hints": self._generate_hints(movie, answer_key)
            }
            
        except Exception as e:
            logger.error(f"Error generating medium question: {str(e)}")
            return self._generate_easy_question()

    def _generate_hints(self, movie: pd.Series, answer_key: str) -> List[str]:
        """Generate helpful hints for the question."""
        hints = []
        
        if answer_key == "Plot":
            # Extract key plot points
            plot_points = movie["Plot"].split(". ")
            if len(plot_points) > 1:
                hints.append(f"Hint: The movie involves {plot_points[0].lower()}")
                if len(plot_points) > 2:
                    hints.append(f"Hint: Later in the story, {plot_points[1].lower()}")
        
        elif answer_key == "Director":
            # Provide director's other works
            director = movie["Director"]
            other_movies = self.movies_df[self.movies_df["Director"] == director].index.tolist()
            if len(other_movies) > 1:
                hints.append(f"Hint: This director also made {other_movies[0]}")
        
        elif answer_key == "Actors":
            # Provide partial actor list
            actors = movie["Actors"].split(", ")
            if len(actors) > 1:
                hints.append(f"Hint: One of the main actors is {actors[0]}")
        
        elif answer_key == "Genre":
            # Provide genre hints
            genres = movie["Genre"].split(", ")
            if len(genres) > 1:
                hints.append(f"Hint: This movie is a {genres[0]}")
        
        elif answer_key == "Year":
            # Provide decade hint
            year = int(movie["Year"])
            decade = (year // 10) * 10
            hints.append(f"Hint: This movie was released in the {decade}s")
        
        return hints

    def start_quiz(self) -> str:
        """Start a new quiz session."""
        self.quiz_score = 0
        self.total_questions = 0
        self.current_quiz = self.generate_quiz_question()
        
        return f"Let's start the movie quiz!\n\nQuestion 1: {self.current_quiz['question']}\nMovie: {self.current_quiz['movie']}"
    
    def answer_quiz(self, answer: str) -> str:
        """Process the user's quiz answer."""
        if self.current_quiz is None:
            return "Please start a new quiz first!"
        
        self.total_questions += 1
        is_correct, confidence, feedback = self.check_answer(answer, self.current_quiz["answer"], self.current_quiz["type"])
        
        if is_correct:
            self.quiz_score += 1
            response = "Correct! "
        else:
            response = f"Sorry, that's not correct. {feedback}\n"
        
        # Generate next question
        self.current_quiz = self.generate_quiz_question()
        response += f"\nQuestion {self.total_questions + 1}: {self.current_quiz['question']}\nMovie: {self.current_quiz['movie']}"
        
        # Add score
        response += f"\n\nCurrent Score: {self.quiz_score}/{self.total_questions}"
        
        return response
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Process chat messages."""
        try:
            # Check for quiz commands
            if message.lower() == "/quiz":
                return self.start_quiz()
            elif self.current_quiz is not None:
                return self.answer_quiz(message)
            
            # Normal chat mode
            response = self.generate_response(
                "Answer the following question about movies.",
                message
            )
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

def create_gradio_interface():
    """Create and launch the Gradio interface."""
    # Initialize the chatbot
    bot = MovieQuizBot()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Roboto', sans-serif;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=custom_css) as interface:
        gr.Markdown("# 🎬 Movie Quiz Chatbot")
        gr.Markdown("""
        Welcome to the Movie Quiz Chatbot! You can:
        - Ask questions about movies
        - Start a quiz by typing `/quiz`
        - Get movie recommendations
        - Learn about actors, directors, and plots
        """)
        
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(
            label="Type your message or '/quiz' to start a quiz",
            placeholder="What's your question about movies?"
        )
        clear = gr.Button("Clear Chat")
        
        # Example questions
        gr.Markdown("### Sample Questions")
        examples = [
            "What's the plot of Inception?",
            "Who directed The Dark Knight?",
            "Recommend some movies like The Matrix",
            "Tell me about Christopher Nolan's movies",
            "/quiz"
        ]
        
        # Add example buttons
        for example in examples:
            gr.Button(example).click(
                lambda x: x,
                [gr.Textbox(value=example, visible=False)],
                msg
            )
        
        # Set up event handlers
        msg.submit(bot.chat, [msg, chatbot], chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
    
    # Launch the interface
    interface.launch(share=True)

def main():
    """Main execution function."""
    try:
        create_gradio_interface()
    except Exception as e:
        logger.error(f"Error in movie quiz chatbot: {str(e)}")
        raise

if __name__ == "__main__":
    main() 