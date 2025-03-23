#!/usr/bin/env python3
"""
RAG Implementation Script
------------------------
This script implements a Retrieval-Augmented Generation system using
the fine-tuned model and vector database. It handles:
1. Loading the fine-tuned model and vector database
2. Implementing retrieval functions
3. Setting up the RAG pipeline
4. Testing the system with sample queries
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import logging
from typing import List, Dict, Tuple
import time
import util

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieRAG:
    def __init__(
        self,
        model_path: str = "models/movie-llm-final",
        encoder_name: str = "all-MiniLM-L6-v2",
        vector_db_path: str = "vector_db",
        top_k: int = 3,
        max_context_length: int = 2048  # Add max context length parameter
    ):
        """
        Initialize the RAG system.
        
        Args:
            model_path (str): Path to the fine-tuned model
            encoder_name (str): Name of the sentence transformer model
            vector_db_path (str): Path to the vector database
            top_k (int): Number of documents to retrieve
            max_context_length (int): Maximum context length for the model
        """
        self.top_k = top_k
        self.max_context_length = max_context_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load the fine-tuned model
            logger.info("Loading fine-tuned model...")
            self.model, self.tokenizer = self._load_model(model_path)
            
            # Load the encoder
            logger.info("Loading encoder model...")
            self.encoder = SentenceTransformer(encoder_name)
            self.encoder.to(self.device)
            
            # Load vector database
            logger.info("Loading vector database...")
            self.index, self.documents = self._load_vector_db(vector_db_path)
            
            # Initialize memory for conversation history
            self.conversation_history = []
            
            logger.info("RAG system initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise RuntimeError("RAG system initialization failed") from e
    
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
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """
        Truncate context to fit within model's context window.
        
        Args:
            context (str): The context to truncate
            max_tokens (int): Maximum number of tokens allowed
            
        Returns:
            str: Truncated context
        """
        tokens = self.tokenizer.encode(context)
        if len(tokens) <= max_tokens:
            return context
            
        # Truncate while keeping complete sentences
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        # Find the last complete sentence
        sentences = truncated_text.split('.')
        if len(sentences) > 1:
            return '.'.join(sentences[:-1]) + '.'
        return truncated_text

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve and rank relevant documents for the query.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: Retrieved and ranked context
        """
        try:
            # Encode the query
            query_vector = self.encoder.encode([query])[0].reshape(1, -1).astype('float32')
            
            # Search for similar documents
            distances, indices = self.index.search(query_vector, self.top_k * 2)  # Get more candidates
            
            # Rank and filter contexts
            contexts = []
            for idx, distance in zip(indices[0], distances[0]):
                relevance_score = 1 / (1 + distance)  # Convert distance to similarity score
                if relevance_score > 0.5:  # Only keep highly relevant contexts
                    contexts.append({
                        'content': self.documents[idx]['content'],
                        'score': relevance_score
                    })
            
            # Sort by relevance and take top_k
            contexts.sort(key=lambda x: x['score'], reverse=True)
            contexts = contexts[:self.top_k]
            
            # Format context with relevance scores
            formatted_contexts = []
            for ctx in contexts:
                formatted_contexts.append(f"[Relevance: {ctx['score']:.2f}]\n{ctx['content']}")
            
            return "\n\n---\n\n".join(formatted_contexts)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""  # Return empty context on error

    def verify_answer(self, answer: str, context: str) -> Tuple[bool, float]:
        """
        Verify the generated answer against the retrieved context.
        
        Args:
            answer (str): The generated answer
            context (str): The retrieved context
            
        Returns:
            Tuple[bool, float]: (is_verified, confidence_score)
        """
        try:
            # Calculate semantic similarity between answer and context
            answer_embedding = self.encoder.encode([answer])[0]
            context_embedding = self.encoder.encode([context])[0]
            
            similarity = util.pytorch_cos_sim(
                answer_embedding.reshape(1, -1),
                context_embedding.reshape(1, -1)
            )
            
            confidence_score = float(similarity[0][0])
            is_verified = confidence_score > 0.7  # Threshold for verification
            
            return is_verified, confidence_score
            
        except Exception as e:
            logger.error(f"Error verifying answer: {str(e)}")
            return False, 0.0

    def generate_response(
        self,
        instruction: str,
        input_text: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response using RAG with answer verification.
        
        Args:
            instruction (str): The instruction for the model
            input_text (str): The input text
            max_length (int): Maximum response length
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            
        Returns:
            str: The generated response
        """
        try:
            # Retrieve context
            context = self.retrieve_context(input_text)
            if not context:
                logger.warning("No relevant context found")
                return "I apologize, but I couldn't find relevant information to answer your question."
            
            # Format prompt with conversation history
            history = self._format_conversation_history()
            prompt = f"""### Instruction: {instruction}

### Previous Conversation:
{history}

### Context:
{context}

### Input: {input_text}

### Response:"""
            
            # Generate response with error handling
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=len(inputs.input_ids[0]) + max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("### Response:")[-1].strip()
                
                # Verify the answer
                is_verified, confidence = self.verify_answer(response, context)
                if not is_verified:
                    logger.warning(f"Low confidence answer (score: {confidence:.2f})")
                    response = f"[Note: This answer has low confidence ({confidence:.2f})]\n{response}"
                
                # Update conversation history
                self.conversation_history.append({
                    "input": input_text,
                    "response": response,
                    "confidence": confidence
                })
                if len(self.conversation_history) > 5:  # Keep last 5 exchanges
                    self.conversation_history.pop(0)
                
                return response
                
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU out of memory, falling back to CPU")
                self.model.to("cpu")
                return self.generate_response(instruction, input_text, max_length, temperature, top_p)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try rephrasing your question."

    def _format_conversation_history(self) -> str:
        """Format the conversation history for context."""
        if not self.conversation_history:
            return "No previous conversation."
            
        history = []
        for exchange in self.conversation_history[-3:]:  # Use last 3 exchanges
            history.append(f"User: {exchange['input']}\nAssistant: {exchange['response']}")
        return "\n\n".join(history)
    
    def chat(self):
        """Interactive chat interface with improved error handling."""
        print("\nMovie RAG Chatbot")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                if user_input.lower() == 'quit':
                    break
                
                # Generate response with timing
                start_time = time.time()
                try:
                    response = self.generate_response(
                        "Answer the following question about movies.",
                        user_input
                    )
                    print(f"\nBot: {response}")
                    print(f"\n(Response generated in {time.time() - start_time:.2f} seconds)")
                    
                except torch.cuda.OutOfMemoryError:
                    logger.error("GPU memory exhausted, attempting recovery...")
                    torch.cuda.empty_cache()
                    self.model.to("cpu")
                    print("\nBot: Let me think about that again...")
                    response = self.generate_response(
                        "Answer the following question about movies.",
                        user_input
                    )
                    print(f"\nBot: {response}")
                    
            except KeyboardInterrupt:
                print("\nExiting gracefully...")
                break
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                print("\nBot: I apologize, but I encountered an error. Please try again.")
                continue

def test_rag_system(rag: MovieRAG):
    """
    Test the RAG system with sample queries.
    
    Args:
        rag (MovieRAG): Initialized RAG system
    """
    test_queries = [
        "What is the plot of The Matrix?",
        "Who are the main actors in Inception?",
        "Tell me about Christopher Nolan's movies.",
        "What are some good sci-fi movies from the 90s?",
        "Compare The Godfather and Goodfellas."
    ]
    
    print("\nTesting RAG System")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            start_time = time.time()
            response = rag.generate_response(
                "Answer the following question about movies.",
                query
            )
            print(f"Response: {response}")
            print(f"(Generated in {time.time() - start_time:.2f} seconds)")
        except Exception as e:
            logger.error(f"Error testing query '{query}': {str(e)}")
            print("Error generating response")
        print("-" * 50)

def main():
    """Main execution function."""
    try:
        # Initialize RAG system
        rag = MovieRAG()
        
        # Run tests
        test_rag_system(rag)
        
        # Start interactive chat
        rag.chat()
        
    except Exception as e:
        logger.error(f"Error in RAG implementation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 