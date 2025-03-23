#!/usr/bin/env python3
"""
Movie Data Preparation Script
----------------------------
This script prepares movie data for fine-tuning and RAG implementation.
It handles:
1. Loading and processing the movie dataset
2. Creating training examples for fine-tuning
3. Setting up the vector database for RAG
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import re
from collections import defaultdict
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('vector_db').mkdir(parents=True, exist_ok=True)

class MovieDataPreprocessor:
    def __init__(self, min_rating: float = 5.0):
        """
        Initialize the movie data preprocessor.
        
        Args:
            min_rating (float): Minimum IMDb rating to include
        """
        self.min_rating = min_rating
        self.stats = defaultdict(int)
        
    def preprocess_movie_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and validate movie data.
        
        Args:
            df (pd.DataFrame): Raw movie data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info("Preprocessing movie data...")
        
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Basic cleaning
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str).str.strip()
        
        # Convert and validate ratings
        df['imdbRating'] = pd.to_numeric(df['imdbRating'], errors='coerce')
        df = df[df['imdbRating'] >= self.min_rating]
        
        # Validate and standardize years
        df['Year'] = df['Year'].apply(self._extract_valid_year)
        df = df[df['Year'].notna()]
        
        # Clean and validate text fields
        df['Title'] = df['Title'].apply(self._clean_title)
        df['Plot'] = df['Plot'].apply(self._clean_text)
        df['Director'] = df['Director'].apply(self._clean_names)
        df['Actors'] = df['Actors'].apply(self._clean_names)
        df['Genre'] = df['Genre'].apply(self._clean_genres)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Title', 'Year'], keep='first')
        
        # Update statistics
        self._update_stats(df)
        
        return df
    
    def _clean_title(self, title: str) -> str:
        """Clean and validate movie title."""
        title = re.sub(r'[^\w\s\-\'\":;,\(\)]', '', title)
        return title.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean and validate text content."""
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-\'\":;,\.\(\)]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def _clean_names(self, names: str) -> str:
        """Clean and validate name lists."""
        if not names:
            return ''
        # Split names, clean each name, and rejoin
        names_list = [name.strip() for name in names.split(',')]
        names_list = [re.sub(r'[^\w\s\-\']', '', name).strip() for name in names_list]
        return ', '.join(name for name in names_list if name)
    
    def _clean_genres(self, genres: str) -> str:
        """Clean and validate genre lists."""
        if not genres:
            return ''
        # Split genres, clean each genre, and rejoin
        genres_list = [genre.strip() for genre in genres.split(',')]
        genres_list = [re.sub(r'[^\w\s\-]', '', genre).strip() for genre in genres_list]
        return ', '.join(genre for genre in genres_list if genre)
    
    def _extract_valid_year(self, year_str: str) -> Optional[int]:
        """Extract and validate year."""
        try:
            # Extract first 4-digit number between 1900 and current year
            year_match = re.search(r'\b(19\d{2}|20[0-2]\d)\b', str(year_str))
            if year_match:
                year = int(year_match.group(1))
                current_year = datetime.now().year
                if 1900 <= year <= current_year:
                    return year
            return None
        except Exception:
            return None
    
    def _update_stats(self, df: pd.DataFrame):
        """Update preprocessing statistics."""
        self.stats['total_movies'] = len(df)
        self.stats['unique_directors'] = df['Director'].nunique()
        self.stats['unique_actors'] = len(set(actor.strip() for actors in df['Actors'].str.split(',') for actor in actors if actor.strip()))
        self.stats['unique_genres'] = len(set(genre.strip() for genres in df['Genre'].str.split(',') for genre in genres if genre.strip()))
        self.stats['avg_rating'] = df['imdbRating'].mean()
        self.stats['movies_by_decade'] = df['Year'].apply(lambda x: f"{x//10*10}s").value_counts().to_dict()

def create_training_examples(movie: Dict) -> List[Dict]:
    """
    Create diverse training examples from a movie entry.
    
    Args:
        movie (Dict): Dictionary containing movie information
        
    Returns:
        List[Dict]: List of training examples
    """
    examples = []
    
    # Basic information examples
    examples.extend([
        {
            'instruction': 'Describe the plot of this movie.',
            'input': f"Movie: {movie['Title']}",
            'output': movie['Plot']
        },
        {
            'instruction': 'List the main actors in this movie.',
            'input': f"Movie: {movie['Title']}",
            'output': movie['Actors']
        },
        {
            'instruction': 'Who directed this movie?',
            'input': f"Movie: {movie['Title']}",
            'output': movie['Director']
        }
    ])
    
    # Genre-based examples
    genres = [g.strip() for g in movie['Genre'].split(',')]
    examples.append({
        'instruction': 'What are the genres of this movie?',
        'input': f"Movie: {movie['Title']}",
        'output': f"This movie belongs to the following genres: {', '.join(genres)}"
    })
    
    # Rating and year examples
    examples.append({
        'instruction': 'What is the rating and release year of this movie?',
        'input': f"Movie: {movie['Title']}",
        'output': f"This movie was released in {movie['Year']} and has an IMDb rating of {movie['imdbRating']}/10"
    })
    
    # Complex examples
    examples.extend([
        {
            'instruction': 'Provide a comprehensive overview of this movie.',
            'input': f"Movie: {movie['Title']}",
            'output': f"'{movie['Title']}' is a {movie['Genre']} film released in {movie['Year']}. "
                     f"Directed by {movie['Director']}, it stars {movie['Actors']}. {movie['Plot']} "
                     f"The movie has an IMDb rating of {movie['imdbRating']}/10."
        },
        {
            'instruction': f"Why might someone who enjoys {genres[0]} movies like this film?",
            'input': f"Movie: {movie['Title']}",
            'output': f"As a {genres[0]} movie, '{movie['Title']}' offers {movie['Plot']} "
                     f"With strong performances from {movie['Actors']} and direction by {movie['Director']}, "
                     f"it delivers what fans of the genre expect."
        }
    ])
    
    return examples

def create_vector_database(
    df: pd.DataFrame,
    encoder_name: str = 'all-MiniLM-L6-v2',
    index_type: str = 'HNSW'
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Create an optimized FAISS vector database from movie data.
    
    Args:
        df (pd.DataFrame): DataFrame containing movie information
        encoder_name (str): Name of the sentence transformer model
        index_type (str): Type of FAISS index to use ('Flat' or 'HNSW')
    """
    # Initialize the encoder model
    encoder = SentenceTransformer(encoder_name)
    encoder.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create documents for vector database
    documents = []
    for _, movie in tqdm(df.iterrows(), total=len(df), desc="Creating documents"):
        # Create a rich document representation
        doc = {
            'id': movie['Title'],
            'content': f"Title: {movie['Title']}\nDirector: {movie['Director']}\n"
                      f"Actors: {movie['Actors']}\nPlot: {movie['Plot']}\n"
                      f"Genre: {movie['Genre']}\nYear: {movie['Year']}\n"
                      f"Rating: {movie['imdbRating']}",
            'metadata': {
                'year': movie['Year'],
                'rating': movie['imdbRating'],
                'genres': movie['Genre'].split(','),
                'director': movie['Director']
            }
        }
        documents.append(doc)
    
    # Generate embeddings in batches
    batch_size = 32
    all_embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Generating embeddings"):
        batch = documents[i:i + batch_size]
        embeddings = encoder.encode([doc['content'] for doc in batch])
        all_embeddings.append(embeddings)
    
    embeddings = np.vstack(all_embeddings)
    dimension = embeddings.shape[1]
    
    # Create appropriate FAISS index
    if index_type == 'HNSW':
        # HNSW index for better search performance
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
        index.hnsw.efConstruction = 200  # More thorough index construction
        index.hnsw.efSearch = 128  # More thorough search
    else:
        # Simple flat index
        index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to index
    index.add(embeddings.astype('float32'))
    
    # Save the index and documents
    logger.info("Saving vector database...")
    faiss.write_index(index, 'vector_db/movie_faiss_index')
    with open('vector_db/movie_documents.pkl', 'wb') as f:
        pickle.dump(documents, f)
    
    return index, documents

def verify_data_quality(
    examples: List[Dict],
    index: faiss.Index,
    documents: List[Dict],
    preprocessor: MovieDataPreprocessor
):
    """
    Perform comprehensive data quality checks.
    
    Args:
        examples (List[Dict]): List of training examples
        index (faiss.Index): FAISS index
        documents (List[Dict]): List of documents
        preprocessor (MovieDataPreprocessor): Preprocessor with statistics
    """
    logger.info("\nData Quality Report:")
    logger.info("=" * 50)
    
    # Dataset statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total movies: {preprocessor.stats['total_movies']}")
    logger.info(f"Unique directors: {preprocessor.stats['unique_directors']}")
    logger.info(f"Unique actors: {preprocessor.stats['unique_actors']}")
    logger.info(f"Unique genres: {preprocessor.stats['unique_genres']}")
    logger.info(f"Average rating: {preprocessor.stats['avg_rating']:.2f}")
    
    # Movies by decade
    logger.info("\nMovies by Decade:")
    for decade, count in sorted(preprocessor.stats['movies_by_decade'].items()):
        logger.info(f"{decade}: {count} movies")
    
    # Training examples analysis
    logger.info("\nTraining Examples Analysis:")
    instruction_types = defaultdict(int)
    for example in examples:
        instruction_types[example['instruction']] += 1
    
    logger.info(f"Total training examples: {len(examples)}")
    logger.info("\nInstruction type distribution:")
    for instruction, count in instruction_types.items():
        logger.info(f"- {instruction}: {count} examples")
    
    # Vector database verification
    logger.info("\nVector Database Verification:")
    logger.info(f"Number of vectors: {index.ntotal}")
    logger.info(f"Number of documents: {len(documents)}")
    
    # Sample data quality check
    logger.info("\nSample Training Examples:")
    for example in examples[:3]:
        logger.info("\n---")
        logger.info(f"Instruction: {example['instruction']}")
        logger.info(f"Input: {example['input']}")
        logger.info(f"Output: {example['output']}")
    
    # Verify vector search
    if len(documents) > 0:
        logger.info("\nVector Search Test:")
        test_doc = documents[0]
        query_vector = encoder.encode([test_doc['content']])[0].reshape(1, -1).astype('float32')
        distances, indices = index.search(query_vector, 5)
        logger.info(f"Successfully retrieved {len(indices[0])} similar documents")

def main():
    """Main execution function with error handling and validation."""
    try:
        # Initialize preprocessor
        preprocessor = MovieDataPreprocessor(min_rating=5.0)
        
        # Load and preprocess the movie dataset
        logger.info("Loading movie dataset...")
        df = pd.read_csv('data/movies.csv')
        df = preprocessor.preprocess_movie_data(df)
        logger.info(f"Processed {len(df)} movies")
        
        # Generate training examples
        logger.info("\nGenerating training examples...")
        all_examples = []
        for _, movie in tqdm(df.iterrows(), total=len(df), desc="Processing movies"):
            examples = create_training_examples(movie.to_dict())
            all_examples.extend(examples)
        
        # Save training examples
        logger.info("Saving training examples...")
        with open('data/processed/training_examples.json', 'w') as f:
            json.dump(all_examples, f, indent=2)
        logger.info(f"Created {len(all_examples)} training examples")
        
        # Create vector database with HNSW index
        logger.info("\nCreating vector database...")
        index, documents = create_vector_database(df, index_type='HNSW')
        
        # Verify data quality
        verify_data_quality(all_examples, index, documents, preprocessor)
        
        logger.info("\nData preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 