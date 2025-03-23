#!/usr/bin/env python3
"""
Script to scrape movie data for fine-tuning and RAG implementation
"""

import os
import requests
import pandas as pd
import time
import random
import json
from tqdm import tqdm
import csv
import re

# Create output directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# OMDB API
OMDB_API_KEY = "YOUR_OMDB_API_KEY"  # Replace with your API key from http://www.omdbapi.com/

# TMDB API 
TMDB_API_KEY = "YOUR_TMDB_API_KEY"  # Replace with your API key from https://www.themoviedb.org/

# IMDb Top 250 Movies (sample)
TOP_MOVIES = [
    "The Shawshank Redemption", "The Godfather", "The Dark Knight", 
    "The Godfather Part II", "12 Angry Men", "Schindler's List",
    "The Lord of the Rings: The Return of the King", "Pulp Fiction",
    "The Lord of the Rings: The Fellowship of the Ring", "Forrest Gump",
    "Inception", "Fight Club", "The Matrix", "Goodfellas",
    "The Lord of the Rings: The Two Towers", "Star Wars: Episode V - The Empire Strikes Back",
    "One Flew Over the Cuckoo's Nest", "The Silence of the Lambs", "Interstellar",
    "Saving Private Ryan", "City of God", "Life Is Beautiful", "The Green Mile",
    "Seven Samurai", "Spirited Away", "Parasite", "The Lion King", "Back to the Future",
    "The Pianist", "Gladiator", "The Departed", "Whiplash", "The Prestige",
    "Casablanca", "Alien", "Apocalypse Now", "Memento", "Raiders of the Lost Ark",
    "Django Unchained", "WALL·E", "The Shining", "The Dark Knight Rises", 
    "Avengers: Infinity War", "Joker", "Oldboy", "Princess Mononoke"
]

# Popular genres to balance the dataset
GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", 
    "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", 
    "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport", 
    "Thriller", "War", "Western"
]

def fetch_omdb_data(title):
    """Fetch movie data from OMDB API"""
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('Response') == 'True':
            return data
    
    return None

def fetch_tmdb_movie_id(title):
    """Fetch movie ID from TMDB API"""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('results') and len(data['results']) > 0:
            return data['results'][0]['id']
    
    return None

def fetch_tmdb_reviews(movie_id):
    """Fetch movie reviews from TMDB API"""
    if not movie_id:
        return []
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        reviews = []
        for review in data.get('results', []):
            content = review.get('content', '')
            if content:
                # Clean and truncate review
                content = re.sub(r'\s+', ' ', content)
                content = content[:500] + '...' if len(content) > 500 else content
                reviews.append(content)
        return reviews
    
    return []

def generate_prompts(movie_data):
    """Generate instruction prompts for fine-tuning"""
    prompts = []
    
    # Basic info prompts
    title = movie_data.get('Title', '')
    year = movie_data.get('Year', '')
    director = movie_data.get('Director', '')
    actors = movie_data.get('Actors', '')
    plot = movie_data.get('Plot', '')
    genre = movie_data.get('Genre', '')
    
    # Reviews
    reviews = movie_data.get('reviews', [])
    review_text = " ".join(reviews)
    
    # Generate different types of prompts
    
    # Movie recommendation
    prompts.append({
        'instruction': f"Recommend movies similar to {title}.",
        'input': f"I enjoyed watching {title} directed by {director}. I particularly like {genre} movies. Can you suggest similar films?",
        'response': f"Based on your enjoyment of {title}, I recommend checking out these similar {genre} films:\n\n1. [Similar Movie 1] - Like {title}, this film features [similar theme/style].\n2. [Similar Movie 2] - Directed by [director], who has a style comparable to {director}.\n3. [Similar Movie 3] - Stars [actor] from {title} in a similar role.\n\nAll of these capture the essence of {title} while offering fresh perspectives on similar themes."
    })
    
    # Movie plot explanation
    prompts.append({
        'instruction': f"Explain the plot of {title}.",
        'input': f"What happens in the movie {title}? Please avoid major spoilers.",
        'response': f"{plot}"
    })
    
    # Movie analysis
    if reviews:
        prompts.append({
            'instruction': f"Analyze critical reception of {title}.",
            'input': f"What did critics think about {title}? Was it well-received?",
            'response': f"Critics had varied opinions on {title} ({year}):\n\n" + 
                       f"Some praised {director}'s direction and the performances of {actors}. " +
                       f"Based on reviews: {review_text[:300]}..." if len(review_text) > 300 else review_text
        })
    
    # Cast information
    prompts.append({
        'instruction': f"List the main cast of {title}.",
        'input': f"Who starred in {title}? What roles did they play?",
        'response': f"The main cast of {title} ({year}) includes:\n\n{actors}\n\nDirected by {director}, this {genre} film features these actors in their respective iconic roles."
    })
    
    # Genre discussion
    prompts.append({
        'instruction': f"Discuss the {genre} elements in {title}.",
        'input': f"How does {title} represent the {genre} genre? What genre tropes does it use or subvert?",
        'response': f"{title} is primarily classified as {genre}. It exemplifies this genre through its [specific elements], while also bringing fresh perspectives by [innovative approaches]. {director}'s direction particularly emphasizes [genre-specific techniques] that have become hallmarks of quality {genre} filmmaking."
    })
    
    return prompts

def main():
    print("Starting movie data collection...")
    
    all_movie_data = []
    all_prompts = []
    
    # Process top movies
    for title in tqdm(TOP_MOVIES, desc="Fetching movie data"):
        try:
            # Get basic movie info from OMDB
            movie_data = fetch_omdb_data(title)
            if not movie_data:
                continue
            
            # Get movie ID from TMDB for reviews
            movie_id = fetch_tmdb_movie_id(title)
            
            # Get reviews
            reviews = fetch_tmdb_reviews(movie_id)
            movie_data['reviews'] = reviews
            
            all_movie_data.append(movie_data)
            
            # Generate instruction prompts
            prompts = generate_prompts(movie_data)
            all_prompts.extend(prompts)
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        except Exception as e:
            print(f"Error processing {title}: {str(e)}")
    
    # Add more movies by genre to balance dataset
    for genre in tqdm(GENRES, desc="Adding genre-specific movies"):
        try:
            # Search movies by genre using TMDB
            url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre}&sort_by=popularity.desc"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                # Take top 3 movies from each genre
                for movie in data.get('results', [])[:3]:
                    title = movie.get('title')
                    
                    # Check if we already have this movie
                    if any(m.get('Title') == title for m in all_movie_data):
                        continue
                    
                    # Get full movie data from OMDB
                    movie_data = fetch_omdb_data(title)
                    if not movie_data:
                        continue
                    
                    # Get reviews
                    reviews = fetch_tmdb_reviews(movie.get('id'))
                    movie_data['reviews'] = reviews
                    
                    all_movie_data.append(movie_data)
                    
                    # Generate instruction prompts
                    prompts = generate_prompts(movie_data)
                    all_prompts.extend(prompts)
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        except Exception as e:
            print(f"Error processing genre {genre}: {str(e)}")
    
    # Save raw movie data
    with open('data/raw/movie_data.json', 'w') as f:
        json.dump(all_movie_data, f, indent=2)
    
    # Convert to dataframe and save
    movies_df = pd.DataFrame([{
        'title': m.get('Title', ''),
        'year': m.get('Year', ''),
        'director': m.get('Director', ''),
        'actors': m.get('Actors', ''),
        'plot': m.get('Plot', ''),
        'genres': m.get('Genre', ''),
        'ratings': str(m.get('Ratings', [])),
        'reviews': '; '.join(m.get('reviews', []))
    } for m in all_movie_data])
    
    movies_df.to_csv('data/processed/movie_data.csv', index=False)
    
    # Save instruction tuning data
    prompts_df = pd.DataFrame(all_prompts)
    prompts_df.to_csv('data/processed/movie_conversations.csv', index=False)
    
    # Create a sample of conversations for easy inspection
    prompts_sample = random.sample(all_prompts, min(10, len(all_prompts)))
    with open('data/processed/sample_conversations.txt', 'w') as f:
        for i, prompt in enumerate(prompts_sample):
            f.write(f"Example {i+1}\n")
            f.write(f"Instruction: {prompt['instruction']}\n")
            f.write(f"Input: {prompt['input']}\n")
            f.write(f"Response: {prompt['response']}\n")
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"Data collection complete. Collected {len(all_movie_data)} movies and generated {len(all_prompts)} conversation examples.")

if __name__ == "__main__":
    main()
