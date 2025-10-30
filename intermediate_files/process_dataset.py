import json
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text):
    """
    Clean and preprocess text content.
    
    Steps:
    1. Lowercase all text
    2. Remove punctuation and numbers
    3. Remove stopwords
    4. Lemmatize words
    5. Remove extra whitespace
    
    Args:
        text (str): Raw text content
        
    Returns:
        str: Cleaned text content
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase all text
    text = text.lower()
    
    # Step 2: Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Step 3: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Step 4: Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Step 5: Remove extra whitespace and join
    cleaned_text = ' '.join(tokens)
    
    # Remove any remaining multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def process_dataset(input_file, output_file):
    """
    Process the parsed dataset JSON file.
    
    Steps:
    1. Load JSON dataset
    2. Extract author_name and text_content fields
    3. Clean text_content for each entry
    4. Save to new JSON file
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    """
    print(f"Loading dataset from {input_file}...")
    
    # Step 1: Load JSON dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Total papers in dataset: {len(dataset)}")
    
    # Step 2 & 3: Extract fields and clean text
    processed_data = []
    
    for idx, paper in enumerate(dataset):
        print(f"Processing paper {idx + 1}/{len(dataset)}...", end='\r')
        
        # Extract only author_name and text_content
        author_name = paper.get('author_name', '')
        text_content = paper.get('text_content', '')
        
        # Skip if text_content is empty or None
        if not text_content:
            print(f"\nWarning: Skipping paper {idx + 1} - Empty text content")
            continue
        
        # Clean the text content
        cleaned_text = clean_text(text_content)
        
        # Skip if cleaned text is empty
        if not cleaned_text:
            print(f"\nWarning: Skipping paper {idx + 1} - Text became empty after cleaning")
            continue
        
        # Create processed entry
        processed_entry = {
            'author_name': author_name,
            'text_content': cleaned_text
        }
        
        processed_data.append(processed_entry)
    
    print(f"\nSuccessfully processed {len(processed_data)} papers")
    
    # Step 4: Save to output file
    print(f"\nSaving processed data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete! Saved to {output_file}")
    print(f"Original papers: {len(dataset)}")
    print(f"Processed papers: {len(processed_data)}")


if __name__ == "__main__":
    # Define input and output file paths
    input_file = "parsed_dataset.json"
    output_file = "cleaned_dataset.json"
    
    # Process the dataset
    process_dataset(input_file, output_file)
