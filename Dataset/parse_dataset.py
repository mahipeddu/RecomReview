#!/usr/bin/env python3
"""
PDF Dataset Parser
Extracts text content from PDF files organized by author folders
and generates a JSON file with (author_name, text_content) pairs.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2
from tqdm import tqdm


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content as a string
    """
    try:
        text_content = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n"
        
        # Clean the text to remove surrogate characters and other problematic Unicode
        # This fixes the UnicodeEncodeError with surrogates
        text_content = text_content.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        return text_content.strip()
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ""


def parse_dataset(dataset_path: str, output_json: str) -> Tuple[List[Dict[str, str]], int]:
    """
    Parse all PDF files in the dataset directory and extract author names and text content.
    
    Args:
        dataset_path: Path to the Dataset directory
        output_json: Path to save the output JSON file
    
    Returns:
        Tuple of (parsed_data, total_pdfs_count)
    """
    parsed_data = []
    total_pdfs = 0
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist!")
        return parsed_data, 0
    
    print(f"Parsing PDFs from: {dataset_path}\n")
    
    # Iterate through each author directory
    for author_name in os.listdir(dataset_path):
        author_path = os.path.join(dataset_path, author_name)
        
        # Skip if not a directory
        if not os.path.isdir(author_path):
            continue
        
        print(f"Processing author: {author_name}")
        
        # Iterate through each PDF in the author's directory
        pdf_count = 0
        for filename in os.listdir(author_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(author_path, filename)
                
                # Extract text from PDF
                text_content = extract_text_from_pdf(pdf_path)
                
                if text_content:
                    parsed_data.append({
                        'author_name': author_name,
                        'paper_title': filename.replace('.pdf', ''),
                        'text_content': text_content
                    })
                    pdf_count += 1
                    total_pdfs += 1
                else:
                    print(f"  ⚠ Warning: Could not extract text from '{filename}'")
        
        print(f"  ✓ Parsed {pdf_count} PDFs for {author_name}\n")
    
    # Save to JSON
    print(f"Saving parsed data to '{output_json}'...")
    with open(output_json, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Data saved to '{output_json}'\n")
    
    return parsed_data, total_pdfs


def main():
    """Main function to run the parser."""
    # Set the dataset path
    dataset_path = "Dataset"
    output_json = "parsed_dataset.json"
    
    # Parse the dataset
    parsed_data, total_pdfs = parse_dataset(dataset_path, output_json)
    
    # Display statistics
    print("\n" + "="*80)
    print(f"PARSING COMPLETE!")
    print(f"Total PDFs parsed: {total_pdfs}")
    print(f"Total entries in JSON: {len(parsed_data)}")
    print("="*80)
    
    # Display sample entry
    if parsed_data:
        print("\nSample entry:")
        print("-"*80)
        sample = parsed_data[0]
        print(f"Author: {sample['author_name']}")
        print(f"Paper Title: {sample['paper_title']}")
        print(f"Text Content (first 500 chars):\n{sample['text_content'][:500]}...")
        print("="*80)


if __name__ == "__main__":
    main()
