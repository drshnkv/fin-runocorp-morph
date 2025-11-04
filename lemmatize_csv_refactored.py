#!/usr/bin/env python3
"""
Process CSV file with refactored lemmatizer

Usage:
    python3 lemmatize_csv_refactored.py input.csv output.csv

Input CSV format:
    poemTitle,text
    "Poem 1","word1 word2 word3"

Output CSV format:
    poem_id,word,lemma,method,confidence
"""

import sys
import csv
from pathlib import Path
from typing import List, Dict
from lemmatizer import FinnishLemmatizer
from lemmatizer_config import LemmatizerConfig

def process_csv(input_file: str, output_file: str, use_sentence_context: bool = True):
    """
    Process CSV file with refactored lemmatizer

    Args:
        input_file: Path to input CSV with poems
        output_file: Path to output CSV with lemmatized results
        use_sentence_context: If True, use lemmatize_sentence() for better accuracy
    """
    print(f"Loading refactored lemmatizer...")
    lemmatizer = FinnishLemmatizer()

    print(f"Reading input: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        poems = list(reader)

    print(f"Processing {len(poems)} poems...")

    results = []
    for i, poem in enumerate(poems, 1):
        poem_id = poem['poemTitle']
        text = poem.get('poemText') or poem.get('text', '')

        # Tokenize (simple whitespace split)
        tokens = text.split()

        if use_sentence_context:
            # Use sentence-level processing (recommended)
            lemma_results = lemmatizer.lemmatize_sentence(tokens)
        else:
            # Use word-by-word processing
            lemma_results = [lemmatizer.lemmatize(token) for token in tokens]

        for result in lemma_results:
            results.append({
                'poem_id': poem_id,
                'word': result['word'],
                'lemma': result['lemma'],
                'method': result.get('method', 'unknown'),
                'confidence': result.get('confidence', 'unknown')
            })

        if i % 5 == 0:
            print(f"  Processed {i}/{len(poems)} poems...")

    print(f"Writing output: {output_file}")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['poem_id', 'word', 'lemma', 'method', 'confidence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Done! Processed {len(results)} words")
    print(f"✓ Output saved to: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 lemmatize_csv_refactored.py input.csv output.csv")
        print()
        print("Example:")
        print("  python3 lemmatize_csv_refactored.py skvr_test_6poems.csv results_refactored.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    process_csv(input_file, output_file)

if __name__ == '__main__':
    main()
