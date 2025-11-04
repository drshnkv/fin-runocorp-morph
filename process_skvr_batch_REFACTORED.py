#!/usr/bin/env python3
"""
========================================================
BATCH CSV PROCESSOR (REFACTORED) - Lemmatize with Refactored Modules
========================================================

This script processes large CSV files containing full poems and lemmatizes
them using the refactored V17 Phase 10 lemmatization system.

USES REFACTORED MODULES:
- lemmatizer.py (Pipeline Layer)
- lemmatizer_core.py (Analysis Layer)
- lemmatizer_config.py (Configuration Layer)

FEATURES:
- 59.9% accuracy (verified match with original)
- Integrates 19,385 validated dialectal variants from SMS
- Confidence-based dictionary override for spelling guesses
- Incremental saves (see results within 2-3 minutes)
- Checkpoint system (resume after interruption)
- Progress tracking with ETA
- Efficient chunked processing

USAGE:
# Basic usage with defaults
python3 process_skvr_batch_REFACTORED.py \\
  --input skvr_runosongs_okt_2025.csv \\
  --output skvr_lemmatized_results_refactored.csv \\
  --chunk-size 100

# Using combined train+test lexicon (recommended for production)
python3 process_skvr_batch_REFACTORED.py \\
  --input skvr_runosongs_okt_2025.csv \\
  --output skvr_lemmatized_results_refactored.csv \\
  --lexicon-path selftraining_lexicon_comb_with_additions.json \\
  --chunk-size 100

# Using custom dialectal dictionary
python3 process_skvr_batch_REFACTORED.py \\
  --input skvr_runosongs_okt_2025.csv \\
  --output skvr_lemmatized_results_refactored.csv \\
  --dialectal-dict-path custom_dialectal_index.json

# Resume after interruption
python3 process_skvr_batch_REFACTORED.py \\
  --input skvr_runosongs_okt_2025.csv \\
  --output skvr_lemmatized_results_refactored.csv \\
  --resume

COMPARISON WITH ORIGINAL:
- Accuracy: 59.9% (exact match verified)
- Code: Modular (3 files) vs monolithic (1 file)
- Features: Identical
- Performance: Identical
"""

import csv
import json
import sys
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Install with: pip install tqdm", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

# Import refactored modules
from lemmatizer import FinnishLemmatizer
from lemmatizer_config import LemmatizerConfig


def tokenize_poem(poem_text: str) -> List[str]:
    """
    Tokenize a poem into individual words

    Handles Finnish characters (ä, ö, å) and preserves word boundaries.
    Removes punctuation but keeps the actual words.

    Parameters:
    - poem_text: Full poem text (multi-line string)

    Returns:
    - List of words (tokens)
    """
    # Use regex to extract words (including Finnish characters)
    # \b = word boundary, \w = word characters
    words = re.findall(r'\b[\wäöåÄÖÅ]+\b', poem_text, re.UNICODE)
    return words


def load_checkpoint(checkpoint_file: Path) -> Optional[Dict]:
    """
    Load checkpoint from previous run

    Returns:
    - Checkpoint data (dict) or None if not found
    """
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load checkpoint: {e}", file=sys.stderr)
            return None
    return None


def save_checkpoint(checkpoint_file: Path, data: Dict):
    """Save checkpoint data for resuming later"""
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"WARNING: Could not save checkpoint: {e}", file=sys.stderr)


def process_batch(args):
    """
    Main batch processing function

    Reads large CSV, processes poems in chunks, saves incrementally
    """

    # STEP 1: Setup
    print("=" * 80)
    print("BATCH CSV PROCESSOR (REFACTORED) - Finnish Runosong Lemmatization")
    print("=" * 80)
    print()
    print("Using refactored modules:")
    print("  ✓ lemmatizer.py (Pipeline Layer)")
    print("  ✓ lemmatizer_core.py (Analysis Layer)")
    print("  ✓ lemmatizer_config.py (Configuration Layer)")
    print()

    input_csv = Path(args.input)
    output_csv = Path(args.output)
    checkpoint_file = Path(args.output).with_suffix('.checkpoint.json')

    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}", file=sys.stderr)
        sys.exit(1)

    # STEP 2: Load checkpoint (if resuming)
    start_index = 0
    checkpoint = None

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            start_index = checkpoint.get('last_processed_index', 0) + 1
            print(f"✓ Resuming from checkpoint: poem {start_index}")
            print(f"  Previous run processed: {checkpoint.get('total_words_processed', 0)} words")
            print()
        else:
            print("WARNING: No checkpoint found, starting from beginning", file=sys.stderr)
            print()

    # STEP 3: Load refactored lemmatizer with configuration
    print("Loading refactored lemmatizer (V17 Phase 10)...", file=sys.stderr)

    # Build configuration with optional custom paths
    config_params = {}
    if args.lexicon_path:
        print(f"  Using lexicon: {args.lexicon_path}", file=sys.stderr)
        config_params['lexicon_path'] = args.lexicon_path
    if args.dialectal_dict_path:
        print(f"  Using dialectal dictionary: {args.dialectal_dict_path}", file=sys.stderr)
        config_params['dialectal_dict_path'] = args.dialectal_dict_path

    if config_params:
        config = LemmatizerConfig(**config_params)
    else:
        config = None  # Uses default lexicon + default dialectal dictionary

    load_start = time.time()
    lemmatizer = FinnishLemmatizer(config=config)
    load_time = time.time() - load_start
    print(f"✓ Lemmatizer loaded in {load_time:.1f} seconds")
    print()

    # STEP 4: Count total poems for progress tracking
    print(f"Counting poems in {input_csv.name}...", file=sys.stderr)
    try:
        # Quick row count
        with open(input_csv, 'r', encoding='utf-8') as f:
            total_poems = sum(1 for line in f) - 1  # Subtract header row
    except Exception as e:
        print(f"WARNING: Could not count poems: {e}", file=sys.stderr)
        total_poems = None

    if total_poems:
        print(f"✓ Found {total_poems:,} poems")
        if start_index > 0:
            print(f"  Remaining to process: {total_poems - start_index:,} poems")
    print()

    # STEP 5: Setup output CSV
    output_mode = 'a' if (args.resume and output_csv.exists()) else 'w'
    write_header = (output_mode == 'w')

    output_fieldnames = ['p_id', 'nro', 'poemTitle', 'word_index', 'word',
                        'lemma', 'method', 'confidence', 'context_score', 'analysis']

    # STEP 6: Process poems in chunks
    print(f"Processing poems (chunk size: {args.chunk_size}, save interval: {args.save_interval}s)...")
    print("=" * 80)
    print()

    chunk_buffer = []
    total_words_processed = checkpoint.get('total_words_processed', 0) if checkpoint else 0
    last_save_time = time.time()
    poems_processed = start_index

    try:
        # Read CSV with pandas for efficient chunking
        csv_reader = pd.read_csv(input_csv, chunksize=args.chunk_size,
                                 skiprows=range(1, start_index + 1) if start_index > 0 else None)

        # Use tqdm for progress bar
        pbar = tqdm(total=total_poems - start_index if total_poems else None,
                   initial=0,
                   desc="Processing",
                   unit="poems")

        for chunk_df in csv_reader:
            # Process each poem in the chunk
            for idx, row in chunk_df.iterrows():
                p_id = row.get('p_id', '')
                nro = row.get('nro', '')
                poem_title = row.get('poemTitle', '')
                poem_text = row.get('poemText', '')

                if not poem_text or pd.isna(poem_text):
                    poems_processed += 1
                    pbar.update(1)
                    continue

                # Tokenize poem into words
                words = tokenize_poem(str(poem_text))

                if not words:
                    poems_processed += 1
                    pbar.update(1)
                    continue

                # Lemmatize all words in the poem (full context with sentence-level API)
                results = lemmatizer.lemmatize_sentence(words)

                # Collect results for this poem
                for word_idx, result in enumerate(results):
                    chunk_buffer.append({
                        'p_id': p_id,
                        'nro': nro,
                        'poemTitle': poem_title,
                        'word_index': word_idx,
                        'word': result.get('word', ''),
                        'lemma': result.get('lemma', ''),
                        'method': result.get('method', ''),
                        'confidence': result.get('confidence', ''),
                        'context_score': result.get('context_score', ''),
                        'analysis': result.get('analysis', '')
                    })

                total_words_processed += len(results)
                poems_processed += 1
                pbar.update(1)

                # STEP 7: Save incrementally (based on time interval)
                current_time = time.time()
                if current_time - last_save_time >= args.save_interval:
                    # Write buffered results to CSV
                    with open(output_csv, output_mode, newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
                        if write_header:
                            writer.writeheader()
                            write_header = False
                        writer.writerows(chunk_buffer)

                    # Clear buffer and switch to append mode
                    chunk_buffer = []
                    output_mode = 'a'
                    last_save_time = current_time

                    # Save checkpoint
                    save_checkpoint(checkpoint_file, {
                        'last_processed_index': poems_processed - 1,
                        'total_words_processed': total_words_processed,
                        'timestamp': datetime.now().isoformat()
                    })

        pbar.close()

        # STEP 8: Final save (flush remaining buffer)
        if chunk_buffer:
            with open(output_csv, output_mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=output_fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(chunk_buffer)

        # Save final checkpoint
        save_checkpoint(checkpoint_file, {
            'last_processed_index': poems_processed - 1,
            'total_words_processed': total_words_processed,
            'timestamp': datetime.now().isoformat(),
            'completed': True
        })

        print()
        print("=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"✓ Total poems processed: {poems_processed:,}")
        print(f"✓ Total words lemmatized: {total_words_processed:,}")
        print(f"✓ Output saved to: {output_csv}")
        print(f"✓ Checkpoint saved to: {checkpoint_file}")
        print()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("PROCESSING INTERRUPTED")
        print("=" * 80)

        # Save remaining buffer
        if chunk_buffer:
            with open(output_csv, output_mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=output_fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(chunk_buffer)

        # Save checkpoint
        save_checkpoint(checkpoint_file, {
            'last_processed_index': poems_processed - 1,
            'total_words_processed': total_words_processed,
            'timestamp': datetime.now().isoformat(),
            'interrupted': True
        })

        print(f"✓ Partial results saved to: {output_csv}")
        print(f"✓ Checkpoint saved to: {checkpoint_file}")
        print(f"  Processed {poems_processed:,} poems ({total_words_processed:,} words)")
        print()
        print("To resume, run:")
        print(f"  python3 {sys.argv[0]} --input {args.input} --output {args.output} --resume")
        print()
        sys.exit(1)

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR DURING PROCESSING")
        print("=" * 80)
        print(f"ERROR: {e}", file=sys.stderr)

        # Save checkpoint on error
        if poems_processed > 0:
            save_checkpoint(checkpoint_file, {
                'last_processed_index': poems_processed - 1,
                'total_words_processed': total_words_processed,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            print(f"✓ Checkpoint saved to: {checkpoint_file}")
            print()
            print("To resume, run:")
            print(f"  python3 {sys.argv[0]} --input {args.input} --output {args.output} --resume")
        print()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Batch process Finnish runosongs with refactored lemmatizer (59.9% accuracy verified)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage with defaults
  python3 process_skvr_batch_REFACTORED.py \\
    --input skvr_runosongs_okt_2025.csv \\
    --output results_refactored.csv \\
    --chunk-size 100

  # Using combined lexicon (recommended)
  python3 process_skvr_batch_REFACTORED.py \\
    --input skvr_runosongs_okt_2025.csv \\
    --output results_refactored.csv \\
    --lexicon-path selftraining_lexicon_comb_with_additions.json

  # Resume after interruption
  python3 process_skvr_batch_REFACTORED.py \\
    --input skvr_runosongs_okt_2025.csv \\
    --output results_refactored.csv \\
    --resume
        '''
    )

    parser.add_argument('--input', required=True, help='Input CSV file with poems')
    parser.add_argument('--output', required=True, help='Output CSV file for lemmatized results')
    parser.add_argument('--lexicon-path', help='Path to training lexicon JSON file (default: selftraining_lexicon_v16_min1.json)')
    parser.add_argument('--dialectal-dict-path', help='Path to SMS dialectal dictionary JSON file (default: sms_dialectal_index_v4_final.json)')
    parser.add_argument('--chunk-size', type=int, default=100, help='Number of poems to process per chunk (default: 100)')
    parser.add_argument('--save-interval', type=int, default=120, help='Save results every N seconds (default: 120)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous checkpoint')

    args = parser.parse_args()

    process_batch(args)


if __name__ == '__main__':
    main()
