#!/usr/bin/env python3
"""
========================================================
BATCH CSV PROCESSOR - Lemmatize large Finnish poem collections
========================================================

This script processes large CSV files containing full poems and lemmatizes
them using the V17 Phase 9 lemmatization system.

WHY WE NEED THIS:
Large CSV files (like SKVR runosongs with 170K poems) take many hours to process.
This script:
1. Shows results within the first 2-3 minutes (incremental saves)
2. Allows resuming if interrupted (checkpoint system)
3. Provides progress tracking with ETA
4. Processes efficiently in chunks

USAGE:
python3 process_skvr_batch.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results.csv \
  --chunk-size 100 \
  --save-interval 120

To resume after interruption:
python3 process_skvr_batch.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results.csv \
  --resume
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

from fin_runocorp_base import OmorfiHfstWithVoikkoV16Hybrid


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
    print("V15 BATCH CSV PROCESSOR - Finnish Runosong Lemmatization")
    print("=" * 80)
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

    # STEP 3: Load lemmatizer
    print("Loading V17 Phase 9 lemmatizer...", file=sys.stderr)
    if args.lexicon_path:
        print(f"  Using lexicon: {args.lexicon_path}", file=sys.stderr)
    load_start = time.time()
    lemmatizer = OmorfiHfstWithVoikkoV16Hybrid(lexicon_path=args.lexicon_path)
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
        csv_reader = pd.read_csv(input_csv, chunksize=args.chunk_size, skiprows=range(1, start_index + 1) if start_index > 0 else None)

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

                # Lemmatize all words in the poem (full context)
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
                        'context_score': result.get('context_score', 0.0),
                        'analysis': result.get('analysis', '')
                    })

                total_words_processed += len(words)
                poems_processed += 1
                pbar.update(1)

                # Save incrementally based on time interval
                current_time = time.time()
                if current_time - last_save_time >= args.save_interval:
                    # Write buffered results to CSV
                    with open(output_csv, output_mode, encoding='utf-8', newline='') as outfile:
                        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
                        if write_header:
                            writer.writeheader()
                            write_header = False
                        writer.writerows(chunk_buffer)

                    # Update output mode to append for future writes
                    output_mode = 'a'

                    # Save checkpoint
                    save_checkpoint(checkpoint_file, {
                        'last_processed_index': poems_processed - 1,
                        'last_processed_p_id': p_id,
                        'total_words_processed': total_words_processed,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Clear buffer and update timer
                    print(f"\n✓ Saved {len(chunk_buffer)} words to {output_csv.name}")
                    print(f"  Total progress: {poems_processed:,} poems, {total_words_processed:,} words")
                    chunk_buffer = []
                    last_save_time = current_time

        pbar.close()

        # STEP 7: Final save (any remaining buffered results)
        if chunk_buffer:
            with open(output_csv, output_mode, encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(chunk_buffer)

            print(f"\n✓ Saved final {len(chunk_buffer)} words to {output_csv.name}")

        # Save final checkpoint
        save_checkpoint(checkpoint_file, {
            'last_processed_index': poems_processed - 1,
            'total_words_processed': total_words_processed,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        })

        print()
        print("=" * 80)
        print("PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"✓ Processed {poems_processed:,} poems")
        print(f"✓ Generated {total_words_processed:,} word lemmatizations")
        print(f"✓ Results saved to: {output_csv}")
        print(f"✓ Checkpoint saved to: {checkpoint_file}")
        print()

    except KeyboardInterrupt:
        print("\n\nINTERRUPTED BY USER (Ctrl+C)")
        print("=" * 80)

        # Save any buffered results
        if chunk_buffer:
            with open(output_csv, output_mode, encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(chunk_buffer)
            print(f"✓ Saved {len(chunk_buffer)} words before exit")

        # Save checkpoint
        save_checkpoint(checkpoint_file, {
            'last_processed_index': poems_processed - 1,
            'total_words_processed': total_words_processed,
            'timestamp': datetime.now().isoformat(),
            'status': 'interrupted'
        })

        print(f"✓ Checkpoint saved at poem {poems_processed - 1}")
        print(f"✓ To resume, run with --resume flag")
        print()
        sys.exit(0)


def main():
    """Main program entry point"""

    parser = argparse.ArgumentParser(
        description='Batch process large CSV files with V15 Finnish lemmatization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire file with progress tracking
  python3 process_skvr_batch.py --input skvr_runosongs_okt_2025.csv --output results.csv

  # Resume interrupted processing
  python3 process_skvr_batch.py --input skvr_runosongs_okt_2025.csv --output results.csv --resume

  # Smaller chunks for faster saves
  python3 process_skvr_batch.py --input skvr_runosongs_okt_2025.csv --output results.csv --chunk-size 50 --save-interval 60
        """
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file (one poem per row with poemText column)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file (one word per row with lemma, method, etc.)')
    parser.add_argument('--lexicon-path', type=str, default=None,
                       help='Path to self-training lexicon JSON file (default: selftraining_lexicon_v16_min1.json)')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Number of poems to process per chunk (default: 100)')
    parser.add_argument('--save-interval', type=int, default=120,
                       help='Save results every N seconds (default: 120 = 2 minutes)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint (if available)')

    args = parser.parse_args()

    # Run batch processing
    process_batch(args)


if __name__ == "__main__":
    main()
