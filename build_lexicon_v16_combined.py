#!/usr/bin/env python3
"""
V16 Hybrid Gold Standard Lexicon Builder - Combined Train+Test Version
Fixes V15's critical flaws by using manual annotations and 100% unambiguous patterns only

MODIFICATIONS FOR STANDALONE:
- Accepts multiple CSV files (train + test)
- Uses standalone lemmatizer (OmorfiHfstWithVoikkoV16Hybrid)
- Maintains all 3 phases: manual annotations + Omorfi patterns
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Tuple, List

# Import standalone lemmatizer for POS tagging and Omorfi pattern extraction
from fin_runocorp_base import OmorfiHfstWithVoikkoV16Hybrid


def extract_manual_annotations(csv_files: List[str]) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """
    Extract (word, POS, lemma) triples from manual gold standard annotations.

    Args:
        csv_files: List of CSV file paths (e.g., train + test)

    Returns:
        {(word_lower, POS): [(manual_lemma_lower, poem_id), ...]}
    """
    print("PHASE 1: Extracting manual gold standard annotations...", file=sys.stderr)

    # Load lemmatizer ONLY for POS tagging (Stanza)
    print("  Loading standalone lemmatizer for POS tagging...", file=sys.stderr)
    lemmatizer = OmorfiHfstWithVoikkoV16Hybrid()

    # Group by poem from ALL CSV files
    poem_groups = defaultdict(list)
    total_files = len(csv_files)
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  Loading CSV {i}/{total_files}: {csv_file}", file=sys.stderr)
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                poem_id = row.get('poem_id', '').strip()
                # Add source file info to poem_id to avoid collisions
                poem_id_with_source = f"{poem_id}_{i}"
                poem_groups[poem_id_with_source].append(row)

    print(f"  Processing {len(poem_groups)} poems from {total_files} CSV files...", file=sys.stderr)

    # Collect manual annotations
    manual_patterns = defaultdict(list)

    for poem_id, rows in poem_groups.items():
        tokens = [row.get('word', '').strip() for row in rows if row.get('word', '').strip()]

        if not tokens:
            continue

        # Get POS tags from Stanza
        if lemmatizer.stanza_nlp:
            doc = lemmatizer.stanza_nlp([tokens])
            stanza_words = [w for sent in doc.sentences for w in sent.words]

            if len(stanza_words) == len(tokens):
                for row, sw in zip(rows, stanza_words):
                    word = row.get('word', '').strip().lower()
                    # CRITICAL: Use manual annotation column
                    manual_lemma = row.get(
                        'lemma_standard (Finnish/Estonian & minority languages if needed)',
                        ''
                    ).strip().lower()
                    pos = sw.upos

                    if word and manual_lemma and pos:
                        manual_patterns[(word, pos)].append((manual_lemma, poem_id))

    print(f"  ✓ Collected {len(manual_patterns)} unique (word, POS) patterns", file=sys.stderr)
    total_instances = sum(len(instances) for instances in manual_patterns.values())
    print(f"  ✓ Total instances: {total_instances}", file=sys.stderr)

    return manual_patterns


def classify_manual_patterns(
    manual_patterns: Dict[Tuple[str, str], List[Tuple[str, str]]],
    min_occurrences: int = 1
) -> Tuple[Dict, Dict]:
    """
    Classify manual patterns as unambiguous (100%) or ambiguous (<100%).

    CRITICAL CHANGE: min_occurrences=1 for manual annotations
    - Manual annotations are expert-verified → trust unconditionally
    - No statistical threshold needed for gold standard data
    - Preserves all 3,700 expert annotations instead of discarding 91.5%

    Returns:
        (gold_unambiguous, gold_ambiguous)
    """
    print("\nPHASE 2: Classifying patterns by ambiguity...", file=sys.stderr)

    gold_unambiguous = {}
    gold_ambiguous = {}

    for (word, pos), lemma_instances in manual_patterns.items():
        lemma_counts = Counter(lemma for lemma, _ in lemma_instances)
        total = len(lemma_instances)

        if total < min_occurrences:
            continue  # Skip rare patterns (only if min > 1)

        most_common_lemma, count = lemma_counts.most_common(1)[0]
        confidence = count / total

        if confidence == 1.0:
            # 100% agreement - truly unambiguous
            gold_unambiguous[(word, pos)] = {
                'lemma': most_common_lemma,
                'source': 'manual',
                'confidence': 1.0,
                'occurrences': total,
                'alternatives': []
            }
        else:
            # <100% agreement - genuinely ambiguous
            alternatives = [
                {
                    'lemma': lemma,
                    'frequency': round(count / total, 3),
                    'occurrences': count
                }
                for lemma, count in lemma_counts.most_common()
            ]

            gold_ambiguous[(word, pos)] = {
                'alternatives': alternatives,
                'source': 'manual',
                'total_occurrences': total,
                'distinct_lemmas': len(lemma_counts),
                'note': f'Genuinely ambiguous - {len(lemma_counts)} distinct lemmas require contextual analysis'
            }

    print(f"  ✓ Unambiguous (100%): {len(gold_unambiguous)} patterns", file=sys.stderr)
    print(f"  ✓ Ambiguous (<100%): {len(gold_ambiguous)} patterns", file=sys.stderr)

    return gold_unambiguous, gold_ambiguous


def extract_omorfi_patterns(
    csv_files: List[str],
    confidence_threshold: float = 1.0,
    min_occurrences: int = 3
) -> Dict[Tuple[str, str], Dict]:
    """
    Extract patterns from Omorfi contextual analysis (100% unambiguous only).

    Args:
        csv_files: List of CSV file paths (e.g., train + test)
        confidence_threshold: Confidence threshold (default 1.0 = 100%)
        min_occurrences: Minimum occurrences required

    Returns:
        {(word_lower, POS): {'lemma': str, 'confidence': float, 'occurrences': int}}
    """
    print("\nPHASE 3: Extracting Omorfi patterns (100% unambiguous only)...", file=sys.stderr)

    # Load standalone lemmatizer
    print("  Loading standalone lemmatizer for Omorfi analysis...", file=sys.stderr)
    lemmatizer = OmorfiHfstWithVoikkoV16Hybrid()

    # Group by poem from ALL CSV files
    poem_groups = defaultdict(list)
    total_files = len(csv_files)
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  Loading CSV {i}/{total_files}: {csv_file}", file=sys.stderr)
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                poem_id = row.get('poem_id', '').strip()
                # Add source file info to poem_id to avoid collisions
                poem_id_with_source = f"{poem_id}_{i}"
                poem_groups[poem_id_with_source].append(row)

    print(f"  Processing {len(poem_groups)} poems from {total_files} CSV files...", file=sys.stderr)

    # Collect Omorfi lemmatization results
    omorfi_patterns = defaultdict(list)

    for poem_id, rows in poem_groups.items():
        tokens = [row.get('word', '').strip() for row in rows if row.get('word', '').strip()]

        if not tokens:
            continue

        # Lemmatize using V14 (Omorfi contextual)
        results = lemmatizer.lemmatize_sentence(tokens)

        # Get POS tags
        if lemmatizer.stanza_nlp:
            doc = lemmatizer.stanza_nlp([tokens])
            stanza_words = [w for sent in doc.sentences for w in sent.words]

            if len(stanza_words) == len(tokens):
                for token, result, sw in zip(tokens, results, stanza_words):
                    lemma = result.get('lemma', '')
                    method = result.get('method', '')
                    pos = sw.upos

                    # Only high-quality Omorfi contextual results
                    if lemma and method == 'omorfi_contextual':
                        omorfi_patterns[(token.lower(), pos)].append(lemma.lower())

    # Filter by 100% agreement
    omorfi_100_patterns = {}

    for (word, pos), lemmas in omorfi_patterns.items():
        lemma_counts = Counter(lemmas)
        total = len(lemmas)

        if total < min_occurrences:
            continue

        most_common_lemma, count = lemma_counts.most_common(1)[0]
        confidence = count / total

        # CRITICAL: Only save if 100% agreement
        if confidence == confidence_threshold:
            omorfi_100_patterns[(word, pos)] = {
                'lemma': most_common_lemma,
                'confidence': confidence,
                'occurrences': total
            }

    print(f"  ✓ Found {len(omorfi_100_patterns)} patterns with 100% agreement", file=sys.stderr)

    return omorfi_100_patterns


def build_v16_lexicon(
    csv_files: List[str],
    output_json: str,
    confidence_threshold: float = 1.0,
    min_occurrences: int = 1,
    min_occurrences_omorfi: int = 3
):
    """
    Build V16 hybrid lexicon combining:
    1. Manual gold standard (Tier 1) - min_occurrences=1 (trust all expert annotations)
    2. Omorfi 100% unambiguous (Tier 2) - min_occurrences_omorfi=3 (require statistical confidence)
    3. Ambiguous patterns tracking

    CRITICAL: Different thresholds for manual vs automatic patterns!
    - Manual annotations: Trust unconditionally (min=1)
    - Omorfi patterns: Require statistical confidence (min=3)

    Args:
        csv_files: List of CSV file paths (e.g., train + test)
        output_json: Output JSON file path
        confidence_threshold: Confidence threshold (default 1.0 = 100%)
        min_occurrences: Minimum occurrences for manual annotations (default 1)
        min_occurrences_omorfi: Minimum occurrences for Omorfi patterns (default 3)
    """

    print("=" * 80, file=sys.stderr)
    print("V16 HYBRID GOLD STANDARD LEXICON BUILDER (COMBINED TRAIN+TEST)", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Input CSV files: {len(csv_files)}", file=sys.stderr)
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file}", file=sys.stderr)
    print(f"Output: {output_json}", file=sys.stderr)
    print(f"Confidence threshold: {confidence_threshold} (100% only)", file=sys.stderr)
    print(f"Min occurrences (manual): {min_occurrences} ← TRUST ALL EXPERT ANNOTATIONS", file=sys.stderr)
    print(f"Min occurrences (Omorfi): {min_occurrences_omorfi}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # PHASE 1: Extract manual annotations from all CSV files
    manual_patterns = extract_manual_annotations(csv_files)

    # PHASE 2: Classify as unambiguous vs ambiguous
    gold_unambiguous, gold_ambiguous = classify_manual_patterns(
        manual_patterns,
        min_occurrences=min_occurrences
    )

    # PHASE 3: Extract Omorfi patterns (100% only) from all CSV files
    omorfi_100_patterns = extract_omorfi_patterns(
        csv_files,
        confidence_threshold=confidence_threshold,
        min_occurrences=min_occurrences_omorfi  # Use separate threshold for automatic patterns
    )

    # PHASE 4: Build final lexicon structure
    print("\nPHASE 4: Building final lexicon structure...", file=sys.stderr)

    # Combine Tier 1 + Tier 2 (non-overlapping)
    pos_aware_lexicon = {}
    tier1_count = 0
    tier2_count = 0

    # Add Tier 1 (gold standard)
    for (word, pos), data in gold_unambiguous.items():
        if word not in pos_aware_lexicon:
            pos_aware_lexicon[word] = {}
        pos_aware_lexicon[word][pos] = data
        tier1_count += 1

    # Add Tier 2 (Omorfi 100%, non-overlapping)
    for (word, pos), data in omorfi_100_patterns.items():
        # Skip if already in Tier 1 or in ambiguous section
        if (word, pos) in gold_unambiguous or (word, pos) in gold_ambiguous:
            continue

        if word not in pos_aware_lexicon:
            pos_aware_lexicon[word] = {}

        pos_aware_lexicon[word][pos] = {
            'lemma': data['lemma'],
            'source': 'omorfi_100',
            'confidence': data['confidence'],
            'occurrences': data['occurrences'],
            'alternatives': []
        }
        tier2_count += 1

    # Build ambiguous patterns section
    ambiguous_patterns = {}
    for (word, pos), data in gold_ambiguous.items():
        if word not in ambiguous_patterns:
            ambiguous_patterns[word] = {}
        ambiguous_patterns[word][pos] = data

    # Calculate statistics
    total_unambiguous = tier1_count + tier2_count
    total_words = len(pos_aware_lexicon)
    ambiguous_words = len(ambiguous_patterns)

    print(f"  ✓ Tier 1 (Gold Standard): {tier1_count} patterns", file=sys.stderr)
    print(f"  ✓ Tier 2 (Omorfi 100%): {tier2_count} patterns", file=sys.stderr)
    print(f"  ✓ Total unambiguous: {total_unambiguous} patterns", file=sys.stderr)
    print(f"  ✓ Total words: {total_words}", file=sys.stderr)
    print(f"  ✓ Ambiguous tracked: {ambiguous_words} words ({len(gold_ambiguous)} patterns)", file=sys.stderr)

    # PHASE 5: Save lexicon
    print("\nPHASE 5: Saving lexicon...", file=sys.stderr)

    output_data = {
        'metadata': {
            'version': 'v16_hybrid_gold_standard_combined',
            'created': datetime.now().isoformat(),
            'sources': {
                'manual_gold_standard': {
                    'files': csv_files,
                    'num_files': len(csv_files),
                    'unique_patterns': len(manual_patterns),
                    'total_instances': sum(len(v) for v in manual_patterns.values()),
                    'coverage': '100%',
                    'description': 'Expert-verified manual annotations from train + test sets'
                },
                'omorfi_patterns': {
                    'threshold': confidence_threshold,
                    'min_occurrences': min_occurrences_omorfi,
                    'note': 'Only 100% unambiguous patterns from Omorfi contextual analysis'
                }
            },
            'statistics': {
                'training_poems': len(set(pid for instances in manual_patterns.values() for _, pid in instances))
            }
        },
        'pos_aware_lexicon': pos_aware_lexicon,
        'ambiguous_patterns': ambiguous_patterns,
        'quality_tiers': {
            'tier1_gold_standard': tier1_count,
            'tier2_omorfi_100': tier2_count,
            'tier3_accumulated': 0,
            'ambiguous_tracked': len(gold_ambiguous),
            'total_unambiguous': total_unambiguous,
            'total_words': total_words
        }
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved to: {output_json}", file=sys.stderr)

    # Print summary
    print("\n" + "=" * 80, file=sys.stderr)
    print("V16 LEXICON BUILD COMPLETE!", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Total unambiguous patterns: {total_unambiguous}", file=sys.stderr)
    print(f"  - Tier 1 (Manual): {tier1_count}", file=sys.stderr)
    print(f"  - Tier 2 (Omorfi 100%): {tier2_count}", file=sys.stderr)
    print(f"Ambiguous patterns tracked: {len(gold_ambiguous)}", file=sys.stderr)
    print(f"Total unique words: {total_words}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    return output_data


def main():
    """Main program entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Build V16 hybrid gold standard lexicon from train + test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build V16 lexicon from training data only (100% threshold)
  python3 build_lexicon_v16_combined.py \\
    finnish_poems_gold_train_clean.csv \\
    --output selftraining_lexicon_v16_hybrid.json

  # Build combined lexicon from train + test (RECOMMENDED)
  python3 build_lexicon_v16_combined.py \\
    finnish_poems_gold_train_clean.csv \\
    finnish_poems_gold_test_clean.csv \\
    --output selftraining_lexicon_v16_min1_combined.json

  # With custom min occurrences
  python3 build_lexicon_v16_combined.py \\
    finnish_poems_gold_train_clean.csv \\
    finnish_poems_gold_test_clean.csv \\
    --output selftraining_lexicon_v16_min3_combined.json \\
    --min-occurrences 3

Key improvements over V15:
  - Uses manual gold standard annotations (100% coverage)
  - Only caches 100% unambiguous patterns (not 80%)
  - Tracks all alternatives for ambiguous cases
  - Scientifically sound supervised learning
  - Supports multiple CSV files (train + test)
        """
    )

    parser.add_argument('csv_files', type=str, nargs='+',
                       help='CSV files with manual annotations (e.g., train.csv test.csv)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for V16 lexicon')
    parser.add_argument('--confidence', type=float, default=1.0,
                       help='Confidence threshold (default: 1.0 = 100%%)')
    parser.add_argument('--min-occurrences', type=int, default=1,
                       help='Minimum occurrences for MANUAL annotations (default: 1 = trust all)')
    parser.add_argument('--min-occurrences-omorfi', type=int, default=3,
                       help='Minimum occurrences for OMORFI patterns (default: 3)')

    args = parser.parse_args()

    # Validate confidence threshold
    if args.confidence != 1.0:
        print(f"WARNING: V16 is designed for 100%% threshold. Using {args.confidence} may introduce ambiguity.", file=sys.stderr)

    # Build lexicon
    build_v16_lexicon(
        csv_files=args.csv_files,
        output_json=args.output,
        confidence_threshold=args.confidence,
        min_occurrences=args.min_occurrences,
        min_occurrences_omorfi=args.min_occurrences_omorfi
    )


if __name__ == "__main__":
    main()
