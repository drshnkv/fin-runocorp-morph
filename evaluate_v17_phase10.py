#!/usr/bin/env python3
"""
========================================================
V17 PHASE 10 EVALUATION SCRIPT - Tests Finnish Dialects Dictionary Integration
========================================================

This script evaluates the V17 Phase 10 lemmatizer (with SMS dialectal dictionary)
against clean test data and compares performance with V17 Phase 9 baseline.

V17 PHASE 10 IMPROVEMENTS TESTED:
1. Finnish Dialects Dictionary (SMS) integration as Tier 6:
   - 19,385 validated dialectal variants from Suomen Murteiden Sanakirja
   - POS-aware filtering using Stanza context
   - High confidence (0.9) for validated dictionary entries
   - CRITICAL FIX: Moved to Tier 6 (before fuzzy matching) for optimal accuracy

2. Tier placement correction:
   - Dialectal dictionary now runs BEFORE fuzzy matching
   - Prevents fuzzy matches from catching dialectal words with wrong lemmas
   - Example fixes: lyhvetä → lyhetä (not hyvä), sknaapu → knaappi (not kapo)

3. Expected impact:
   - Target: +3-6 percentage points accuracy improvement (58.8% → 62-65%)
   - Expected: 150-250 dialectal variants correctly lemmatized per test set
   - Better precision (fewer fuzzy errors) AND better recall (more variants caught)

BASED ON V17 Phase 9: Morphology-Aware Fuzzy Matching (58.8% accuracy)

COMPARISON METRICS:
- Overall accuracy (V17 Phase 10 vs Phase 9)
- dialectal_dictionary usage and accuracy rate
- fuzzy_lexicon_morphological vs dialectal_dictionary comparison
- Identity_fallback reduction
- Method distribution changes

INPUT: Test data with manual annotations (answer key)
OUTPUT: Accuracy metrics + detailed comparison CSV with dialectal dictionary analysis
"""

import csv
import sys
from collections import defaultdict

# Import the Phase 10 lemmatizer
sys.path.insert(0, '.')
from fin_runocorp_base_v2_dialectal_dict_integrated import OmorfiHfstWithVoikkoV16Hybrid


def evaluate_v17_phase10(test_csv: str, results_csv: str):
    """
    Test V17 Phase 10 lemmatizer on unseen data and measure accuracy

    Evaluates:
    1. Overall accuracy vs manual gold standard
    2. Dialectal dictionary effectiveness (Tier 6)
    3. Comparison with fuzzy matching (would have been used without dictionary)
    4. Identity_fallback reduction
    5. Method distribution changes

    Parameters:
    - test_csv: File with test poems and manual annotations
    - results_csv: Where to save detailed comparison results

    Returns:
    - stats: Dictionary with accuracy numbers and method breakdown
    """

    # STEP 1: Load V17 Phase 10 lemmatizer
    print("=" * 80, file=sys.stderr)
    print("Loading V17 Phase 10 Lemmatizer (Finnish Dialects Dictionary)...", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    lemmatizer = OmorfiHfstWithVoikkoV16Hybrid()
    print("", file=sys.stderr)

    # Display lexicon statistics
    total_unambiguous = lemmatizer.quality_tiers.get('total_unambiguous', 0)
    tier1_count = lemmatizer.quality_tiers.get('tier1_gold_standard', 0)
    tier2_count = lemmatizer.quality_tiers.get('tier2_omorfi_100', 0)
    ambiguous_tracked = lemmatizer.quality_tiers.get('ambiguous_tracked', 0)

    print("V17 Phase 10 Lexicon Statistics:", file=sys.stderr)
    print(f"  Training lexicon patterns: {total_unambiguous}", file=sys.stderr)
    print(f"    - Tier 1 (Gold Standard): {tier1_count}", file=sys.stderr)
    print(f"    - Tier 2 (Omorfi 100%): {tier2_count}", file=sys.stderr)
    print(f"  Ambiguous patterns tracked: {ambiguous_tracked}", file=sys.stderr)
    print("", file=sys.stderr)

    # Display dialectal dictionary info
    dialectal_count = len(lemmatizer.dialectal_dict.get('variant_to_lemma', {}))
    print("V17 Phase 10 New Feature:", file=sys.stderr)
    print(f"  ✓ Finnish Dialects Dictionary (SMS): {dialectal_count:,} validated variants", file=sys.stderr)
    print(f"  ✓ Tier 6 placement (before fuzzy matching)", file=sys.stderr)
    print(f"  ✓ High confidence (0.9) for validated entries", file=sys.stderr)
    print(f"  ✓ Expected: 150-250 dialectal variants caught correctly", file=sys.stderr)
    print("", file=sys.stderr)

    # STEP 2: Set up tracking for results
    results = []
    stats = {
        'total': 0,
        'exact_match': 0,
        # V17-specific method tracking
        'v17_lexicon_tier1_match': 0,
        'v17_lexicon_tier1_pos_x_match': 0,
        'v17_lexicon_tier2_match': 0,
        'v17_lexicon_tier2_pos_x_match': 0,
        'v17_lexicon_tier3_match': 0,
        'v17_lexicon_tier3_pos_x_match': 0,
        'omorfi_contextual_match': 0,
        'voikko_omorfi_match': 0,
        'omorfi_direct_match': 0,
        'voikko_normalized_match': 0,
        'exception_lexicon_match': 0,
        'enhanced_voikko_match': 0,
        'dialectal_dictionary_match': 0,  # NEW: Phase 10 dialectal dictionary
        'fuzzy_lexicon_morphological_match': 0,  # Phase 9 morphological fuzzy
        'fuzzy_lexicon_aggressive_match': 0,  # Phase 8 aggressive fuzzy
        'identity_fallback_match': 0,
        'unknown': 0,
        'mismatch': 0,
        # Phase 10 specific tracking
        'dialectal_dictionary_used': 0,  # How many times dialectal_dictionary was used
        'dialectal_dictionary_correct': 0,  # How many times it was correct
        'identity_fallback_used': 0,
        'identity_fallback_correct': 0,
    }

    # Track dialectal_dictionary words for analysis
    dialectal_words = []

    # Track identity_fallback words
    identity_fallback_words = []

    # STEP 3: Read test data and organize by poem
    poem_groups = {}
    with open(test_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            poem_id = row.get('poem_id', '').strip()
            if poem_id not in poem_groups:
                poem_groups[poem_id] = []
            poem_groups[poem_id].append(row)

    print(f"Processing {len(poem_groups)} test poems...", file=sys.stderr)
    print("", file=sys.stderr)

    # STEP 4: Process each poem and compare results
    processed = 0
    for poem_idx, (poem_id, rows) in enumerate(poem_groups.items()):
        # Get all words from this poem
        tokens = [row.get('word', '').strip() for row in rows if row.get('word', '').strip()]

        # Get manually verified correct lemmas (answer key)
        manual_lemmas = [row.get('lemma_standard (Finnish/Estonian & minority languages if needed)', '').strip()
                        for row in rows if row.get('word', '').strip()]

        if not tokens:
            continue

        # Lemmatize entire poem with V17 Phase 10 system
        poem_results = lemmatizer.lemmatize_sentence(tokens)

        # Compare each prediction with correct answer
        for i, (token, manual_lemma, row) in enumerate(zip(tokens, manual_lemmas, [r for r in rows if r.get('word', '').strip()])):
            if not manual_lemma:
                continue

            stats['total'] += 1
            processed += 1

            # Get V17 Phase 10 prediction
            result = poem_results[i] if i < len(poem_results) else {'lemma': '', 'method': 'unknown'}
            predicted_lemma = result.get('lemma', '') or ''
            method = result.get('method', '')

            # Track dialectal_dictionary usage
            is_dialectal = (method == 'dialectal_dictionary')
            if is_dialectal:
                stats['dialectal_dictionary_used'] += 1

            # Track identity_fallback usage
            is_identity_fallback = (method == 'identity_fallback')
            if is_identity_fallback:
                stats['identity_fallback_used'] += 1

            # Check if prediction is correct
            is_exact_match = predicted_lemma.lower() == manual_lemma.lower() if predicted_lemma else False

            if is_exact_match:
                stats['exact_match'] += 1

                # Track dialectal_dictionary success
                if is_dialectal:
                    stats['dialectal_dictionary_correct'] += 1
                    dialectal_words.append({
                        'word': token,
                        'lemma': predicted_lemma,
                        'method': method,
                        'poem_id': poem_id,
                        'correct': 'YES'
                    })

                # Track identity_fallback success
                if is_identity_fallback:
                    stats['identity_fallback_correct'] += 1
                    identity_fallback_words.append({
                        'word': token,
                        'lemma': predicted_lemma,
                        'method': method,
                        'poem_id': poem_id,
                        'correct': 'YES'
                    })

                # Track which method successfully found correct lemma
                if method == 'v17_lexicon_tier1':
                    stats['v17_lexicon_tier1_match'] += 1
                elif method == 'v17_lexicon_tier1_pos_x':
                    stats['v17_lexicon_tier1_pos_x_match'] += 1
                elif method == 'v17_lexicon_tier2':
                    stats['v17_lexicon_tier2_match'] += 1
                elif method == 'v17_lexicon_tier2_pos_x':
                    stats['v17_lexicon_tier2_pos_x_match'] += 1
                elif method == 'v17_lexicon_tier3':
                    stats['v17_lexicon_tier3_match'] += 1
                elif method == 'v17_lexicon_tier3_pos_x':
                    stats['v17_lexicon_tier3_pos_x_match'] += 1
                elif method == 'omorfi_contextual':
                    stats['omorfi_contextual_match'] += 1
                elif method == 'voikko_omorfi':
                    stats['voikko_omorfi_match'] += 1
                elif method == 'omorfi_direct':
                    stats['omorfi_direct_match'] += 1
                elif method == 'voikko_normalized':
                    stats['voikko_normalized_match'] += 1
                elif method == 'exception_lexicon':
                    stats['exception_lexicon_match'] += 1
                elif method == 'enhanced_voikko':
                    stats['enhanced_voikko_match'] += 1
                elif method == 'dialectal_dictionary':
                    stats['dialectal_dictionary_match'] += 1
                elif method == 'fuzzy_lexicon_morphological':
                    stats['fuzzy_lexicon_morphological_match'] += 1
                elif method == 'fuzzy_lexicon_aggressive':
                    stats['fuzzy_lexicon_aggressive_match'] += 1
                elif method == 'identity_fallback':
                    stats['identity_fallback_match'] += 1
            else:
                # Wrong prediction
                if method == 'unknown':
                    stats['unknown'] += 1
                else:
                    stats['mismatch'] += 1

                # Track failed dialectal_dictionary
                if is_dialectal:
                    dialectal_words.append({
                        'word': token,
                        'lemma': predicted_lemma,
                        'manual_lemma': manual_lemma,
                        'method': method,
                        'poem_id': poem_id,
                        'correct': 'NO'
                    })

                # Track failed identity_fallback
                if is_identity_fallback:
                    identity_fallback_words.append({
                        'word': token,
                        'lemma': predicted_lemma,
                        'manual_lemma': manual_lemma,
                        'method': method,
                        'poem_id': poem_id,
                        'correct': 'NO'
                    })

            # Save detailed results
            result_entry = {
                'poem_id': poem_id,
                'word': token,
                'manual_lemma': manual_lemma,
                'predicted_lemma': predicted_lemma,
                'method': method,
                'exact_match': 'YES' if is_exact_match else 'NO',
                'is_dialectal': 'YES' if is_dialectal else 'NO',
                'is_identity_fallback': 'YES' if is_identity_fallback else 'NO',
                'verse': row.get('verse', ''),
                'place_name': row.get('place_name', ''),
                'year': row.get('year', ''),
            }
            results.append(result_entry)

        # Progress indicator
        if (poem_idx + 1) % 5 == 0:
            print(f"  Processed {poem_idx + 1}/{len(poem_groups)} poems ({processed} words)...", file=sys.stderr)

    print("", file=sys.stderr)

    # STEP 5: Save detailed results to CSV
    with open(results_csv, 'w', encoding='utf-8', newline='') as outfile:
        fieldnames = ['poem_id', 'word', 'manual_lemma', 'predicted_lemma',
                     'method', 'exact_match', 'is_dialectal', 'is_identity_fallback',
                     'verse', 'place_name', 'year']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # STEP 6: Save dialectal_dictionary analysis
    if dialectal_words:
        dialectal_csv = results_csv.replace('.csv', '_dialectal_dictionary_analysis.csv')
        with open(dialectal_csv, 'w', encoding='utf-8', newline='') as outfile:
            fieldnames = ['word', 'lemma', 'manual_lemma', 'method', 'poem_id', 'correct']
            filtered_words = []
            for w in dialectal_words:
                if w.get('correct') == 'YES':
                    filtered_words.append({
                        'word': w['word'],
                        'lemma': w['lemma'],
                        'manual_lemma': '',
                        'method': w['method'],
                        'poem_id': w['poem_id'],
                        'correct': w['correct']
                    })
                else:
                    filtered_words.append({
                        'word': w['word'],
                        'lemma': w['lemma'],
                        'manual_lemma': w.get('manual_lemma', ''),
                        'method': w['method'],
                        'poem_id': w['poem_id'],
                        'correct': w['correct']
                    })
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_words)
        stats['dialectal_csv'] = dialectal_csv

    # STEP 7: Save identity_fallback analysis
    if identity_fallback_words:
        identity_csv = results_csv.replace('.csv', '_identity_fallback_analysis.csv')
        with open(identity_csv, 'w', encoding='utf-8', newline='') as outfile:
            fieldnames = ['word', 'lemma', 'manual_lemma', 'method', 'poem_id', 'correct']
            filtered_words = []
            for w in identity_fallback_words:
                if w.get('correct') == 'YES':
                    filtered_words.append({
                        'word': w['word'],
                        'lemma': w['lemma'],
                        'manual_lemma': '',
                        'method': w['method'],
                        'poem_id': w['poem_id'],
                        'correct': w['correct']
                    })
                else:
                    filtered_words.append({
                        'word': w['word'],
                        'lemma': w['lemma'],
                        'manual_lemma': w.get('manual_lemma', ''),
                        'method': w['method'],
                        'poem_id': w['poem_id'],
                        'correct': w['correct']
                    })
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_words)
        stats['identity_fallback_csv'] = identity_csv

    # STEP 8: Calculate accuracy
    if stats['total'] > 0:
        stats['accuracy'] = stats['exact_match'] / stats['total'] * 100
    else:
        stats['accuracy'] = 0.0

    # Calculate dialectal_dictionary success rate
    if stats['dialectal_dictionary_used'] > 0:
        stats['dialectal_dictionary_success_rate'] = stats['dialectal_dictionary_correct'] / stats['dialectal_dictionary_used'] * 100
    else:
        stats['dialectal_dictionary_success_rate'] = 0.0

    # Calculate identity_fallback success rate
    if stats['identity_fallback_used'] > 0:
        stats['identity_fallback_success_rate'] = stats['identity_fallback_correct'] / stats['identity_fallback_used'] * 100
    else:
        stats['identity_fallback_success_rate'] = 0.0

    return stats


def main():
    """
    Main program - runs V17 Phase 10 evaluation and displays comprehensive results
    """

    # Specify input and output files
    test_csv = 'finnish_poems_gold_test_clean.csv'  # 24 test poems, ~1,468 words
    results_csv = 'finnish_lemma_evaluation_v17_phase10.csv'

    # Display header
    print()
    print("=" * 80)
    print("V17 PHASE 10 EVALUATION - Finnish Dialects Dictionary Integration")
    print("=" * 80)
    print()
    print("Testing V17 Phase 10 improvements over Phase 9:")
    print("  ✓ Finnish Dialects Dictionary (SMS) - 19,385 validated variants")
    print("  ✓ Tier 6 placement (before fuzzy matching) for optimal accuracy")
    print("  ✓ High confidence (0.9) for validated dictionary entries")
    print("  ✓ Target: +3-6pp accuracy improvement (58.8% → 62-65%)")
    print()
    print(f"Test data: {test_csv} (24 Finnish poems, ~1,468 words)")
    print("=" * 80)
    print()

    # Run evaluation
    stats = evaluate_v17_phase10(test_csv, results_csv)

    # Display results
    print("=" * 80)
    print("V17 PHASE 10 - EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"Total test words:    {stats['total']}")
    print(f"Exact matches:       {stats['exact_match']} ({stats['accuracy']:.1f}%)")
    print()

    # V17 Phase 10 specific results
    print("V17 Phase 10 Finnish Dialects Dictionary Statistics:")
    print(f"  Dialectal dictionary used:    {stats['dialectal_dictionary_used']} times")
    print(f"  Dialectal dictionary correct: {stats['dialectal_dictionary_correct']} ({stats['dialectal_dictionary_success_rate']:.1f}%)")
    if stats.get('dialectal_csv'):
        print(f"  ✓ Dialectal dictionary analysis saved to: {stats['dialectal_csv']}")
    print()

    print(f"  Identity fallback used:       {stats['identity_fallback_used']} times")
    print(f"  Identity fallback correct:    {stats['identity_fallback_correct']} ({stats['identity_fallback_success_rate']:.1f}%)")
    if stats.get('identity_fallback_csv'):
        print(f"  ✓ Identity fallback analysis saved to: {stats['identity_fallback_csv']}")
    print()

    # Tier-by-tier breakdown
    print("Lexicon Performance:")
    tier1_total = stats['v17_lexicon_tier1_match'] + stats['v17_lexicon_tier1_pos_x_match']
    tier2_total = stats['v17_lexicon_tier2_match'] + stats['v17_lexicon_tier2_pos_x_match']
    tier3_total = stats['v17_lexicon_tier3_match'] + stats['v17_lexicon_tier3_pos_x_match']

    print(f"  Tier 1 (Gold Standard):  {tier1_total} correct")
    print(f"  Tier 2 (Omorfi 100%):    {tier2_total} correct")
    if tier3_total > 0:
        print(f"  Tier 3 (Accumulated):    {tier3_total} correct")

    total_lexicon = tier1_total + tier2_total + tier3_total
    print(f"  Total lexicon hits:      {total_lexicon}")
    print()

    # Contextual fallback breakdown
    print("Contextual Fallback Performance:")
    print(f"  Omorfi contextual:          {stats['omorfi_contextual_match']} correct")
    print(f"  Voikko + Omorfi:            {stats['voikko_omorfi_match']} correct")
    print(f"  Omorfi direct:              {stats['omorfi_direct_match']} correct")
    print(f"  Voikko normalized:          {stats['voikko_normalized_match']} correct")
    if stats['exception_lexicon_match'] > 0:
        print(f"  Exception lexicon:          {stats['exception_lexicon_match']} correct")
    print(f"  Enhanced Voikko:            {stats['enhanced_voikko_match']} correct")
    print(f"  Dialectal Dictionary:       {stats['dialectal_dictionary_match']} correct (NEW - Phase 10)")
    print(f"  Fuzzy morphological:        {stats['fuzzy_lexicon_morphological_match']} correct")
    print(f"  Fuzzy aggressive:           {stats['fuzzy_lexicon_aggressive_match']} correct")
    print(f"  Identity fallback:          {stats['identity_fallback_match']} correct")
    print()

    # Errors
    print("Errors:")
    print(f"  Mismatches:          {stats['mismatch']} (wrong predictions)")
    print(f"  Unknown:             {stats['unknown']} (couldn't process)")
    print()
    print("=" * 80)
    print()
    print(f"✓ Overall accuracy: {stats['accuracy']:.1f}%")
    print(f"✓ Detailed results saved to: {results_csv}")
    print()

    # V17 Phase 9 comparison baseline
    print("=" * 80)
    print("COMPARISON WITH V17 PHASE 9 (BASELINE)")
    print("=" * 80)
    print()
    print("V17 Phase 9 baseline:")
    print("  Accuracy: 58.8%")
    print("  No dialectal dictionary")
    print("  fuzzy_lexicon_morphological: Used for dialectal variants (many errors)")
    print()
    print("V17 Phase 10 improvements:")
    print(f"  Accuracy: {stats['accuracy']:.1f}%")
    print(f"  Dialectal dictionary: {stats['dialectal_dictionary_used']} uses, {stats['dialectal_dictionary_success_rate']:.1f}% success rate")
    print(f"  Expected impact: 150-250 dialectal variants now correct")
    print()

    if stats['accuracy'] > 58.8:
        improvement = stats['accuracy'] - 58.8
        print(f"  ✓ Accuracy improvement over Phase 9: +{improvement:.1f} percentage points")

        # Calculate if we hit the target
        if improvement >= 3.0:
            print(f"  ✓ Target achieved! (Expected: +3-6pp, Actual: +{improvement:.1f}pp)")
        else:
            print(f"  ⚠ Below target (Expected: +3-6pp, Actual: +{improvement:.1f}pp)")
    else:
        decline = 58.8 - stats['accuracy']
        print(f"  ⚠ Accuracy vs Phase 9: -{decline:.1f} percentage points (investigate)")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
