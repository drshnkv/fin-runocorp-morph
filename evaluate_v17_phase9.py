#!/usr/bin/env python3
"""
========================================================
V17 PHASE 9 EVALUATION SCRIPT - Tests Morphology-Aware Fuzzy Matching
========================================================

This script evaluates the V17 Phase 9 lemmatizer against clean test data and compares
performance with V17 Phase 8 baseline.

V17 PHASE 9 IMPROVEMENTS TESTED:
1. Morphology-aware fuzzy lexicon matching (Omorfi UFEATS integration):
   - Universal Features (UFEATS) extraction via Analysis.get_ufeats()
   - Feature-aware candidate ranking (Case, Number, Tense agreement)
   - Morphological plausibility scoring for fuzzy matches
   - Feature similarity computation (bonuses: 0.2-0.3 per feature)
2. Enhanced fuzzy matching with morphological awareness:
   - Prefers candidates with matching morphological features
   - Adjusted distance = raw_distance - feature_similarity_bonus
   - Feature-based confidence scoring (medium-high/medium/medium-low)
3. Target: +2-3 percentage points accuracy improvement (58.5% → 60.5-61.5%)
BASED ON V17 Phase 8: Aggressive Fuzzy Matching (58.5% accuracy, 1 identity_fallback)

EXPECTED IMPACT:
- V17 Phase 8 baseline: 58.5% accuracy, 1 identity_fallback
- V17 Phase 9 target: 60.5-61.5% accuracy (+2-3pp)
- Expected improvement: Better fuzzy match selection via morphological plausibility

COMPARISON METRICS:
- Overall accuracy (V17 Phase 9 vs Phase 8)
- fuzzy_lexicon_morphological vs fuzzy_lexicon comparison
- Morphological feature bonus impact analysis
- Confidence distribution analysis
- Method distribution changes

WHY WE NEED THIS:
V17 Phase 8 achieved 98% identity_fallback reduction but fuzzy matching still has ~70% error rate.
Root cause: Phonetic similarity doesn't guarantee morphological plausibility.
Solution: Use Omorfi UFEATS to rank candidates by morphological agreement with dialectal forms.

INPUT: Test data with manual annotations (answer key)
OUTPUT: Accuracy metrics + detailed comparison CSV with morphological feature analysis
"""

import csv
import sys
from collections import defaultdict
from fin_runocorp_base import OmorfiHfstWithVoikkoV16Hybrid


def evaluate_v17_phase9(test_csv: str, results_csv: str):
    """
    Test V17 Phase 7 lemmatizer on unseen data and measure accuracy

    Evaluates:
    1. Overall accuracy vs manual gold standard
    2. Identity_fallback reduction vs V17 Phase 6 baseline (289 words)
    3. Fuzzy lexicon effectiveness (weighted edit distance matching)
    4. False positive rate in fuzzy matches
    5. Comparison with V17 Phase 6 baseline

    Parameters:
    - test_csv: File with test poems and manual annotations
    - results_csv: Where to save detailed comparison results

    Returns:
    - stats: Dictionary with accuracy numbers and method breakdown
    """

    # STEP 1: Load V17 Phase 7 lemmatizer
    print("=" * 80, file=sys.stderr)
    print("Loading V17 Phase 8 Lemmatizer (Fuzzy Lexicon Matching)...", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    lemmatizer = OmorfiHfstWithVoikkoV16Hybrid()
    print("", file=sys.stderr)

    # Display lexicon statistics
    total_unambiguous = lemmatizer.quality_tiers.get('total_unambiguous', 0)
    tier1_count = lemmatizer.quality_tiers.get('tier1_gold_standard', 0)
    tier2_count = lemmatizer.quality_tiers.get('tier2_omorfi_100', 0)
    ambiguous_tracked = lemmatizer.quality_tiers.get('ambiguous_tracked', 0)

    print("V17 Phase 7 Lexicon Statistics (same as Phase 4):", file=sys.stderr)
    print(f"  Total unambiguous patterns: {total_unambiguous}", file=sys.stderr)
    print(f"    - Tier 1 (Gold Standard): {tier1_count}", file=sys.stderr)
    print(f"    - Tier 2 (Omorfi 100%): {tier2_count}", file=sys.stderr)
    print(f"  Ambiguous patterns tracked: {ambiguous_tracked}", file=sys.stderr)
    print("", file=sys.stderr)
    print("V17 Phase 7 New Features:", file=sys.stderr)
    print("  ✓ Fuzzy lexicon lookup - NEW tier before identity_fallback", file=sys.stderr)
    print("  ✓ Weighted edit distance (threshold 2.0) with dialectal awareness", file=sys.stderr)
    print("  ✓ POS-aware matching against all lexicon entries (Tier 1-3)", file=sys.stderr)
    print("  ✓ Target: Reduce identity_fallback from 289 to ~200 words (30% reduction)", file=sys.stderr)
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
        'voikko_omorfi_match': 0,  # NEW: Voikko-assisted Omorfi (contextual)
        'omorfi_direct_match': 0,
        'voikko_normalized_match': 0,
        'exception_lexicon_match': 0,
        'enhanced_voikko_match': 0,  # Phase 4: aggressive Voikko tier
        'fuzzy_lexicon_match': 0,  # NEW: Phase 8 aggressive fuzzy lexicon matching
        'identity_fallback_match': 0,  # Track identity_fallback usage
        'unknown': 0,
        'mismatch': 0,
        # Ambiguous case tracking
        'ambiguous_contextual_correct': 0,
        'ambiguous_contextual_wrong': 0,
        # V17 Phase 7 specific tracking
        'identity_fallback_used': 0,  # How many times identity_fallback was used
        'identity_fallback_correct': 0,  # How many times it was correct
        'enhanced_voikko_used': 0,  # How many times enhanced_voikko was used
        'enhanced_voikko_correct': 0,  # How many times it was correct
        'fuzzy_lexicon_used': 0,  # NEW: How many times fuzzy_lexicon was used
        'fuzzy_lexicon_correct': 0,  # NEW: How many times it was correct
    }

    # Track ambiguous words encountered
    ambiguous_words_seen = defaultdict(int)
    ambiguous_results = []

    # Track identity_fallback words for analysis
    identity_fallback_words = []

    # Track enhanced_voikko words for analysis
    enhanced_voikko_words = []

    # Track fuzzy_lexicon words for analysis
    fuzzy_lexicon_words = []

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

        # Lemmatize entire poem with V17 Phase 6 system
        poem_results = lemmatizer.lemmatize_sentence(tokens)

        # Compare each prediction with correct answer
        for i, (token, manual_lemma, row) in enumerate(zip(tokens, manual_lemmas, [r for r in rows if r.get('word', '').strip()])):
            if not manual_lemma:
                continue

            stats['total'] += 1
            processed += 1

            # Get V17 Phase 6 prediction
            result = poem_results[i] if i < len(poem_results) else {'lemma': '', 'method': 'unknown'}
            predicted_lemma = result.get('lemma', '') or ''
            method = result.get('method', '')

            # Track identity_fallback usage
            is_identity_fallback = (method == 'identity_fallback')
            if is_identity_fallback:
                stats['identity_fallback_used'] += 1

            # Track enhanced_voikko usage
            is_enhanced_voikko = (method == 'enhanced_voikko')
            if is_enhanced_voikko:
                stats['enhanced_voikko_used'] += 1

            # Check if POS='X' fallback was used
            is_pos_x_fallback = 'pos_x' in method

            # Check if this word is in ambiguous_patterns
            token_lower = token.lower()
            is_ambiguous = token_lower in lemmatizer.ambiguous_patterns

            if is_ambiguous:
                ambiguous_words_seen[token_lower] += 1

            # Check if prediction is correct
            is_exact_match = predicted_lemma.lower() == manual_lemma.lower() if predicted_lemma else False

            if is_exact_match:
                stats['exact_match'] += 1

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

                # Track enhanced_voikko success
                if is_enhanced_voikko:
                    stats['enhanced_voikko_correct'] += 1
                    enhanced_voikko_words.append({
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
                    if is_ambiguous:
                        stats['ambiguous_contextual_correct'] += 1
                elif method == 'voikko_omorfi':  # NEW: Voikko-assisted contextual
                    stats['voikko_omorfi_match'] += 1
                    if is_ambiguous:
                        stats['ambiguous_contextual_correct'] += 1
                elif method == 'omorfi_direct':
                    stats['omorfi_direct_match'] += 1
                elif method == 'voikko_normalized':
                    stats['voikko_normalized_match'] += 1
                elif method == 'exception_lexicon':
                    stats['exception_lexicon_match'] += 1
                elif method == 'enhanced_voikko':
                    stats['enhanced_voikko_match'] += 1
                elif method == 'identity_fallback':
                    stats['identity_fallback_match'] += 1
            else:
                # Wrong prediction
                if method == 'unknown':
                    stats['unknown'] += 1
                else:
                    stats['mismatch'] += 1
                    if is_ambiguous and method in ['omorfi_contextual', 'voikko_omorfi']:
                        stats['ambiguous_contextual_wrong'] += 1

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

                # Track failed enhanced_voikko
                if is_enhanced_voikko:
                    enhanced_voikko_words.append({
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
                'is_ambiguous': 'YES' if is_ambiguous else 'NO',
                'is_identity_fallback': 'YES' if is_identity_fallback else 'NO',
                'is_enhanced_voikko': 'YES' if is_enhanced_voikko else 'NO',
                'verse': row.get('verse', ''),
                'place_name': row.get('place_name', ''),
                'year': row.get('year', ''),
                'context_score': result.get('context_score', 0.0)
            }
            results.append(result_entry)

            # Track ambiguous case results separately
            if is_ambiguous:
                ambiguous_results.append(result_entry)

        # Progress indicator
        if (poem_idx + 1) % 5 == 0:
            print(f"  Processed {poem_idx + 1}/{len(poem_groups)} poems ({processed} words)...", file=sys.stderr)

    print("", file=sys.stderr)

    # STEP 5: Save detailed results to CSV
    with open(results_csv, 'w', encoding='utf-8', newline='') as outfile:
        fieldnames = ['poem_id', 'word', 'manual_lemma', 'predicted_lemma',
                     'method', 'exact_match', 'is_ambiguous', 'is_identity_fallback', 'is_enhanced_voikko',
                     'verse', 'place_name', 'year', 'context_score']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # STEP 6: Save ambiguous case analysis
    if ambiguous_results:
        ambiguous_csv = results_csv.replace('.csv', '_ambiguous_analysis.csv')
        with open(ambiguous_csv, 'w', encoding='utf-8', newline='') as outfile:
            fieldnames = ['poem_id', 'word', 'manual_lemma', 'predicted_lemma',
                         'method', 'exact_match', 'is_ambiguous', 'is_identity_fallback', 'is_enhanced_voikko',
                         'verse', 'place_name', 'year', 'context_score']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ambiguous_results)
        stats['ambiguous_csv'] = ambiguous_csv
        stats['ambiguous_words_count'] = len(ambiguous_words_seen)
        stats['ambiguous_instances'] = sum(ambiguous_words_seen.values())

    # STEP 7: Save identity_fallback analysis
    if identity_fallback_words:
        identity_csv = results_csv.replace('.csv', '_identity_fallback_analysis.csv')
        with open(identity_csv, 'w', encoding='utf-8', newline='') as outfile:
            fieldnames = ['word', 'lemma', 'manual_lemma', 'method', 'poem_id', 'correct']
            # Filter out None values
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

    # STEP 7.5: Save flipped selections analysis
    flipped_selections = lemmatizer.morph_stats.get('flipped_selections', [])
    if flipped_selections:
        flipped_csv = results_csv.replace('.csv', '_flipped_selections.csv')
        with open(flipped_csv, 'w', encoding='utf-8', newline='') as outfile:
            fieldnames = ['word', 'lemma_without_features', 'lemma_with_features',
                         'score_without', 'score_with']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flipped_selections)
        print(f"  ✓ Flipped selections saved to: {flipped_csv}")

    # STEP 8: Save enhanced_voikko analysis
    if enhanced_voikko_words:
        enhanced_voikko_csv = results_csv.replace('.csv', '_enhanced_voikko_analysis.csv')
        with open(enhanced_voikko_csv, 'w', encoding='utf-8', newline='') as outfile:
            fieldnames = ['word', 'lemma', 'manual_lemma', 'method', 'poem_id', 'correct']
            # Filter out None values
            filtered_words = []
            for w in enhanced_voikko_words:
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
        stats['enhanced_voikko_csv'] = enhanced_voikko_csv

    # STEP 9: Calculate accuracy
    if stats['total'] > 0:
        stats['accuracy'] = stats['exact_match'] / stats['total'] * 100
    else:
        stats['accuracy'] = 0.0

    # Calculate identity_fallback success rate
    if stats['identity_fallback_used'] > 0:
        stats['identity_fallback_success_rate'] = stats['identity_fallback_correct'] / stats['identity_fallback_used'] * 100
    else:
        stats['identity_fallback_success_rate'] = 0.0

    # Calculate enhanced_voikko success rate
    if stats['enhanced_voikko_used'] > 0:
        stats['enhanced_voikko_success_rate'] = stats['enhanced_voikko_correct'] / stats['enhanced_voikko_used'] * 100
    else:
        stats['enhanced_voikko_success_rate'] = 0.0

    return stats, lemmatizer  # Return lemmatizer for morph_stats access


def main():
    """
    Main program - runs V17 Phase 9 evaluation and displays comprehensive results
    """

    # Specify input and output files
    test_csv = 'finnish_poems_gold_test_clean.csv'  # 24 test poems, ~1,468 words
    results_csv = 'finnish_lemma_evaluation_v17_phase9.csv'

    # Display header
    print()
    print("=" * 80)
    print("V17 PHASE 9 EVALUATION - Morphology-Aware Fuzzy Matching")
    print("=" * 80)
    print()
    print("Testing V17 Phase 9 improvements over Phase 8:")
    print("  ✓ Omorfi Universal Features (UFEATS) extraction and integration")
    print("  ✓ Feature-aware fuzzy candidate ranking (Case, Number, Tense agreement)")
    print("  ✓ Morphological plausibility scoring for fuzzy matches")
    print("  ✓ Target: +2-3pp accuracy improvement (58.5% → 60.5-61.5%)")
    print()
    print(f"Test data: {test_csv} (24 Finnish poems, ~1,468 words)")
    print("=" * 80)
    print()

    # Run evaluation
    stats, lemmatizer = evaluate_v17_phase9(test_csv, results_csv)

    # Display results
    print("=" * 80)
    print("V17 PHASE 9 - EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"Total test words:    {stats['total']}")
    print(f"Exact matches:       {stats['exact_match']} ({stats['accuracy']:.1f}%)")
    print()

    # V17 Phase 9 specific results - REFACTORED
    print("V17 Phase 9 Morphological Feature Statistics (Tier 1 - Refactored):")

    # Extract morphological stats from lemmatizer
    morph_stats = lemmatizer.morph_stats

    print(f"  Words with multiple Omorfi analyses:  {morph_stats['words_with_multiple_analyses']}")
    print(f"  Features extracted successfully:       {morph_stats['features_extracted_successfully']}")
    print(f"  Feature similarity computed:           {morph_stats['feature_similarity_computed']} times")
    print(f"  Feature bonus applied (bonus > 0):     {morph_stats['feature_bonus_applied']} times")
    print(f"  Selection CHANGED by features:         {morph_stats['selection_changed_by_features']} times")

    if morph_stats['feature_bonus_applied'] > 0:
        avg_bonus = morph_stats['total_feature_bonus'] / morph_stats['feature_bonus_applied']
        print(f"  Average feature bonus:                 {avg_bonus:.3f}")
        print(f"  Maximum feature bonus:                 {morph_stats['max_feature_bonus']:.3f}")

        # Calculate impact percentage
        if morph_stats['words_with_multiple_analyses'] > 0:
            change_rate = (morph_stats['selection_changed_by_features'] / morph_stats['words_with_multiple_analyses']) * 100
            print(f"  Impact rate:                           {change_rate:.1f}% (changed selections / words with multiple analyses)")

    # V17 Option B tracking
    if 'option_b_stanza_used' in morph_stats:
        print(f"  Option B - Stanza used in Tier 3:      {morph_stats['option_b_stanza_used']} times")

    print()
    print("V17 Phase 9 Legacy Fuzzy Tier (deprecated - moved to Tier 1):")
    print(f"  Morphology-aware fuzzy used: {stats.get('fuzzy_lexicon_morphological', 0)} times")
    print()
    print(f"  Identity fallback used:    {stats['identity_fallback_used']} times")
    print(f"  Identity fallback correct: {stats['identity_fallback_correct']} ({stats['identity_fallback_success_rate']:.1f}%)")
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
    print(f"  Omorfi contextual:       {stats['omorfi_contextual_match']} correct")
    print(f"  Voikko + Omorfi:         {stats['voikko_omorfi_match']} correct (Voikko-assisted)")
    print(f"  Omorfi direct:           {stats['omorfi_direct_match']} correct")
    print(f"  Voikko normalized:       {stats['voikko_normalized_match']} correct")
    if stats['exception_lexicon_match'] > 0:
        print(f"  Exception lexicon:       {stats['exception_lexicon_match']} correct")
    print(f"  Enhanced Voikko:         {stats['enhanced_voikko_match']} correct")
    print(f"  Fuzzy lexicon:           {stats.get('fuzzy_lexicon_match', 0)} correct (Phase 6)")
    print(f"  Identity fallback:       {stats['identity_fallback_match']} correct")
    print()

    # Ambiguous case analysis
    if stats.get('ambiguous_instances', 0) > 0:
        print("Ambiguous Pattern Handling:")
        print(f"  Ambiguous words encountered: {stats.get('ambiguous_words_count', 0)} unique")
        print(f"  Ambiguous instances:         {stats.get('ambiguous_instances', 0)} total")
        print(f"  Contextual correct:          {stats['ambiguous_contextual_correct']}")
        print(f"  Contextual wrong:            {stats['ambiguous_contextual_wrong']}")
        if (stats['ambiguous_contextual_correct'] + stats['ambiguous_contextual_wrong']) > 0:
            ambig_accuracy = stats['ambiguous_contextual_correct'] / (stats['ambiguous_contextual_correct'] + stats['ambiguous_contextual_wrong']) * 100
            print(f"  Ambiguous accuracy:          {ambig_accuracy:.1f}%")
        print()
        print(f"  ✓ Ambiguous analysis saved to: {stats.get('ambiguous_csv', 'N/A')}")
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

    # V17 Phase 6 comparison baseline
    print("=" * 80)
    print("COMPARISON WITH V17 PHASE 6")
    print("=" * 80)
    print()
    print("V17 Phase 6 baseline:")
    print("  Accuracy: 58.3%")
    print("  voikko_omorfi errors: 197/325 uses (61% error rate)")
    print("  voikko_omorfi correct: 128 (39% success rate)")
    print()
    print("V17 Phase 7 improvements:")
    print(f"  Voikko ranking applied: Multi-criteria selection (edit distance, POS, frequency)")
    print(f"  voikko_omorfi correct: {stats['voikko_omorfi_match']}")
    print()
    phase6_voikko_correct = 128  # From Phase 6 results
    if stats['voikko_omorfi_match'] > phase6_voikko_correct:
        improvement_count = stats['voikko_omorfi_match'] - phase6_voikko_correct
        print(f"  ✓ voikko_omorfi improvement: +{improvement_count} correct ({phase6_voikko_correct} → {stats['voikko_omorfi_match']})")
    else:
        decline_count = phase6_voikko_correct - stats['voikko_omorfi_match']
        print(f"  ⚠ voikko_omorfi declined: -{decline_count} correct (investigate)")
    print()
    if stats['accuracy'] > 58.3:
        improvement = stats['accuracy'] - 58.3
        print(f"  ✓ Accuracy improvement over Phase 6: +{improvement:.1f} percentage points")
    else:
        decline = 58.3 - stats['accuracy']
        print(f"  ⚠ Accuracy vs Phase 6: -{decline:.1f} percentage points (investigate)")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
