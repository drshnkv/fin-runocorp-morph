#!/usr/bin/env python3
"""
Finnish dialectal lemmatizer - Pipeline orchestration module.

This module provides the high-level FinnishLemmatizer class that orchestrates
the 9-tier lemmatization pipeline.
"""

import sys
import re
from typing import List, Dict, Tuple, Optional, Any

from lemmatizer_config import LemmatizerConfig, VERSION
from lemmatizer_core import LemmatizerCore

# Optional imports
try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False


class FinnishLemmatizer:
    """
    Finnish lemmatizer with 9-tier hybrid pipeline

    Pipeline Architecture:
    =====================
    Tier 1: V16 Lexicon (Gold Standard)
           - Manual annotations (100% trusted)
           - Omorfi 100% unambiguous (non-overlapping)
           - Future production accumulation

    Tier 2: Omorfi Contextual Analysis
           - Direct Omorfi analysis with smart alternative selection
           - Highest confidence for known words

    Tier 3: Voikko + Omorfi
           - Spelling normalization via Voikko suggestions (10 max)
           - Re-analysis with Omorfi

    Tier 4: Omorfi Guesser
           - Guesser analysis for unknown words
           - Medium confidence

    Tier 5: Fuzzy Lexicon (Morphological)
           - Morphology-aware fuzzy matching
           - UFEATS feature similarity ranking
           - Conservative threshold (2.0)

    Tier 6: Fuzzy Lexicon (Aggressive)
           - Aggressive fuzzy matching with suffix stripping
           - Relaxed threshold (3.5)
           - Character normalization

    Tier 7: Finnish Dialects Dictionary (SMS)
           - 19,385 validated dialectal variants
           - High confidence for exact matches

    Tier 8: Enhanced Voikko (Aggressive)
           - Up to 30 Voikko suggestions
           - More aggressive than Tier 3

    Tier 9: Identity Fallback
           - Returns word.lower() as lemma
           - Helps with proper nouns and nominative forms

    Version: V17 Phase 11
    """

    def __init__(self, config: Optional[LemmatizerConfig] = None,
                 model_dir: Optional[str] = None,
                 voikko_path: Optional[str] = None,
                 lang: str = 'fi') -> None:
        """Initialize the Finnish lemmatizer

        Args:
            config: LemmatizerConfig instance (if None, creates from other params)
            model_dir: DEPRECATED - use config instead
            voikko_path: DEPRECATED - use config instead
            lang: DEPRECATED - use config instead
        """
        # Handle backward compatibility
        if config is None:
            config = LemmatizerConfig(
                model_dir=model_dir,
                voikko_path=voikko_path,
                lang=lang
            )

        # Initialize the core lemmatizer
        self.core = LemmatizerCore(config)

        # Expose core attributes for backward compatibility
        self.config = self.core.config
        self.model_dir = self.core.model_dir
        self.lang = self.core.lang
        self.describe_path = self.core.describe_path
        self.analyzer = self.core.analyzer
        self.guesser = self.core.guesser
        self.guesser_effective = self.core.guesser_effective
        self.voikko_path = self.core.voikko_path
        self.voikko = self.core.voikko
        self.voikko_sukija = self.core.voikko_sukija
        self.voikko_old_finnish = self.core.voikko_old_finnish
        self.stanza_nlp = self.core.stanza_nlp
        self.pos_aware_lexicon = self.core.pos_aware_lexicon
        self.ambiguous_patterns = self.core.ambiguous_patterns
        self.quality_tiers = self.core.quality_tiers
        self.dialectal_normalizer = self.core.dialectal_normalizer
        self.dialectal_dict = self.core.dialectal_dict
        self.feature_extractor = self.core.feature_extractor
        self.similarity_scorer = self.core.similarity_scorer
        self.morph_stats = self.core.morph_stats

    def lemmatize(self, word: str) -> Dict[str, Any]:
        """
        Complete 9-tier lemmatization pipeline with smart alternative selection

        Pipeline Execution Order:
        ========================
        1. Tier 1: V16 Lexicon lookup (gold standard)
        2. Tier 2: Omorfi contextual analysis (direct + smart alternatives)
        3. Tier 3: Voikko normalization + Omorfi re-analysis (10 suggestions)
        4. Tier 4: Omorfi guesser (unknown word analysis)
        5. Tier 5: Morphology-aware fuzzy lexicon (UFEATS matching, threshold 2.0)
        6. Tier 6: Aggressive fuzzy lexicon (threshold 3.5, suffix stripping)
        7. Tier 7: Finnish Dialects Dictionary (SMS - 19,385 variants)
        8. Tier 8: Enhanced Voikko (30 suggestions, aggressive mode)
        9. Tier 9: Identity fallback (word.lower())

        Args:
            word: Word to lemmatize

        Returns:
            Dictionary with keys:
                - word: Original word
                - lemma: Lemmatized form (or None if failed)
                - analysis: Omorfi analysis string (or None)
                - weight: Omorfi weight (or None)
                - method: Processing method used
                - confidence: Confidence level (high/medium/low)
                - alternatives: List of alternative analyses (if any)
        """
        # Tier 1: V16 Lexicon (Gold Standard)
        result = self.tier1_v16_lexicon(word)
        if result:
            return result

        # Tier 2: Omorfi Contextual Analysis
        result = self.tier2_omorfi_contextual(word)
        if result:
            return result

        # Tier 3: Voikko + Omorfi
        result = self.tier3_voikko_omorfi(word)
        if result:
            return result

        # Tier 4: Omorfi Guesser
        result = self.tier4_omorfi_guesser(word)
        if result:
            return result

        # Tier 5: Morphology-Aware Fuzzy Lexicon
        result = self.tier5_fuzzy_lexicon_morphological(word)
        if result:
            return result

        # Tier 6: Aggressive Fuzzy Lexicon
        result = self.tier6_fuzzy_lexicon_aggressive(word)
        if result:
            return result

        # Tier 7: Finnish Dialects Dictionary (SMS)
        result = self.tier7_dialectal_dictionary(word)
        if result:
            return result

        # Tier 8: Enhanced Voikko (Aggressive)
        result = self.tier8_voikko_aggressive(word)
        if result:
            return result

        # Tier 9: Identity Fallback
        return self.tier9_identity_fallback(word)

    def lemmatize_sentence(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Lemmatize a sentence with V16 hybrid lexicon + contextual disambiguation

        Priority order:
        1. Check if (word, POS) in ambiguous_patterns → skip lexicon, use contextual
        2. Check pos_aware_lexicon (Tier 1 + Tier 2) → return cached lemma
        3. Fall back to Omorfi contextual analysis

        NEW IN V16: Ambiguous pattern detection prevents incorrect caching

        This method provides sentence-level context via Stanza for better
        POS tagging and morphological feature extraction, enabling more
        accurate lemmatization than word-by-word processing.

        Args:
            tokens: List of word tokens in the sentence

        Returns:
            List of dictionaries with lemmatization results for each token
        """
        if not self.core.stanza_nlp or not tokens:
            # Fallback to word-level without context
            return [self.lemmatize(token) for token in tokens]

        try:
            # Get Stanza analysis for context
            doc = self.core.stanza_nlp([tokens])  # Pre-tokenized
            stanza_words = [w for sent in doc.sentences for w in sent.words]

            if len(stanza_words) != len(tokens):
                # Tokenization mismatch, fallback
                return [self.lemmatize(token) for token in tokens]

            results = []
            for i, token in enumerate(tokens):
                # Get Stanza context
                sw = stanza_words[i]
                stanza_upos = sw.upos
                stanza_feats = sw.feats or ""

                # V17 Phase 1 Fix #1: Normalize archaic long s (ſ → s)
                token_lower = token.lower().replace('ſ', 's')

                # V16 CRITICAL: Check if this (word, POS) is genuinely ambiguous
                # If yes, skip lexicon and use contextual analysis
                if token_lower in self.core.ambiguous_patterns:
                    pos_ambig_dict = self.core.ambiguous_patterns[token_lower]
                    if stanza_upos in pos_ambig_dict:
                        # This (word, POS) has multiple valid lemmas
                        # Fall through to contextual analysis below
                        pass
                    else:
                        # Word is ambiguous in general, but not for this POS
                        # Check unambiguous lexicon
                        if token_lower in self.core.pos_aware_lexicon:
                            pos_dict = self.core.pos_aware_lexicon[token_lower]
                            if stanza_upos in pos_dict:
                                entry = pos_dict[stanza_upos]
                                lemma = entry['lemma']
                                source = entry.get('source', 'unknown')
                                tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                                results.append({
                                    'word': token,
                                    'lemma': lemma,
                                    'analysis': f'[V16_LEXICON:{token_lower}+{stanza_upos}→{lemma}][SOURCE:{source}][TIER:{tier}]',
                                    'method': f'v16_lexicon_{tier}',
                                    'confidence': 'high',
                                    'context_score': 1.0
                                })
                                continue
                            # V17 Phase 1 Fix #2: POS='X' fallback for unclassified POS in training
                            elif 'X' in pos_dict:
                                entry = pos_dict['X']
                                lemma = entry['lemma']
                                source = entry.get('source', 'unknown')
                                tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                                results.append({
                                    'word': token,
                                    'lemma': lemma,
                                    'analysis': f'[V17_LEXICON_POS_X:{token_lower}+X→{lemma}][SOURCE:{source}][TIER:{tier}]',
                                    'method': f'v17_lexicon_{tier}_pos_x',
                                    'confidence': 'high',
                                    'context_score': 1.0
                                })
                                continue
                else:
                    # Not in ambiguous section, check unambiguous lexicon
                    if token_lower in self.core.pos_aware_lexicon:
                        pos_dict = self.core.pos_aware_lexicon[token_lower]
                        if stanza_upos in pos_dict:
                            entry = pos_dict[stanza_upos]
                            lemma = entry['lemma']
                            source = entry.get('source', 'unknown')
                            tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                            results.append({
                                'word': token,
                                'lemma': lemma,
                                'analysis': f'[V16_LEXICON:{token_lower}+{stanza_upos}→{lemma}][SOURCE:{source}][TIER:{tier}]',
                                'method': f'v16_lexicon_{tier}',
                                'confidence': 'high',
                                'context_score': 1.0
                            })
                            continue
                        # V17 Phase 1 Fix #2: POS='X' fallback for unclassified POS in training
                        elif 'X' in pos_dict:
                            entry = pos_dict['X']
                            lemma = entry['lemma']
                            source = entry.get('source', 'unknown')
                            tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                            results.append({
                                'word': token,
                                'lemma': lemma,
                                'analysis': f'[V17_LEXICON_POS_X:{token_lower}+X→{lemma}][SOURCE:{source}][TIER:{tier}]',
                                'method': f'v17_lexicon_{tier}_pos_x',
                                'confidence': 'high',
                                'context_score': 1.0
                            })
                            continue

                # If we get here: either ambiguous or not in lexicon
                # Use V14-style contextual analysis

                # Get Omorfi analyses
                analyses = self.core.analyze_direct(token)
                voikko_used = False  # Track if Voikko provided suggestions

                if not analyses:
                    # V17 Phase 7: Voikko with multi-criteria ranking
                    # Rank suggestions before trying them (not just first-match)
                    voikko_suggestions = self.core.voikko_suggest(token, max_n=20)
                    if voikko_suggestions:
                        # Rank suggestions by edit distance, POS match, frequency, etc.
                        ranked_suggestions = self.core.rank_voikko_suggestions(token, voikko_suggestions, stanza_upos)
                        # Try ranked suggestions in order (best first)
                        for suggestion, _ in ranked_suggestions:
                            analyses = self.core.analyze_direct(suggestion)
                            if analyses:
                                voikko_used = True  # Mark that Voikko was used
                                break

                if analyses:
                    # V17 Phase 9 TRACKING: Count words with multiple Omorfi analyses
                    if len(analyses) > 1:
                        self.core.morph_stats['words_with_multiple_analyses'] += 1

                    # Score with context (V17 Phase 9: now uses rich morphological features!)
                    best_score = -1000.0
                    best_result = None
                    scored_candidates = []  # Track all scored candidates for comparison

                    for analysis, weight in analyses[:5]:
                        score = self.core.score_candidate_contextual(
                            token, analysis, weight,
                            stanza_upos, stanza_feats
                        )

                        # Extract lemma (may be compound first component only)
                        lemma = self.core.normalize_lemma(self.core.extract_lemma(analysis) or '')

                        # V17 Phase 12: Check for compound and reconstruct if needed
                        if self.core.compound_classifier and self.core.has_compound_boundary(analysis):
                            classification = self.core.compound_classifier.classify_compound(
                                analysis,
                                original_word=token
                            )
                            if classification['is_true_compound'] and classification['reconstructed']:
                                lemma = classification['reconstructed']  # Use reconstructed compound

                        scored_candidates.append((lemma, score, analysis))

                        if score > best_score:
                            best_score = score
                            # Use different method name if Voikko assisted
                            method_name = 'voikko_omorfi' if voikko_used else 'omorfi_contextual'
                            best_result = {
                                'word': token,
                                'lemma': lemma,  # Now contains reconstructed compound if applicable
                                'analysis': analysis,
                                'method': method_name,
                                'confidence': 'medium-high' if voikko_used else 'high',
                                'context_score': score
                            }

                    # V17 Phase 9 TRACKING: Check if morphological features changed selection
                    # (This would require scoring WITHOUT features too, which we skip for now)
                    # We already track feature_bonus_applied in score_candidate_contextual()

                    results.append(best_result or self.lemmatize(token))

                    # V17 Phase 10: Override voikko_omorfi (spelling guesses) with dictionary if available
                    if results and results[-1] and results[-1].get('method') == 'voikko_omorfi':
                        dict_result = self.core.lemmatize_with_dialectal_dict(token, stanza_upos)
                        if dict_result and dict_result['lemma'] != results[-1]['lemma']:
                            # Dictionary has different lemma - use it instead of spelling guess
                            original_lemma = results[-1]['lemma']
                            results[-1] = dict_result
                            results[-1]['overridden_method'] = 'voikko_omorfi'
                            results[-1]['original_lemma'] = original_lemma
                else:
                    results.append(self.lemmatize(token))

            return results

        except Exception as e:
            # Fallback on any error
            print(f"⚠ Contextual disambiguation failed: {e}", file=sys.stderr)
            return [self.lemmatize(token) for token in tokens]

    # =========================================================================
    # TIER 1: V16 HYBRID LEXICON (GOLD STANDARD)
    # =========================================================================

    def tier1_v16_lexicon(self, word: str, stanza_upos: Optional[str] = None) -> Optional[Dict]:
        """
        Tier 1: V16 Hybrid Gold Standard Lexicon lookup

        Three-tier hybrid system:
        - Tier 1a: Manual annotations (100% trusted)
        - Tier 1b: Omorfi 100% unambiguous (non-overlapping with manual)
        - Tier 1c: Future production accumulation

        Ambiguity Tracking:
        - Never caches genuinely ambiguous words
        - Tracks ambiguous patterns separately

        Args:
            word: Word to look up
            stanza_upos: Optional POS tag from Stanza for disambiguation

        Returns:
            Lemmatization result if found in lexicon, None otherwise
        """
        if not self.pos_aware_lexicon:
            return None

        word_lower = word.lower()

        # Check ambiguous patterns first
        if word_lower in self.ambiguous_patterns:
            ambiguous_pos = self.ambiguous_patterns[word_lower]

            # If we have Stanza POS, try to disambiguate
            if stanza_upos and stanza_upos in ambiguous_pos:
                entry = ambiguous_pos[stanza_upos]
                lemma = entry['lemma']
                source = entry.get('source', 'unknown')
                tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                return {
                    'word': word,
                    'lemma': lemma,
                    'analysis': f'[V16_LEXICON_DISAMBIGUATED:{word}→{lemma}][POS:{stanza_upos}][TIER:{tier}]',
                    'weight': None,
                    'method': f'v16_lexicon_{tier}',
                    'confidence': 'high',
                    'alternatives': []
                }

            # Ambiguous without context - skip to next tier
            return None

        # Check unambiguous lexicon
        if word_lower not in self.pos_aware_lexicon:
            return None

        pos_dict = self.pos_aware_lexicon[word_lower]

        # If we have Stanza POS, use it for disambiguation
        if stanza_upos and stanza_upos in pos_dict:
            entry = pos_dict[stanza_upos]
            lemma = entry['lemma']
            source = entry.get('source', 'unknown')
            tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

            return {
                'word': word,
                'lemma': lemma,
                'analysis': f'[V16_LEXICON:{word}→{lemma}][POS:{stanza_upos}][TIER:{tier}]',
                'weight': None,
                'method': f'v16_lexicon_{tier}',
                'confidence': 'high',
                'alternatives': []
            }

        # No Stanza POS - if unambiguous (single POS), return it
        if len(pos_dict) == 1:
            pos, entry = next(iter(pos_dict.items()))
            lemma = entry['lemma']
            source = entry.get('source', 'unknown')
            tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

            return {
                'word': word,
                'lemma': lemma,
                'analysis': f'[V16_LEXICON:{word}→{lemma}][POS:{pos}][TIER:{tier}]',
                'weight': None,
                'method': f'v16_lexicon_{tier}',
                'confidence': 'high',
                'alternatives': []
            }

        # Multiple POS without context - skip to next tier for disambiguation
        return None

    # =========================================================================
    # TIER 2: OMORFI CONTEXTUAL ANALYSIS
    # =========================================================================

    def tier2_omorfi_contextual(self, word: str, strict_proper_noun_check: bool = True) -> Optional[Dict]:
        """
        Tier 2: Direct Omorfi analysis with smart alternative selection

        Process:
        1. Get all Omorfi analyses for the word
        2. Use smart alternative selection to choose best lemma
        3. Apply proper noun heuristics (reject 1-2 char lemmas for capitalized words)

        Args:
            word: Word to lemmatize
            strict_proper_noun_check: If True, reject short lemmas for capitalized words

        Returns:
            Lemmatization result or None if no valid analysis
        """
        return self.core.lemmatize_direct(word, strict_proper_noun_check=strict_proper_noun_check)

    # =========================================================================
    # TIER 3: VOIKKO + OMORFI
    # =========================================================================

    def tier3_voikko_omorfi(self, word: str) -> Optional[Dict]:
        """
        Tier 3: Voikko spelling normalization + Omorfi re-analysis

        Process:
        1. Get up to 10 Voikko spelling suggestions
        2. Try Omorfi analysis on each suggestion
        3. Return first successful analysis

        Use Case: Handles dialectal spellings and typos

        Args:
            word: Word to normalize and lemmatize

        Returns:
            Lemmatization result with normalized form, or None
        """
        if not self.voikko_path and not self.voikko:
            return None

        return self.core.lemmatize_with_voikko(word)

    # =========================================================================
    # TIER 4: OMORFI GUESSER
    # =========================================================================

    def tier4_omorfi_guesser(self, word: str) -> Optional[Dict]:
        """
        Tier 4: Omorfi guesser for unknown words

        Process:
        1. Use Omorfi guesser transducer for unknown word analysis
        2. Apply smart alternative selection
        3. Reject analyses with '?' markers (low confidence)

        Use Case: Handles neologisms, rare inflections, and unknown compounds

        Args:
            word: Unknown word to analyze

        Returns:
            Lemmatization result with medium confidence, or None
        """
        if not self.guesser:
            return None

        return self.core.lemmatize_guesser(word)

    # =========================================================================
    # TIER 5: MORPHOLOGY-AWARE FUZZY LEXICON
    # =========================================================================

    def tier5_fuzzy_lexicon_morphological(self, word: str, stanza_upos: Optional[str] = None) -> Optional[Dict]:
        """
        Tier 5: Morphology-aware fuzzy lexicon matching

        Process:
        1. Extract morphological features (UFEATS) from dialectal word via Omorfi
        2. Find fuzzy candidates from V16 lexicon (edit distance ≤ 2.0)
        3. Extract morphological features from each candidate
        4. Compute feature similarity bonus
        5. Re-rank by: edit_distance - feature_similarity_bonus
        6. Return best morphologically plausible match

        Use Case: Dialectal variations with morphological context preservation
        Example: "taloista" (dialectal) → "talo" (standard) via Case=Ela match

        Args:
            word: Dialectal word form to lemmatize
            stanza_upos: Optional POS tag from Stanza for filtering

        Returns:
            Lemmatization result with morphological confidence, or None
        """
        if not (self.dialectal_normalizer and self.pos_aware_lexicon):
            return None

        return self.core.lemmatize_with_fuzzy_lexicon_morphological(word, stanza_upos)

    # =========================================================================
    # TIER 6: AGGRESSIVE FUZZY LEXICON
    # =========================================================================

    def tier6_fuzzy_lexicon_aggressive(self, word: str, stanza_upos: Optional[str] = None) -> Optional[Dict]:
        """
        Tier 6: Aggressive fuzzy lexicon with suffix stripping and normalization

        Process:
        1. Generate word variants by stripping common dialectal suffixes
        2. Apply character normalization (w→v, c→k, j→i in compounds)
        3. Find fuzzy matches with relaxed threshold (≤ 3.5)
        4. Try POS-aware matching first, fall back to POS-agnostic

        Use Case: Heavy dialectal variations, archaic forms, compound variants
        Example: "colmjkymment" → "kolmekymmentä" via suffix strip + j→i + fuzzy

        Suffixes Handled:
        - ntkaane, kaane, staan, sessa, mast, ille, ttaa, taa, lla, s

        Args:
            word: Heavily dialectal word form
            stanza_upos: Optional POS tag from Stanza for filtering

        Returns:
            Lemmatization result with low confidence, or None
        """
        if not (self.dialectal_normalizer and self.pos_aware_lexicon):
            return None

        return self.core.lemmatize_with_fuzzy_lexicon_aggressive(word, stanza_upos)

    # =========================================================================
    # TIER 7: FINNISH DIALECTS DICTIONARY (SMS)
    # =========================================================================

    def tier7_dialectal_dictionary(self, word: str, stanza_upos: Optional[str] = None) -> Optional[Dict]:
        """
        Tier 7: Finnish Dialects Dictionary (SMS) lookup

        Source: Suomen murteiden sanakirja (Dictionary of Finnish Dialects)
        Coverage: 19,385 validated dialectal variants

        Process:
        1. Exact lookup in dialectal variant index
        2. Apply POS filtering if Stanza context available
        3. Select best candidate by confidence score

        Use Case: Validated dialectal forms not in standard dictionaries
        Example: "huomena" (dialectal) → "huominen" (standard)

        Args:
            word: Word to look up in dialectal dictionary
            stanza_upos: Optional POS tag from Stanza for filtering

        Returns:
            Lemmatization result with high confidence, or None
        """
        if not self.dialectal_dict:
            return None

        return self.core.lemmatize_with_dialectal_dict(word, stanza_upos)

    # =========================================================================
    # TIER 8: ENHANCED VOIKKO (AGGRESSIVE)
    # =========================================================================

    def tier8_voikko_aggressive(self, word: str) -> Optional[Dict]:
        """
        Tier 8: Enhanced Voikko with aggressive suggestions

        Process:
        1. Get up to 30 Voikko spelling suggestions (vs 10 in Tier 3)
        2. Try Omorfi analysis on each suggestion
        3. Return first successful analysis

        Use Case: Catches more unknown words before identity fallback
        Target: Reduce identity_fallback from 291 words (19.8%) to <150 words

        Trade-off: Lower confidence than Tier 3 due to more aggressive matching

        Args:
            word: Word to aggressively normalize

        Returns:
            Lemmatization result with low confidence, or None
        """
        if not self.voikko_path and not self.voikko:
            return None

        return self.core.lemmatize_with_voikko_enhanced(word)

    # =========================================================================
    # TIER 9: IDENTITY FALLBACK
    # =========================================================================

    def tier9_identity_fallback(self, word: str) -> Dict[str, Any]:
        """
        Tier 9: Identity fallback (return word.lower() as lemma)

        Rationale:
        - Proper nouns in nominative form are already correct lemmas
        - Unknown words in base form don't need transformation
        - Prevents total failure for unanalyzable words

        Use Case: Proper nouns, foreign words, neologisms, errors

        Args:
            word: Word that failed all other tiers

        Returns:
            Lemmatization result with word.lower() as lemma (low confidence)
        """
        identity_lemma = word.lower()
        return {
            'word': word,
            'lemma': identity_lemma,
            'analysis': f'[IDENTITY_FALLBACK:{word}→{identity_lemma}]',
            'weight': None,
            'method': 'identity_fallback',
            'confidence': 'low',
            'alternatives': []
        }

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    def lemmatize_batch(self, words: List[str]) -> Dict[str, Any]:
        """
        Lemmatize multiple words and return statistics

        Args:
            words: List of words to lemmatize

        Returns:
            Dictionary with keys:
                - results: List of lemmatization results
                - statistics: Processing statistics by method
        """
        results = []
        stats = {
            'total': len(words),
            'v16_lexicon_tier1': 0,
            'v16_lexicon_tier2': 0,
            'v16_lexicon_tier3': 0,
            'omorfi_contextual': 0,
            'voikko_omorfi': 0,
            'omorfi_direct': 0,
            'omorfi_guesser': 0,
            'voikko_normalized': 0,
            'exception_lexicon': 0,
            'enhanced_voikko': 0,
            'fuzzy_lexicon': 0,
            'fuzzy_lexicon_morphological': 0,
            'dialectal_dictionary': 0,
            'fuzzy_lexicon_aggressive': 0,
            'identity_fallback': 0,
            'unknown': 0
        }

        for word in words:
            result = self.lemmatize(word)
            results.append(result)

            method = result['method']
            if method in stats:
                stats[method] += 1
            else:
                stats['unknown'] += 1

        stats['coverage'] = (stats['total'] - stats['unknown']) / stats['total'] * 100 if stats['total'] > 0 else 0

        return {
            'results': results,
            'statistics': stats
        }

    def lemmatize_text(self, text: str) -> Dict[str, Any]:
        """
        Lemmatize text with automatic tokenization

        Args:
            text: Text to lemmatize (will be tokenized automatically)

        Returns:
            Dictionary with keys:
                - results: List of lemmatization results
                - statistics: Processing statistics by method
        """
        # Simple tokenization (word characters including Finnish letters)
        words = re.findall(r'\b[\wäöåÄÖÅ]+\b', text, re.UNICODE)
        return self.lemmatize_batch(words)

    def lemmatize_conllu(self, conllu_text: str) -> str:
        """
        Process CoNLL-U formatted text and add lemmatization

        Args:
            conllu_text: Text in CoNLL-U format

        Returns:
            CoNLL-U text with updated lemma column
        """
        output_lines = []

        for line in conllu_text.split('\n'):
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                output_lines.append(line)
                continue

            # Parse CoNLL-U line
            fields = line.split('\t')
            if len(fields) < 10:
                output_lines.append(line)
                continue

            # Extract word form
            word = fields[1]

            # Lemmatize
            result = self.lemmatize(word)

            # Update lemma field (column 2)
            fields[2] = result['lemma'] if result['lemma'] else '_'

            # Add processing method as comment
            output_lines.append('\t'.join(fields))

        return '\n'.join(output_lines)

    # =========================================================================
    # STANZA CONTEXT INTEGRATION (for future contextual tiers)
    # =========================================================================

    def get_stanza_context(self, sentence: str) -> Dict[str, Dict[str, str]]:
        """
        Get contextual POS and morphological features from Stanza

        Args:
            sentence: Sentence to analyze

        Returns:
            Dictionary mapping word forms to their context:
                {
                    'word': {
                        'upos': 'NOUN',
                        'feats': 'Case=Nom|Number=Sing'
                    }
                }
        """
        if not self.stanza_nlp:
            return {}

        context = {}

        try:
            doc = self.stanza_nlp(sentence)

            for sent in doc.sentences:
                for word in sent.words:
                    context[word.text] = {
                        'upos': word.upos,
                        'feats': word.feats if word.feats else ''
                    }
        except Exception as e:
            print(f"⚠ Stanza analysis failed: {e}", file=sys.stderr)

        return context


def main() -> None:
    """CLI interface for Finnish lemmatizer"""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Finnish lemmatization with 9-tier hybrid pipeline (v{VERSION})"
    )
    parser.add_argument('text', nargs='?', help='Text to lemmatize (or use stdin)')
    parser.add_argument('--model-dir', default=None,
                       help='Path to Omorfi models (default: ~/.omorfi)')
    parser.add_argument('--voikko-path', default=None,
                       help='Path to voikkospell (default: auto-detect)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed analysis')
    parser.add_argument('--show-alternatives', action='store_true',
                       help='Show all alternative lemmas')
    parser.add_argument('--show-unknown', action='store_true',
                       help='Only show unknown words')
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')

    args = parser.parse_args()

    # Get input text
    if args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    # Initialize lemmatizer
    try:
        lemmatizer = FinnishLemmatizer(
            model_dir=args.model_dir,
            voikko_path=args.voikko_path
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Lemmatize text
    batch_result = lemmatizer.lemmatize_text(text)

    # Print results
    print()
    print("=" * 80)
    print(f"FINNISH LEMMATIZER v{VERSION} - 9-TIER HYBRID PIPELINE")
    print("=" * 80)
    print()

    # Statistics
    stats = batch_result['statistics']
    print(f"Total words:         {stats['total']}")
    print(f"V16 Tier 1 (Gold):   {stats['v16_lexicon_tier1']} ({stats['v16_lexicon_tier1']/stats['total']*100:.1f}%)")
    print(f"V16 Tier 2 (Omorfi): {stats['v16_lexicon_tier2']} ({stats['v16_lexicon_tier2']/stats['total']*100:.1f}%)")

    if stats['v16_lexicon_tier3'] > 0:
        print(f"V16 Tier 3 (Accum):  {stats['v16_lexicon_tier3']} ({stats['v16_lexicon_tier3']/stats['total']*100:.1f}%)")

    print(f"Omorfi contextual:   {stats['omorfi_contextual']} ({stats['omorfi_contextual']/stats['total']*100:.1f}%)")
    print(f"Omorfi direct:       {stats['omorfi_direct']} ({stats['omorfi_direct']/stats['total']*100:.1f}%)")

    if lemmatizer.guesser_effective:
        print(f"Omorfi guesser:      {stats['omorfi_guesser']} ({stats['omorfi_guesser']/stats['total']*100:.1f}%)")

    print(f"Voikko normalized:   {stats['voikko_normalized']} ({stats['voikko_normalized']/stats['total']*100:.1f}%)")

    if stats['dialectal_dictionary'] > 0:
        print(f"Dialectal Dictionary: {stats['dialectal_dictionary']} ({stats['dialectal_dictionary']/stats['total']*100:.1f}%)")

    if stats['fuzzy_lexicon_morphological'] > 0:
        print(f"Fuzzy (Morphological): {stats['fuzzy_lexicon_morphological']} ({stats['fuzzy_lexicon_morphological']/stats['total']*100:.1f}%)")

    if stats['fuzzy_lexicon_aggressive'] > 0:
        print(f"Fuzzy (Aggressive):  {stats['fuzzy_lexicon_aggressive']} ({stats['fuzzy_lexicon_aggressive']/stats['total']*100:.1f}%)")

    if stats['identity_fallback'] > 0:
        print(f"Identity fallback:   {stats['identity_fallback']} ({stats['identity_fallback']/stats['total']*100:.1f}%)")

    print(f"Unknown:             {stats['unknown']} ({stats['unknown']/stats['total']*100:.1f}%)")
    print(f"Coverage:            {stats['coverage']:.1f}%")
    print()
    print("=" * 80)
    print()

    # Word-level results
    for result in batch_result['results']:
        if args.show_unknown and result['method'] != 'identity_fallback':
            continue

        original_word = result.get('original_word', result['word'])
        word = result['word']
        lemma = result['lemma'] if result['lemma'] else '_UNKNOWN_'
        method = result['method']
        confidence = result['confidence']

        if confidence == 'high':
            status = '✓'
        elif confidence == 'medium' or 'medium' in confidence:
            status = '~'
        elif confidence == 'low':
            status = '◌'
        else:
            status = '?'

        # Format output
        if result.get('normalized_word') and original_word != result['normalized_word']:
            print(f"{status} {original_word:15} → {result['normalized_word']:15} → {lemma:15} ({method})")
        else:
            print(f"{status} {original_word:20} → {lemma:20} ({method})")

        if result.get('note'):
            print(f"    Note: {result['note']}")

        # Show alternatives
        if result.get('alternatives') and len(result['alternatives']) > 0:
            if args.show_alternatives:
                alts = result['alternatives']
            else:
                alts = result['alternatives'][:2]

            seen_lemmas = {lemma}
            unique_lemmas = []
            for alt in alts:
                alt_lemma = alt.get('lemma')
                if alt_lemma and alt_lemma not in seen_lemmas:
                    seen_lemmas.add(alt_lemma)
                    unique_lemmas.append(alt_lemma)

            if unique_lemmas:
                alt_str = ", ".join(unique_lemmas)
                print(f"    Also: {alt_str}")

        # Verbose mode
        if args.verbose and result.get('analysis'):
            print(f"    Analysis: {result['analysis']}")
            if result.get('weight') is not None:
                print(f"    Weight: {result['weight']:.2f}")
            if result.get('normalized_word'):
                print(f"    Voikko normalized: {result['original_word']} → {result['normalized_word']}")
            if args.show_alternatives and result.get('alternatives'):
                print(f"    Alternative analyses:")
                for alt in result['alternatives']:
                    if alt['lemma']:
                        print(f"      → {alt['lemma']:20} (weight: {alt['weight']:.2f}, score: {alt.get('score', 0):.2f})")

    # Summary
    if stats['identity_fallback'] > 0:
        print()
        print("=" * 80)
        print(f"◌ {stats['identity_fallback']} words fell back to identity (may need manual review)")
        print("  Run with --show-unknown to see only these words")
        print("=" * 80)

    print()


if __name__ == "__main__":
    main()
