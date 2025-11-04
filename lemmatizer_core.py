#!/usr/bin/env python3
"""
Core lemmatization functionality for Finnish dialectal texts.

This module contains the LemmatizerCore class which handles:
- Loading and managing Omorfi/HFST morphological analyzers
- Voikko spelling suggestion and ranking
- Morphological feature extraction and scoring
- Lemma candidate selection with contextual disambiguation

Extracted from fin_runocorp_base_v2_dialectal_dict_integrated_v17_phase11.py
as part of Phase 12 refactoring to support compound word analysis.
"""

import hfst
import os
import sys
import re
import subprocess
import shlex
import json
from typing import List, Dict, Tuple, Optional, Any, Set, Union

# Import from our new config module
from lemmatizer_config import (
    LemmatizerConfig,
    MorphologicalFeatureExtractor,
    FeatureSimilarityScorer,
    LemmatizerConstants,
    OMORFI_POS_UD,
    OMORFI_CASE_UD
)

# Optional imports
try:
    import libvoikko
    LIBVOIKKO_AVAILABLE = True
except ImportError:
    LIBVOIKKO_AVAILABLE = False

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False

from dialectal_normalizer import DialectalNormalizer
from compound_classifier import CompoundClassifier


class LemmatizerCore:
    """
    Core lemmatization engine using Omorfi HFST, Voikko, and Stanza.

    This class provides the fundamental morphological analysis and lemma
    selection functionality. It manages:
    - HFST transducers (analyzer and guesser)
    - Voikko dictionaries (Sukija, Old Finnish, system)
    - Stanza contextual analysis
    - Morphological feature extraction and scoring
    - Dialectal normalization

    Attributes:
        config: LemmatizerConfig instance with paths and settings
        analyzer: HFST transducer for morphological analysis
        guesser: HFST transducer for unknown word guessing (optional)
        voikko_sukija: Voikko instance for historical/dialectal texts
        voikko_old_finnish: Voikko instance for Old Finnish texts
        voikko: Primary Voikko instance
        stanza_nlp: Stanza pipeline for contextual analysis
        feature_extractor: MorphologicalFeatureExtractor instance
        similarity_scorer: FeatureSimilarityScorer instance
        dialectal_normalizer: DialectalNormalizer instance
        pos_aware_lexicon: Dict mapping (word, POS) → lemma
        ambiguous_patterns: Dict tracking ambiguous (word, POS) pairs
        quality_tiers: Statistics about lexicon quality tiers
        dialectal_dict: Finnish Dialects Dictionary (SMS) data
        morph_stats: Statistics about morphological feature usage
    """

    def __init__(self, config: Optional[LemmatizerConfig] = None,
                 model_dir: Optional[str] = None,
                 voikko_path: Optional[str] = None,
                 lang: str = 'fi') -> None:
        """
        Initialize LemmatizerCore with Omorfi, Voikko, and Stanza resources.

        Args:
            config: LemmatizerConfig instance (if None, creates from other params)
            model_dir: DEPRECATED - use config instead
            voikko_path: DEPRECATED - use config instead
            lang: DEPRECATED - use config instead

        Raises:
            FileNotFoundError: If Omorfi models not found at specified path
        """
        # Handle backward compatibility
        if config is None:
            config = LemmatizerConfig(
                model_dir=model_dir,
                voikko_path=voikko_path,
                lang=lang
            )
        self.config = config

        # For backward compatibility, keep these attributes
        self.model_dir = self.config.model_dir
        self.lang = self.config.lang
        self.describe_path = self.config.analyzer_path

        # Initialize morphological feature extractors
        self.feature_extractor = MorphologicalFeatureExtractor()
        self.similarity_scorer = FeatureSimilarityScorer()

        # Load resources
        self._load_stanza()
        self._load_voikko_dictionaries()
        self._load_omorfi_analyzer()
        self._load_omorfi_guesser()
        self._load_hybrid_lexicon()
        self._load_dialectal_normalizer()
        self._load_dialectal_dictionary()
        self._load_compound_classifier()

        # V17 Phase 9: Track morphological feature usage in detail
        self.morph_stats = {
            'words_with_multiple_analyses': 0,      # Words that had >1 Omorfi analysis
            'features_extracted_successfully': 0,    # Times we got UFEATS from word form
            'feature_similarity_computed': 0,        # Times we computed similarity score
            'feature_bonus_applied': 0,              # Times feature_bonus > 0
            'selection_changed_by_features': 0,      # Times features changed which candidate won
            'total_feature_bonus': 0.0,              # Sum of all feature bonuses
            'max_feature_bonus': 0.0                 # Maximum feature bonus seen
        }

    def _load_stanza(self) -> None:
        """Load Stanza Finnish-TDT model for contextual analysis."""
        self.stanza_nlp = None
        if STANZA_AVAILABLE:
            try:
                print("Loading Stanza Finnish-TDT model...", file=sys.stderr)
                self.stanza_nlp = stanza.Pipeline(
                    lang="fi",
                    processors="tokenize,pos,lemma",
                    use_gpu=False,
                    tokenize_pretokenized=True
                )
                print("✓ Stanza loaded", file=sys.stderr)
            except Exception as e:
                print(f"⚠ Could not load Stanza: {e}", file=sys.stderr)
                self.stanza_nlp = None

    def _load_voikko_dictionaries(self) -> None:
        """Load Voikko dictionaries (Sukija, Old Finnish, system fallback)."""
        # Initialize libvoikko with multiple dictionaries (Phase 4)
        # We'll try to load both Sukija (historical/dialectal) and Old Finnish
        self.voikko_sukija = None
        self.voikko_old_finnish = None
        self.voikko = None  # Will point to primary dictionary

        if LIBVOIKKO_AVAILABLE:
            # Try Sukija dictionary (historical/dialectal texts)
            try:
                sukija_path = self.config.voikko_sukija_path
                if os.path.exists(sukija_path):
                    self.voikko_sukija = libvoikko.Voikko("fi", path=sukija_path)
                    print(f"✓ Voikko Sukija loaded from {sukija_path}", file=sys.stderr)
            except Exception as e:
                print(f"⚠ Could not load Sukija dictionary: {e}", file=sys.stderr)

            # Try Old Finnish dictionary
            try:
                old_finnish_path = self.config.voikko_old_finnish_path
                if os.path.exists(old_finnish_path):
                    self.voikko_old_finnish = libvoikko.Voikko("fi", path=old_finnish_path)
                    print(f"✓ Voikko Old Finnish loaded from {old_finnish_path}", file=sys.stderr)
            except Exception as e:
                print(f"⚠ Could not load Old Finnish dictionary: {e}", file=sys.stderr)

            # Set primary dictionary (Sukija preferred, fallback to Old Finnish)
            if self.voikko_sukija:
                self.voikko = self.voikko_sukija
            elif self.voikko_old_finnish:
                self.voikko = self.voikko_old_finnish
            else:
                # Fall back to system dictionary
                try:
                    self.voikko = libvoikko.Voikko("fi")
                    print("✓ Voikko loaded with system dictionary", file=sys.stderr)
                except Exception as e:
                    print(f"⚠ Could not load libvoikko: {e}", file=sys.stderr)
                    self.voikko = None

        # Keep voikkospell path as fallback if libvoikko not available
        if self.config.voikko_path is None and not self.voikko:
            # Try common locations
            paths = [
                '/usr/local/Cellar/libvoikko/4.3.3/bin/voikkospell',
                '/usr/local/bin/voikkospell',
                '/opt/homebrew/bin/voikkospell'
            ]
            for path in paths:
                if os.path.exists(path):
                    self.voikko_path = path
                    break
            else:
                # Try to find in PATH
                try:
                    result = subprocess.run(['which', 'voikkospell'],
                                          capture_output=True, text=True, check=True)
                    self.voikko_path = result.stdout.strip()
                except:
                    self.voikko_path = None
        else:
            self.voikko_path = self.config.voikko_path if not self.voikko else None

    def _load_omorfi_analyzer(self) -> None:
        """Load Omorfi HFST analyzer transducer."""
        if not os.path.exists(self.describe_path):
            raise FileNotFoundError(
                f"Omorfi models not found at {self.describe_path}\n"
                f"Download with: See INSTALLATION_GUIDE.md"
            )

        print(f"Loading Omorfi analyzer from: {self.model_dir}", file=sys.stderr)
        input_stream = hfst.HfstInputStream(self.describe_path)
        self.analyzer = input_stream.read()
        input_stream.close()
        print(f"✓ Omorfi analyzer loaded", file=sys.stderr)

    def _load_omorfi_guesser(self) -> None:
        """Load Omorfi HFST guesser transducer (optional)."""
        self.guesser = None
        self.guesser_effective = False
        guesser_path = f'{self.model_dir}/omorfi-guesser.hfst'

        if os.path.exists(guesser_path):
            print(f"Loading guesser from: {guesser_path}", file=sys.stderr)
            try:
                input_stream = hfst.HfstInputStream(guesser_path)
                self.guesser = input_stream.read()
                input_stream.close()
                print(f"✓ Guesser loaded", file=sys.stderr)
            except Exception as e:
                print(f"⚠ Could not load guesser: {e}", file=sys.stderr)

        # Check Voikko availability
        if self.voikko_path and os.path.exists(self.voikko_path):
            print(f"✓ Voikko found: {self.voikko_path}", file=sys.stderr)
            try:
                result = subprocess.run(
                    [self.voikko_path, '-h'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 or 'usage' in result.stderr.lower():
                    print(f"✓ Voikko working", file=sys.stderr)
                else:
                    print(f"⚠ Voikko not working properly", file=sys.stderr)
                    self.voikko_path = None
            except Exception as e:
                print(f"⚠ Voikko test failed: {e}", file=sys.stderr)
                self.voikko_path = None
        else:
            print(f"⚠ Voikko not found", file=sys.stderr)
            self.voikko_path = None

    def _load_hybrid_lexicon(self) -> None:
        """Load V16 hybrid gold standard lexicon with three tiers + ambiguous tracking."""
        self.pos_aware_lexicon = {}
        self.ambiguous_patterns = {}
        self.quality_tiers = {}

        # Use V16 min=1 lexicon (trusts all manual annotations - 13.6× more patterns!)
        lexicon_path = self.config.lexicon_path
        if os.path.exists(lexicon_path):
            try:
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    lexicon_data = json.load(f)
                    self.pos_aware_lexicon = lexicon_data.get('pos_aware_lexicon', {})
                    self.ambiguous_patterns = lexicon_data.get('ambiguous_patterns', {})
                    self.quality_tiers = lexicon_data.get('quality_tiers', {})

                total_patterns = sum(len(pos_dict) for pos_dict in self.pos_aware_lexicon.values())
                ambiguous_count = sum(len(pos_dict) for pos_dict in self.ambiguous_patterns.values())

                print(f"✓ Loaded V16 hybrid lexicon:", file=sys.stderr)
                print(f"  - Total unambiguous: {total_patterns} (word, POS) patterns", file=sys.stderr)
                print(f"    · Tier 1 (Gold Standard): {self.quality_tiers.get('tier1_gold_standard', 0)}", file=sys.stderr)
                print(f"    · Tier 2 (Omorfi 100%): {self.quality_tiers.get('tier2_omorfi_100', 0)}", file=sys.stderr)
                print(f"  - Ambiguous patterns tracked: {ambiguous_count}", file=sys.stderr)

            except Exception as e:
                print(f"⚠ Could not load V16 hybrid lexicon: {e}", file=sys.stderr)
        else:
            print(f"⚠ V16 hybrid lexicon not found: {lexicon_path}", file=sys.stderr)
            print(f"  Build with: python3 build_lexicon_v16.py --train <train.csv> --output {lexicon_path}", file=sys.stderr)

    def _load_dialectal_normalizer(self) -> None:
        """Initialize dialectal normalizer for edit distance calculations."""
        try:
            self.dialectal_normalizer = DialectalNormalizer()
            print(f"✓ Dialectal normalizer initialized", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Could not initialize dialectal normalizer: {e}", file=sys.stderr)
            self.dialectal_normalizer = None

    def _load_dialectal_dictionary(self) -> None:
        """Load Finnish Dialects Dictionary (SMS)."""
        self.dialectal_dict = {}
        dialectal_dict_path = self.config.dialectal_dict_path
        if os.path.exists(dialectal_dict_path):
            try:
                with open(dialectal_dict_path, 'r', encoding='utf-8') as f:
                    self.dialectal_dict = json.load(f)

                variant_count = len(self.dialectal_dict.get('variant_to_lemma', {}))
                print(f"✓ Loaded Finnish Dialects Dictionary: {variant_count:,} variants", file=sys.stderr)
            except Exception as e:
                print(f"⚠ Could not load dialectal dictionary: {e}", file=sys.stderr)
        else:
            print(f"⚠ Dialectal dictionary not found: {dialectal_dict_path}", file=sys.stderr)

    def _load_compound_classifier(self) -> None:
        """Initialize compound classifier for Phase 12 integration."""
        try:
            self.compound_classifier = CompoundClassifier(
                lexicon_path=self.config.lexicon_path,
                sms_dict_path=self.config.dialectal_dict_path,
                voikko_path=self.config.voikko_path
            )
            print(f"✓ Compound classifier initialized", file=sys.stderr)

            # V17 Phase 12: Track compound classification statistics
            self.compound_stats = {
                'total_compounds_detected': 0,
                'true_compounds_reconstructed': 0,
                'false_compounds_rejected': 0,
                'possessive_with_compound_marker': 0,
                'fallback_to_non_compound': 0
            }
        except Exception as e:
            print(f"⚠ Could not initialize compound classifier: {e}", file=sys.stderr)
            self.compound_classifier = None
            self.compound_stats = {}

    # ========================================================================
    # Omorfi/HFST Analysis Methods
    # ========================================================================

    def analyze_direct(self, word: str) -> List[Tuple[str, float]]:
        """
        Direct analysis with Omorfi analyzer.

        Args:
            word: Word to analyze

        Returns:
            List of (analysis, weight) tuples sorted by weight (lower is better)
        """
        return self.analyzer.lookup(word.lower())

    def analyze_guesser(self, word: str) -> List[Tuple[str, float]]:
        """
        Guesser analysis for unknown words (if guesser is loaded).

        Args:
            word: Word to analyze

        Returns:
            List of (analysis, weight) tuples or empty list if no guesser
        """
        if self.guesser is None:
            return []

        return self.guesser.lookup(word.lower())

    def extract_lemma(self, analysis: str) -> Optional[str]:
        """
        Extract lemma from Omorfi analysis string.

        Args:
            analysis: Omorfi analysis string like '[WORD_ID=talo][UPOS=NOUN][NUM=SG][CASE=NOM]'

        Returns:
            Extracted lemma (e.g., 'talo') or None if not found

        Example:
            >>> extract_lemma('[WORD_ID=talo][UPOS=NOUN][NUM=SG][CASE=NOM]')
            'talo'
        """
        match = re.search(r'\[WORD_ID=([^\]]+)\]', analysis)
        if match:
            return match.group(1)
        return None

    def parse_omorfi_tags(self, analysis: str) -> Dict[str, str]:
        """
        Parse Omorfi analysis into structured tags for contextual scoring.

        Args:
            analysis: Omorfi analysis string

        Returns:
            Dictionary with 'pos', 'case', and 'number' keys (values may be None)

        Example:
            >>> parse_omorfi_tags('[WORD_ID=talo][UPOS=NOUN][NUM=SG][CASE=NOM]')
            {'pos': 'NOUN', 'case': 'NOM', 'number': 'SG'}
        """
        tags = {'pos': None, 'case': None, 'number': None}

        pos_match = re.search(r'\[UPOS=([^\]]+)\]', analysis)
        if pos_match:
            tags['pos'] = pos_match.group(1)

        case_match = re.search(r'\[CASE=([^\]]+)\]', analysis)
        if case_match:
            tags['case'] = case_match.group(1)

        num_match = re.search(r'\[NUM=([^\]]+)\]', analysis)
        if num_match:
            tags['number'] = 'SG' if num_match.group(1) == 'SG' else 'PL'

        return tags

    # ========================================================================
    # Voikko Suggestion Methods
    # ========================================================================

    def voikko_suggest(self, word: str, max_n: int = 5) -> List[str]:
        """
        Get spelling suggestions using Voikko (libvoikko or subprocess fallback).
        V17 Phase 4: Merges results from both Sukija and Old Finnish dictionaries.

        Args:
            word: Word to get suggestions for
            max_n: Maximum number of suggestions

        Returns:
            List of spelling suggestions (merged from all available dictionaries)
        """
        # V17 Phase 4: Try both dictionaries and merge results
        if self.voikko_sukija or self.voikko_old_finnish:
            try:
                all_suggestions = []

                # Try Sukija dictionary first (historical/dialectal texts)
                if self.voikko_sukija:
                    # Check if word is correct in Sukija
                    if self.voikko_sukija.spell(word):
                        return [word]

                    # Get Sukija suggestions
                    sukija_suggestions = self.voikko_sukija.suggest(word)
                    if sukija_suggestions:
                        all_suggestions.extend(sukija_suggestions)

                # Try Old Finnish dictionary (additional coverage)
                if self.voikko_old_finnish:
                    # Check if word is correct in Old Finnish (if not already found)
                    if not all_suggestions and self.voikko_old_finnish.spell(word):
                        return [word]

                    # Get Old Finnish suggestions
                    old_finnish_suggestions = self.voikko_old_finnish.suggest(word)
                    if old_finnish_suggestions:
                        all_suggestions.extend(old_finnish_suggestions)

                # Remove duplicates while preserving order (Sukija priority)
                seen = set()
                unique_suggestions = []
                for s in all_suggestions:
                    if s not in seen:
                        seen.add(s)
                        unique_suggestions.append(s)

                # Apply confusion-weighted ranking if dialectal normalizer available
                if unique_suggestions and self.dialectal_normalizer:
                    ranked = self.dialectal_normalizer.rank_suggestions(word, unique_suggestions)
                    suggestions = [lemma for lemma, distance in ranked[:max_n]]
                else:
                    suggestions = unique_suggestions[:max_n]

                return suggestions

            except Exception as e:
                # Fall through to subprocess method
                pass

        # Fallback to subprocess method if libvoikko not available
        if not self.voikko_path:
            return []

        try:
            cmd = f"{self.voikko_path} -s -d {self.lang}"
            result = subprocess.run(
                shlex.split(cmd),
                input=(word + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
                check=False
            )

            suggestions = []
            for line in result.stdout.decode("utf-8", "ignore").splitlines():
                if line.startswith("S: "):
                    sugg = line[3:].strip()
                    suggestions.append(sugg)
                    if len(suggestions) >= 20:
                        break
                elif line.startswith("C: "):
                    return [word]

            # Apply confusion-weighted ranking if dialectal normalizer available
            if suggestions and self.dialectal_normalizer:
                ranked = self.dialectal_normalizer.rank_suggestions(word, suggestions)
                suggestions = [lemma for lemma, distance in ranked[:max_n]]
            else:
                suggestions = suggestions[:max_n]

            return suggestions

        except Exception as e:
            return []

    def rank_voikko_suggestions(self, word: str, suggestions: List[str],
                               stanza_upos: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        V17 Phase 7: Rank Voikko suggestions using multi-criteria scoring.

        Ranking criteria (lower score = better):
        1. Edit distance from original word (weighted Levenshtein)
        2. POS agreement bonus (if Stanza POS available and matches)
        3. Lexicon frequency (prefer common words)
        4. Proper noun penalty (unless expected)

        Args:
            word: Original dialectal word
            suggestions: List of Voikko suggestions
            stanza_upos: Optional Stanza POS tag for filtering

        Returns:
            List of (suggestion, score) tuples, sorted by score (best first)
        """
        if not suggestions:
            return []

        scored_suggestions = []

        for suggestion in suggestions:
            score = 0.0

            # 1. Edit distance from original word (main criterion)
            if self.dialectal_normalizer:
                edit_dist = self.dialectal_normalizer.weighted_edit_distance(word.lower(), suggestion.lower())
                score += edit_dist * LemmatizerConstants.EDIT_DISTANCE_WEIGHT  # Weight edit distance heavily
            else:
                # Fallback to simple character difference
                score += abs(len(word) - len(suggestion)) * LemmatizerConstants.LENGTH_DIFF_WEIGHT

            # 2. POS agreement bonus (if available)
            if stanza_upos:
                # Try to get Omorfi analysis for this suggestion
                analyses = self.analyze_direct(suggestion)
                if analyses:
                    for analysis, _ in analyses:
                        tags = self.parse_omorfi_tags(analysis)
                        if tags['pos'] and stanza_upos in OMORFI_POS_UD.get(tags['pos'], set()):
                            score -= LemmatizerConstants.POS_MATCH_BONUS_VOIKKO  # Strong bonus for POS match
                            break

            # 3. Lexicon frequency: check if in our training lexicon (prefer known words)
            if self.pos_aware_lexicon and suggestion.lower() in self.pos_aware_lexicon:
                score -= LemmatizerConstants.LEXICON_FREQUENCY_BONUS  # Bonus for being in training data

            # 4. Proper noun penalty (unless expected)
            if suggestion and suggestion[0].isupper():
                if stanza_upos == "PROPN":
                    score -= LemmatizerConstants.PROPER_NOUN_EXPECTED_BONUS  # Bonus if proper noun expected
                else:
                    score += LemmatizerConstants.PROPER_NOUN_NOT_EXPECTED_PENALTY  # Penalty if proper noun not expected

            # 5. Length similarity bonus
            len_diff = abs(len(word) - len(suggestion))
            if len_diff <= LemmatizerConstants.LENGTH_SIMILARITY_THRESHOLD:
                score -= LemmatizerConstants.LENGTH_SIMILARITY_BONUS  # Small bonus for similar length

            scored_suggestions.append((suggestion, score))

        # Sort by score (lower = better)
        scored_suggestions.sort(key=lambda x: x[1])
        return scored_suggestions

    # ========================================================================
    # Scoring Methods
    # ========================================================================

    def normalize_lemma(self, lemma: str) -> str:
        """
        Normalize lemma (remove trailing hyphens).

        Args:
            lemma: Raw lemma from Omorfi

        Returns:
            Normalized lemma without trailing hyphen

        Example:
            >>> normalize_lemma('talo-')
            'talo'
        """
        if lemma and lemma.endswith('-'):
            return lemma[:-1]
        return lemma

    def score_alternative(self, lemma: str, word: str, weight: float, analysis: str) -> float:
        """
        Score an alternative lemma (lower is better).

        Heuristics:
        1. Penalize stem forms (ending with -)
        2. Penalize short lemmas (likely compound first component)
        3. Prefer lemmas close to word length
        4. Small NOM bonus for nouns
        5. Penalize ADV forms when lemma==word
        6. Prefer infinitives for verbs
        7. Consider Omorfi weight

        Args:
            lemma: Candidate lemma
            word: Original word
            weight: Omorfi weight
            analysis: Full Omorfi analysis string

        Returns:
            Score (lower is better)
        """
        score = weight  # Start with Omorfi's weight

        # Penalize stem forms heavily
        if lemma.endswith('-'):
            score += LemmatizerConstants.STEM_PENALTY

        # Penalize very short lemmas (compound over-analysis)
        lemma_len = len(lemma.rstrip('-'))
        word_len = len(word)
        length_ratio = lemma_len / word_len if word_len > 0 else 1.0

        if length_ratio < LemmatizerConstants.SHORT_LEMMA_THRESHOLD_1:
            score += LemmatizerConstants.SHORT_LEMMA_PENALTY_1
        elif length_ratio < LemmatizerConstants.SHORT_LEMMA_THRESHOLD_2:
            score += LemmatizerConstants.SHORT_LEMMA_PENALTY_2

        # Prefer lemmas closer to word length
        length_diff = abs(lemma_len - word_len)
        if length_diff > LemmatizerConstants.LENGTH_DIFF_THRESHOLD:
            score += LemmatizerConstants.LENGTH_DIFF_PENALTY

        # Small NOM bonus for nouns
        if '[UPOS=NOUN]' in analysis or '[UPOS=PROPN]' in analysis:
            if '[CASE=NOM]' in analysis:
                if length_ratio >= LemmatizerConstants.SHORT_LEMMA_THRESHOLD_1:
                    score -= LemmatizerConstants.NOM_CASE_BONUS

        # Penalize ADV forms when lemma==word (lemma leakage fix)
        if '[UPOS=ADV]' in analysis and lemma.lower() == word.lower():
            score += LemmatizerConstants.ADV_LEMMA_LEAKAGE_PENALTY

        # Prefer infinitive forms for verbs
        if '[UPOS=VERB]' in analysis:
            if not (lemma.endswith('a') or lemma.endswith('ä')):
                score += LemmatizerConstants.VERB_NON_INFINITIVE_PENALTY

        return score

    def score_candidate_contextual(self, word: str, analysis: str, weight: float,
                                   stanza_upos: str, stanza_feats: str) -> float:
        """
        V17 Phase 9: Score candidate with rich morphological features (UFEATS).

        Scoring components:
        1. Base score (lemma quality, edit distance)
        2. POS agreement (strong signal: +6.0)
        3. V17 Phase 9 NEW: Rich morphological feature agreement
           - Case: +3.0
           - Number: +2.0
           - Tense: +2.0
           - VerbForm: +1.5
           - Person: +1.0
           - Mood: +1.0
        4. Proper noun bonus: +0.8

        Args:
            word: Original word
            analysis: Omorfi analysis string
            weight: Omorfi weight
            stanza_upos: Stanza universal POS tag
            stanza_feats: Stanza morphological features

        Returns:
            Score (higher = better)
        """
        # Start with inverse of base score (higher = better now)
        base_score = self.score_alternative(self.extract_lemma(analysis) or '', word, weight, analysis)
        score = LemmatizerConstants.BASE_SCORE_INVERSION - base_score

        tags = self.parse_omorfi_tags(analysis)

        # POS match (strong signal)
        if tags['pos']:
            if stanza_upos in OMORFI_POS_UD.get(tags['pos'], set()):
                score += LemmatizerConstants.POS_MATCH_BONUS

        # V17 Phase 9: Rich morphological feature agreement
        # Extract UFEATS from analysis (if available)
        candidate_features = self.get_ufeats_from_analysis(analysis, weight)

        if candidate_features:
            # Track that we're using morphological features
            self.morph_stats['features_extracted_successfully'] += 1

            # Parse Stanza features into dictionary
            stanza_feature_dict = {}
            if stanza_feats:
                for feat in stanza_feats.split("|"):
                    if "=" in feat:
                        key, value = feat.split("=", 1)
                        stanza_feature_dict[key] = value

            # Compute feature agreement bonuses
            feature_bonus = 0.0

            # Case agreement (strong signal)
            if 'Case' in candidate_features and 'Case' in stanza_feature_dict:
                if candidate_features['Case'] == stanza_feature_dict['Case']:
                    feature_bonus += LemmatizerConstants.FEATURE_CASE_BONUS
                    self.morph_stats['feature_similarity_computed'] += 1

            # Number agreement
            if 'Number' in candidate_features and 'Number' in stanza_feature_dict:
                if candidate_features['Number'] == stanza_feature_dict['Number']:
                    feature_bonus += LemmatizerConstants.FEATURE_NUMBER_BONUS
                    self.morph_stats['feature_similarity_computed'] += 1

            # Tense agreement (for verbs)
            if 'Tense' in candidate_features and 'Tense' in stanza_feature_dict:
                if candidate_features['Tense'] == stanza_feature_dict['Tense']:
                    feature_bonus += LemmatizerConstants.FEATURE_TENSE_BONUS
                    self.morph_stats['feature_similarity_computed'] += 1

            # VerbForm agreement
            if 'VerbForm' in candidate_features and 'VerbForm' in stanza_feature_dict:
                if candidate_features['VerbForm'] == stanza_feature_dict['VerbForm']:
                    feature_bonus += LemmatizerConstants.FEATURE_VERBFORM_BONUS
                    self.morph_stats['feature_similarity_computed'] += 1

            # Person agreement
            if 'Person' in candidate_features and 'Person' in stanza_feature_dict:
                if candidate_features['Person'] == stanza_feature_dict['Person']:
                    feature_bonus += LemmatizerConstants.FEATURE_PERSON_BONUS
                    self.morph_stats['feature_similarity_computed'] += 1

            # Mood agreement
            if 'Mood' in candidate_features and 'Mood' in stanza_feature_dict:
                if candidate_features['Mood'] == stanza_feature_dict['Mood']:
                    feature_bonus += LemmatizerConstants.FEATURE_MOOD_BONUS
                    self.morph_stats['feature_similarity_computed'] += 1

            if feature_bonus > 0.0:
                score += feature_bonus
                self.morph_stats['feature_bonus_applied'] += 1
                self.morph_stats['total_feature_bonus'] += feature_bonus
                if feature_bonus > self.morph_stats['max_feature_bonus']:
                    self.morph_stats['max_feature_bonus'] = feature_bonus

        # Proper noun preference for capitalized words
        if word[0].isupper() and stanza_upos == "PROPN" and tags['pos'] == 'N':
            score += LemmatizerConstants.PROPER_NOUN_BONUS

        return score

    # ========================================================================
    # Phase 12: Compound Classification Helper Methods
    # ========================================================================

    def has_compound_boundary(self, analysis: str) -> bool:
        """Check if analysis contains [BOUNDARY=COMPOUND] marker."""
        return '[BOUNDARY=COMPOUND]' in analysis

    def has_possessive_marker(self, analysis: str) -> bool:
        """Check if analysis contains [POSS=...] marker."""
        return '[POSS=' in analysis

    def select_best_alternative(self, word: str, analyses: List[Tuple[str, float]]) -> Tuple[str, float, str, List[Dict]]:
        """
        Select best lemma from alternatives using smart heuristics.

        V17 Phase 9 REFACTORED: Now uses morphological features when multiple analyses exist!

        CRITICAL: This method will be enhanced in Phase 12 to integrate compound word analysis.
        The compound analyzer will provide additional candidates that will be scored alongside
        the standard Omorfi analyses using the same morphological feature awareness.

        Args:
            word: Original word
            analyses: List of (analysis, weight) tuples from Omorfi

        Returns:
            Tuple of (best_lemma, best_weight, best_analysis, alternatives_list)
            Returns (None, None, None, []) if no analyses available
        """
        if not analyses:
            return None, None, None, []

        # V17 Phase 9 TRACKING: Count words with multiple analyses
        if len(analyses) > 1:
            self.morph_stats['words_with_multiple_analyses'] += 1

        # V17 Phase 9: Extract morphological features from input word's first analysis
        # This represents the actual morphological context of the word form
        word_features = None
        if len(analyses) > 1:  # Only if there are multiple alternatives to choose from
            word_features = self.get_ufeats_from_analysis(analyses[0][0], analyses[0][1])
            if word_features:
                self.morph_stats['features_extracted_successfully'] += 1

        # Extract all lemmas with scores
        candidates = []
        for analysis, weight in analyses[:5]:  # Consider top 5
            lemma = self.extract_lemma(analysis)
            if lemma:
                # Base score from existing heuristics
                base_score = self.score_alternative(lemma, word, weight, analysis)

                # V17 Phase 9: Apply morphological feature similarity bonus
                feature_bonus = 0.0
                if word_features:
                    # Get features for this candidate analysis
                    candidate_features = self.get_ufeats_from_analysis(analysis, weight)
                    if candidate_features:
                        self.morph_stats['feature_similarity_computed'] += 1

                        # Compute similarity and apply as bonus (subtract from score)
                        feature_bonus = self.compute_feature_similarity(word_features, candidate_features)

                        # V17 Phase 9 TRACKING: Track feature bonus statistics
                        if feature_bonus > 0.0:
                            self.morph_stats['feature_bonus_applied'] += 1
                            self.morph_stats['total_feature_bonus'] += feature_bonus
                            if feature_bonus > self.morph_stats['max_feature_bonus']:
                                self.morph_stats['max_feature_bonus'] = feature_bonus

                # Lower score is better, so subtract feature bonus
                adjusted_score = base_score - (feature_bonus * LemmatizerConstants.FEATURE_BONUS_SCALE)  # Scale feature bonus

                candidates.append({
                    'lemma': lemma,
                    'normalized_lemma': self.normalize_lemma(lemma),
                    'analysis': analysis,
                    'weight': weight,
                    'score': adjusted_score,
                    'base_score': base_score,
                    'feature_bonus': feature_bonus
                })

        if not candidates:
            return None, None, None, []

        # V17 Phase 9 TRACKING: Check if features changed the selection
        # Sort by base score (without features) to see what would have won
        candidates_by_base = sorted(candidates, key=lambda x: x['base_score'])
        winner_without_features = candidates_by_base[0]['normalized_lemma']

        # Sort by adjusted score (with features) - this is the actual selection
        candidates.sort(key=lambda x: x['score'])
        winner_with_features = candidates[0]['normalized_lemma']

        # Track if features changed the selection
        if winner_without_features != winner_with_features:
            self.morph_stats['selection_changed_by_features'] += 1

        # Best candidate (with features)
        best = candidates[0]

        # ========================================================================
        # V17 Phase 12: Compound Classification Integration
        # ========================================================================
        #
        # Three-stage approach:
        # Stage 1: Filter analyses by type (prefer possessive → regular → compound)
        # Stage 2: Select best from chosen group (already done above)
        # Stage 3: Process compound if needed (reconstruct true, reject false)
        #
        # Key insight: If best candidate has [BOUNDARY=COMPOUND], check if it's
        # a true compound or false positive (possessive suffix misidentified)
        # ========================================================================

        if self.compound_classifier and self.has_compound_boundary(best['analysis']):
            self.compound_stats['total_compounds_detected'] += 1

            # Check if this is a true compound or false positive
            classification = self.compound_classifier.classify_compound(
                best['analysis'],
                original_word=word
            )

            if classification['is_true_compound']:
                # True compound detected - use reconstructed lemma if available
                if classification['reconstructed']:
                    self.compound_stats['true_compounds_reconstructed'] += 1
                    best['normalized_lemma'] = classification['reconstructed']
                    # Note: We keep the original analysis for documentation
            else:
                # False compound (likely possessive) - try to fallback to non-compound analysis
                self.compound_stats['false_compounds_rejected'] += 1

                # Check if it's a possessive with compound marker
                if self.has_possessive_marker(best['analysis']):
                    self.compound_stats['possessive_with_compound_marker'] += 1

                # Look for non-compound alternatives in the candidate list
                non_compound_candidates = [c for c in candidates if not self.has_compound_boundary(c['analysis'])]

                if non_compound_candidates:
                    # Use first non-compound alternative
                    self.compound_stats['fallback_to_non_compound'] += 1
                    best = non_compound_candidates[0]
                # else: Keep original compound analysis (no better alternative)

        # Build alternatives list (excluding best)
        alternatives = [
            {
                'lemma': c['normalized_lemma'],
                'analysis': c['analysis'],
                'weight': c['weight'],
                'score': c['score'],
                'feature_bonus': c.get('feature_bonus', 0.0)
            }
            for c in candidates[1:]
        ]

        return best['normalized_lemma'], best['weight'], best['analysis'], alternatives

    # ========================================================================
    # Morphological Feature Methods
    # ========================================================================

    def get_ufeats_from_analysis(self, analysis_str: str, weight: float = 0.0) -> Optional[Dict]:
        """
        Extract Universal Features from HFST analysis string.

        Delegates to MorphologicalFeatureExtractor for actual extraction.

        Args:
            analysis_str: Omorfi analysis string
            weight: Omorfi weight (unused, for compatibility)

        Returns:
            Dictionary of Universal Dependencies features or None
        """
        return self.feature_extractor.get_ufeats_from_analysis(analysis_str, weight)

    def compute_feature_similarity(self, feats1: Optional[Dict], feats2: Optional[Dict]) -> float:
        """
        Compute morphological feature similarity.

        Delegates to FeatureSimilarityScorer for actual computation.

        Args:
            feats1: First feature dictionary
            feats2: Second feature dictionary

        Returns:
            Similarity score (higher = more similar)
        """
        return self.similarity_scorer.compute_similarity(feats1, feats2)

    # =========================================================================
    # HIGH-LEVEL LEMMATIZATION METHODS (Tier implementations)
    # =========================================================================

    def lemmatize_direct(self, word: str, strict_proper_noun_check: bool = False) -> Optional[Dict]:
        """
        Direct lemmatization using Omorfi analyzer with smart alternative selection

        Args:
            word: Word to lemmatize
            strict_proper_noun_check: If True, reject 1-2 char lemmas for capitalized words

        Returns:
            Lemmatization result or None
        """
        analyses = self.analyze_direct(word)

        if not analyses:
            return None

        # Use smart alternative selection
        lemma, weight, best_analysis, alternatives = self.select_best_alternative(word, analyses)

        if not lemma:
            return None

        # PROPER-NOUN OVERRIDE: Only in Tier 1 (strict mode)
        if strict_proper_noun_check and word[:1].isupper() and len(lemma) <= 2:
            return None

        return {
            'word': word,
            'lemma': lemma,
            'analysis': best_analysis,
            'weight': weight,
            'method': 'omorfi_direct',
            'confidence': 'high',
            'alternatives': alternatives
        }

    def lemmatize_guesser(self, word: str) -> Optional[Dict]:
        """
        Guesser fallback for unknown words
        """
        guesses = self.analyze_guesser(word)

        if not guesses:
            return None

        # Use smart alternative selection
        lemma, weight, best_analysis, alternatives = self.select_best_alternative(word, guesses)

        if not lemma or '+?' in best_analysis:
            return None

        self.guesser_effective = True

        return {
            'word': word,
            'lemma': lemma,
            'analysis': best_analysis,
            'weight': weight,
            'method': 'omorfi_guesser',
            'confidence': 'medium',
            'alternatives': alternatives
        }

    def lemmatize_with_voikko(self, word: str) -> Optional[Dict]:
        """
        Tier 3: Voikko spelling normalization with smart alternative selection
        """
        suggestions = self.voikko_suggest(word, max_n=10)

        if not suggestions:
            return None

        for suggestion in suggestions:
            result = self.lemmatize_direct(suggestion, strict_proper_noun_check=False)
            if result:
                result['method'] = 'voikko_normalized'
                result['original_word'] = word
                result['normalized_word'] = suggestion
                result['confidence'] = 'medium'
                return result

        return None

    def lemmatize_with_voikko_enhanced(self, word: str) -> Optional[Dict]:
        """
        V17 Phase 3: Enhanced Voikko with aggressive suggestions (up to 30)

        This tier tries more suggestions than regular Voikko (Tier 3) to catch
        more unknown words before falling back to identity_fallback.

        Target: Reduce identity_fallback from 291 words (19.8%) to <150 words
        """
        # Try up to 30 suggestions (vs 10 in regular Voikko)
        suggestions = self.voikko_suggest(word, max_n=30)

        if not suggestions:
            return None

        # Try ALL suggestions (aggressive mode)
        for suggestion in suggestions:
            result = self.lemmatize_direct(suggestion, strict_proper_noun_check=False)
            if result:
                result['method'] = 'enhanced_voikko'
                result['original_word'] = word
                result['normalized_word'] = suggestion
                result['confidence'] = 'low'  # Lower confidence than regular Voikko
                return result

        return None

    def lemmatize_with_fuzzy_lexicon_morphological(self, word: str, stanza_upos: str = None) -> Optional[Dict]:
        """
        V17 Phase 9: Morphology-aware fuzzy lexicon lookup

        Enhances Phase 7 fuzzy matching with morphological feature awareness.
        Uses Omorfi UFEATS to rank candidates by morphological plausibility.

        Process:
        1. Get Omorfi analysis of dialectal word form → extract UFEATS
        2. Find fuzzy candidates from lexicon (threshold 2.0)
        3. Get Omorfi analysis of each candidate → extract UFEATS
        4. Compute feature similarity bonus
        5. Re-rank by: edit_distance - feature_similarity_bonus
        6. Return best morphologically plausible match

        Args:
            word: The dialectal word form to lemmatize
            stanza_upos: Optional POS tag from Stanza for filtering

        Returns:
            Lemmatization result with morphological confidence, or None if no match
        """
        if not (self.dialectal_normalizer and self.pos_aware_lexicon):
            return None

        # Get Omorfi analysis of input word to extract morphological features
        word_analyses = self.analyze_direct(word)
        word_features = None

        if word_analyses:
            # Use first (best weight) analysis for feature extraction
            word_features = self.get_ufeats_from_analysis(word_analyses[0][0], word_analyses[0][1])

        word_lower = word.lower()
        candidates = []

        # Find fuzzy matches in lexicon
        for lexicon_word, pos_dict in self.pos_aware_lexicon.items():
            distance = self.dialectal_normalizer.weighted_edit_distance(word_lower, lexicon_word)

            if distance <= LemmatizerConstants.FUZZY_THRESHOLD_STANDARD:  # Standard fuzzy threshold
                for pos, entry in pos_dict.items():
                    # Filter by POS if available
                    if stanza_upos and pos != stanza_upos:
                        continue

                    lemma = entry['lemma']
                    source = entry.get('source', 'unknown')
                    tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                    # Get morphological features of candidate lemma
                    candidate_features = None
                    candidate_analyses = self.analyze_direct(lemma)
                    if candidate_analyses:
                        candidate_features = self.get_ufeats_from_analysis(
                            candidate_analyses[0][0],
                            candidate_analyses[0][1]
                        )

                    # Compute morphological similarity bonus
                    feature_bonus = 0.0
                    if word_features and candidate_features:
                        feature_bonus = self.compute_feature_similarity(word_features, candidate_features)

                    # Adjusted score: lower is better
                    adjusted_distance = distance - feature_bonus

                    candidates.append((
                        lemma,
                        distance,
                        adjusted_distance,
                        feature_bonus,
                        lexicon_word,
                        pos,
                        tier
                    ))

        if candidates:
            # Sort by adjusted distance (morphologically aware), then tier priority
            tier_priority = {'tier1': 0, 'tier2': 1, 'tier3': 2}
            best_lemma, raw_dist, adj_dist, feat_bonus, matched_word, matched_pos, matched_tier = min(
                candidates,
                key=lambda x: (x[2], tier_priority[x[6]])  # Sort by adjusted_distance, then tier
            )

            # Determine confidence based on feature bonus
            if feat_bonus >= LemmatizerConstants.MORPH_CONFIDENCE_HIGH_THRESHOLD:
                confidence = 'medium-high'  # Strong morphological agreement
            elif feat_bonus >= LemmatizerConstants.MORPH_CONFIDENCE_MEDIUM_THRESHOLD:
                confidence = 'medium'  # Moderate morphological agreement
            else:
                confidence = 'medium-low'  # Weak or no morphological agreement

            return {
                'word': word,
                'lemma': best_lemma,
                'analysis': f'[FUZZY_MORPH:{word}→{matched_word}({matched_pos})={best_lemma}@{raw_dist:.2f}-{feat_bonus:.2f}={adj_dist:.2f}]',
                'weight': None,
                'method': 'fuzzy_lexicon_morphological',
                'confidence': confidence,
                'alternatives': [],
                'morphological_bonus': feat_bonus,
                'raw_distance': raw_dist,
                'adjusted_distance': adj_dist
            }

        return None

    def lemmatize_with_fuzzy_lexicon_aggressive(self, word: str, stanza_upos: str = None) -> Optional[Dict]:
        """
        V17 Phase 8: Aggressive fuzzy lexicon lookup with suffix stripping and normalization

        This is a more aggressive version of lemmatize_with_fuzzy_lexicon() designed to catch
        dialectal forms with heavy suffix variations and character substitutions.

        Key differences from regular fuzzy_lexicon:
        - Relaxed distance threshold: 3.5 (vs 2.0)
        - Suffix stripping: Removes common dialectal suffixes before matching
        - Character normalization: Handles dialectal spelling variants (w→v, c→k, etc.)
        - POS-optional: Falls back to POS-agnostic matching if POS-aware fails

        Target: Reduce identity_fallback from 50 words to ~10-20 (60-80% reduction)

        Args:
            word: The word to lemmatize
            stanza_upos: Optional POS tag from Stanza for filtering

        Returns:
            Lemmatization result if match found, None otherwise
        """
        if not (self.dialectal_normalizer and self.pos_aware_lexicon):
            return None

        word_lower = word.lower()

        # Common dialectal suffixes to strip (ordered by specificity)
        dialectal_suffixes = [
            'ntkaane', 'kaane', 'staan', 'sessa', 'mast', 'ille', 'ttaa', 'taa', 'lla', 's'
        ]

        # Try with and without suffix stripping
        word_variants = [word_lower]

        # Generate variants by stripping suffixes
        for suffix in dialectal_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                stripped = word_lower[:-len(suffix)]
                if stripped not in word_variants:
                    word_variants.append(stripped)

        # Generate normalized variants (dialectal character substitutions)
        normalized_variants = []
        for variant in word_variants:
            # Common Finnish dialectal character variations
            normalized = variant.replace('w', 'v').replace('c', 'k')
            # Context-dependent j→i (only in certain positions)
            if 'j' in normalized and len(normalized) > 3:
                # Try j→i substitution for compound boundaries
                normalized_j_i = normalized.replace('colmj', 'colmi').replace('nelj', 'neli')
                if normalized_j_i != normalized:
                    normalized_variants.append(normalized_j_i)
            if normalized != variant:
                normalized_variants.append(normalized)

        # Combine all variants
        all_variants = word_variants + normalized_variants

        # Try POS-aware matching first
        candidates = []
        for test_variant in all_variants:
            for lexicon_word, pos_dict in self.pos_aware_lexicon.items():
                distance = self.dialectal_normalizer.weighted_edit_distance(test_variant, lexicon_word)

                if distance <= LemmatizerConstants.FUZZY_THRESHOLD_AGGRESSIVE:  # Relaxed threshold
                    for pos, entry in pos_dict.items():
                        # Filter by POS if available
                        if stanza_upos and pos != stanza_upos:
                            continue

                        lemma = entry['lemma']
                        source = entry.get('source', 'unknown')
                        tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                        candidates.append((lemma, distance, lexicon_word, pos, tier, test_variant))

        # If POS-aware matching failed and we have a POS tag, try POS-agnostic
        if not candidates and stanza_upos:
            for test_variant in all_variants:
                for lexicon_word, pos_dict in self.pos_aware_lexicon.items():
                    distance = self.dialectal_normalizer.weighted_edit_distance(test_variant, lexicon_word)

                    if distance <= LemmatizerConstants.FUZZY_THRESHOLD_AGGRESSIVE:
                        for pos, entry in pos_dict.items():
                            # No POS filtering
                            lemma = entry['lemma']
                            source = entry.get('source', 'unknown')
                            tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                            candidates.append((lemma, distance, lexicon_word, pos, tier, test_variant))

        if candidates:
            # Return best match (lowest distance, prefer tier1 > tier2 > tier3)
            tier_priority = {'tier1': 0, 'tier2': 1, 'tier3': 2}
            best_lemma, best_dist, matched_word, matched_pos, matched_tier, matched_variant = min(
                candidates,
                key=lambda x: (x[1], tier_priority[x[4]])  # Sort by distance, then tier
            )

            return {
                'word': word,
                'lemma': best_lemma,
                'analysis': f'[FUZZY_AGGRESSIVE:{word}→{matched_variant}→{matched_word}({matched_pos})={best_lemma}@{best_dist:.2f}]',
                'weight': None,
                'method': 'fuzzy_lexicon_aggressive',
                'confidence': 'low',  # Lower than regular fuzzy due to aggressive matching
                'alternatives': []
            }

        return None

    def lemmatize_with_dialectal_dict(self, word: str, stanza_upos: Optional[str] = None) -> Optional[Dict]:
        """
        Tier 7: Finnish Dialects Dictionary (SMS) lookup

        Searches 19,385 validated dialectal variants with POS filtering.

        Args:
            word: The word to lemmatize
            stanza_upos: Optional POS tag from Stanza for filtering

        Returns:
            Lemmatization result if match found, None otherwise
        """
        if not self.dialectal_dict:
            return None

        variant_index = self.dialectal_dict.get('variant_to_lemma', {})
        word_lower = word.lower()

        if word_lower not in variant_index:
            return None

        candidates = variant_index[word_lower]

        # Filter by POS if available (Stanza context)
        if stanza_upos and candidates:
            filtered = [c for c in candidates if c.get('pos') == stanza_upos]
            if filtered:
                candidates = filtered

        if not candidates:
            return None

        # Select best candidate (highest confidence)
        best = max(candidates, key=lambda x: x.get('confidence', 0.0))

        return {
            'word': word,
            'lemma': best['lemma'],
            'analysis': f'[DIALECTAL_DICT:{word}→{best["lemma"]}][POS:{best.get("pos")}]',
            'weight': None,
            'method': 'dialectal_dictionary',
            'confidence': 'high',
            'alternatives': []
        }
