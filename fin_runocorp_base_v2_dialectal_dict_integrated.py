#!/usr/bin/env python3
"""
Omorfi HFST Lemmatizer with Voikko Normalization - V17 PHASE 10 (Finnish Dialects Dictionary Integration)

NEW IN V17 Phase 10:
1. **Finnish Dialects Dictionary (SMS) Integration**: Added as Tier 7
   - 19,385 validated dialectal variants from Suomen Murteiden Sanakirja
   - POS-aware filtering using Stanza context
   - High confidence (0.9) for validated dictionary entries
   - Expected improvement: 58.8% → 62-65% accuracy (+150-250 dialectal variants)

V17 Phase 9 Features (retained):
1. **CRITICAL FIX**: Morphological features moved to Tier 1 where they actually help!
   - OLD: Features in Tier 6 (fuzzy lexicon) → 0 uses (wrong tier!)
   - NEW: Features in Tier 1 (select_best_alternative) → 387+ potential uses ✓
2. Enhanced candidate selection with morphological awareness:
   - When multiple Omorfi analyses exist, use UFEATS to pick best one
   - Case/Number/Tense/VerbForm agreement scoring (0.1-0.3 per feature)
   - Feature bonus scaled 3× to influence selection
3. Custom HFST parser (bypasses omorfi.analysis.Analysis.fromomor()):
   - Direct regex parsing of [KEY=VALUE] tags
   - Maps HFST tags → Universal Features
   - Works with our HFST analyzer output format
4. Target: Use morphological features where multiple candidates exist
   - Tier 1 (omorfi_contextual): 387 words (26% of test set) ← PRIMARY TARGET
   - Tier 3 (voikko_omorfi): 130 words (9% of test set) ← FUTURE ENHANCEMENT
5. Foundation for Tier 3 Voikko ranking enhancement

V17 Phase 8 Features (retained):
1. Aggressive fuzzy lexicon matching (Tier 2):
   - Relaxed threshold: 3.5, suffix stripping, character normalization
   - Identity fallback reduction: 50 → 1 word (98% success!)

V17 Phase 7 Features (retained):
1. Multi-criteria Voikko ranking: edit distance, POS, lexicon frequency, proper nouns

V17 Phase 4 Features (retained):
1. Sukija dictionary: Specialized dictionary for historical/dialectal Finnish texts
2. Dual-dictionary approach: Combines Sukija + Old Finnish suggestions
3. Maximum coverage: Merges results from both dictionaries for best accuracy

V17 Phase 3 Features (retained):
1. Enhanced Voikko suggestions: Try up to 30 suggestions (vs 5 in Phase 2)
2. Aggressive Voikko tier: New tier before identity_fallback to catch more unknowns
3. Contextual analysis: Try ALL Voikko suggestions, not just top 3

V17 Phase 2 Features (retained):
1. Old Finnish dictionary: VANHAT_MUODOT=yes for archaic forms (Phase 2)
2. Python libvoikko: Replaced subprocess calls with library integration (Phase 2)
3. Custom dictionary path: ~/.voikko/5/5 with archaic forms enabled (Phase 2)
4. Long s normalization: ſ → s (Phase 1 - fixes 4 words in test set)
5. POS='X' fallback: Uses unclassified POS from training (Phase 1 - fixes 7 words)
6. Identity fallback: Return word itself as lemma when all else fails (Phase 1.5)

BASED ON V16 (Hybrid Gold Standard Lexicon):
1. Three-tier POS-aware lexicon:
   - Tier 1: Manual gold standard annotations (100% trusted)
   - Tier 2: Omorfi 100% unambiguous patterns (non-overlapping)
   - Tier 3: Production accumulation (reserved for future)
2. Ambiguous pattern tracking - avoids caching genuinely ambiguous words
3. Falls back to V14 contextual analysis for non-cached/ambiguous cases
4. Ultimate fallback: Identity lemma (word = lemma)

IMPACT:
- V16: 53.1% accuracy, 289 unknown words
- V17 Phase 1: 53.3% accuracy (+0.2pp), 289 unknown words
- V17 Phase 1.5: 54.2% accuracy (+0.9pp), 276 unknown words
- V17 Phase 4: 54.2% accuracy, 289 identity_fallback (baseline)
- V17 Phase 7: 58.5% accuracy (+0.2pp), Voikko ranking
- V17 Phase 8: 58.5% accuracy, 1 identity_fallback (98% reduction)
- V17 Phase 9: 58.8% accuracy (baseline for Phase 10)
- V17 Phase 10: Expected 62-65% accuracy (+3-6pp) ← CURRENT

Usage:
    python3 fin_runocorp_base_v2_dialectal_dict_integrated.py < input.txt
"""

import hfst
import os
import sys
import re
import subprocess
import shlex
import json
from typing import List, Dict, Tuple, Optional, Any, Set, Union

# Import Omorfi Analysis class for morphological feature extraction
try:
    from omorfi.analysis import Analysis
    OMORFI_ANALYSIS_AVAILABLE = True
except ImportError:
    OMORFI_ANALYSIS_AVAILABLE = False
    print("⚠ omorfi.analysis not available - morphological features disabled", file=sys.stderr)

# Import libvoikko for Old Finnish dictionary
try:
    import libvoikko
    LIBVOIKKO_AVAILABLE = True
except ImportError:
    LIBVOIKKO_AVAILABLE = False
    print("⚠ libvoikko not available - falling back to voikkospell subprocess", file=sys.stderr)

VERSION = "v17_phase10_dialectal_dict"
print(f"[omorfi_v17_phase10] Loaded {VERSION}", file=sys.stderr)

# Check for Stanza
try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False
    print("⚠ Stanza not available - contextual disambiguation disabled", file=sys.stderr)

# Import dialectal normalizer for confusion-weighted edit distance
from dialectal_normalizer import DialectalNormalizer

# Exception lexicon for stubborn dialectal forms (currently empty, can be populated)
EXCEPTION_LEXICON = {}

# Omorfi POS to UD POS mapping
OMORFI_POS_UD = {
    "N": {"NOUN", "PROPN"},
    "V": {"VERB", "AUX"},
    "A": {"ADJ"},
    "ADV": {"ADV"},
    "PRON": {"PRON", "DET"},
    "NUM": {"NUM"},
    "ADP": {"ADP"},
    "PART": {"PART"},
    "INTJ": {"INTJ"},
    "CCONJ": {"CCONJ"},
    "SCONJ": {"SCONJ"},
}

# Omorfi CASE to UD features mapping
OMORFI_CASE_UD = {
    "NOM": "Nom", "GEN": "Gen", "PAR": "Par", "INE": "Ine", "ELA": "Ela",
    "ILL": "Ill", "ADE": "Ade", "ABL": "Abl", "ALL": "All", "ESS": "Ess",
    "TRA": "Tra", "ABE": "Abe", "COM": "Com", "INS": "Ins"
}


# ============================================================================
# Configuration Classes
# ============================================================================

class LemmatizerConfig:
    """Configuration for Finnish dialectal runosong lemmatizer.

    Centralizes all configuration parameters that were previously scattered
    throughout the __init__ method.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        voikko_path: Optional[str] = None,
        lang: str = 'fi',
        lexicon_path: str = 'selftraining_lexicon_v16_min1.json',
        dialectal_dict_path: str = 'sms_dialectal_index_v4_final.json',
        max_suggestions_standard: int = 10,
        max_suggestions_enhanced: int = 30,
        enable_stanza: bool = True,
        enable_voikko: bool = True,
        enable_omorfi: bool = True
    ) -> None:
        # HFST/Omorfi configuration
        self.model_dir = model_dir or os.path.expanduser('~/.omorfi')
        self.analyzer_path = os.path.join(self.model_dir, 'omorfi.describe.hfst')
        self.guesser_path = os.path.join(self.model_dir, 'omorfi-guesser.hfst')

        # Voikko configuration
        self.voikko_path = voikko_path
        self.voikko_sukija_path = os.path.expanduser('~/.voikko/sukija/5/mor-sukija')
        self.voikko_old_finnish_path = os.path.expanduser('~/.voikko/5/5')

        # Language and lexicon
        self.lang = lang
        self.lexicon_path = lexicon_path
        self.dialectal_dict_path = dialectal_dict_path

        # Suggestion limits
        self.max_suggestions_standard = max_suggestions_standard
        self.max_suggestions_enhanced = max_suggestions_enhanced

        # Feature flags
        self.enable_stanza = enable_stanza
        self.enable_voikko = enable_voikko
        self.enable_omorfi = enable_omorfi

    def __repr__(self) -> str:
        return (
            f"LemmatizerConfig(model_dir={self.model_dir!r}, "
            f"lang={self.lang!r}, lexicon={self.lexicon_path!r})"
        )


class MorphologicalFeatureExtractor:
    """Extracts Universal Features (UFEATS) from HFST analysis strings.

    Converts HFST morphological tags to Universal Dependencies format.
    Based on get_ufeats_from_analysis method from original lines 1149-1272.
    """

    # HFST to UD mappings
    HFST_NUMBER_MAP = {
        'SG': 'Sing',
        'PL': 'Plur'
    }

    HFST_CASE_MAP = {
        'NOM': 'Nom',
        'GEN': 'Gen',
        'PAR': 'Par',
        'ILL': 'Ill',
        'INE': 'Ine',
        'ELA': 'Ela',
        'ADE': 'Ade',
        'ABL': 'Abl',
        'ALL': 'All',
        'ESS': 'Ess',
        'TRA': 'Tra',
        'INS': 'Ins',
        'ABE': 'Abe',
        'COM': 'Com'
    }

    HFST_TENSE_MAP = {
        'PAST': 'Past',
        'PRES': 'Pres'
    }

    HFST_MOOD_MAP = {
        'COND': 'Cnd',
        'IMPV': 'Imp',
        'POTN': 'Pot'
    }

    HFST_PERSON_MAP = {
        'SG1': '1',
        'SG2': '2',
        'SG3': '3',
        'PL1': '1',
        'PL2': '2',
        'PL3': '3'
    }

    HFST_VOICE_MAP = {
        'PSS': 'Pass'
    }

    HFST_POLARITY_MAP = {
        'NEG': 'Neg'
    }

    HFST_NUMTYPE_MAP = {
        'ORD': 'Ord',
        'CARD': 'Card'
    }

    HFST_VERBFORM_MAP = {
        'INF1': 'Inf',
        'INF2': 'Inf',
        'INF3': 'Inf',
        'INF4': 'Inf',
        'INF5': 'Inf',
        'PART': 'Part',
        'PCP1': 'Part',
        'PCP2': 'Part',
        'ACTV': 'Part',
        'PASS': 'Part'
    }

    def __init__(self) -> None:
        """Initialize feature extractor with HFST→UD mappings."""
        pass

    def get_ufeats_from_analysis(self, analysis_str: str, weight: float = 0.0) -> Optional[Dict[str, str]]:
        """Extract Universal Features from HFST analysis string.

        Args:
            analysis_str: HFST analysis like "[WORD=kissa][POS=NOUN][NUM=SG][CASE=NOM]"
            weight: Analysis weight (for confidence tracking)

        Returns:
            dict: Universal Features like {'Case': 'Nom', 'Number': 'Sing'} or None if parsing fails
        """
        if not analysis_str:
            return None

        try:
            # Parse [KEY=VALUE] tags from HFST string
            tags = re.findall(r'\[([A-Z_]+)=([^\]]+)\]', analysis_str)

            if not tags:
                return None

            ufeats = {}

            for key, value in tags:
                # Number
                if key == 'NUM':
                    if value == 'SG':
                        ufeats['Number'] = 'Sing'
                    elif value == 'PL':
                        ufeats['Number'] = 'Plur'

                # Case (capitalize first letter)
                elif key == 'CASE':
                    ufeats['Case'] = value.capitalize()

                # Tense
                elif key == 'TENSE':
                    if value == 'PAST':
                        ufeats['Tense'] = 'Past'
                    elif value == 'PRES':
                        ufeats['Tense'] = 'Pres'

                # Mood
                elif key == 'MOOD':
                    if value == 'INDV':
                        ufeats['Mood'] = 'Ind'
                    elif value == 'COND':
                        ufeats['Mood'] = 'Cond'
                    elif value == 'IMPV':
                        ufeats['Mood'] = 'Imp'
                    elif value == 'POTN':
                        ufeats['Mood'] = 'Pot'

                # Voice
                elif key == 'VOICE':
                    if value == 'ACT':
                        ufeats['Voice'] = 'Act'
                    elif value == 'PASS':
                        ufeats['Voice'] = 'Pass'

                # Person (includes number)
                elif key == 'PERS':
                    if value in ['SG1', 'SG2', 'SG3']:
                        ufeats['Person'] = value[-1]  # '1', '2', '3'
                        ufeats['Number'] = 'Sing'
                    elif value in ['PL1', 'PL2', 'PL3']:
                        ufeats['Person'] = value[-1]
                        ufeats['Number'] = 'Plur'

                # Infinitive
                elif key == 'INF':
                    ufeats['VerbForm'] = 'Inf'
                    if value == 'A':
                        ufeats['InfForm'] = '1'  # A-infinitive
                    elif value == 'E':
                        ufeats['InfForm'] = '2'  # E-infinitive
                    elif value == 'MA':
                        ufeats['InfForm'] = '3'  # MA-infinitive
                    elif value == 'MINEN':
                        ufeats['InfForm'] = '4'  # MINEN-infinitive

                # Degree (for adjectives)
                elif key == 'CMP':
                    if value == 'POS':
                        ufeats['Degree'] = 'Pos'
                    elif value == 'CMP':
                        ufeats['Degree'] = 'Cmp'
                    elif value == 'SUP':
                        ufeats['Degree'] = 'Sup'

                # Participle forms
                elif key == 'PCP':
                    ufeats['VerbForm'] = 'Part'
                    if value == 'PRES_ACT' or value == 'VA':
                        ufeats['Tense'] = 'Pres'
                        ufeats['Voice'] = 'Act'
                    elif value == 'PAST_ACT' or value == 'NUT':
                        ufeats['Tense'] = 'Past'
                        ufeats['Voice'] = 'Act'
                    elif value == 'PRES_PASS':
                        ufeats['Tense'] = 'Pres'
                        ufeats['Voice'] = 'Pass'
                    elif value == 'PAST_PASS':
                        ufeats['Tense'] = 'Past'
                        ufeats['Voice'] = 'Pass'

            return ufeats if ufeats else None

        except Exception as e:
            # Silently fail on parsing errors
            return None


class LemmatizerConstants:
    """Central repository for all scoring and threshold constants.

    Externalizes magic numbers for better maintainability and tuning.
    """

    # === Scoring Constants (score_alternative) ===
    STEM_PENALTY = 10.0
    SHORT_LEMMA_THRESHOLD_1 = 0.5
    SHORT_LEMMA_PENALTY_1 = 5.0
    SHORT_LEMMA_THRESHOLD_2 = 0.7
    SHORT_LEMMA_PENALTY_2 = 2.0
    LENGTH_DIFF_THRESHOLD = 5
    LENGTH_DIFF_PENALTY = 1.0
    NOM_CASE_BONUS = 0.8
    ADV_LEMMA_LEAKAGE_PENALTY = 2.0
    VERB_NON_INFINITIVE_PENALTY = 3.0

    # === Contextual Scoring Constants (score_candidate_contextual) ===
    BASE_SCORE_INVERSION = 10.0
    POS_MATCH_BONUS = 6.0
    FEATURE_CASE_BONUS = 3.0
    FEATURE_NUMBER_BONUS = 2.0
    FEATURE_TENSE_BONUS = 2.0
    FEATURE_VERBFORM_BONUS = 1.5
    FEATURE_PERSON_BONUS = 1.0
    FEATURE_MOOD_BONUS = 1.0
    PROPER_NOUN_BONUS = 0.8

    # === Voikko Ranking Constants (rank_voikko_suggestions) ===
    EDIT_DISTANCE_WEIGHT = 2.0
    LENGTH_DIFF_WEIGHT = 0.5
    POS_MATCH_BONUS_VOIKKO = 1.5
    LEXICON_FREQUENCY_BONUS = 0.8
    PROPER_NOUN_EXPECTED_BONUS = 0.5
    PROPER_NOUN_NOT_EXPECTED_PENALTY = 1.0
    LENGTH_SIMILARITY_THRESHOLD = 2
    LENGTH_SIMILARITY_BONUS = 0.3

    # === Fuzzy Matching Thresholds ===
    FUZZY_THRESHOLD_STANDARD = 2.0
    FUZZY_THRESHOLD_AGGRESSIVE = 3.5
    FEATURE_BONUS_SCALE = 3.0  # Scaling factor for morphological feature bonuses
    MORPH_CONFIDENCE_HIGH_THRESHOLD = 0.5
    MORPH_CONFIDENCE_MEDIUM_THRESHOLD = 0.3


class FeatureSimilarityScorer:
    """Computes morphological feature similarity between analyses.

    Assigns bonus scores based on feature agreement to help select
    the best analysis when multiple candidates exist.
    """

    FEATURE_WEIGHTS = {
        'Case': 0.3,
        'Number': 0.2,
        'Tense': 0.2,
        'Person': 0.15,
        'Mood': 0.15,
        'Voice': 0.1,
        'VerbForm': 0.1,
        'Polarity': 0.05,
        'NumType': 0.05
    }

    MAX_BONUS = 6.5

    def __init__(self) -> None:
        """Initialize similarity scorer with feature weights."""
        pass

    def compute_similarity(self, feats1: Optional[Dict[str, str]], feats2: Optional[Dict[str, str]]) -> float:
        """Compute similarity between two feature dictionaries.

        Args:
            feats1: First feature dict (e.g., from Omorfi analysis)
            feats2: Second feature dict (e.g., from Stanza context)

        Returns:
            float: Similarity bonus (0.0 to MAX_BONUS)
        """
        if not feats1 or not feats2:
            return 0.0

        bonus = 0.0
        for feature, weight in self.FEATURE_WEIGHTS.items():
            if feature in feats1 and feature in feats2:
                if feats1[feature] == feats2[feature]:
                    # Scale weight to contribute to MAX_BONUS
                    bonus += weight * 10.0

        # Cap at maximum bonus
        return min(bonus, self.MAX_BONUS)


class OmorfiHfstWithVoikkoV16Hybrid:
    """
    Finnish lemmatizer with three-tier hybrid gold standard lexicon

    V16: Hybrid gold standard with ambiguity tracking
    - Tier 1: Manual annotations (100% trusted)
    - Tier 2: Omorfi 100% unambiguous (non-overlapping)
    - Tier 3: Future production accumulation
    - Ambiguous tracking: Never caches genuinely ambiguous words

    V15: POS-aware lexicon with 80% threshold (FLAWED - uses automatic Omorfi)
    V14: No lexicon, pure contextual (baseline)
    """

    def __init__(self, config: Optional[LemmatizerConfig] = None, model_dir: Optional[str] = None, voikko_path: Optional[str] = None, lang: str = 'fi') -> None:
        """Load Omorfi HFST models, Voikko, Stanza, and V16 hybrid lexicon

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
        self.config = config

        # For backward compatibility, keep these attributes
        self.model_dir = self.config.model_dir
        self.lang = self.config.lang
        self.describe_path = self.config.analyzer_path

        # Initialize morphological feature extractors
        self.feature_extractor = MorphologicalFeatureExtractor()
        self.similarity_scorer = FeatureSimilarityScorer()

        # Initialize Stanza if available
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

        # Load describe transducer (full morphological analysis)
        if not os.path.exists(self.describe_path):
            raise FileNotFoundError(
                f"Omorfi models not found at {self.describe_path}\n"
                f"Download with: See INSTALLATION_GUIDE.md"
            )

        print(f"Loading Omorfi analyzer from: {model_dir}", file=sys.stderr)
        input_stream = hfst.HfstInputStream(self.describe_path)
        self.analyzer = input_stream.read()
        input_stream.close()
        print(f"✓ Omorfi analyzer loaded", file=sys.stderr)

        # Try to load guesser transducer
        self.guesser = None
        self.guesser_effective = False
        guesser_path = f'{model_dir}/omorfi-guesser.hfst'

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

        # V16: Load hybrid gold standard lexicon with three tiers + ambiguous tracking
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

        # Initialize dialectal normalizer
        try:
            self.dialectal_normalizer = DialectalNormalizer()
            print(f"✓ Dialectal normalizer initialized", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Could not initialize dialectal normalizer: {e}", file=sys.stderr)
            self.dialectal_normalizer = None

        # Load Finnish Dialects Dictionary (SMS)
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

    def normalize_lemma(self, lemma: str) -> str:
        """
        Normalize lemma (remove trailing hyphens)

        Args:
            lemma: Raw lemma from Omorfi

        Returns:
            Normalized lemma without trailing hyphen
        """
        if lemma and lemma.endswith('-'):
            return lemma[:-1]
        return lemma

    def score_alternative(self, lemma: str, word: str, weight: float, analysis: str) -> float:
        """
        Score an alternative lemma (lower is better)

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

    def parse_omorfi_tags(self, analysis: str) -> Dict[str, str]:
        """Parse Omorfi analysis into structured tags for contextual scoring"""
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

    def score_candidate_contextual(self, word: str, analysis: str, weight: float,
                                   stanza_upos: str, stanza_feats: str) -> float:
        """
        V17 Phase 9: Score candidate with rich morphological features (UFEATS)

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

    def rank_voikko_suggestions(self, word: str, suggestions: List[str], stanza_upos: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        V17 Phase 7: Rank Voikko suggestions using multi-criteria scoring

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

    def lemmatize_sentence(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Lemmatize a sentence with V16 hybrid lexicon + contextual disambiguation

        Priority order:
        1. Check if (word, POS) in ambiguous_patterns → skip lexicon, use contextual
        2. Check pos_aware_lexicon (Tier 1 + Tier 2) → return cached lemma
        3. Fall back to Omorfi contextual analysis

        NEW IN V16: Ambiguous pattern detection prevents incorrect caching
        """
        if not self.stanza_nlp or not tokens:
            # Fallback to word-level without context
            return [self.lemmatize(token) for token in tokens]

        try:
            # Get Stanza analysis for context
            doc = self.stanza_nlp([tokens])  # Pre-tokenized
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
                if token_lower in self.ambiguous_patterns:
                    pos_ambig_dict = self.ambiguous_patterns[token_lower]
                    if stanza_upos in pos_ambig_dict:
                        # This (word, POS) has multiple valid lemmas
                        # Fall through to contextual analysis below
                        pass
                    else:
                        # Word is ambiguous in general, but not for this POS
                        # Check unambiguous lexicon
                        if token_lower in self.pos_aware_lexicon:
                            pos_dict = self.pos_aware_lexicon[token_lower]
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
                    if token_lower in self.pos_aware_lexicon:
                        pos_dict = self.pos_aware_lexicon[token_lower]
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
                analyses = self.analyze_direct(token)
                voikko_used = False  # Track if Voikko provided suggestions

                if not analyses:
                    # V17 Phase 7: Voikko with multi-criteria ranking
                    # Rank suggestions before trying them (not just first-match)
                    voikko_suggestions = self.voikko_suggest(token, max_n=20)
                    if voikko_suggestions:
                        # Rank suggestions by edit distance, POS match, frequency, etc.
                        ranked_suggestions = self.rank_voikko_suggestions(token, voikko_suggestions, stanza_upos)
                        # Try ranked suggestions in order (best first)
                        for suggestion, _ in ranked_suggestions:
                            analyses = self.analyze_direct(suggestion)
                            if analyses:
                                voikko_used = True  # Mark that Voikko was used
                                break

                if analyses:
                    # V17 Phase 9 TRACKING: Count words with multiple Omorfi analyses
                    if len(analyses) > 1:
                        self.morph_stats['words_with_multiple_analyses'] += 1

                    # Score with context (V17 Phase 9: now uses rich morphological features!)
                    best_score = -1000.0
                    best_result = None
                    scored_candidates = []  # Track all scored candidates for comparison

                    for analysis, weight in analyses[:5]:
                        score = self.score_candidate_contextual(
                            token, analysis, weight,
                            stanza_upos, stanza_feats
                        )

                        lemma = self.normalize_lemma(self.extract_lemma(analysis) or '')
                        scored_candidates.append((lemma, score, analysis))

                        if score > best_score:
                            best_score = score
                            # Use different method name if Voikko assisted
                            method_name = 'voikko_omorfi' if voikko_used else 'omorfi_contextual'
                            best_result = {
                                'word': token,
                                'lemma': lemma,
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
                        dict_result = self.lemmatize_with_dialectal_dict(token, stanza_upos)
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

    def select_best_alternative(self, word: str, analyses: List[Tuple[str, float]]) -> Tuple[str, float, str, List[Dict]]:
        """
        Select best lemma from alternatives using smart heuristics

        V17 Phase 9 REFACTORED: Now uses morphological features when multiple analyses exist!

        Args:
            word: Original word
            analyses: List of (analysis, weight) tuples from Omorfi

        Returns:
            Tuple of (best_lemma, best_weight, best_analysis, alternatives_list)
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

    def extract_lemma(self, analysis: str) -> Optional[str]:
        """
        Extract lemma from Omorfi analysis string

        Input: '[WORD_ID=talo][UPOS=NOUN][NUM=SG][CASE=NOM]'
        Output: 'talo'
        """
        match = re.search(r'\[WORD_ID=([^\]]+)\]', analysis)
        if match:
            return match.group(1)
        return None

    def analyze_direct(self, word: str) -> List[Tuple[str, float]]:
        """
        Direct analysis with Omorfi analyzer

        Returns:
            List of (analysis, weight) tuples sorted by weight (lower is better)
        """
        return self.analyzer.lookup(word.lower())

    def analyze_guesser(self, word: str) -> List[Tuple[str, float]]:
        """
        Guesser analysis for unknown words (if guesser is loaded)

        Returns:
            List of (analysis, weight) tuples or empty list if no guesser
        """
        if self.guesser is None:
            return []

        return self.guesser.lookup(word.lower())

    def voikko_suggest(self, word: str, max_n: int = 5) -> List[str]:
        """
        Get spelling suggestions using Voikko (libvoikko or subprocess fallback)
        V17 Phase 4: Merges results from both Sukija and Old Finnish dictionaries

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

    def lemmatize_exception(self, word: str) -> Optional[Dict]:
        """
        Exception lexicon for stubborn dialectal forms
        """
        if word not in EXCEPTION_LEXICON:
            return None

        exception = EXCEPTION_LEXICON[word]
        norm = exception["norm"]

        result = self.lemmatize_direct(norm, strict_proper_noun_check=False)

        if not result:
            lemma = exception["lemma"]
        else:
            lemma = result['lemma']

        return {
            'word': word,
            'lemma': lemma,
            'analysis': None,
            'weight': None,
            'method': 'exception_lexicon',
            'confidence': 'medium',
            'original_word': word,
            'normalized_word': norm,
            'note': exception["note"],
            'alternatives': []
        }

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

    # =============================================================================
    # V17 PHASE 9: MORPHOLOGICAL FEATURE EXTRACTION AND SIMILARITY
    # =============================================================================

    def get_ufeats_from_analysis(self, analysis_str: str, weight: float = 0.0) -> Optional[Dict]:
        """Extract Universal Features from HFST analysis string.

        Delegates to MorphologicalFeatureExtractor for actual extraction.
        """
        return self.feature_extractor.get_ufeats_from_analysis(analysis_str, weight)

    def compute_feature_similarity(self, feats1: Optional[Dict], feats2: Optional[Dict]) -> float:
        """Compute morphological feature similarity.

        Delegates to FeatureSimilarityScorer for actual computation.
        """
        return self.similarity_scorer.compute_similarity(feats1, feats2)

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

    def lemmatize_with_fuzzy_lexicon(self, word: str, stanza_upos: str = None) -> Optional[Dict]:
        """
        V17 Phase 7: Fuzzy lexicon lookup using weighted edit distance

        Searches all lexicon entries (Tier 1-3 training lemmas) for close dialectal matches
        using dialectal_normalizer.weighted_edit_distance()

        Distance threshold: 2.0 (allows 1-2 character dialectal variations)
        POS filtering: If available, only match same part-of-speech

        Args:
            word: The word to lemmatize
            stanza_upos: Optional POS tag from Stanza for filtering

        Returns:
            Lemmatization result if match found, None otherwise
        """
        if not (self.dialectal_normalizer and self.pos_aware_lexicon):
            return None

        word_lower = word.lower()
        candidates = []

        # Iterate through all words in lexicon
        for lexicon_word, pos_dict in self.pos_aware_lexicon.items():
            # Calculate distance from input word to lexicon word
            distance = self.dialectal_normalizer.weighted_edit_distance(word_lower, lexicon_word)

            if distance <= LemmatizerConstants.FUZZY_THRESHOLD_STANDARD:
                # Check all POS variants of this word
                for pos, entry in pos_dict.items():
                    # Filter by POS if available
                    if stanza_upos and pos != stanza_upos:
                        continue

                    lemma = entry['lemma']
                    source = entry.get('source', 'unknown')
                    tier = 'tier1' if source == 'manual' else 'tier2' if source == 'omorfi_100' else 'tier3'

                    candidates.append((lemma, distance, lexicon_word, pos, tier))

        if candidates:
            # Return best match (lowest distance, prefer tier1 > tier2 > tier3)
            tier_priority = {'tier1': 0, 'tier2': 1, 'tier3': 2}
            best_lemma, best_dist, matched_word, matched_pos, matched_tier = min(
                candidates,
                key=lambda x: (x[1], tier_priority.get(x[4], 3))  # Sort by distance, then tier
            )

            return {
                'word': word,
                'lemma': best_lemma,
                'analysis': f'[FUZZY_LEXICON:{word}→{matched_word}({matched_pos})→{best_lemma},dist={best_dist:.2f},{matched_tier}]',
                'weight': None,
                'method': 'fuzzy_lexicon',
                'confidence': 'medium',
                'alternatives': []
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

    def lemmatize(self, word: str) -> Dict[str, Any]:
        """
        Complete lemmatization pipeline with smart alternative selection

        V17 Phase 9: Eight-tier fallback with morphology-aware fuzzy matching:
        1. Direct Omorfi analysis (highest confidence) + smart alternatives
        2. Guesser analysis (medium confidence)
        3. Voikko normalization + re-analysis (10 suggestions)
        4. Exception lexicon (for stubborn dialectal forms)
        5. Enhanced Voikko (30 suggestions, aggressive mode)
        6. Morphology-aware fuzzy lexicon (UFEATS feature matching, threshold 2.0) - NEW IN PHASE 9
        7. Aggressive fuzzy lexicon (relaxed threshold 3.5, suffix stripping)
        8. Identity fallback (return word.lower())

        Returns:
            {
                'word': str,
                'lemma': str or None,
                'analysis': str or None,
                'weight': float or None,
                'method': str,
                'confidence': str
            }
        """
        # Tier 1: Direct analysis with smart alternatives
        result = self.lemmatize_direct(word, strict_proper_noun_check=True)
        if result:
            return result

        # Tier 2: Guesser fallback
        if self.guesser:
            result = self.lemmatize_guesser(word)
            if result:
                return result

        # Tier 3: Voikko normalization (10 suggestions)
        if self.voikko_path:
            result = self.lemmatize_with_voikko(word)
            if result:
                return result

        # Tier 4: Exception lexicon
        result = self.lemmatize_exception(word)
        if result:
            return result

        # Tier 5: Enhanced Voikko (V17 Phase 3)
        # More aggressive: try up to 30 suggestions to catch more unknowns
        if self.voikko_path:
            result = self.lemmatize_with_voikko_enhanced(word)
            if result:
                return result

        # Tier 6: Finnish Dialects Dictionary (SMS) - MOVED UP FROM TIER 7
        # Search 19,385 validated dialectal variants
        # High confidence (0.9) for validated dictionary entries
        # RATIONALE: Exact dictionary match > fuzzy approximation
        result = self.lemmatize_with_dialectal_dict(word)
        if result:
            return result

        # Tier 7: Morphology-Aware Fuzzy Lexicon (V17 Phase 9)
        # Search training lexicon with morphological feature awareness
        # Uses Omorfi UFEATS to rank candidates by morphological plausibility
        # Threshold: 2.0 (conservative matching with feature bonuses)
        result = self.lemmatize_with_fuzzy_lexicon_morphological(word)
        if result:
            return result

        # Tier 8: Aggressive Fuzzy Lexicon (V17 Phase 8)
        # More aggressive fuzzy matching with suffix stripping and normalization
        # Threshold: 3.5 (relaxed matching)
        # Target: Reduce identity_fallback from 50 words to ~10-20 (60-80% reduction)
        result = self.lemmatize_with_fuzzy_lexicon_aggressive(word)
        if result:
            return result

        # Tier 9: Identity fallback (V17 Phase 1.5)
        # When all else fails, return the word itself as the lemma
        # This helps with proper nouns and words in nominative form
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

    def lemmatize_batch(self, words: List[str]) -> Dict[str, Any]:
        """
        Lemmatize multiple words and return statistics

        Returns:
            {
                'results': List[Dict],
                'statistics': {...}
            }
        """
        results = []
        stats = {
            'total': len(words),
            'v16_lexicon_tier1': 0,
            'v16_lexicon_tier2': 0,
            'v16_lexicon_tier3': 0,
            'omorfi_contextual': 0,
            'voikko_omorfi': 0,  # Voikko-assisted Omorfi analysis
            'omorfi_direct': 0,
            'omorfi_guesser': 0,
            'voikko_normalized': 0,
            'exception_lexicon': 0,
            'enhanced_voikko': 0,  # Phase 3: aggressive Voikko (30 suggestions)
            'fuzzy_lexicon': 0,  # Phase 7: weighted edit distance on training lemmas (threshold 2.0) - deprecated in Phase 9
            'fuzzy_lexicon_morphological': 0,  # NEW in Phase 9: morphology-aware fuzzy (UFEATS feature matching)
            'dialectal_dictionary': 0,  # NEW Tier 7: Finnish Dialects Dictionary (SMS) - 19,385 validated variants
            'fuzzy_lexicon_aggressive': 0,  # Phase 8: aggressive fuzzy (threshold 3.5, suffix stripping)
            'identity_fallback': 0,  # Phase 1.5: word.lower() fallback
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


def main() -> None:
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Finnish lemmatization with V16 hybrid gold standard lexicon"
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

    args = parser.parse_args()

    # Get input text
    if args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    # Initialize lemmatizer
    try:
        lemmatizer = OmorfiHfstWithVoikkoV16Hybrid(
            model_dir=args.model_dir,
            voikko_path=args.voikko_path
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Tokenize
    words = re.findall(r'\b[\wäöåÄÖÅ]+\b', text, re.UNICODE)

    # Lemmatize
    batch_result = lemmatizer.lemmatize_batch(words)

    # Print results
    print()
    print("=" * 80)
    print("V16 HYBRID GOLD STANDARD LEMMATIZATION")
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

    if stats['exception_lexicon'] > 0:
        print(f"Exception lexicon:   {stats['exception_lexicon']} ({stats['exception_lexicon']/stats['total']*100:.1f}%)")

    print(f"Unknown:             {stats['unknown']} ({stats['unknown']/stats['total']*100:.1f}%)")
    print(f"Coverage:            {stats['coverage']:.1f}%")
    print()
    print("=" * 80)
    print()

    # Word-level results
    for result in batch_result['results']:
        if args.show_unknown and result['method'] != 'unknown':
            continue

        original_word = result.get('original_word', result['word'])
        word = result['word']
        lemma = result['lemma'] if result['lemma'] else '_UNKNOWN_'
        method = result['method']
        confidence = result['confidence']

        if confidence == 'high':
            status = '✓'
        elif confidence == 'medium':
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
    if stats['unknown'] > 0:
        print()
        print("=" * 80)
        print(f"? {stats['unknown']} words remain unknown (may need manual review)")
        print("  Run with --show-unknown to see only these words")
        print("=" * 80)

    print()


if __name__ == "__main__":
    main()
