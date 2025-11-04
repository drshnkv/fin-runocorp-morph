#!/usr/bin/env python3
"""
Configuration module for Finnish dialectal lemmatizer.

This module contains all configuration classes, constants, and utility functions
that were previously scattered throughout the monolithic lemmatizer script.

Classes:
    LemmatizerConfig: Main configuration holder
    MorphologicalFeatureExtractor: HFST → Universal Dependencies feature conversion
    FeatureSimilarityScorer: Feature agreement scoring for candidate selection
    LemmatizerConstants: Centralized scoring thresholds and magic numbers
"""

import os
import re
from typing import Dict, Optional


# ============================================================================
# Global Constants
# ============================================================================

VERSION = "v17_phase11_refactored_with_compounds"

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
        """Initialize lemmatizer configuration.

        Args:
            model_dir: Path to Omorfi HFST models directory (default: ~/.omorfi)
            voikko_path: Path to Voikko executable (deprecated)
            lang: Language code (default: 'fi')
            lexicon_path: Path to training lexicon JSON file
            dialectal_dict_path: Path to SMS dialectal dictionary JSON file
            max_suggestions_standard: Max Voikko suggestions for standard tier (default: 10)
            max_suggestions_enhanced: Max Voikko suggestions for enhanced tier (default: 30)
            enable_stanza: Enable Stanza contextual analysis (default: True)
            enable_voikko: Enable Voikko spell checking (default: True)
            enable_omorfi: Enable Omorfi morphological analysis (default: True)
        """
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


# ============================================================================
# Morphological Feature Extraction
# ============================================================================

class MorphologicalFeatureExtractor:
    """Extracts Universal Features (UFEATS) from HFST analysis strings.

    Converts HFST morphological tags to Universal Dependencies format.
    Based on get_ufeats_from_analysis method from original implementation.
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
            analysis_str: HFST analysis like "[WORD_ID=kissa][UPOS=NOUN][NUM=SG][CASE=NOM]"
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


# ============================================================================
# Scoring and Similarity
# ============================================================================

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
