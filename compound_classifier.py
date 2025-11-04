#!/usr/bin/env python3
"""
Finnish Compound Word Classifier and Reconstructor

This module provides intelligent classification of Omorfi compound analyses
to distinguish true compounds from false positives (inflectional endings
misidentified as compound components).

Integration-ready for fin_runocorp_base_v2_dialectal_dict_integrated.py
"""

import re
from typing import Dict, Tuple, Optional, List
import json


class CompoundClassifier:
    """
    Classifies Omorfi compound analyses as TRUE or FALSE compounds.

    Uses linguistic heuristics to filter out:
    - Verb inflectional suffixes misidentified as compounds
    - Case/clitic endings misidentified as compounds
    - Invalid semantic combinations
    """

    # Finnish verb suffixes and clitics (commonly misidentified as compounds)
    VERB_SUFFIXES = {
        'sinko', 'sinka', 'sinkä',  # Conditional + question clitic
        'kaan', 'kään',              # Negative clitic
        'han', 'hän',                # Emphatic clitic
        'pa', 'pä',                  # Emphatic clitic
        'ko', 'kö',                  # Question clitic (short)
        'kin',                       # Also/too clitic
        'kse', 'ksi',                # Translative case / imperative
        'lla', 'llä',                # Adessive case
        'lta', 'ltä',                # Ablative case
        'lle',                       # Allative case
        'ssa', 'ssä',                # Inessive case
        'sta', 'stä',                # Elative case
        'tta', 'ttä',                # Abessive case
    }

    # Finnish possessive suffixes (often misidentified as compound components)
    POSSESSIVE_SUFFIXES = {
        'ni',    # 1st person singular (minun)
        'si',    # 2nd person singular (sinun)
        'nsa', 'nsä',  # 3rd person (hänen, heidän)
        'mme',   # 1st person plural (meidän)
        'nne',   # 2nd person plural (teidän)
        'an', 'än',    # 3rd person after vowel
    }

    # Known Finnish nouns/adjectives that can be compound components
    # This is a small set - for full validation, use lexicon
    KNOWN_SECOND_COMPONENTS = {
        # Body parts
        'pää', 'silmä', 'käsi', 'jalka', 'sormi', 'polvi', 'suu', 'nenä',
        'korva', 'hammas', 'kieli', 'selkä', 'vartalo', 'leuka', 'niska',
        # Nature
        'maa', 'vesi', 'meri', 'suo', 'niitty', 'metsä', 'puu', 'kivi',
        'lehti', 'kukka', 'lintu', 'kala', 'eläin',
        # Common objects
        'talo', 'huone', 'ovi', 'ikkuna', 'lattia', 'seinä', 'katto',
        'pöytä', 'tuoli', 'sänky', 'vaate', 'kenkä',
        # Directions/places
        'puoli', 'paikka', 'reuna', 'kohta', 'kulma',
        # Time
        'aika', 'hetki', 'päivä', 'yö', 'vuosi', 'kerta',
        # Abstract
        'mieli', 'henki', 'voima', 'tapa', 'laatu', 'osa', 'koko',
        # Animals/nature specific
        'hevonen', 'lehmä', 'sika', 'koira', 'kissa', 'lammas',
    }

    def __init__(self, lexicon_path: Optional[str] = None, sms_dict_path: Optional[str] = None, voikko_path: Optional[str] = None):
        """
        Initialize compound classifier.

        Args:
            lexicon_path: Optional path to lexicon JSON for validation
            sms_dict_path: Optional path to SMS dialectal dictionary JSON
            voikko_path: Optional path to Voikko dictionary for validation
        """
        self.lexicon = None
        self.sms_dict = None

        if lexicon_path:
            self.load_lexicon(lexicon_path)
        if sms_dict_path:
            self.load_sms_dialectal(sms_dict_path)

        # Initialize Voikko for word validation
        self.voikko = None
        self.voikko_sukija = None
        self.init_voikko(voikko_path)

        self.stats = {
            'total_analyzed': 0,
            'true_compounds': 0,
            'false_compounds': 0,
            'rejected_verb_suffix': 0,
            'rejected_possessive_suffix': 0,
            'rejected_too_short': 0,
            'rejected_verb_pos': 0,
            'rejected_not_validated': 0,
            'accepted_noun_noun': 0,
            'accepted_lexicon_validated': 0,
            'accepted_sms_validated': 0,
            'accepted_voikko_validated': 0,
        }

    def load_lexicon(self, lexicon_path: str):
        """Load lexicon for second component validation."""
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract unique words from pos_aware_lexicon
                self.lexicon = set(data.get('pos_aware_lexicon', {}).keys())
                print(f"✓ Loaded lexicon with {len(self.lexicon)} unique words for compound validation")
        except Exception as e:
            print(f"⚠ Could not load lexicon: {e}")
            self.lexicon = None

    def load_sms_dialectal(self, sms_dict_path: str):
        """Load SMS dialectal dictionary for second component validation."""
        try:
            with open(sms_dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract lemmas from variant_to_lemma mappings
                variant_to_lemma = data.get('variant_to_lemma', {})
                lemmas = set()
                for variant, lemma_list in variant_to_lemma.items():
                    # Add both the variant and the lemmas
                    lemmas.add(variant.lower())
                    if isinstance(lemma_list, list):
                        for lemma_entry in lemma_list:
                            if isinstance(lemma_entry, dict):
                                lemmas.add(lemma_entry.get('lemma', '').lower())
                            elif isinstance(lemma_entry, str):
                                lemmas.add(lemma_entry.lower())

                self.sms_dict = lemmas
                print(f"✓ Loaded SMS dialectal dictionary with {len(self.sms_dict)} unique words for compound validation")
        except Exception as e:
            print(f"⚠ Could not load SMS dialectal dictionary: {e}")
            self.sms_dict = None

    def init_voikko(self, voikko_path: Optional[str] = None):
        """Initialize Voikko for word validation."""
        try:
            import libvoikko

            # Try to load Sukija (Old Finnish) dictionary
            sukija_path = voikko_path or "/Users/kaarelveskis/.voikko/sukija/5"
            try:
                self.voikko_sukija = libvoikko.Voikko("fi", path=sukija_path)
                print(f"✓ Loaded Voikko Sukija from {sukija_path}")
            except:
                pass

            # Load standard Finnish dictionary
            std_path = voikko_path or "/Users/kaarelveskis/.voikko/5"
            try:
                self.voikko = libvoikko.Voikko("fi", path=std_path)
                print(f"✓ Loaded Voikko standard Finnish")
            except:
                pass

            if not self.voikko and not self.voikko_sukija:
                print("⚠ Voikko not available - will use lexicon-only validation")

        except ImportError:
            print("⚠ libvoikko not installed - will use lexicon-only validation")
            self.voikko = None
            self.voikko_sukija = None

    def parse_compound_components(self, analysis: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Parse compound components from Omorfi analysis.

        Args:
            analysis: Omorfi analysis string with [BOUNDARY=COMPOUND]

        Returns:
            Tuple of (first_word, first_pos, second_word, second_pos) or None
        """
        if '[BOUNDARY=COMPOUND]' not in analysis:
            return None

        try:
            # Find first component
            first_word_match = re.search(r'\[WORD_ID=([^\]]+)\]', analysis)
            if not first_word_match:
                return None
            first_word = first_word_match.group(1)

            # Find first POS (before BOUNDARY=COMPOUND)
            before_boundary = analysis.split('[BOUNDARY=COMPOUND]')[0]
            first_pos_match = re.search(r'\[UPOS=([^\]]+)\]', before_boundary)
            first_pos = first_pos_match.group(1) if first_pos_match else None

            # Find second component (after BOUNDARY=COMPOUND)
            after_boundary = analysis.split('[BOUNDARY=COMPOUND]')[1]
            second_word_match = re.search(r'\[WORD_ID=([^\]]+)\]', after_boundary)
            if not second_word_match:
                return None
            second_word = second_word_match.group(1)

            # Find second POS
            second_pos_match = re.search(r'\[UPOS=([^\]]+)\]', after_boundary)
            second_pos = second_pos_match.group(1) if second_pos_match else None

            return (first_word, first_pos, second_word, second_pos)

        except Exception as e:
            return None

    def is_verb_suffix(self, component: str) -> bool:
        """Check if component is a known verb suffix or clitic."""
        return component.lower() in self.VERB_SUFFIXES

    def is_possessive_suffix(self, component: str) -> bool:
        """Check if component is a known possessive suffix."""
        return component.lower() in self.POSSESSIVE_SUFFIXES

    def has_possessive_marker(self, analysis: str) -> bool:
        """Check if analysis contains [POSS=...] marker."""
        return '[POSS=' in analysis

    def is_known_word(self, word: str) -> bool:
        """
        Check if word is a known Finnish word.

        Uses fallback chain: Voikko → lexicon → SMS dict → built-in set.
        """
        word_lower = word.lower()

        # 1. Check Voikko dictionaries (most comprehensive)
        if self.voikko_sukija:
            try:
                if self.voikko_sukija.spell(word_lower):
                    return True
            except:
                pass

        if self.voikko:
            try:
                if self.voikko.spell(word_lower):
                    return True
            except:
                pass

        # 2. Check lexicon if available
        if self.lexicon and word_lower in self.lexicon:
            return True

        # 3. Check SMS dialectal dictionary if available
        if self.sms_dict and word_lower in self.sms_dict:
            return True

        # 4. Fall back to built-in set
        return word_lower in self.KNOWN_SECOND_COMPONENTS

    def classify_compound(self, analysis: str, original_word: str = None) -> Dict:
        """
        Classify a compound analysis as TRUE or FALSE compound.

        Args:
            analysis: Omorfi analysis string
            original_word: Optional original word form

        Returns:
            Dict with classification result:
            {
                'is_true_compound': bool,
                'first_component': str,
                'second_component': str,
                'reconstructed': str or None,
                'reason': str,
                'confidence': float (0.0-1.0)
            }
        """
        self.stats['total_analyzed'] += 1

        # Rule 0: HIGHEST PRIORITY - Reject if analysis contains [POSS=...] marker
        # This indicates a possessive suffix, not a true compound
        if self.has_possessive_marker(analysis):
            components = self.parse_compound_components(analysis)
            if components:
                first_word, first_pos, second_word, second_pos = components
                self.stats['false_compounds'] += 1
                self.stats['rejected_possessive_suffix'] += 1
                return {
                    'is_true_compound': False,
                    'first_component': first_word,
                    'second_component': second_word,
                    'reconstructed': None,
                    'reason': f'Analysis contains [POSS=...] marker - possessive suffix, not compound',
                    'confidence': 0.99
                }

        components = self.parse_compound_components(analysis)
        if not components:
            return {
                'is_true_compound': False,
                'first_component': None,
                'second_component': None,
                'reconstructed': None,
                'reason': 'Could not parse compound components',
                'confidence': 0.0
            }

        first_word, first_pos, second_word, second_pos = components

        # Rule 1: Reject if second component is a known possessive suffix
        if self.is_possessive_suffix(second_word):
            self.stats['false_compounds'] += 1
            self.stats['rejected_possessive_suffix'] += 1
            return {
                'is_true_compound': False,
                'first_component': first_word,
                'second_component': second_word,
                'reconstructed': None,
                'reason': f'Second component "{second_word}" is possessive suffix',
                'confidence': 0.98
            }

        # Rule 1: Reject if second component is a known suffix/clitic
        if self.is_verb_suffix(second_word):
            self.stats['false_compounds'] += 1
            self.stats['rejected_verb_suffix'] += 1
            return {
                'is_true_compound': False,
                'first_component': first_word,
                'second_component': second_word,
                'reconstructed': None,
                'reason': f'Second component "{second_word}" is verb suffix/clitic',
                'confidence': 0.95
            }

        # Rule 2: Reject if second component is very short (≤2 characters)
        if len(second_word) <= 2:
            self.stats['false_compounds'] += 1
            self.stats['rejected_too_short'] += 1
            return {
                'is_true_compound': False,
                'first_component': first_word,
                'second_component': second_word,
                'reconstructed': None,
                'reason': f'Second component too short ({len(second_word)} chars)',
                'confidence': 0.90
            }

        # Rule 3: Reject if first component is VERB (verbs rarely form true compounds)
        if first_pos == 'VERB':
            self.stats['false_compounds'] += 1
            self.stats['rejected_verb_pos'] += 1
            return {
                'is_true_compound': False,
                'first_component': first_word,
                'second_component': second_word,
                'reconstructed': None,
                'reason': f'First component is VERB (rare for true compounds)',
                'confidence': 0.85
            }

        # Rule 4: Accept if NOUN+NOUN and second component ≥4 chars and is known word
        if first_pos == 'NOUN' and second_pos == 'NOUN' and len(second_word) >= 4:
            # Check if second component is a known word
            if self.is_known_word(second_word):
                self.stats['true_compounds'] += 1
                self.stats['accepted_noun_noun'] += 1

                # Track validation source (priority order)
                if self.voikko and (self.voikko.spell(second_word.lower()) or
                                   (self.voikko_sukija and self.voikko_sukija.spell(second_word.lower()))):
                    self.stats['accepted_voikko_validated'] += 1
                    validation_source = "Voikko"
                elif self.lexicon and second_word.lower() in self.lexicon:
                    self.stats['accepted_lexicon_validated'] += 1
                    validation_source = "lexicon"
                elif self.sms_dict and second_word.lower() in self.sms_dict:
                    self.stats['accepted_sms_validated'] += 1
                    validation_source = "SMS"
                else:
                    validation_source = "built-in"

                reconstructed = first_word + second_word
                return {
                    'is_true_compound': True,
                    'first_component': first_word,
                    'second_component': second_word,
                    'reconstructed': reconstructed,
                    'reason': f'NOUN+NOUN, both ≥4 chars, "{second_word}" validated ({validation_source})',
                    'confidence': 0.90
                }
            else:
                # NOUN+NOUN but second component not validated
                # Mark as uncertain/false to be conservative
                self.stats['false_compounds'] += 1
                self.stats['rejected_not_validated'] += 1
                return {
                    'is_true_compound': False,
                    'first_component': first_word,
                    'second_component': second_word,
                    'reconstructed': None,
                    'reason': f'NOUN+NOUN but "{second_word}" not validated',
                    'confidence': 0.60
                }

        # Rule 5: Accept if NOUN+ADJ and second component ≥5 chars and is known word
        if first_pos == 'NOUN' and second_pos == 'ADJ' and len(second_word) >= 5:
            if self.is_known_word(second_word):
                self.stats['true_compounds'] += 1
                reconstructed = first_word + second_word
                return {
                    'is_true_compound': True,
                    'first_component': first_word,
                    'second_component': second_word,
                    'reconstructed': reconstructed,
                    'reason': f'NOUN+ADJ, ≥5 chars, "{second_word}" validated',
                    'confidence': 0.80
                }

        # Default: Reject to be conservative (precision over recall)
        self.stats['false_compounds'] += 1
        return {
            'is_true_compound': False,
            'first_component': first_word,
            'second_component': second_word,
            'reconstructed': None,
            'reason': f'Conservative rejection (POS={first_pos}+{second_pos}, len={len(second_word)})',
            'confidence': 0.70
        }

    def get_statistics(self) -> Dict:
        """Get classification statistics."""
        return self.stats.copy()

    def print_statistics(self):
        """Print classification statistics."""
        print("\n" + "="*70)
        print("COMPOUND CLASSIFICATION STATISTICS")
        print("="*70)
        print(f"Total analyzed:           {self.stats['total_analyzed']}")
        print(f"True compounds:           {self.stats['true_compounds']} ({100*self.stats['true_compounds']/max(1,self.stats['total_analyzed']):.1f}%)")
        print(f"False compounds:          {self.stats['false_compounds']} ({100*self.stats['false_compounds']/max(1,self.stats['total_analyzed']):.1f}%)")
        print()
        print("Rejection reasons:")
        print(f"  - Possessive suffix:    {self.stats['rejected_possessive_suffix']}")
        print(f"  - Verb suffix/clitic:   {self.stats['rejected_verb_suffix']}")
        print(f"  - Too short (≤2):       {self.stats['rejected_too_short']}")
        print(f"  - VERB POS:             {self.stats['rejected_verb_pos']}")
        print(f"  - Not validated:        {self.stats['rejected_not_validated']}")
        print()
        print("Acceptance reasons:")
        print(f"  - NOUN+NOUN validated:  {self.stats['accepted_noun_noun']}")
        print(f"    - Voikko validated:   {self.stats['accepted_voikko_validated']}")
        print(f"    - Lexicon validated:  {self.stats['accepted_lexicon_validated']}")
        print(f"    - SMS validated:      {self.stats['accepted_sms_validated']}")
        print("="*70)


# Example usage and testing
if __name__ == '__main__':
    import sys

    # Initialize classifier with both lexicons
    classifier = CompoundClassifier(
        lexicon_path='selftraining_lexicon_comb_with_additions.json',
        sms_dict_path='sms_dialectal_index_v4_final.json'
    )

    # Test cases
    test_cases = [
        # TRUE compounds (should reconstruct)
        ('[WORD_ID=jouhi][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=leuka][UPOS=NOUN][NUM=SG][CASE=GEN]',
         'jouhileuan', 'Should be TRUE: jouhi+leuka (horsehair+jaw)'),
        ('[WORD_ID=meri][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=puoli][UPOS=NOUN][NUM=SG][CASE=NOM]',
         'meripuolen', 'Should be TRUE: meri+puoli (sea+side)'),

        # FALSE compounds (should reject)
        ('[WORD_ID=toivo][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=sinko][UPOS=VERB]',
         'toivosinko', 'Should be FALSE: sinko is verb suffix'),
        ('[WORD_ID=päivä][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=seksi][UPOS=NOUN]',
         'päiväseksi', 'Should be FALSE: seksi is case ending'),
        ('[WORD_ID=polvi][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=ansa][UPOS=NOUN][NUM=SG][CASE=NOM][POSS=3]',
         'polviansa', 'Should be FALSE: has [POSS=3] marker'),
        ('[WORD_ID=kanta][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=ja][UPOS=NOUN][NUM=SG][CASE=GEN][POSS=SG1]',
         'kantajani', 'Should be FALSE: has [POSS=SG1] marker + ja too short'),
    ]

    print("\nTEST CASES:")
    print("="*70)
    for analysis, original, description in test_cases:
        result = classifier.classify_compound(analysis, original)
        status = "✓" if result['is_true_compound'] else "✗"
        print(f"\n{status} {description}")
        print(f"   Original: {original}")
        print(f"   Components: {result['first_component']} + {result['second_component']}")
        print(f"   Reconstructed: {result['reconstructed']}")
        print(f"   Reason: {result['reason']}")
        print(f"   Confidence: {result['confidence']:.2f}")

    # Print statistics
    classifier.print_statistics()
