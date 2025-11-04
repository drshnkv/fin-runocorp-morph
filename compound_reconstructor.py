#!/usr/bin/env python3
"""
Compound Reconstructor - Production-ready implementation for Finnish compound reconstruction.

This module provides functions to determine whether Omorfi-detected compounds
should be reconstructed into full compound lemmas or kept as first-component-only.

Usage:
    from compound_reconstructor import CompoundReconstructor

    reconstructor = CompoundReconstructor()
    result = reconstructor.process_analysis(word, lemma, omorfi_analysis)
    print(f"Word: {word}, Lemma: {result['lemma']}, Confidence: {result['confidence']}")
"""

import re
from typing import Dict, Optional, Tuple


class CompoundReconstructor:
    """
    Handles compound word reconstruction from Omorfi analyses.
    """

    def __init__(self, conservative=True, min_confidence=0.75):
        """
        Initialize reconstructor with configuration.

        Args:
            conservative: If True, only reconstruct high-confidence cases
            min_confidence: Minimum confidence threshold for reconstruction (0.0-1.0)
        """
        self.conservative = conservative
        self.min_confidence = min_confidence

        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'reconstructed': 0,
            'kept_original': 0,
            'uncertain': 0,
            'parse_failed': 0
        }

    def parse_omorfi_analysis(self, analysis_str: str) -> Optional[Dict]:
        """
        Parse Omorfi analysis string to extract compound components.

        Args:
            analysis_str: Omorfi analysis string containing [BOUNDARY=COMPOUND]

        Returns:
            Dict with first_component, second_component, first_pos, second_pos
            or None if parsing fails
        """
        if not isinstance(analysis_str, str) or '[BOUNDARY=COMPOUND]' not in analysis_str:
            return None

        result = {
            'first_component': None,
            'first_pos': None,
            'second_component': None,
            'second_pos': None,
            'full_analysis': analysis_str
        }

        # Extract first component (before BOUNDARY=COMPOUND)
        first_match = re.search(
            r'\[WORD_ID=([^\]]+)\](.*?)\[BOUNDARY=COMPOUND\]',
            analysis_str
        )
        if first_match:
            result['first_component'] = first_match.group(1)
            # Extract POS from tags between WORD_ID and BOUNDARY
            pos_match = re.search(r'\[UPOS=([^\]]+)\]', first_match.group(2))
            if pos_match:
                result['first_pos'] = pos_match.group(1)

        # Extract second component (after BOUNDARY=COMPOUND)
        second_match = re.search(
            r'\[BOUNDARY=COMPOUND\]\[WORD_ID=([^\]]+)\]',
            analysis_str
        )
        if second_match:
            result['second_component'] = second_match.group(1)
            # Extract POS for second component
            after_boundary = analysis_str.split('[BOUNDARY=COMPOUND]')[1]
            pos_match = re.search(r'\[UPOS=([^\]]+)\]', after_boundary)
            if pos_match:
                result['second_pos'] = pos_match.group(1)

        return result

    def classify_compound(
        self,
        first_component: str,
        second_component: str,
        first_pos: Optional[str] = None,
        second_pos: Optional[str] = None
    ) -> Tuple[Optional[bool], str, float]:
        """
        Classify whether compound should be reconstructed.

        Args:
            first_component: First part of compound
            second_component: Second part of compound
            first_pos: POS tag of first component
            second_pos: POS tag of second component

        Returns:
            Tuple of (should_reconstruct, reason, confidence)
            where should_reconstruct is True/False/None (uncertain)
        """
        first_len = len(first_component)
        second_len = len(second_component)

        # PRIORITY 1: REJECT if second component very short
        if second_len <= 2:
            return (
                False,
                f"Second component ≤2 chars (likely inflectional): '{second_component}'",
                0.80
            )

        # PRIORITY 1: REJECT if both components very short
        if first_len <= 3 and second_len <= 3:
            return (
                False,
                f"Both components ≤3 chars (too short): {first_len}+{second_len}",
                0.75
            )

        # PRIORITY 2: ACCEPT if NOUN+NOUN with substantial components
        if first_pos == "NOUN" and second_pos == "NOUN":
            if first_len >= 4 and second_len >= 4:
                return (
                    True,
                    f"NOUN+NOUN with both ≥4 chars: {first_len}+{second_len}",
                    0.85
                )

        # PRIORITY 2: ACCEPT if second component is long content word
        if second_len >= 5 and second_pos in ["NOUN", "ADJ"]:
            return (
                True,
                f"Long second component ({second_len} chars) with {second_pos} POS",
                0.80
            )

        # PRIORITY 3: UNCERTAIN if second component is 3 chars
        if second_len == 3:
            return (
                None,
                f"3-char second component '{second_component}' - needs validation",
                0.50
            )

        # PRIORITY 3: UNCERTAIN if NOUN+ADJ
        if first_pos == "NOUN" and second_pos == "ADJ":
            return (
                None,
                f"NOUN+ADJ combination - check semantics",
                0.60
            )

        # DEFAULT: UNCERTAIN
        return (
            None,
            "No clear rule matches - needs manual review",
            0.40
        )

    def reconstruct_compound_lemma(
        self,
        first_component: str,
        second_component: str
    ) -> str:
        """
        Reconstruct compound lemma from components.

        Note: This is a simplified concatenation. In production, you might want
        to handle vowel harmony, consonant gradation, etc.

        Args:
            first_component: First part of compound
            second_component: Second part of compound

        Returns:
            Reconstructed compound lemma
        """
        # Simple concatenation for now
        # TODO: Consider vowel harmony rules if needed
        return first_component + second_component

    def process_analysis(
        self,
        word: str,
        current_lemma: str,
        analysis: str
    ) -> Dict:
        """
        Process a word with Omorfi analysis and decide on lemma reconstruction.

        Args:
            word: Original word form
            current_lemma: Current lemma (usually first component only)
            analysis: Omorfi analysis string

        Returns:
            Dict with:
                - lemma: Final lemma decision
                - reconstructed: Was compound reconstructed?
                - confidence: Confidence score (0.0-1.0)
                - reason: Human-readable explanation
                - components: Dict with parsing details (if successful)
        """
        self.stats['total_processed'] += 1

        result = {
            'word': word,
            'lemma': current_lemma,  # Default to keeping original
            'reconstructed': False,
            'confidence': 1.0,  # High confidence in keeping original
            'reason': 'Not a compound',
            'components': None
        }

        # Parse Omorfi analysis
        parsed = self.parse_omorfi_analysis(analysis)

        if not parsed or not parsed['first_component'] or not parsed['second_component']:
            # Not a compound or parsing failed
            self.stats['parse_failed'] += 1
            result['reason'] = 'Parsing failed or not a compound'
            return result

        # Store component information
        result['components'] = parsed

        # Classify compound
        should_reconstruct, reason, confidence = self.classify_compound(
            parsed['first_component'],
            parsed['second_component'],
            parsed['first_pos'],
            parsed['second_pos']
        )

        result['confidence'] = confidence
        result['reason'] = reason

        # Decision logic
        if should_reconstruct is True and confidence >= self.min_confidence:
            # HIGH confidence TRUE compound - reconstruct
            reconstructed = self.reconstruct_compound_lemma(
                parsed['first_component'],
                parsed['second_component']
            )
            result['lemma'] = reconstructed
            result['reconstructed'] = True
            self.stats['reconstructed'] += 1

        elif should_reconstruct is False:
            # FALSE compound - keep original
            result['lemma'] = current_lemma
            result['reconstructed'] = False
            self.stats['kept_original'] += 1

        else:  # should_reconstruct is None (UNCERTAIN)
            # UNCERTAIN - apply conservative or aggressive strategy
            if self.conservative:
                # Conservative: keep original lemma for uncertain cases
                result['lemma'] = current_lemma
                result['reconstructed'] = False
                result['reason'] += " (kept original - conservative mode)"
                self.stats['uncertain'] += 1
            else:
                # Aggressive: reconstruct if confidence above threshold
                if confidence >= self.min_confidence:
                    reconstructed = self.reconstruct_compound_lemma(
                        parsed['first_component'],
                        parsed['second_component']
                    )
                    result['lemma'] = reconstructed
                    result['reconstructed'] = True
                    result['reason'] += " (reconstructed - aggressive mode)"
                    self.stats['reconstructed'] += 1
                else:
                    result['lemma'] = current_lemma
                    result['reconstructed'] = False
                    result['reason'] += " (confidence too low)"
                    self.stats['uncertain'] += 1

        return result

    def get_statistics(self) -> Dict:
        """
        Get processing statistics.

        Returns:
            Dict with counts and percentages
        """
        total = self.stats['total_processed']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'reconstructed_pct': self.stats['reconstructed'] / total * 100,
            'kept_original_pct': self.stats['kept_original'] / total * 100,
            'uncertain_pct': self.stats['uncertain'] / total * 100,
            'parse_failed_pct': self.stats['parse_failed'] / total * 100
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0


def main():
    """
    Example usage and testing.
    """
    reconstructor = CompoundReconstructor(conservative=True, min_confidence=0.75)

    # Test cases
    test_cases = [
        # (word, lemma, analysis, expected_result)
        (
            "jouhileuan",
            "jouhi",
            "[WORD_ID=jouhi][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=leuka][UPOS=NOUN][NUM=SG][CASE=GEN]",
            "Should reconstruct to jouhileuka"
        ),
        (
            "kantajani",
            "kanta",
            "[WORD_ID=kanta][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=ja][UPOS=NOUN][NUM=SG][CASE=GEN][POSS=SG1]",
            "Should keep as kanta (false compound)"
        ),
        (
            "polviansa",
            "polvi",
            "[WORD_ID=polvi][UPOS=NOUN][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=ansa][UPOS=NOUN][NUM=SG][CASE=NOM]",
            "Uncertain - keep original in conservative mode"
        ),
    ]

    print("=" * 80)
    print("COMPOUND RECONSTRUCTOR TEST")
    print("=" * 80)
    print(f"Mode: {'Conservative' if reconstructor.conservative else 'Aggressive'}")
    print(f"Min confidence: {reconstructor.min_confidence}")
    print()

    for word, lemma, analysis, expected in test_cases:
        result = reconstructor.process_analysis(word, lemma, analysis)

        print(f"Word: {word}")
        print(f"  Current lemma: {lemma}")
        print(f"  Result lemma: {result['lemma']}")
        print(f"  Reconstructed: {result['reconstructed']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Reason: {result['reason']}")
        print(f"  Expected: {expected}")
        print()

    # Print statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = reconstructor.get_statistics()
    for key, value in stats.items():
        if key.endswith('_pct'):
            print(f"{key}: {value:.1f}%")
        else:
            print(f"{key}: {value}")


if __name__ == '__main__':
    main()
