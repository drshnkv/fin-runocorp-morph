#!/usr/bin/env python3
"""
Dialectal Normalization Module for Finnish Runosong Lemmatization
Uses confusion-weighted edit distance based on corpus analysis
"""

import sys
from typing import List, Tuple, Dict

class DialectalNormalizer:
    """
    Normalizes dialectal Finnish words using weighted edit distance
    Based on systematic patterns from 2,729 coverage failure analysis
    """

    def __init__(self):
        """Initialize dialectal normalization weights"""

        # Archaic orthography (cost = 0.05 - nearly free)
        # From analysis: w→v (155), ſ→s (124), c→k (85), z→s (35)
        self.ARCHAIC_ORTHOGRAPHY = {
            ('w', 'v'): 0.05,
            ('v', 'w'): 0.05,
            ('ſ', 's'): 0.05,
            ('s', 'ſ'): 0.05,
            ('c', 'k'): 0.05,
            ('k', 'c'): 0.05,
            ('z', 's'): 0.05,
            ('s', 'z'): 0.05,
        }

        # Systematic sound changes (cost = 0.2-0.3)
        # From analysis: d→t (98), i↔e (183), o→a (83), etc.
        self.SOUND_CHANGES = {
            ('d', 't'): 0.2,
            ('t', 'd'): 0.2,
            ('g', 'k'): 0.25,
            ('k', 'g'): 0.25,
            ('i', 'e'): 0.3,
            ('e', 'i'): 0.3,
            ('o', 'a'): 0.3,
            ('a', 'o'): 0.3,
            ('ä', 'e'): 0.3,
            ('e', 'ä'): 0.3,
            ('ö', 'o'): 0.3,
            ('o', 'ö'): 0.3,
            ('ü', 'u'): 0.3,
            ('u', 'ü'): 0.3,
            ('y', 'i'): 0.3,
            ('i', 'y'): 0.3,
            ('u', 'o'): 0.35,
            ('o', 'u'): 0.35,
        }

        # Consonant gradation (cost = 0.3)
        self.GRADATION = {
            ('kk', 'k'): 0.3,
            ('k', 'kk'): 0.3,
            ('pp', 'p'): 0.3,
            ('p', 'pp'): 0.3,
            ('tt', 't'): 0.3,
            ('t', 'tt'): 0.3,
            ('gg', 'g'): 0.3,
            ('g', 'gg'): 0.3,
        }

        # Case ending characters (high deletion/insertion frequency)
        self.CASE_ENDING_CHARS = set('niets')

        # H-variation cost
        self.H_VARIATION_COST = 0.1

        # Default costs
        self.DEFAULT_SUB_COST = 1.0
        self.DEFAULT_INDEL_COST = 0.5
        self.END_POSITION_COST = 0.1  # For case endings

    def get_substitution_cost(self, c1: str, c2: str) -> float:
        """Get cost of substituting c1 with c2"""
        if c1 == c2:
            return 0.0

        # Check archaic orthography
        if (c1, c2) in self.ARCHAIC_ORTHOGRAPHY:
            return self.ARCHAIC_ORTHOGRAPHY[(c1, c2)]

        # Check sound changes
        if (c1, c2) in self.SOUND_CHANGES:
            return self.SOUND_CHANGES[(c1, c2)]

        # Default
        return self.DEFAULT_SUB_COST

    def get_deletion_cost(self, char: str, position: int, word_length: int) -> float:
        """Get cost of deleting char at position"""
        # H-variation
        if char.lower() == 'h':
            return self.H_VARIATION_COST

        # Case endings at end of word
        if position >= word_length - 2 and char.lower() in self.CASE_ENDING_CHARS:
            return self.END_POSITION_COST

        # Default
        return self.DEFAULT_INDEL_COST

    def get_insertion_cost(self, char: str, position: int, lemma_length: int) -> float:
        """Get cost of inserting char at position"""
        # H-variation
        if char.lower() == 'h':
            return self.H_VARIATION_COST

        # Case endings at end of lemma
        if position >= lemma_length - 2 and char.lower() in self.CASE_ENDING_CHARS:
            return self.END_POSITION_COST

        # Default
        return self.DEFAULT_INDEL_COST

    def weighted_edit_distance(self, word: str, lemma: str) -> float:
        """
        Calculate weighted edit distance between word and lemma
        Uses dialectal-aware costs for operations

        Returns:
            Float distance (lower = more similar)
        """
        word = word.lower()
        lemma = lemma.lower()

        m, n = len(word), len(lemma)

        # Initialize DP table
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        # Base cases
        dp[0][0] = 0.0
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + self.get_deletion_cost(word[i-1], i-1, m)
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + self.get_insertion_cost(lemma[j-1], j-1, n)

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word[i-1] == lemma[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Match
                else:
                    # Check for consonant gradation (2-char patterns)
                    gradation_cost = float('inf')
                    if i >= 2 and j >= 1:
                        two_char_word = word[i-2:i]
                        one_char_lemma = lemma[j-1]
                        if (two_char_word, one_char_lemma) in self.GRADATION:
                            gradation_cost = dp[i-2][j-1] + self.GRADATION[(two_char_word, one_char_lemma)]

                    if i >= 1 and j >= 2:
                        one_char_word = word[i-1]
                        two_char_lemma = lemma[j-2:j]
                        if (one_char_word, two_char_lemma) in self.GRADATION:
                            gradation_cost = min(gradation_cost,
                                                dp[i-1][j-2] + self.GRADATION[(one_char_word, two_char_lemma)])

                    # Standard operations
                    sub_cost = dp[i-1][j-1] + self.get_substitution_cost(word[i-1], lemma[j-1])
                    del_cost = dp[i-1][j] + self.get_deletion_cost(word[i-1], i-1, m)
                    ins_cost = dp[i][j-1] + self.get_insertion_cost(lemma[j-1], j-1, n)

                    dp[i][j] = min(sub_cost, del_cost, ins_cost, gradation_cost)

        return dp[m][n]

    def rank_suggestions(self, word: str, suggestions: List[str]) -> List[Tuple[str, float]]:
        """
        Rank suggestions by weighted edit distance

        Args:
            word: Original dialectal word
            suggestions: List of candidate lemmas

        Returns:
            List of (lemma, distance) tuples, sorted by distance (best first)
        """
        scored = []
        for suggestion in suggestions:
            distance = self.weighted_edit_distance(word, suggestion)
            scored.append((suggestion, distance))

        # Sort by distance (lower is better)
        scored.sort(key=lambda x: x[1])
        return scored

    def normalize_archaic_orthography(self, word: str) -> str:
        """
        Quick normalization of archaic orthography
        Useful as pre-processing step before main lemmatization

        Returns:
            Normalized word with archaic chars replaced
        """
        result = word
        result = result.replace('w', 'v')
        result = result.replace('W', 'V')
        result = result.replace('ſ', 's')
        result = result.replace('c', 'k')
        result = result.replace('C', 'K')
        result = result.replace('z', 's')
        result = result.replace('Z', 'S')
        return result


def test_dialectal_normalizer():
    """Test dialectal normalizer with examples from analysis"""
    normalizer = DialectalNormalizer()

    print("="*80)
    print("TESTING DIALECTAL NORMALIZER")
    print("="*80)

    # Test archaic orthography (should have very low distance)
    print("\n1. ARCHAIC ORTHOGRAPHY (expected distance < 0.1):")
    test_cases = [
        ("wesi", "vesi"),    # water: w→v
        ("ſaa", "saa"),      # get: ſ→s
        ("cansa", "kansa"),  # people: c→k
    ]
    for word, lemma in test_cases:
        dist = normalizer.weighted_edit_distance(word, lemma)
        print(f"   {word:15s} → {lemma:15s}: {dist:.3f}")

    # Test sound changes (should have moderate distance)
    print("\n2. SOUND CHANGES (expected distance < 0.5):")
    test_cases = [
        ("kidu", "kito"),    # d→t
        ("kivi", "keve"),    # i→e
        ("koko", "kaka"),    # o→a
    ]
    for word, lemma in test_cases:
        dist = normalizer.weighted_edit_distance(word, lemma)
        print(f"   {word:15s} → {lemma:15s}: {dist:.3f}")

    # Test case endings (should have low distance at end)
    print("\n3. CASE ENDINGS (expected distance < 0.3 for end positions):")
    test_cases = [
        ("vesin", "vesi"),      # Genitive -n
        ("vettä", "vesi"),      # Partitive
        ("vedellä", "vesi"),    # Adessive
    ]
    for word, lemma in test_cases:
        dist = normalizer.weighted_edit_distance(word, lemma)
        print(f"   {word:15s} → {lemma:15s}: {dist:.3f}")

    # Test ranking with multiple suggestions
    print("\n4. RANKING SUGGESTIONS:")
    word = "wesi"
    suggestions = ["vesi", "väsi", "kesi", "tesi", "mesi"]
    ranked = normalizer.rank_suggestions(word, suggestions)
    print(f"   Word: {word}")
    print(f"   Ranked suggestions:")
    for lemma, dist in ranked:
        print(f"      {lemma:15s}: {dist:.3f}")

    # Test archaic normalization
    print("\n5. ARCHAIC NORMALIZATION (pre-processing):")
    test_words = ["wesi", "ſaa", "cansa", "zouti"]
    for word in test_words:
        normalized = normalizer.normalize_archaic_orthography(word)
        print(f"   {word:15s} → {normalized}")

    print("\n" + "="*80)
    print("✓ Dialectal normalizer tests complete!")


if __name__ == '__main__':
    test_dialectal_normalizer()
