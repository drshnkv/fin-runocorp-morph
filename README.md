# Finnish Runosong Corpus Morphological Lemmatizer

A hybrid Finnish dialectal poetry lemmatization system combining multiple NLP tools (Stanza, Omorfi, Voikko) with morphological feature analysis for improved accuracy on dialectal Finnish texts.

## Overview

This repository contains a production-ready lemmatization system specifically designed for Finnish Kalevala-meter poetry (*runokorpus*), achieving 58.8% exact match accuracy on dialectal Finnish gold standard test data. The system uses a multi-tier fallback strategy with morphological feature awareness to handle the linguistic complexity of historical Finnish dialects.

## System Architecture

### Core Components

1. **`fin_runocorp_base.py`** - Main lemmatization engine
   - Multi-tier fallback chain (lexicon → Omorfi → Voikko → fuzzy matching)
   - Morphological feature extraction and similarity scoring
   - Dialectal normalization integration
   - POS-aware lemma selection

2. **`dialectal_normalizer.py`** - Dialectal variant normalization
   - Handles h-variation (h-insertion/deletion)
   - Geminate consonant normalization
   - Vowel length standardization
   - Case-ending variations

3. **`evaluate_v17_phase9.py`** - Evaluation script
   - Gold standard comparison
   - Detailed accuracy metrics
   - Ambiguous pattern analysis
   - Method performance tracking

### Supporting Resources

- **`selftraining_lexicon_v16_min1.json`** - Self-training lexicon (681 KB)
  - 3,626 unambiguous (word, POS) patterns
  - 74 ambiguous patterns tracked
  - Tier 1 (Gold Standard) entries

- **`finnish_poems_gold_test_clean.csv`** - Test dataset (187 KB)
  - 24 Finnish poems, ~1,468 words
  - Manual gold standard lemmatization
  - Dialect/region annotations

- **`finnish_poems_gold_train_clean.csv`** - Training dataset
  - Training data for lexicon development
  - Additional dialectal examples

## Lemmatization Strategy

### 8-Tier Fallback Chain

1. **Lexicon exact match** (Tier 1: Gold Standard)
2. **Omorfi contextual analysis** with morphological features
3. **Voikko + Omorfi hybrid** with multi-criteria ranking
4. **Omorfi direct** (single candidate)
5. **Voikko normalized** (dialectal variants)
6. **Enhanced Voikko** (expanded analysis)
7. **Fuzzy lexicon matching** (Levenshtein distance ≤ 2.0)
8. **Identity fallback** (word form = lemma)

### Morphological Feature Integration

- **Universal Features extraction**: Case, Number, Tense, Mood, Voice, Person
- **Feature similarity scoring**: Weighted agreement between candidates
- **POS-aware selection**: Part-of-speech filtering and preferences
- **Dialectal awareness**: Normalized forms considered in matching

## Performance Metrics

### V17 Phase 9 Results (Current Version)

```
Total test words:    1,468
Exact matches:       863 (58.8%)

Method Performance:
- Lexicon (Tier 1):      7 correct
- Omorfi contextual:     391 correct
- Voikko + Omorfi:       130 correct
- Fuzzy lexicon:         0 correct
- Identity fallback:     0 correct (1 attempt)

Ambiguous Handling:
- Ambiguous accuracy:    79.5% (35/44 correct)
- Words with ambiguity:  20 unique forms
```

### Key Improvements

- **+0.5% over Phase 6** (58.3% → 58.8%)
- **Voikko ranking**: Multi-criteria selection (+2 correct instances)
- **Feature-aware matching**: 2,051 feature comparisons performed
- **Ambiguity resolution**: High success rate on polysemous forms

## Installation

### Requirements

```bash
# Python 3.8+
pip install stanza
pip install omorfi
pip install voikko

# Download Stanza Finnish model
python -c "import stanza; stanza.download('fi')"
```

### Voikko Dictionaries

- **Voikko Sukija**: Old Finnish dictionary
- **Voikko Standard**: Modern Finnish
- Paths configured via `voikko_path` parameter

### Optional Dependencies

```bash
pip install hfst  # For Omorfi morphological analysis
```

## Usage

### Basic Lemmatization

```python
from fin_runocorp_base import FinnishRunosongLemmatizer

# Initialize lemmatizer
lemmatizer = FinnishRunosongLemmatizer(
    model_dir=None,  # Uses default Stanza model
    voikko_path='/path/to/.voikko',
    lang='fi'
)

# Lemmatize text
text = "Vanhoilda silmät valu"
results = lemmatizer.lemmatize_text(text)

for token in results:
    print(f"{token['word']} → {token['lemma']} ({token['method']})")
```

### Evaluation

```bash
python3 evaluate_v17_phase9.py
```

Generates two CSV files with evaluation results:

**1. `finnish_lemma_evaluation_v17_phase9.csv`** (Main evaluation results)
- Complete word-by-word evaluation of all 1,468 test words
- Shows predicted lemma vs. manual gold standard for each word
- Includes which lemmatization method was used (lexicon, omorfi_contextual, voikko_omorfi, etc.)
- Contains poem metadata (poem_id, verse, location, year)
- Indicates whether each prediction was correct
- Use this file for detailed error analysis and method performance comparison

**2. `finnish_lemma_evaluation_v17_phase9_ambiguous_analysis.csv`** (Ambiguous words analysis)
- Subset of 44 instances involving 20 unique word forms that have multiple possible lemmas
- Example: "on" (verb "to be") can appear in different grammatical contexts
- Shows how well the system handles polysemous/homonymous words
- Achieves 79.5% accuracy (35/44 correct) on these challenging cases
- Useful for understanding ambiguity resolution performance

## Configuration

### LemmatizerConfig Class

```python
from fin_runocorp_base import LemmatizerConfig

config = LemmatizerConfig(
    fuzzy_threshold=2.0,              # Levenshtein distance threshold
    min_similarity_score=0.6,         # Minimum similarity for fuzzy matching
    feature_weight_case=2.0,          # Case agreement weight
    feature_weight_number=2.0,        # Number agreement weight
    feature_weight_tense=1.5,         # Tense agreement weight
    voikko_max_candidates=10,         # Max Voikko candidates to consider
    enable_feature_scoring=True,      # Enable morphological features
    enable_fuzzy_fallback=True        # Enable fuzzy lexicon matching
)

lemmatizer = FinnishRunosongLemmatizer(config=config)
```

## Key Features

### 1. Morphological Feature Extraction
- Parses HFST/Omorfi tags → Universal Dependencies features
- Extracts: Case, Number, Tense, Mood, Voice, Person, Degree
- Handles compound tags and nested structures

### 2. Feature Similarity Scoring
- Weighted agreement calculation
- Configurable weights per feature type
- Bonus system for multi-feature matches

### 3. Dialectal Normalization
- h-variation handling (e.g., *tukkia* ↔ *tuhkia*)
- Geminate consonant patterns
- Vowel length variations
- Regional morphology differences

### 4. Ambiguity Resolution
- Contextual POS preferences
- Frequency-based ranking
- Morphological plausibility checks
- Multi-candidate tracking

### 5. Performance Optimization
- Cached Omorfi analyses
- Efficient Voikko lookups
- Lexicon-first strategy
- Early termination on exact matches

## Corpus Context

This lemmatizer was developed for the **Finnish Kalevala-meter poetry corpus** (*Suomen Kansan Vanhat Runot*, SKVR), which presents unique challenges:

- **Dialectal variation**: Multiple regional Finnish dialects
- **Archaic forms**: Historical Finnish from 17th-20th centuries
- **Poetic license**: Non-standard morphology for meter
- **Compound words**: Complex nominal and verbal compounds
- **h-variation**: Systematic h-insertion/deletion patterns

## Accuracy Considerations

### Strong Performance Areas
- High-frequency standard forms (lexicon coverage)
- Contextual disambiguation with clear POS signals
- Voikko-assisted dialectal normalization

### Challenging Cases
- Rare dialectal forms not in lexicon
- Ambiguous word forms requiring semantic context
- Compound words with non-standard splitting
- Poetic contractions and elisions

### Known Limitations
- Identity fallback rare but necessary (1 case in test set)
- Some proper nouns require specialized handling
- Fuzzy matching currently contributes 0 correct instances

## Development Status

**Current Version**: V17 Phase 9 (Morphology-Aware Fuzzy Matching)
**Status**: Production-ready
**Last Updated**: 2025-11-02

### Version History
- **V17 Phase 6**: Baseline (58.3% accuracy)
- **V17 Phase 7**: Voikko ranking improvements
- **V17 Phase 8**: Fuzzy lexicon matching
- **V17 Phase 9**: Morphological feature integration (58.8% accuracy)

## Citation

If you use this lemmatizer in your research, please cite:

```
Finnish Runosong Corpus Morphological Lemmatizer (2025)
https://github.com/drshnkv/fin-runocorp-morph
```

## License

[Specify license here]

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.

## Acknowledgments

- **Stanza**: Stanford NLP Group
- **Omorfi**: Open Morphology of Finnish
- **Voikko**: Finnish linguistic software
- **SKVR Corpus**: Finnish Literature Society (*Suomalaisen Kirjallisuuden Seura*)
