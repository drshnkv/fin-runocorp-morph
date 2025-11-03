# Finnish Runosong Corpus Morphological Lemmatizer

A hybrid Finnish dialectal poetry lemmatization system combining multiple NLP tools (Stanza, Omorfi, Voikko) with morphological feature analysis for improved accuracy on dialectal Finnish texts.

## Usage and Contact

If you intend to use these resources, please contact first: kaarel.veskis@kirmus.ee

## Overview

This repository contains a production-ready lemmatization system specifically designed for Finnish Kalevala-meter poetry (*runokorpus*), achieving 59.0% exact match accuracy on dialectal Finnish gold standard test data with V2 dialectal dictionary integration. The system uses a multi-tier fallback strategy with morphological feature awareness and 19,385 validated dialectal variants to handle the linguistic complexity of historical Finnish dialects.

## System Architecture

### Core Components

1. **`fin_runocorp_base_v2_dialectal_dict_integrated.py`** - V2 lemmatization engine (recommended)
   - Multi-tier fallback chain with dialectal dictionary integration
   - 19,385 validated dialectal variants from Suomen Murteiden Sanakirja (SMS)
   - Confidence-based override for spelling correction guesses
   - Morphological feature extraction and similarity scoring
   - Dialectal normalization integration
   - POS-aware lemma selection
   - **59.0% accuracy** on test set

2. **`fin_runocorp_base.py`** - V1 lemmatization engine (baseline)
   - Multi-tier fallback chain (lexicon → Omorfi → Voikko → fuzzy matching)
   - Morphological feature extraction and similarity scoring
   - Dialectal normalization integration
   - POS-aware lemma selection
   - **58.8% accuracy** on test set

3. **`dialectal_normalizer.py`** - Dialectal variant normalization
   - Handles h-variation (h-insertion/deletion)
   - Geminate consonant normalization
   - Vowel length standardization
   - Case-ending variations

4. **`evaluate_v17_phase10.py`** - V2 evaluation script (current)
   - Gold standard comparison with dialectal dictionary tracking
   - Detailed accuracy metrics
   - Dictionary usage and success rate analysis
   - Method performance tracking

5. **`evaluate_v17_phase9.py`** - V1 evaluation script (baseline)
   - Gold standard comparison
   - Detailed accuracy metrics
   - Ambiguous pattern analysis
   - Method performance tracking

### Supporting Resources

**Dialectal Dictionary:**

- **`sms_dialectal_index_v4_final.json`** - Finnish Dialectal Dictionary (2.2 MB)
  - 19,385 validated dialectal variants from Suomen Murteiden Sanakirja (SMS)
  - POS-tagged entries with confidence scores (0.9)
  - Alphabetically sorted for efficient lookup
  - **Used in V2 lemmatizer** for dialectal variant recognition
  - 80% success rate (4/5 correct) on test set

**Lexicons:**

- **`selftraining_lexicon_v16_min1.json`** - Train-only lexicon (681 KB)
  - 3,626 unambiguous (word, POS) patterns
  - 74 ambiguous patterns tracked
  - Tier 1 (Gold Standard) entries from training data
  - **Used for V1 and V2 evaluation metrics**

- **`selftraining_lexicon_v16_min1_combined.json`** - Combined train+test lexicon (874 KB)
  - 4,612 unambiguous (word, POS) patterns
  - 109 ambiguous patterns tracked
  - Combines both training and test gold standard annotations
  - **Recommended for batch processing** (see usage below)
  - Not used for evaluation to avoid test data leakage

**Test/Train Data:**

- **`finnish_poems_gold_test_clean.csv`** - Test dataset (187 KB)
  - 24 Finnish poems, ~1,468 words
  - Manual gold standard lemmatization
  - Dialect/region annotations

- **`finnish_poems_gold_train_clean.csv`** - Training dataset
  - Training data for lexicon development
  - Additional dialectal examples

## Lemmatization Strategy

### V2 Tier Architecture (9-Tier Fallback Chain with Dialectal Dictionary)

1. **Lexicon exact match** (Tier 1: Gold Standard)
2. **Omorfi contextual analysis** with morphological features
3. **Voikko + Omorfi hybrid** with multi-criteria ranking (confidence-based override by dictionary)
4. **Omorfi direct** (single candidate)
5. **Voikko normalized** (dialectal variants)
6. **Enhanced Voikko** (expanded analysis)
7. **Dialectal Dictionary** (SMS - 19,385 validated variants) - **NEW in V2**
8. **Fuzzy lexicon matching** (Levenshtein distance ≤ 2.0)
9. **Identity fallback** (word form = lemma)

**V2 Enhancement:** Dialectal dictionary can also override Tier 3 (Voikko + Omorfi) results when they are spelling correction guesses (medium-high confidence) and dictionary has a different lemma with matching POS.

### V1 Tier Architecture (8-Tier Fallback Chain - Baseline)

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
- Fuzzy lexicon:         74 correct (handles archaic orthography)
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

### V17 Phase 10 Results (V2 with Dialectal Dictionary)

The V2 lemmatizer integrates **19,385 validated dialectal variants** from the Finnish Dialectal Dictionary (Suomen Murteiden Sanakirja - SMS) with confidence-based override logic.

```
Total test words:    1,468
Exact matches:       866 (59.0%)

V2 Improvements:
- Dialectal dictionary used:    5 times
- Dictionary success rate:      80.0% (4/5 correct)
- Accuracy improvement:         +0.2pp over Phase 9 (58.8% → 59.0%)

Method Performance:
- Lexicon (Tier 1):         7 correct
- Omorfi contextual:        391 correct
- Voikko + Omorfi:          129 correct (4 overridden by dictionary)
- Dialectal Dictionary:     4 correct (NEW in Phase 10)
- Fuzzy morphological:      72 correct
- Fuzzy aggressive:         2 correct
```

**Key V2 Features:**
- **Confidence-based override**: Dictionary overrides only spelling correction guesses (voikko_omorfi), not high-confidence direct analyses
- **Fixed spelling errors**: `morsien`→`morsian`, `hahti`→`haahti`, `noit`→`nuo`, `kakla`→`kaula`
- **No regression**: All standard words preserved (high-confidence results protected)
- **POS-aware filtering**: Dictionary lookups respect Stanza part-of-speech context
- **Surgical precision**: Only targets medium-confidence spelling guesses, not direct Omorfi analyses

**Files:**
- **Lemmatizer**: `fin_runocorp_base_v2_dialectal_dict_integrated.py`
- **Dictionary**: `sms_dialectal_index_v4_final.json` (19,385 dialectal variants)
- **Batch Script**: `process_skvr_batch_v2.py`
- **Evaluation**: `evaluate_v17_phase10.py`

**Expected Performance on Dialectal Corpus:**
The current test set contains mostly standard Finnish with manual annotations. When processing true dialectal runosong corpus (SKVR), dictionary usage is expected to be significantly higher (150-250+ instances) with corresponding accuracy improvements.

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

### Batch Processing (Large Corpora)

For processing large Finnish poetry corpora (e.g., SKVR with 170,668 poems), use the V2 batch processing script with dialectal dictionary integration (recommended):

```bash
# V2 with train-only lexicon + dialectal dictionary (default)
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --chunk-size 100 \
  --save-interval 120

# V2 with combined train+test lexicon (recommended for better coverage)
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --lexicon-path selftraining_lexicon_v16_min1_combined.json \
  --chunk-size 100 \
  --save-interval 120

# V2 with custom dialectal dictionary
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --dialectal-dict-path custom_dialectal_index.json
```

**Features:**
- **Dialectal dictionary integration** - 19,385 validated variants from SMS
- **Confidence-based override** - Surgical fixes for spelling correction guesses
- **Incremental saves** - Results saved every 120 seconds (configurable)
- **Checkpoint/resume** - Automatically resumes from interruptions
- **Progress tracking** - Real-time progress bar with ETA
- **CSV output** - 10 columns: p_id, nro, poemTitle, word_index, word, lemma, method, confidence, context_score, analysis
- **Lexicon selection** - Use `--lexicon-path` to specify train-only or combined lexicon
- **Dictionary selection** - Use `--dialectal-dict-path` for custom dialectal dictionary

**Performance:**
- **Test set (standard Finnish):** 58.8% → 59.0% (+0.2pp, +3 correct)
- **Dialectal corpus (SKVR):** Expected 150-250+ dictionary uses with higher accuracy gains

**Resume after interruption:**
```bash
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --resume
```

**For long-running processing (full SKVR corpus):**

The full corpus takes 2-4 days to process. Use tmux or nohup to run in background:

```bash
# Option 1: Using tmux (recommended)
tmux new -s skvr_processing
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --lexicon-path selftraining_lexicon_v16_min1_combined.json
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t skvr_processing

# Option 2: Using nohup
nohup python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --lexicon-path selftraining_lexicon_v16_min1_combined.json \
  > skvr_processing_v2.log 2>&1 &

# Monitor progress
tail -f skvr_processing_v2.log
wc -l skvr_lemmatized_results_v2.csv  # Check word count
```

**Processing Time Estimates:**
- 10 poems: ~2-3 minutes
- 100 poems: ~20-30 minutes
- 170,668 poems (full SKVR): ~2-4 days

**Note:** Tested with `skvr_test_6poems.csv` containing 6 poems (2,022 words processed in 2.6 minutes). The file has multi-line CSV records due to embedded newlines in metadata fields.

**V1 Baseline Script:** The original `process_skvr_batch.py` (without dialectal dictionary) remains available for baseline comparisons and backward compatibility.

### Basic Lemmatization

```python
from fin_runocorp_base import FinnishRunosongLemmatizer

# Initialize lemmatizer with train-only lexicon (default)
lemmatizer = FinnishRunosongLemmatizer(
    model_dir=None,  # Uses default Stanza model
    voikko_path='/path/to/.voikko',
    lang='fi'
)

# Or use combined lexicon for better coverage
lemmatizer = FinnishRunosongLemmatizer(
    model_dir=None,
    voikko_path='/path/to/.voikko',
    lang='fi',
    lexicon_path='selftraining_lexicon_v16_min1_combined.json'
)

# Lemmatize text
text = "Vanhoilda silmät valu"
results = lemmatizer.lemmatize_text(text)

for token in results:
    print(f"{token['word']} → {token['lemma']} ({token['method']})")
```

### Evaluation

**V2 Evaluation (with Dialectal Dictionary):**
```bash
python3 evaluate_v17_phase10.py
```

Generates detailed CSV files with evaluation results:

**1. `finnish_lemma_evaluation_v17_phase10.csv`** (Main evaluation results)
- Complete word-by-word evaluation of all 1,468 test words
- Shows predicted lemma vs. manual gold standard for each word
- Includes which lemmatization method was used (lexicon, omorfi_contextual, voikko_omorfi, dialectal_dictionary, etc.)
- Contains poem metadata (poem_id, verse, location, year)
- Indicates whether each prediction was correct
- Use this file for detailed error analysis and method performance comparison

**2. `finnish_lemma_evaluation_v17_phase10_dialectal_dictionary_analysis.csv`** (Dictionary usage analysis)
- Shows all instances where dialectal dictionary was used
- Includes original method that was overridden
- 5 instances total with 80% success rate (4/5 correct)
- Useful for understanding dictionary override effectiveness

**V1 Baseline Evaluation (for comparison):**
```bash
python3 evaluate_v17_phase9.py
```

Generates V1 baseline results without dialectal dictionary for performance comparison.

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
- Morphological feature weights not yet POS-specific

## Development Status

**Current Version**: V17 Phase 9 (Morphology-Aware Fuzzy Matching)
**Status**: Production-ready
**Last Updated**: 2025-11-02

### Version History
- **V17 Phase 6**: Baseline (58.3% accuracy)
- **V17 Phase 7**: Voikko ranking improvements
- **V17 Phase 8**: Fuzzy lexicon matching
- **V17 Phase 9**: Morphological feature integration (58.8% accuracy)

## Contact

For questions or issues, please contact the repository maintainer (kaarel.veskis@kirmus.ee).

## Acknowledgments

- **Stanza**: Stanford NLP Group
- **Omorfi**: Open Morphology of Finnish
- **Voikko**: Finnish linguistic software
- **SKVR Corpus**: Finnish Literature Society (*Suomalaisen Kirjallisuuden Seura*)
