# Finnish Runosong Corpus Morphological Lemmatizer

A hybrid Finnish dialectal poetry lemmatization system combining multiple NLP tools (Stanza, Omorfi, Voikko) with morphological feature analysis for improved accuracy on dialectal Finnish texts.


## Overview

This repository contains a production-ready lemmatization system specifically designed for Finnish runosongs, achieving 60.4% exact match accuracy with Phase 12 compound integration on manual annotations test set with expanded lexicon and V2 dialectal dictionary integration. The system uses a multi-tier fallback strategy with morphological feature awareness, compound reconstruction, and 19,385 validated dialectal variants to handle the linguistic complexity of historical Finnish dialects.

## System Architecture

### Core Components

1. **`fin_runocorp_base_v2_dialectal_dict_integrated.py`** - V2 lemmatization engine (recommended)
   - Multi-tier fallback chain with dialectal dictionary integration
   - 19,385 dialectal variants from Suomen Murteiden Sanakirja (SMS)
   - Confidence-based override for spelling correction guesses
   - Morphological feature extraction and similarity scoring
   - Dialectal normalization integration
   - POS-aware lemma selection
   - **59.9% accuracy** on test set with expanded lexicon

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

- **`selftraining_lexicon_v16_min1.json`** - Train-only baseline lexicon (681 KB)
  - 3,626 unambiguous (word, POS) patterns
  - 74 ambiguous patterns tracked
  - Tier 1 (Gold Standard) entries from training data
  - **Baseline performance: 59.0% accuracy**

- **`selftraining_lexicon_train_with_additions.json`** - Expanded train-only lexicon (1.1 MB) **[RECOMMENDED]**
  - 5,484 unambiguous (word, POS) patterns (+57.8% over baseline)
  - Adds 2,009 entries from word_normalised and word_lemmatised fields
  - **Achieves 59.9% accuracy (+0.9pp improvement)**
  - Better handling of archaic orthography and dialectal variants
  - **Recommended for evaluation and production use**

- **`selftraining_lexicon_v16_min1_combined.json`** - Combined train+test baseline lexicon (874 KB)
  - 4,612 unambiguous (word, POS) patterns
  - 109 ambiguous patterns tracked
  - Combines both training and test gold standard annotations
  - Not used for evaluation to avoid test data leakage

- **`selftraining_lexicon_comb_with_additions.json`** - Expanded combined lexicon (1.3 MB)
  - 6,864 unambiguous (word, POS) patterns (+48.8% over baseline combined)
  - Maximum coverage for batch processing
  - Includes both baseline and expansion entries

**Test/Train Data:**

Manual gold standard lemmatization of a selected part of Finnish runosong corpus (not currently included in github repo)

- **`finnish_poems_gold_test_clean.csv`** - Test dataset (187 KB)
  - 24 Finnish poems, ~1,468 words
 
- **`finnish_poems_gold_train_clean.csv`** - Training dataset
  - Training data for lexicon development
  

**Results and Output Files:**

- **`skvr_lemmatized_results_expanded.csv`** - SKVR corpus lemmatization with expanded lexicon (20 MB) **[RECOMMENDED]**
  - Sample lemmatization using expanded lexicon (`selftraining_lexicon_comb_with_additions.json`)
  - Generated using V2 lemmatizer with dialectal dictionary integration
  - 147,375 lemmatized words from SKVR runosong corpus
  - 10 columns: p_id, nro, poemTitle, word_index, word, lemma, method, confidence, context_score, analysis
  - **Improved lemmatization quality** over baseline (see `batch_lemma_differences.csv` for comparison)

- **`batch_lemma_differences.csv`** - Lemmatization quality comparison (7.4 KB)
  - 99 lemma differences between baseline and expanded lexicon (first 2,000 words)
  - Shows word, baseline_lemma, baseline_method, expanded_lemma, expanded_method
  - Demonstrates improvement patterns from lexicon expansion

- **`skvr_lemmatized_results_with_lexicon.csv`** - SKVR corpus baseline results
  - Generated using baseline combined lexicon (`selftraining_lexicon_v16_min1_combined.json`)
  - Used for comparison with expanded lexicon results

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

### V17 Phase 10 Results (Current Version - with Dialectal Dictionary)

The V2 lemmatizer with Finnish Dialectal Dictionary integration.

### V17 Phase 9 Results (Baseline)

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

### V17 Phase 10 Results (V2 with Dialectal Dictionary - Baseline)

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

### V17 Phase 10 with Expanded Lexicon (Current Best) **[RECOMMENDED]**

The expanded lexicon adds **2,009 word forms** from `word_normalised` and `word_lemmatised` training data fields, improving accuracy by **+0.9pp** over baseline.

```
Total test words:    1,468
Exact matches:       879 (59.9%)

Lexicon Expansion:
- Baseline lexicon:         3,626 patterns (59.0% accuracy)
- Expanded lexicon:         5,484 patterns (+2,009 entries)
- Accuracy improvement:     +0.9pp (59.0% → 59.9%)
- Coverage improvement:     +57.8% more word forms

Method Performance:
- Lexicon (Tier 1):         7 correct
- Lexicon (Tier 3):         66 correct (NEW - from expansion)
- Omorfi contextual:        339 correct
- Voikko + Omorfi:          131 correct
- Dialectal Dictionary:     3 correct
- Fuzzy morphological:      69 correct
- Fuzzy aggressive:         1 correct
```

**Expansion Benefits:**
- **Archaic orthography**: Better handling of historical spelling variants (ſ→s, ŋ→ng, w→v)
- **Dialectal coverage**: More regional word forms from training data
- **Alternative spellings**: Additional variants improve recognition
- **Production-ready**: Recommended for all new lemmatization tasks

**Files:**
- **Lemmatizer**: `fin_runocorp_base_v2_dialectal_dict_integrated.py`
- **Lexicon**: `selftraining_lexicon_train_with_additions.json` (5,484 words)
- **Dictionary**: `sms_dialectal_index_v4_final.json` (19,385 dialectal variants)
- **Batch Script**: `process_skvr_batch_v2.py`
- **Evaluation**: `evaluate_train_expanded_FAIR.py`
- **Results**: `finnish_lemma_evaluation_train_expanded_FAIR.csv`

### V17 Phase 12 with Compound Integration (Current Production) **[RECOMMENDED]**

The Phase 12 refactoring adds **compound reconstruction** for better handling of Finnish compound words through modular refactored components, achieving **60.4% accuracy** (+0.5pp improvement).

```
Total test words:    1,468
Exact matches:       886 (60.4%)

Phase 12 Improvements:
- Compound reconstruction:      +0.5pp accuracy improvement (59.9% → 60.4%)
- True compounds fixed:         15/79 (19% compound accuracy)
- Conservative classification:  Prevents false positives from possessives
- Refactored architecture:      Modular components (lemmatizer.py, lemmatizer_core.py)

Compound Examples Fixed:
- valdaherra → valtaherra (mighty lord)
- kirjalehti → kirjalehti (book page)
- olvitynnyri → olvitynnyri (beer barrel)
- pahnakimppu → pahnakimppu (chaff bundle)
- maakukka → maakukka (ground flower)
```

**Refactored Components:**
- **lemmatizer.py** - Main lemmatizer with compound integration
- **lemmatizer_core.py** - Core processing logic with compound reconstruction
- **lemmatizer_config.py** - Configuration management
- **compound_classifier.py** - Conservative compound classification
- **compound_reconstructor.py** - Compound reconstruction logic
- **process_skvr_batch_REFACTORED.py** - Batch processing with compound support
- **evaluate_train_expanded_FAIR_REFACTORED.py** - Updated evaluation script

**Documentation:**
- See `BATCH_PROCESSING_REFACTORED.md` for batch processing guide
- See `REFACTORED_USAGE_GUIDE.md` for API documentation


## Installation


See: INSTALLATION.md

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

For processing large Finnish poetry corpora (e.g., SKVR with 170,668 poems), use the V2 batch processing script with dialectal dictionary integration and expanded lexicon (recommended):

```bash
# V2 with expanded train-only lexicon + dialectal dictionary (RECOMMENDED)
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --lexicon-path selftraining_lexicon_train_with_additions.json \
  --chunk-size 100 \
  --save-interval 120

# V2 with expanded combined lexicon (maximum coverage)
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --lexicon-path selftraining_lexicon_comb_with_additions.json \
  --chunk-size 100 \
  --save-interval 120

# V2 with baseline train-only lexicon (for comparison)
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --chunk-size 100 \
  --save-interval 120

# V2 with custom dialectal dictionary
python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --dialectal-dict-path custom_dialectal_index.json
```

**Features:**
- **Expanded lexicon** - 5,484 word forms for improved accuracy
- **Dialectal dictionary integration** - 19,385 validated variants from SMS
- **Confidence-based override** - Surgical fixes for spelling correction guesses
- **Incremental saves** - Results saved every 120 seconds (configurable)
- **Checkpoint/resume** - Automatically resumes from interruptions
- **Progress tracking** - Real-time progress bar with ETA
- **CSV output** - 10 columns: p_id, nro, poemTitle, word_index, word, lemma, method, confidence, context_score, analysis
- **Lexicon selection** - Use `--lexicon-path` to specify train-only or combined lexicon
- **Dictionary selection** - Use `--dialectal-dict-path` for custom dialectal dictionary

**Performance:**
- **Test set with expanded lexicon:** 59.0% → 59.9% (+0.9pp, +13 correct)
- **Test set baseline:** 58.8% → 59.0% (+0.2pp, +3 correct)
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
  --lexicon-path selftraining_lexicon_comb_with_additions.json
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t skvr_processing

# Option 2: Using nohup
nohup python3 process_skvr_batch_v2.py \
  --input skvr_runosongs_okt_2025.csv \
  --output skvr_lemmatized_results_v2.csv \
  --lexicon-path selftraining_lexicon_comb_with_additions.json \
  > skvr_processing_v2.log 2>&1 &

# Monitor progress
tail -f skvr_processing_v2.log
wc -l skvr_lemmatized_results_v2.csv  # Check word count
```

**Processing Time Estimates:**
- 10 poems: ~2-3 minutes
- 100 poems: ~20-30 minutes
- 170,668 poems (full SKVR): ~2-4 days


**V1 Baseline Script:** The original `process_skvr_batch.py` (without dialectal dictionary) remains available for baseline comparisons and backward compatibility.

### Basic Lemmatization

```python
from fin_runocorp_base import FinnishRunosongLemmatizer

# Initialize lemmatizer with expanded train-only lexicon (RECOMMENDED)
lemmatizer = FinnishRunosongLemmatizer(
    model_dir=None,  # Uses default Stanza model
    voikko_path='/path/to/.voikko',
    lang='fi',
    lexicon_path='selftraining_lexicon_train_with_additions.json'
)

# Or use expanded combined lexicon for maximum coverage
lemmatizer = FinnishRunosongLemmatizer(
    model_dir=None,
    voikko_path='/path/to/.voikko',
    lang='fi',
    lexicon_path='selftraining_lexicon_comb_with_additions.json'
)

# Or use baseline train-only lexicon (for comparison)
lemmatizer = FinnishRunosongLemmatizer(
    model_dir=None,
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

**V2 Evaluation with Expanded Lexicon (RECOMMENDED):**
```bash
python3 evaluate_train_expanded_FAIR.py
```

Generates detailed CSV files with evaluation results using expanded lexicon:

**1. `finnish_lemma_evaluation_train_expanded_FAIR.csv`** (Main evaluation results - 59.9% accuracy)
- Complete word-by-word evaluation of all 1,468 test words
- Shows predicted lemma vs. manual gold standard for each word
- Includes which lemmatization method was used (lexicon, omorfi_contextual, voikko_omorfi, dialectal_dictionary, etc.)
- Contains poem metadata (poem_id, verse, location, year)
- Indicates whether each prediction was correct
- **Best performance: 59.9% accuracy with expanded lexicon**
- Use this file for detailed error analysis and method performance comparison

**2. `batch_lemma_differences.csv`** (Comparison with baseline)
- Shows 99 lemma differences between baseline and expanded lexicon
- Demonstrates quality improvements from lexicon expansion
- Useful for understanding expansion benefits

**V2 Baseline Evaluation (with Dialectal Dictionary):**
```bash
python3 evaluate_v17_phase10.py
```

Generates detailed CSV files with baseline evaluation results (59.0% accuracy):

**1. `finnish_lemma_evaluation_v17_phase10.csv`** (Baseline evaluation results)
- Complete word-by-word evaluation with baseline lexicon
- 59.0% accuracy (baseline for comparison)

**2. `finnish_lemma_evaluation_v17_phase10_dialectal_dictionary_analysis.csv`** (Dictionary usage analysis)
- Shows all instances where dialectal dictionary was used
- Includes original method that was overridden
- 5 instances total with 80% success rate (4/5 correct)
- Useful for understanding dictionary override effectiveness

**V1 Baseline Evaluation (for comparison):**
```bash
python3 evaluate_v17_phase9.py
```

Generates V1 baseline results without dialectal dictionary for performance comparison (58.8% accuracy).

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

**Current Version**: V17 Phase 12 with Compound Integration (60.4% accuracy)
**Status**: Production-ready
**Last Updated**: 2025-11-04

### Version History
- **V17 Phase 6**: Baseline (58.3% accuracy)
- **V17 Phase 7**: Voikko ranking improvements
- **V17 Phase 8**: Fuzzy lexicon matching
- **V17 Phase 9**: Morphological feature integration (58.8% accuracy)
- **V17 Phase 10**: Expanded lexicon (59.9% accuracy)
- **V17 Phase 12**: Compound integration + refactored modules (60.4% accuracy) ✅

## Contact

For questions or issues, please contact the repository maintainer (kaarel.veskis@kirmus.ee).

## Acknowledgments

- **Stanza**: Stanford NLP Group
- **Omorfi**: Open Morphology of Finnish
- **Voikko**: Finnish linguistic software
- **SKVR Corpus**: Finnish Literature Society (*Suomalaisen Kirjallisuuden Seura*)

This repository contains a derived JSON lexicon index built from  
*Suomen murteiden sanakirja* (Dictionary of Finnish Dialects), Institute for the Languages of Finland (Kotus).

The original dictionary is available online via Kotus at:  
- Suomen murteiden sanakirja info page: https://kotus.fi/sanakirjat/suomen-murteiden-sanakirja/  
- Online dictionary interface: http://kaino.kotus.fi/sms/

Original data © Kotimaisten kielten keskus (Kotus), licensed under  
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The data in `data/sms_lexicon.json` has been transformed and restructured from the original XML release;  
any errors or omissions are the responsibility of this project.


	
