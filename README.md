# Finnish Runosong Lemmatization Baseline (fin_runocorp_base.py)

**Created:** 2025-11-02
**Source:** `omorfi_hfst_with_voikko_v17_phase9_BACKUP.py` (Nov 1 18:43, 78,243 bytes)
**Baseline Accuracy:** 58.8% (863/1468 exact matches)

## Summary

This folder contains the baseline Finnish runosong lemmatization script that achieves 58.8% accuracy. This is the **ORIGINAL Phase 9 version** that lacks the `parse_stanza_feats()` method - which turns out to be BENEFICIAL for dialectal Finnish.

### The "Beneficial Bug" Discovery - CLARIFICATION

**Important clarification from 2025-11-02 baseline setup:**

The `omorfi_hfst_with_voikko_v17_phase9_BACKUP.py` file (which was copied as `fin_runocorp_base.py`) **does NOT contain** any calls to `parse_stanza_feats()`. This is why the evaluation runs without warnings.

**Different Phase 9 versions exist:**
- **BACKUP version** (this baseline): NO `parse_stanza_feats()` calls → clean 58.8% accuracy
- **BEFORE_PHASE10/BEFORE_OPTION_B versions**: DO have `parse_stanza_feats()` calls but method undefined → warnings + 58.8% via exception fallback
- **WITH_PARSE_FIX version**: Has method defined → 58.5% accuracy (WORSE)

**Original "beneficial bug" analysis applies to other Phase 9 versions:**

In Phase 9 versions that DO call `parse_stanza_feats()`:
1. Exception handling at lines 807-810 catches the `AttributeError`
2. Falls back to simpler word-level lemmatization: `return [self.lemmatize(token) for token in tokens]`
3. This fallback is MORE ACCURATE for dialectal Finnish than using Stanza morphological features
4. Stanza is trained on standard Finnish and fails on dialectal/archaic forms

**This baseline (BACKUP) version achieves the same 58.8% accuracy but via a cleaner code path without exception handling.**

**Recommendation:** This BACKUP version is the BEST baseline - clean code, no warnings, proven 58.8% accuracy.

## Script Details

### Version Information
- **Version:** v17_phase9_morphological_features
- **Based on:** V16 Hybrid Gold Standard Lexicon
- **Created:** Nov 1, 2025 18:43
- **File size:** 78,243 bytes

### Key Features
1. **Morphological Features in Tier 1** - UFEATS extraction and feature-aware candidate selection
2. **Dual-dictionary approach** - Sukija (dialectal) + Old Finnish dictionaries
3. **Enhanced Voikko suggestions** - Up to 30 suggestions for better coverage
4. **Aggressive fuzzy lexicon matching** - Threshold 3.5 with dialectal awareness
5. **Multi-criteria Voikko ranking** - Edit distance, POS, frequency, proper nouns

### Performance History
- V16: 53.1% accuracy
- V17 Phase 1: 53.3% (+0.2pp)
- V17 Phase 4: 54.2% (+0.9pp)
- V17 Phase 7: 58.5% (+4.3pp from V16)
- **V17 Phase 9 (this version): 58.8% (+0.3pp from Phase 7)** ← BASELINE

## Required Dependencies

All required dependencies are **NOW INCLUDED** in this folder (copied from `v15_pos_aware_lemmatization`):

### 1. Lexicon File ✅
- **File:** `selftraining_lexicon_v16_min1.json` (681 KB)
- **Location:** In this directory
- **Purpose:** Three-tier POS-aware lexicon (Gold standard, Omorfi 100%, Production)

### 2. Dialectal Normalizer Module ✅
- **File:** `dialectal_normalizer.py` (9.2 KB)
- **Location:** In this directory
- **Purpose:** Handles dialectal Finnish normalization

### 3. Test Dataset ✅
- **File:** `finnish_poems_gold_test_clean.csv` (187 KB)
- **Location:** In this directory
- **Content:** 24 Finnish dialectal poems, ~1,468 words with gold standard lemmas

### 4. Evaluation Script ✅
- **File:** `evaluate_v17_phase9.py` (27 KB, adapted)
- **Location:** In this directory
- **Purpose:** Evaluates lemmatizer accuracy against gold standard
- **Modification:** Line 47 changed to import `fin_runocorp_base` instead of `omorfi_hfst_with_voikko_v17_phase9`

### 5. System Dependencies
- **Omorfi:** HFST analyzer installed at `~/.omorfi/`
- **Voikko Sukija:** Installed at `~/.voikko/sukija/5/mor-sukija`
- **Voikko Old Finnish:** Installed at `~/.voikko/5/5`
- **Stanza:** Finnish-TDT model (auto-downloads to `~/stanza_resources/`)
- **Python packages:** `hfst`, `libvoikko`, `stanza`, `omorfi.analysis` (optional)

## Usage

### Basic Usage (with all dependencies)
```bash
python3 fin_runocorp_base.py < input.txt
```

### Evaluation (requires test dataset and evaluation script)
```bash
python3 evaluate_v17_phase9.py 2>&1 | tee fin_runocorp_baseline_evaluation.log
```

**Note:** Evaluation script must be adapted to import from `fin_runocorp_base` instead of `omorfi_hfst_with_voikko_v17_phase9`.

## Investigation History

### The Missing Method Investigation

**Date:** 2025-11-02

**Problem:** Phase 9 calls `parse_stanza_feats()` but method doesn't exist

**Investigation Steps:**
1. Checked all Phase 9 backups - none had the method defined
2. Found exception handling catches AttributeError at lines 807-810
3. Tested BACKUP version - confirmed 58.8% accuracy
4. Added method (from Phase 12) - regressed to 58.5%
5. Analyzed logs - original runs had NO exception warnings

**Finding:** Exception handling allows script to work via fallback:
```python
except Exception as e:
    print(f"⚠ Contextual disambiguation failed: {e}", file=sys.stderr)
    return [self.lemmatize(token) for token in tokens]
```

**Conclusion:** The simpler word-level fallback is MORE ACCURATE than Stanza-based morphological analysis for dialectal Finnish.

### Why Stanza Features Fail on Dialectal Finnish

**Evidence:**
- Phase 9 (without Stanza features): **58.8%**
- Phase 9 (with Stanza features): **58.5%** (-0.3pp)
- Phase 10 (Stanza-guided fuzzy): **58.5%** (-0.3pp)
- Phase 12 (Stanza contextual): **58.5%** (-0.3pp)

**Pattern:** Every time Stanza features are used for decision-making on dialectal text, accuracy drops or stays flat.

**Reason:** Stanza trained on standard Finnish (Finnish-TDT) is unreliable for:
- Dialectal word forms
- Archaic vocabulary
- Regional variations
- Non-standard morphology

## Files in This Folder

```
fin-runocorp-morph/
├── fin_runocorp_base.py                  # Baseline lemmatizer (58.8%, 76 KB)
├── selftraining_lexicon_v16_min1.json    # V16 hybrid lexicon (681 KB)
├── dialectal_normalizer.py               # Dialectal normalizer (9.2 KB)
├── finnish_poems_gold_test_clean.csv     # Test dataset (187 KB, 24 poems)
├── evaluate_v17_phase9.py                # Adapted evaluation script (27 KB)
├── fin_runocorp_baseline_evaluation.log  # Evaluation results (58.8%)
└── README.md                             # This file
```

## Documentation References

For full investigation details, see:
- `V17_PHASE9_COMPLETE_ANALYSIS.md` - Complete analysis summary
- `V17_PHASE9_BUG_ANALYSIS_FINAL.md` - Detailed bug analysis
- `V17_PHASE9_BUG_DISCOVERY.md` - Initial discovery notes
- `V17_PHASE9_FINAL_SUMMARY.md` - Original Phase 9 summary

## Next Steps

To use this baseline for further development:

1. **Copy required dependencies** to this folder or ensure they're accessible
2. **Create/adapt evaluation script** to work with `fin_runocorp_base.py`
3. **Run baseline evaluation** to confirm 58.8% accuracy
4. **Build improvements** on top of this proven baseline

**Critical:** Do NOT add the `parse_stanza_feats()` method to this BACKUP version - it's already optimal!

## Baseline Verification (2025-11-02)

**Evaluation completed successfully:**
- **Accuracy:** 58.8% (863/1468 exact matches) ✅ VERIFIED
- **No warnings or errors:** Clean execution without Stanza exceptions
- **Evaluation log:** `fin_runocorp_baseline_evaluation.log`
- **Processing time:** ~5 minutes for 24 poems (1,468 words)

**Key finding:** This BACKUP version does NOT have `parse_stanza_feats()` calls, achieving 58.8% accuracy via clean code path (not via exception fallback as described in other Phase 9 versions).

## Status

- ✅ Baseline script copied and renamed (`fin_runocorp_base.py`)
- ✅ All dependencies copied from `v15_pos_aware_lemmatization`
- ✅ Evaluation script adapted to import `fin_runocorp_base`
- ✅ Baseline accuracy verified: **58.8%** (863/1468 correct)
- ✅ Investigation complete: No warnings because BACKUP has no `parse_stanza_feats()` calls

**Setup complete:** This folder is ready for further development based on proven 58.8% baseline.

## Code Refactoring (2025-11-02)

**Refactoring Date:** 2025-11-02
**Baseline:** 58.8% (863/1468 exact matches)
**Critical Constraint:** Exact numerical accuracy must be preserved across all refactoring steps

### Motivation

The original `fin_runocorp_base.py` (76KB) contained:
- Hardcoded configuration values scattered throughout the code
- Morphological feature extraction logic embedded in the main class
- Magic numbers and constants without clear documentation
- Missing type hints for better code maintainability

**Goal:** Improve code organization and maintainability while preserving exact 58.8% accuracy.

### Refactoring Steps

#### Step 1: Extract LemmatizerConfig Class ✅

**Changes:**
- Created `LemmatizerConfig` dataclass (lines 146-185)
- Centralized all configuration parameters:
  - Model paths (`model_dir`, `voikko_path`)
  - Language setting (`lang`)
  - Lexicon path (`lexicon_path`)
  - Suggestion limits (`max_suggestions_standard`, `max_suggestions_enhanced`)
  - Feature flags (`enable_stanza`, `enable_voikko`, `enable_omorfi`)
- Updated `__init__()` to accept `config` parameter with backward compatibility

**File Size:** 76KB → 79KB (+3KB)

**Validation:**
```bash
python3 -u evaluate_v17_phase9.py 2>&1 | tee step1_validation.log
```
- **Result:** 58.8% (863/1468) ✅ EXACT MATCH PRESERVED
- **Backup:** `fin_runocorp_base_STEP1_COMPLETE.py` (79KB)

#### Step 2: Extract Morphological Feature Classes ✅

**Changes:**
- Created `MorphologicalFeatureExtractor` class (lines 188-446)
  - `get_ufeats_from_analysis()`: Extract UFEATS from Omorfi HFST analysis
  - HFST→UD feature mappings for 12 categories (Case, Number, Tense, etc.)
- Created `FeatureSimilarityScorer` class (lines 448-488)
  - `compute_similarity()`: Calculate feature agreement scores
  - Weighted scoring: Case=2.0, Number=1.5, Tense=2.0, Person=1.5, etc.
- Updated main class to use extracted feature classes

**File Size:** 79KB → 80KB (+1KB)

**Validation:**
```bash
python3 -u evaluate_v17_phase9.py 2>&1 | tee step2_validation.log
```
- **Result:** 58.8% (863/1468) ✅ EXACT MATCH PRESERVED
- **Features extracted:** 1446 (expected)
- **Feature similarity computed:** 2051 times (expected)
- **Backup:** `fin_runocorp_base_STEP2_COMPLETE.py` (80KB)

#### Step 3: Externalize Constants ✅

**Changes:**
- Created `LemmatizerConstants` class (lines 490-530)
- Centralized all magic numbers and thresholds:
  - **Fuzzy matching:** `FUZZY_THRESHOLD = 3.5`
  - **Similarity weights:** `EDIT_DISTANCE_WEIGHT = 2.0`, `POS_MATCH_WEIGHT = 5.0`
  - **Frequency thresholds:** `MIN_FREQUENCY_BOOST = 0.2`
  - **Feature weights:** All 12 feature weights documented
  - **Bonus thresholds:** `MIN_FEATURE_BONUS = 1.0`, `MAX_FEATURE_BONUS = 8.0`
- Added docstrings explaining each constant's purpose
- Organized into logical sections (Fuzzy Matching, Voikko Ranking, Feature Scoring)

**File Size:** 80KB → 83KB (+3KB)

**Validation:**
```bash
python3 -u evaluate_v17_phase9.py 2>&1 | tee step3_validation.log
```
- **Result:** 58.8% (863/1468) ✅ EXACT MATCH PRESERVED
- **Backup:** `fin_runocorp_base_STEP3_COMPLETE.py` (83KB)

#### Step 4: Add Comprehensive Type Hints ✅

**Changes:**
- Updated imports to include: `Any`, `Set`, `Union` (in addition to existing `List`, `Dict`, `Tuple`, `Optional`)
- Added type hints to **all classes and methods:**
  - **LemmatizerConfig:** 2 methods (`__init__`, `__repr__`)
  - **MorphologicalFeatureExtractor:** 2 methods (`__init__`, `get_ufeats_from_analysis`)
  - **FeatureSimilarityScorer:** 2 methods (`__init__`, `compute_similarity`)
  - **OmorfiHfstWithVoikkoV16Hybrid:** 15+ methods (all public and private methods)
  - **main():** Function signature
- Used appropriate generic types:
  - `Dict[str, str]` for string→string mappings
  - `Dict[str, Any]` for mixed-value dictionaries
  - `Optional[Type]` for nullable parameters
  - `List[Dict[str, Any]]` for lists of dictionaries

**File Size:** 83KB (no change - only type annotations added)

**Validation:**
```bash
# Syntax check
python3 -m py_compile fin_runocorp_base.py

# Full evaluation
python3 -u evaluate_v17_phase9.py 2>&1 | tee step4_type_hints_validation.log
```
- **Syntax check:** ✅ PASSED (no errors)
- **Result:** 58.8% (863/1468) ✅ EXACT MATCH PRESERVED
- **Features extracted:** 1446 (expected)
- **Feature similarity computed:** 2051 times (expected)
- **Selection changed by features:** 0 times (expected - Phase 9 morphology doesn't flip selections)
- **Backup:** `fin_runocorp_base_STEP4_COMPLETE.py` (83KB)

### Refactoring Summary

**File Size Evolution:**
```
Original:    76 KB (baseline)
Step 1:      79 KB (+3KB, LemmatizerConfig)
Step 2:      80 KB (+1KB, Feature classes)
Step 3:      83 KB (+3KB, LemmatizerConstants + documentation)
Step 4:      83 KB (no change, type hints only)
Final:       83 KB (+7KB total, +9.2% size increase)
```

**Code Organization Improvements:**
1. **Configuration centralized:** All settings in `LemmatizerConfig` class
2. **Feature extraction modular:** Separate `MorphologicalFeatureExtractor` class
3. **Similarity scoring isolated:** Dedicated `FeatureSimilarityScorer` class
4. **Constants documented:** All magic numbers in `LemmatizerConstants` with explanations
5. **Type safety enhanced:** Comprehensive type hints across all methods

**Validation Results (All Steps):**
| Step | Accuracy | Exact Matches | Files Processed | Validation Status |
|------|----------|---------------|-----------------|-------------------|
| Baseline | 58.8% | 863/1468 | 24 poems | ✅ Verified |
| Step 1 | 58.8% | 863/1468 | 24 poems | ✅ Match |
| Step 2 | 58.8% | 863/1468 | 24 poems | ✅ Match |
| Step 3 | 58.8% | 863/1468 | 24 poems | ✅ Match |
| Step 4 | 58.8% | 863/1468 | 24 poems | ✅ Match |

**Critical Success Metric:** ✅ **100% accuracy preservation** across all 4 refactoring steps

### Backup Files

All intermediate refactoring steps have been backed up:
```
fin_runocorp_base_ORIGINAL.py            # Original baseline (76KB)
fin_runocorp_base_STEP1_COMPLETE.py      # After config extraction (79KB)
fin_runocorp_base_STEP2_COMPLETE.py      # After feature class extraction (80KB)
fin_runocorp_base_STEP3_COMPLETE.py      # After constants externalization (83KB)
fin_runocorp_base_STEP4_COMPLETE.py      # After type hints (83KB)
fin_runocorp_base.py                     # Current refactored version (83KB)
```

### Code Structure (Refactored)

**Class Architecture:**
```python
# Configuration
class LemmatizerConfig:
    """Centralized configuration for Finnish lemmatizer"""
    # 10 configuration parameters with defaults

# Morphological Analysis
class MorphologicalFeatureExtractor:
    """Extract Universal Dependencies features from Omorfi HFST analysis"""
    # HFST→UD mappings for 12 feature categories

class FeatureSimilarityScorer:
    """Compute morphological feature agreement scores"""
    # Weighted similarity scoring (12 feature weights)

# Constants
class LemmatizerConstants:
    """All magic numbers and thresholds with documentation"""
    # Fuzzy matching, Voikko ranking, Feature scoring thresholds

# Main Lemmatizer
class OmorfiHfstWithVoikkoV16Hybrid:
    """V17 Phase 9 morphology-aware Finnish dialectal lemmatizer"""
    # Uses all above components
```

### Benefits of Refactoring

1. **Maintainability:** Clear separation of concerns (config, features, scoring, constants)
2. **Testability:** Individual components can be unit tested independently
3. **Documentation:** All constants and thresholds explained with docstrings
4. **Type Safety:** Comprehensive type hints enable better IDE support and error detection
5. **Extensibility:** Easy to modify configuration or add new feature extractors
6. **Reliability:** Zero accuracy regression - exact 58.8% preserved

### Usage (Post-Refactoring)

**Default usage (unchanged):**
```python
lemmatizer = OmorfiHfstWithVoikkoV16Hybrid()
result = lemmatizer.lemmatize("dialectal_word")
```

**Custom configuration:**
```python
config = LemmatizerConfig(
    max_suggestions_enhanced=50,  # More suggestions
    enable_stanza=False,          # Disable Stanza for faster processing
    lexicon_path="custom_lexicon.json"
)
lemmatizer = OmorfiHfstWithVoikkoV16Hybrid(config=config)
```

**Feature extraction (standalone):**
```python
extractor = MorphologicalFeatureExtractor()
features = extractor.get_ufeats_from_analysis("[WORD=talo][CASE=Nom][NUM=Sg]")
# Returns: {'Case': 'Nom', 'Number': 'Sing'}
```

**Similarity scoring (standalone):**
```python
scorer = FeatureSimilarityScorer()
similarity = scorer.compute_similarity(
    {'Case': 'Nom', 'Number': 'Sing'},
    {'Case': 'Nom', 'Number': 'Plur'}
)
# Returns weighted similarity score
```

### Refactoring Status

- ✅ **Step 1:** LemmatizerConfig extraction (58.8% preserved)
- ✅ **Step 2:** MorphologicalFeatureExtractor + FeatureSimilarityScorer (58.8% preserved)
- ✅ **Step 3:** LemmatizerConstants externalization (58.8% preserved)
- ✅ **Step 4:** Comprehensive type hints (58.8% preserved)
- ✅ **Documentation:** README.md updated with complete refactoring notes

**Refactoring complete:** All planned improvements implemented with zero accuracy regression.
