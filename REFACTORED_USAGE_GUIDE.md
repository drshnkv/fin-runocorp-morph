# Using the Refactored Finnish Lemmatizer

## Quick Start

The refactored lemmatizer is split into 3 clean modules that you can use independently:

- **`lemmatizer_config.py`** - Configuration and constants
- **`lemmatizer_core.py`** - Core morphological analysis tools
- **`lemmatizer.py`** - Main lemmatizer with 9-tier pipeline

## Installation

No installation needed! Just make sure you're in the project directory:

```bash
cd /Users/kaarelveskis/Downloads/eesti_murrete_sonaraamat_2025/claude_code/fin-runocorp-morph-standalone
```

## Usage Examples

### 1. Single Word Lemmatization

```python
#!/usr/bin/env python3
from lemmatizer import FinnishLemmatizer

# Initialize lemmatizer
lemmatizer = FinnishLemmatizer()

# Lemmatize a word
result = lemmatizer.lemmatize("talossa")

print(f"Word:       {result['word']}")
print(f"Lemma:      {result['lemma']}")
print(f"Method:     {result['method']}")
print(f"Confidence: {result['confidence']}")
```

**Output:**
```
Word:       talossa
Lemma:      talo
Method:     omorfi_contextual
Confidence: high
```

### 2. Sentence-Level Processing (RECOMMENDED)

For better accuracy, use `lemmatize_sentence()` which provides contextual POS tagging:

```python
#!/usr/bin/env python3
from lemmatizer import FinnishLemmatizer

lemmatizer = FinnishLemmatizer()

# Process entire sentence with context
tokens = ["Minun", "talo", "on", "suuri", "ja", "kaunis"]
results = lemmatizer.lemmatize_sentence(tokens)

for result in results:
    print(f"{result['word']:15} → {result['lemma']:15} [{result['method']}]")
```

**Output:**
```
Minun           → minä            [omorfi_contextual]
talo            → talo            [v16_lexicon_tier1]
on              → olla            [omorfi_contextual]
suuri           → suuri           [v16_lexicon_tier1]
ja              → ja              [v16_lexicon_tier1]
kaunis          → kaunis          [v16_lexicon_tier1]
```

### 3. Batch Processing

```python
#!/usr/bin/env python3
from lemmatizer import FinnishLemmatizer

lemmatizer = FinnishLemmatizer()

# Process multiple words
words = ["talo", "talossa", "talot", "talojen", "talossa"]
batch_results = lemmatizer.lemmatize_batch(words)

print(f"Processed {batch_results['total_words']} words in {batch_results['processing_time']:.2f}s")
for result in batch_results['results']:
    print(f"  {result['word']:15} → {result['lemma']}")
```

### 4. Process Raw Text

```python
#!/usr/bin/env python3
from lemmatizer import FinnishLemmatizer

lemmatizer = FinnishLemmatizer()

# Process raw text (automatically tokenized)
text = "Vanha Väinämöinen lauloi kanteleella."
results = lemmatizer.lemmatize_text(text)

print("Original:", text)
lemmas = ' '.join([r['lemma'] for r in results['results']])
print("Lemmatized:", lemmas)
```

**Output:**
```
Original: Vanha Väinämöinen lauloi kanteleella.
Lemmatized: vanha Väinämöinen laulaa kantele
```

### 5. Custom Configuration

```python
#!/usr/bin/env python3
from lemmatizer import FinnishLemmatizer
from lemmatizer_config import LemmatizerConfig

# Custom configuration
config = LemmatizerConfig(
    lexicon_path='selftraining_lexicon_train_with_additions.json',
    dialectal_dict_path='sms_dialectal_index_v4_final.json',
    enable_stanza=True,      # Enable contextual analysis (recommended)
    enable_voikko=True,      # Enable spelling suggestions
    enable_omorfi=True       # Enable morphological analysis
)

lemmatizer = FinnishLemmatizer(config=config)

# Use with custom config
result = lemmatizer.lemmatize("kantajani")
print(f"{result['word']} → {result['lemma']}")
```

## Command-Line Scripts

### Process CSV Files

**Script:** `lemmatize_csv_refactored.py`

```bash
# Basic usage
python3 lemmatize_csv_refactored.py input.csv output.csv

# Example with sample data
python3 lemmatize_csv_refactored.py skvr_test_6poems.csv results_refactored.csv
```

**Input CSV format:**
```csv
poemTitle,poemText
"Poem 1","vaan se on vanha väinämöinen"
"Poem 2","niin tuo portto"
```

**Output CSV format:**
```csv
poem_id,word,lemma,method,confidence
"Poem 1",vaan,vaan,v16_lexicon_tier1,high
"Poem 1",se,se,v16_lexicon_tier1,high
"Poem 1",on,olla,omorfi_contextual,high
```

### Interactive Mode

**Script:** `lemmatize_interactive.py`

```bash
python3 lemmatize_interactive.py
```

**Interactive session:**
```
Finnish Dialectal Lemmatizer (Refactored)
================================================================================

Loading lemmatizer...
✓ Ready!

Enter Finnish words or sentences to lemmatize.
Commands:
  - Type a word: lemmatize single word
  - Type a sentence: lemmatize with context
  - 'quit' or 'exit': exit program

Finnish text > talo

  Word:       talo
  Lemma:      talo
  Method:     v16_lexicon_tier1
  Confidence: high

Finnish text > Minun talo on suuri

  Word                 Lemma                Method                    Confidence
  ---------------------------------------------------------------------------
  Minun                minä                 omorfi_contextual         high
  talo                 talo                 v16_lexicon_tier1         high
  on                   olla                 omorfi_contextual         high
  suuri                suuri                v16_lexicon_tier1         high

Finnish text > quit
Goodbye!
```

## API Reference

### FinnishLemmatizer Class

#### Constructor

```python
FinnishLemmatizer(config: Optional[LemmatizerConfig] = None, lexicon_path: Optional[str] = None)
```

**Parameters:**
- `config` - Optional custom configuration object
- `lexicon_path` - Optional path to training lexicon JSON file

#### Methods

##### lemmatize(word: str) → Dict

Lemmatize a single word using 9-tier fallback pipeline.

**Returns:**
```python
{
    'word': str,           # Original word
    'lemma': str,          # Lemmatized form
    'method': str,         # Method used (e.g., 'v16_lexicon_tier1')
    'confidence': str,     # Confidence level ('high', 'medium', 'low')
    'analysis': str,       # Omorfi analysis string (if available)
    'weight': float,       # Omorfi weight (if available)
    'alternatives': List   # Alternative analyses (if available)
}
```

##### lemmatize_sentence(tokens: List[str]) → List[Dict]

**RECOMMENDED** for better accuracy. Lemmatize entire sentence with Stanza contextual analysis.

**Parameters:**
- `tokens` - List of word tokens in the sentence

**Returns:**
List of dictionaries (same format as `lemmatize()` for each token)

##### lemmatize_batch(words: List[str]) → Dict

Process multiple words efficiently.

**Returns:**
```python
{
    'results': List[Dict],    # List of lemmatization results
    'total_words': int,       # Number of words processed
    'processing_time': float  # Total processing time in seconds
}
```

##### lemmatize_text(text: str) → Dict

Process raw text with automatic tokenization.

**Parameters:**
- `text` - Raw Finnish text

**Returns:**
```python
{
    'results': List[Dict],    # List of lemmatization results
    'original_text': str,     # Original input text
    'processing_time': float  # Total processing time in seconds
}
```

### LemmatizerConfig Class

#### Constructor

```python
LemmatizerConfig(
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
)
```

**Parameters:**
- `model_dir` - Directory for Stanza models (default: None = automatic)
- `voikko_path` - Path to Voikko installation (default: None = system default)
- `lang` - Language code (default: 'fi')
- `lexicon_path` - Path to training lexicon JSON file
- `dialectal_dict_path` - Path to SMS dialectal dictionary JSON file
- `max_suggestions_standard` - Max Voikko suggestions for standard tier (default: 10)
- `max_suggestions_enhanced` - Max Voikko suggestions for enhanced tier (default: 30)
- `enable_stanza` - Enable Stanza contextual analysis (default: True)
- `enable_voikko` - Enable Voikko spell checking (default: True)
- `enable_omorfi` - Enable Omorfi morphological analysis (default: True)

## 9-Tier Lemmatization Pipeline

The refactored lemmatizer uses a 9-tier fallback pipeline for maximum accuracy:

**Tier 1: V16 Lexicon (Gold Standard)**
- Manual annotations (100% trusted)
- Omorfi 100% unambiguous entries
- **Confidence**: High

**Tier 2: Omorfi Contextual Analysis**
- Direct Omorfi analysis with smart alternative selection
- **Confidence**: High

**Tier 3: Voikko + Omorfi**
- Spelling normalization via Voikko suggestions (10 max)
- Re-analysis with Omorfi
- **Confidence**: Medium-High

**Tier 4: Omorfi Guesser**
- Guesser analysis for unknown words
- **Confidence**: Medium

**Tier 5: Fuzzy Lexicon (Morphological)**
- Morphology-aware fuzzy matching
- UFEATS feature similarity ranking
- Conservative threshold (2.0)
- **Confidence**: Medium

**Tier 6: Fuzzy Lexicon (Aggressive)**
- Aggressive fuzzy matching
- Suffix stripping
- Threshold 3.5
- **Confidence**: Medium-Low

**Tier 7: Finnish Dialects Dictionary (SMS)**
- 19,385 validated dialectal variants
- SMS dictionary lookup
- **Confidence**: High (0.9)

**Tier 8: Enhanced Voikko**
- 30 suggestions, aggressive mode
- **Confidence**: Low

**Tier 9: Identity Fallback**
- Returns word.lower() as lemma
- **Confidence**: Very Low

## Performance

### Accuracy

- **Test Set**: 1,468 words from 24 Finnish dialectal poems
- **Accuracy**: 59.9% (879/1468 correct)
- **Exact match** with original monolithic script

### Processing Speed

- **Single word**: ~50-100ms
- **Sentence (10 words)**: ~200-500ms (with Stanza context)
- **Large corpus (1,468 words)**: ~3-4 minutes

### Resource Requirements

- **Memory**: ~500MB (Stanza models loaded)
- **Disk**: ~2GB (Stanza models, Omorfi, Voikko dictionaries)
- **Python**: 3.8+ required

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'lemmatizer'
```

**Solution**: Make sure you're in the project directory:
```bash
cd /Users/kaarelveskis/Downloads/eesti_murrete_sonaraamat_2025/claude_code/fin-runocorp-morph-standalone
```

### Stanza Models Not Found

```
FileNotFoundError: Stanza models not found
```

**Solution**: Stanza will automatically download models on first run. Make sure you have internet connection.

### Omorfi Not Found

```
⚠ Omorfi not found
```

**Solution**: Check that Omorfi is installed at `~/.omorfi/`. The lemmatizer will still work with reduced functionality.

### Low Accuracy

If you're getting poor results:

1. **Use `lemmatize_sentence()` instead of `lemmatize()`** - Sentence-level context significantly improves accuracy
2. **Check lexicon path** - Make sure `selftraining_lexicon_train_with_additions.json` is present
3. **Enable all tools** - Set `enable_stanza=True`, `enable_voikko=True`, `enable_omorfi=True`

## Comparison: Original vs Refactored

| Feature | Original Script | Refactored Modules |
|---------|----------------|-------------------|
| **Lines of code** | 2,176 (monolithic) | 2,710 (3 modules) |
| **Accuracy** | 59.9% | 59.9% ✅ |
| **Modularity** | Single file | 3 clean layers |
| **Testability** | Difficult | Easy (23 unit tests) |
| **Documentation** | Minimal | Comprehensive |
| **API** | Class-based | Class-based + scripts |
| **Maintainability** | Hard | Easy |
| **Extensibility** | Limited | Excellent |

## Next Steps

### Phase 12: Compound Classification Integration

The refactored code is ready for compound word classification integration:

**Target method**: `select_best_alternative()` in `lemmatizer_core.py:903-937`

**Compound classifier**: `compound_classifier.py` (already enhanced, 3/4 tests passing)

**Expected improvement**: Better handling of possessive suffixes vs true compounds

---

## Support

For issues or questions:
- Check `REFACTORING_SUCCESS_SUMMARY.md` for technical details
- See `REFACTORING_SESSION_STATE.md` for development history
- Review unit tests in `test_lemmatizer_config.py` for usage examples

## License

Same as original fin-runocorp-morph project.

---

**Last Updated**: 2025-11-04
**Version**: Refactored v1.0 (59.9% accuracy verified)
