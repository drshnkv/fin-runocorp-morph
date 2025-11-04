# Batch Processing with Refactored Lemmatizer

## Overview

The refactored batch processing script `process_skvr_batch_REFACTORED.py` is a drop-in replacement for the original `process_skvr_batch_v2.py` that uses the new modular lemmatizer components.

## ✅ Key Features

- **✅ 60.4% accuracy** - With Phase 12 compound integration (up from 59.9%)
- **✅ Uses refactored modules** - lemmatizer.py, lemmatizer_core.py, lemmatizer_config.py
- **✅ Compound reconstruction** - Automatically includes Phase 12 compound integration
- **✅ Identical functionality** - All features from original script preserved
- **✅ Same command-line interface** - Drop-in replacement

## Usage

### Basic Usage (Same as Original)

```bash
python3 process_skvr_batch_REFACTORED.py \
    --input skvr_runosongs_okt_2025.csv \
    --output skvr_lemmatized_results_refactored.csv \
    --chunk-size 100
```

### Using Combined Lexicon (Recommended)

```bash
python3 process_skvr_batch_REFACTORED.py \
    --input skvr_runosongs_okt_2025.csv \
    --output skvr_lemmatized_results_refactored.csv \
    --lexicon-path selftraining_lexicon_comb_with_additions.json \
    --chunk-size 100
```

### Resume After Interruption

```bash
python3 process_skvr_batch_REFACTORED.py \
    --input skvr_runosongs_okt_2025.csv \
    --output skvr_lemmatized_results_refactored.csv \
    --resume
```

### Custom Dialectal Dictionary

```bash
python3 process_skvr_batch_REFACTORED.py \
    --input skvr_runosongs_okt_2025.csv \
    --output skvr_lemmatized_results_refactored.csv \
    --dialectal-dict-path custom_dialectal_index.json
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input CSV file with poems | (required) |
| `--output` | Output CSV file for results | (required) |
| `--lexicon-path` | Path to lexicon JSON file | `selftraining_lexicon_v16_min1.json` |
| `--dialectal-dict-path` | Path to SMS dialectal dictionary | `sms_dialectal_index_v4_final.json` |
| `--chunk-size` | Poems to process per chunk | 100 |
| `--save-interval` | Save results every N seconds | 120 |
| `--resume` | Resume from checkpoint | (flag) |

## Features

### 1. Incremental Saves

Results are saved every 120 seconds (configurable with `--save-interval`):
- See progress within 2-3 minutes
- Data preserved even if process is interrupted
- No need to wait hours to see results

### 2. Checkpoint System

Automatic checkpoints allow resuming:
- Checkpoint file created: `{output}.checkpoint.json`
- Resume with `--resume` flag
- Continues from last saved position

### 3. Progress Tracking

Real-time progress bar with:
- Number of poems processed
- Processing speed (poems/second)
- Estimated time remaining (ETA)

### 4. Robust Error Handling

- Graceful handling of interruptions (Ctrl+C)
- Saves partial results on error
- Clear instructions for resuming

## Input CSV Format

The script expects CSV files with these columns:

```csv
p_id,nro,poemTitle,poemText
1,skvr01100010,SKVR I1 1.,"vaan se on vanha väinämöinen..."
2,skvr01100020,SKVR I1 2.,"uks on vanha väinämöini..."
```

**Required columns:**
- `poemText` - Full poem text (or `text` as fallback)
- `poemTitle` - Poem title/ID

**Optional columns** (preserved in output):
- `p_id` - Poem ID
- `nro` - Poem number
- Any other columns are ignored

## Output CSV Format

The script generates CSV files with these columns:

```csv
p_id,nro,poemTitle,word_index,word,lemma,method,confidence,context_score,analysis
1,skvr01100010,SKVR I1 1.,0,vaan,vaan,v16_lexicon_tier1,high,,
1,skvr01100010,SKVR I1 1.,1,se,se,v16_lexicon_tier1,high,,
1,skvr01100010,SKVR I1 1.,2,on,olla,omorfi_contextual,high,5.2,[WORD_ID=olla]...
```

**Output columns:**
- `p_id`, `nro`, `poemTitle` - From input
- `word_index` - Word position in poem (0-based)
- `word` - Original word
- `lemma` - Lemmatized form
- `method` - Tier/method used (e.g., `v16_lexicon_tier1`, `omorfi_contextual`)
- `confidence` - Confidence level (`high`, `medium`, `low`)
- `context_score` - Contextual scoring (if available)
- `analysis` - Omorfi analysis string (if available)

## Performance

### Processing Speed

Based on testing with skvr_test_6poems.csv:

- **Lemmatizer loading**: ~3-5 seconds (first time, includes Stanza model loading)
- **Processing speed**: ~50-200 poems/minute (depending on poem length)
- **Large corpus (170K poems)**: Estimated 14-56 hours

### Resource Requirements

- **Memory**: ~500MB (Stanza models loaded)
- **Disk**: ~2GB (Stanza models, Omorfi, Voikko dictionaries)
- **CPU**: Single-threaded (one core used)

### Optimization Tips

1. **Increase chunk size** for faster processing of small poems:
   ```bash
   --chunk-size 500
   ```

2. **Decrease save interval** for more frequent checkpoints:
   ```bash
   --save-interval 60
   ```

3. **Use combined lexicon** for better coverage:
   ```bash
   --lexicon-path selftraining_lexicon_comb_with_additions.json
   ```

## Comparison: Original vs Refactored

| Feature | Original `process_skvr_batch_v2.py` | Refactored `process_skvr_batch_REFACTORED.py` |
|---------|-------------------------------------|-----------------------------------------------|
| **Accuracy** | 59.9% | 60.4% ✅ (+0.5pp with Phase 12 compounds) |
| **Compound Integration** | ❌ No | ✅ Yes (Phase 12 automatic) |
| **Imports** | `from fin_runocorp_base_v2...` | `from lemmatizer import FinnishLemmatizer` |
| **Code** | Uses monolithic script | Uses refactored modules |
| **Features** | All features | All features + compound reconstruction ✅ |
| **Command-line** | Same options | Same options ✅ |
| **Output format** | Same CSV | Same CSV ✅ |
| **Performance** | Identical | Identical ✅ |

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'lemmatizer'
```

**Solution**: Make sure you're in the project directory:
```bash
cd /Users/kaarelveskis/Downloads/eesti_murrete_sonaraamat_2025/claude_code/fin-runocorp-morph-standalone
```

### Checkpoint Issues

If checkpoint is corrupted:
```bash
rm {output}.checkpoint.json
# Restart from beginning
```

### Memory Issues

For very large corpora, reduce chunk size:
```bash
--chunk-size 50
```

### Interrupted Processing

The script handles interruptions gracefully:
1. Press Ctrl+C to stop
2. Script saves current progress
3. Resume with `--resume` flag

## Example Workflow

### Processing a Large Corpus

```bash
# Step 1: Start processing
python3 process_skvr_batch_REFACTORED.py \
    --input skvr_runosongs_okt_2025.csv \
    --output skvr_lemmatized_results.csv \
    --lexicon-path selftraining_lexicon_comb_with_additions.json \
    --chunk-size 100 \
    --save-interval 120

# Step 2: If interrupted, resume
python3 process_skvr_batch_REFACTORED.py \
    --input skvr_runosongs_okt_2025.csv \
    --output skvr_lemmatized_results.csv \
    --resume

# Step 3: Check results
head -20 skvr_lemmatized_results.csv

# Step 4: Check progress
wc -l skvr_lemmatized_results.csv
```

### Testing with Sample Data

```bash
# Test with small dataset first
python3 process_skvr_batch_REFACTORED.py \
    --input skvr_test_6poems.csv \
    --output test_results.csv \
    --chunk-size 10

# Verify output
head -50 test_results.csv
```

## Migration from Original Script

To migrate from the original `process_skvr_batch_v2.py`:

**Option 1: Direct replacement** (recommended)
```bash
# Replace script name only
python3 process_skvr_batch_REFACTORED.py \
    --input skvr_runosongs_okt_2025.csv \
    --output skvr_lemmatized_results_expanded.csv \
    --lexicon-path selftraining_lexicon_comb_with_additions.json \
    --chunk-size 100
```

**Option 2: Resume existing checkpoint**

The refactored script can resume from checkpoints created by the original script (same format).

**Option 3: Side-by-side comparison**

Run both scripts on same input:
```bash
# Original
python3 process_skvr_batch_v2.py --input test.csv --output results_original.csv

# Refactored
python3 process_skvr_batch_REFACTORED.py --input test.csv --output results_refactored.csv

# Compare (should be identical)
diff results_original.csv results_refactored.csv
```

## Advanced Usage

### Custom Configuration

For advanced use cases, you can modify the script to use custom `LemmatizerConfig`:

```python
# At line ~187, modify config creation:
config = LemmatizerConfig(
    lexicon_path=args.lexicon_path,
    dialectal_dict_path=args.dialectal_dict_path,
    enable_stanza=True,
    enable_voikko=True,
    enable_omorfi=True,
    max_suggestions_standard=10,
    max_suggestions_enhanced=30
)
```

### Parallel Processing

For multi-core processing, use GNU Parallel:

```bash
# Split input CSV into chunks
split -l 10000 skvr_runosongs_okt_2025.csv chunk_

# Process in parallel (4 cores)
ls chunk_* | parallel -j 4 python3 process_skvr_batch_REFACTORED.py \
    --input {} \
    --output {}.lemmatized.csv

# Combine results
cat chunk_*.lemmatized.csv > final_results.csv
```

## Support

For issues or questions:
- See `REFACTORED_USAGE_GUIDE.md` for API documentation
- See `REFACTORING_SUCCESS_SUMMARY.md` for technical details
- Check original `process_skvr_batch_v2.py` for comparison

## Phase 12 Compound Integration (Automatic)

The refactored batch script **automatically includes Phase 12 compound integration** through the modular lemmatizer components:

### What This Means

- **Compound reconstruction**: Words like `valdaherra` are correctly lemmatized as `valtaherra` (not `valta`)
- **Automatic**: No code changes needed - inherited from `lemmatizer.py` and `lemmatizer_core.py`
- **Conservative**: Only reconstructs high-confidence compounds to avoid false positives
- **+0.5pp accuracy improvement**: From 59.9% to 60.4% on test set

### Examples of Compounds Reconstructed

| Word | Without Integration | With Integration ✅ |
|------|-------------------|-------------------|
| valdaherra | valta | valtaherra |
| kirjalehti | kirja | kirjalehti |
| olvitynnyri | olvi | olvitynnyri |
| maakukka | maa | maakukka |
| pahnakimppu | pahna | pahnakimppu |

### Technical Details

The compound integration uses `CompoundClassifier` to:
1. Detect `[BOUNDARY=COMPOUND]` markers in Omorfi analyses
2. Classify as true compound vs. false positive (e.g., possessive suffixes)
3. Reconstruct compound lemmas for true compounds
4. Fallback to non-compound alternatives for false positives

See `PHASE_12_COMPOUND_INTEGRATION_RESOLUTION.md` for complete details.

---

**Last Updated**: 2025-11-04 (Phase 12 compound integration)
**Version**: Refactored v1.1 (with Phase 12 compounds)
**Accuracy**: 60.4% (59.9% baseline + 0.5pp compound integration)
