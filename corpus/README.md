# Finnish Runosong Corpus V2

Pre-built corpus with morphological analysis of 185,376 Finnish runosongs from SKVR and JR collections.

## Files

| File | Size | Description |
|------|------|-------------|
| `finnish_runosong_corpus_v2.json.gz` | 31 MB | Full corpus (JSON format) |
| `finnish_runosong_corpus_v2.db.gz.partaa` | 94 MB | SQLite database (part 1) |
| `finnish_runosong_corpus_v2.db.gz.partab` | 40 MB | SQLite database (part 2) |

## Reassembling SQLite Database

The SQLite database is split into parts due to GitHub's 100MB file limit:

```bash
# Reassemble the gzipped file
cat finnish_runosong_corpus_v2.db.gz.part* > finnish_runosong_corpus_v2.db.gz

# Decompress
gunzip finnish_runosong_corpus_v2.db.gz
```

## Statistics

- 185,376 poems (89,247 SKVR + 96,129 JR)
- 7,439,688 tokens analyzed
- 701,670 unique word forms
- 98,687 unique lemmas

## JSON Structure

```json
{
  "metadata": {
    "version": "v2",
    "build_date": "2025-12-09",
    "total_poems": 185376,
    "total_tokens": 7439688,
    "unique_words": 701670,
    "unique_lemmas": 98687
  },
  "words": {
    "wordform": {
      "count": 12345,
      "lemmas": {
        "lemma1": {
          "count": 10000,
          "morphology": {...}
        }
      }
    }
  },
  "lemma_index": {
    "lemma": ["wordform1", "wordform2", ...]
  }
}
```
