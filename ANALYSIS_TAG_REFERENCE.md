# Finnish Runosong Corpus - Analysis Tag Reference

**Generated:** 2025-12-09
**Source:** `output_batches_skvr/` and `output_batches_jr/` batch files
**Total unique tags:** 39

---

## Analysis Format

Each token has an `analysis` field with bracketed tags:

```
[TAG1=VALUE1][TAG2=VALUE2]...
```

Example:
```
[WORD_ID=olla][UPOS=AUX][VOICE=ACT][MOOD=INDV][TENSE=PRESENT][PERS=SG3]
```

---

## 1. Morphological Features (UD-style)

These follow Universal Dependencies conventions for Finnish morphology.

| Tag | Count | Values | Description |
|-----|-------|--------|-------------|
| **UPOS** | 112,775 | ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PRON, PROPN, PUNCT, SCONJ, VERB | Universal POS tag |
| **CASE** | 82,887 | NOM, GEN, PAR, ADE, ALL, ABL, INE, ELA, ILL, ESS, TRA, ABE, COM, LAT, ACC, TRN | Grammatical case |
| **NUM** | 81,127 | SG, PL | Number (singular/plural) |
| **VOICE** | 27,404 | ACT, PSS | Voice (active/passive) |
| **MOOD** | 22,175 | INDV, COND, IMPV, OPT, POTN | Verbal mood |
| **TENSE** | 17,794 | PAST, PRESENT | Tense |
| **PERS** | 22,859 | SG0, SG1, SG2, SG3, PL1, PL2, PL3, PE4 | Person |
| **POSS** | 5,903 | SG1, SG2, 3, PL1, PL2 | Possessive suffix |
| **PRONTYPE** | 4,343 | DEM, IND, INT, PRS, REC, REL | Pronoun type |
| **CMP** | 12,388 | POS, CMP, SUP | Comparison (positive/comparative/superlative) |
| **PCP** | 3,149 | AGENT, NEG, NUT, VA | Participle type |
| **INF** | 2,573 | A, E, MA | Infinitive type |
| **DRV** | 1,824 | INEN, JA, MINEN, NEN, LLINEN, HKO, ISA, IN², MAISILLA, MPI, etc. | Derivational suffix |
| **NEG** | 757 | CON | Negative/connegative |
| **NUMTYPE** | 1,182 | CARD, ORD, MULT, FRAC | Numeral type |
| **ADPTYPE** | 1,178 | POST, PREP | Adposition type |
| **CLIT** | 3,917 | HAN, KA, KAAN, KIN, KO, PA, S | Clitic particle |

### Case Values Explained

| Case | Finnish Name | Example |
|------|--------------|---------|
| NOM | Nominatiivi | talo (house) |
| GEN | Genetiivi | talon (of house) |
| PAR | Partitiivi | taloa |
| ADE | Adessiivi | talolla (at/on house) |
| ALL | Allatiivi | talolle (to house) |
| ABL | Ablatiivi | talolta (from house) |
| INE | Inessiivi | talossa (in house) |
| ELA | Elatiivi | talosta (from inside) |
| ILL | Illatiivi | taloon (into house) |
| ESS | Essiivi | talona (as house) |
| TRA | Translatiivi | taloksi (becoming house) |
| ABE | Abessiivi | talotta (without house) |
| COM | Komitatiivi | taloineen (with house) |

---

## 2. Metadata Tags

| Tag | Count | Values | Description |
|-----|-------|--------|-------------|
| **WORD_ID** | 112,775 | (lemma forms) | Lemma from Omorfi analysis |
| **SOURCE** | 60,375 | `manual`, `v17_expansion_word_lemmatised`, `v17_expansion_word_normalised` | Lexicon source |
| **TIER** | 60,977 | `tier1`, `tier3` | Lexicon tier (tier1=gold, tier3=POS_X) |
| **POS** | 1,465 | ADJ, ADP, ADV, CCONJ, INTJ, NOUN, PART, PRON, PROPN, VERB, None | Part-of-speech (lexicon) |
| **STYLE** | 6,207 | ARCHAIC, DIALECTAL, NONSTANDARD, RARE | Register/style marker |
| **SEM** | 1,703 | COUNTRY, CURRENCY, INHABITANT, LANGUAGE, MEASURE, TIME, TITLE | Semantic class |
| **SUBCAT** | 950 | NEG, QUANTIFIER, REFLEXIVE, ROMAN, SUFFIX | Subcategory |
| **LEX** | 136 | ADE, INE, LAT, LOC, SEP, STI, TRA, TTAIN | Lexicalized adverb type |
| **HOMONYM** | 160 | (marks ambiguous analyses) | Homonym disambiguation |
| **ABBR** | 256 | ABBREVIATION, ACRONYM | Abbreviation type |
| **FOREIGN** | 30 | FOREIGN | Foreign word marker |
| **PROPER** | 7 | MISC, ORG | Named entity type |

---

## 3. Method/Source Tags

These indicate which lemmatization method produced the analysis.

| Tag | Count | Description |
|-----|-------|-------------|
| **V16_LEXICON** | 53,627 | V16 self-training lexicon match |
| **V17_LEXICON_POS_X** | 7,350 | V17 expansion with POS=X |
| **FUZZY_MORPH** | 17,476 | Fuzzy matching with morphological context |
| **FUZZY_AGGRESSIVE** | 2,852 | More permissive fuzzy matching |
| **DIALECTAL_DICT** | 860 | SMS dialectal dictionary match |
| **DIALECTAL_DICT_FUZZY** | 3 | Fuzzy dialectal dictionary match |
| **IDENTITY_FALLBACK** | 130 | Word form = lemma (last resort) |
| **BLACKLIST** | 8,231 | Blocked analysis (`OMORFI`, `TOOSHORTFORCOMPOUND`) |
| **BOUNDARY** | 11,309 | Compound word boundary marker |
| **COMPOUND_FORM** | 305 | Compound form status (`MISSING-`) |

### Method Tag Formats

**V16_LEXICON:**
```
[V16_LEXICON:word+POS→lemma]
```

**FUZZY_MORPH:**
```
[FUZZY_MORPH:input→match(POS)=lemma@score]
```

**FUZZY_AGGRESSIVE:**
```
[FUZZY_AGGRESSIVE:input→input→match(POS)=lemma@score]
```

**DIALECTAL_DICT:**
```
[DIALECTAL_DICT:dialectal_form→standard_form]
```

---

## 4. Examples by Tag

### STYLE (register/dialect markers)

| Word | Lemma | Analysis |
|------|-------|----------|
| petäjäissä | petäjäinen | `[WORD_ID=petäjäinen][UPOS=NOUN][NUM=SG][CASE=ESS][STYLE=ARCHAIC]` |
| sie | sie | `[WORD_ID=sie][UPOS=PRON][PRONTYPE=PRS][PERS=SG2][STYLE=DIALECTAL][NUM=SG][CASE=NOM]` |
| miulle | mie | `[WORD_ID=mie][UPOS=PRON][PRONTYPE=PRS][PERS=SG1][STYLE=DIALECTAL][CASE=ALL]` |

*Marks dialectal pronouns (sie, mie vs standard sinä, minä) and archaic forms.*

### SEM (semantic classes)

| Word | Lemma | Tag | Meaning |
|------|-------|-----|---------|
| seppolan | seppäolka | `SEM=TITLE` | Profession/title (seppä = blacksmith) |
| jäsinensä | jäsen | `SEM=TITLE` | Member/title |
| vaingo | vai | `SEM=LANGUAGE` | Language name |

### SUBCAT (subcategories)

| Word | Lemma | Analysis |
|------|-------|----------|
| keskikertaiset | keski--kertainen | `[SUBCAT=SUFFIX]` - derivational suffix |
| muilla | muu | `[SUBCAT=QUANTIFIER]` - quantifier pronoun |
| muita | muu | `[SUBCAT=QUANTIFIER]` |

### LEX (lexicalized adverb types)

| Word | Lemma | Tag | Meaning |
|------|-------|-----|---------|
| kiirehesti | kiireesti | `LEX=STI` | -sti adverb (manner) |
| täällä | täällä | `LEX=ADE` | Locative adverb (adessive-like) |
| alasti | alasti | `LEX=STI` | -sti adverb |

### HOMONYM (ambiguous lemmas)

| Word | Lemma | Analysis |
|------|-------|----------|
| tapasi | tavata | `[WORD_ID=tavata][HOMONYM=2[UPOS=VERB][VOICE=ACT][MOOD=INDV][TENSE=PAST][PERS=SG3]` |
| tapaat | tavata | `[WORD_ID=tavata][HOMONYM=2[UPOS=VERB][VOICE=ACT][MOOD=INDV][TENSE=PRESENT][PERS=SG2]` |

*HOMONYM=2 indicates second possible lemma (tavata "to meet" vs tapa "custom").*

### ABBR (abbreviations)

| Word | Lemma | Analysis |
|------|-------|----------|
| e | e | `[ABBR=ACRONYM][SEM=CURRENCY]` - euro |
| l | l | `[ABBR=ACRONYM][SEM=MEASURE]` - liter |
| it | it | `[ABBR=ACRONYM]` |

### FOREIGN

| Word | Lemma | Analysis |
|------|-------|----------|
| sta | sta | `[UPOS=X][FOREIGN=FOREIGN]` |

*Marks non-Finnish words.*

### PROPER (named entity type)

| Word | Lemma | Analysis |
|------|-------|----------|
| hakoavi | hako | `...[PROPER=ORG]...` |

*Note: Some PROPER=ORG tags are noise from compound analysis.*

---

## 5. Compound Word Analysis

Compound words have multiple `[WORD_ID=...]` segments separated by `[BOUNDARY=COMPOUND]`:

```
[WORD_ID=seppä][UPOS=NOUN][SEM=TITLE][NUM=SG][CASE=NOM][BOUNDARY=COMPOUND][WORD_ID=olka][UPOS=NOUN][NUM=SG][CASE=GEN]
```

This represents: seppä + olka → seppäolka (blacksmith's shoulder)

---

## 6. Tag Frequency Summary

| Category | Tags | Total Occurrences |
|----------|------|-------------------|
| Core morphology | UPOS, CASE, NUM, VOICE, MOOD, TENSE, PERS | ~370,000 |
| Lexicon methods | V16_LEXICON, V17_LEXICON_POS_X, TIER, SOURCE | ~180,000 |
| Fuzzy methods | FUZZY_MORPH, FUZZY_AGGRESSIVE | ~20,000 |
| Style/semantic | STYLE, SEM, SUBCAT | ~9,000 |
| Compound analysis | BOUNDARY, BLACKLIST, COMPOUND_FORM | ~20,000 |
| Rare tags | FOREIGN, PROPER, ABBR, HOMONYM | ~500 |

---

## References

- **Universal Dependencies:** https://universaldependencies.org/
- **Omorfi:** https://github.com/flammie/omorfi
- **SMS (Suomen Murteiden Sanakirja):** https://kaino.kotus.fi/sms/
