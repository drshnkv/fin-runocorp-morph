# Installation Guide

**Finnish Runosong Corpus Morphological Lemmatizer**

This guide provides step-by-step instructions for installing and setting up the Finnish dialectal poetry lemmatization system.

---

## Quick Start

```bash
# Install Python dependencies
pip install stanza omorfi voikko pandas tqdm

# Download Stanza Finnish model
python -c "import stanza; stanza.download('fi')"

# Test installation
python3 -c "from fin_runocorp_base import OmorfiHfstWithVoikkoV16Hybrid; print('✓ Installation successful')"
```

---

## Prerequisites

### System Requirements

- **Operating System:** macOS, Linux, or Windows
- **Python:** 3.8 or later
- **Disk Space:** ~500 MB (models + dependencies)
- **RAM:** 4 GB minimum, 8 GB recommended

### Check Python Version

```bash
python3 --version
```

**Required:** Python 3.8+

If Python is not installed:
- **macOS:** `brew install python3`
- **Linux:** `sudo apt-get install python3 python3-pip`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

---

## Installation Steps

### Step 1: Install Core Dependencies

```bash
pip install stanza
pip install omorfi
pip install voikko
```

**Expected installation time:** ~2-3 minutes

### Step 2: Install Optional Dependencies

```bash
pip install pandas    # For CSV processing
pip install tqdm      # For progress bars
pip install hfst      # For Omorfi morphological analysis (optional but recommended)
```

### Step 3: Download Stanza Finnish Model

```bash
python3 -c "import stanza; stanza.download('fi')"
```

**Expected download:** ~350 MB
**Installation time:** ~2-5 minutes (depending on connection speed)

---

## Voikko Dictionary Setup

### macOS Installation

```bash
# Install Voikko via Homebrew
brew install libvoikko

# Verify installation
which voikkospell
```

### Linux Installation

```bash
# Debian/Ubuntu
sudo apt-get install libvoikko1 voikko-fi

# Fedora/RHEL
sudo dnf install libvoikko voikko-fi
```

### Optional: Old Finnish Dictionary (VANHAT_MUODOT)

For better handling of archaic and dialectal Finnish forms:

```bash
# Clone Voikko repository
cd /tmp
git clone https://github.com/voikko/corevoikko.git
cd corevoikko/voikko-fi

# Build with old Finnish forms enabled
make vvfst VANHAT_MUODOT=yes

# Install to user directory
mkdir -p ~/.voikko/5
make vvfst-install DESTDIR=$HOME/.voikko/5
```

The lemmatizer will automatically use the custom dictionary if found at `~/.voikko/5/5/mor-standard/`.

---

## Omorfi Model Setup

### Option A: Automatic Installation (Recommended)

The Omorfi package typically includes pre-built models. Verify installation:

```bash
python3 -c "import omorfi; print('✓ Omorfi installed')"
```

### Option B: Manual Model Download

If models are not included:

```bash
# Create Omorfi directory
mkdir -p ~/.omorfi

# Download Omorfi release
cd /tmp
curl -L -o omorfi-0.9.11.tar.gz \
  https://github.com/flammie/omorfi/releases/download/0.9.11/omorfi-0.9.11.tar.gz

# Extract and copy models
tar -xzf omorfi-0.9.11.tar.gz
cp omorfi-0.9.11/share/omorfi/*.hfst ~/.omorfi/
```

---

## Verification

### Test Installation

```bash
cd fin-runocorp-morph-standalone

python3 -c "
from fin_runocorp_base import OmorfiHfstWithVoikkoV16Hybrid
import stanza

print('✓ Imports successful')
print('✓ Installation complete')
"
```

### Test Lemmatization

```bash
python3 test_batch_6poems.py
```

**Expected output:** Processes 6 test poems successfully

### Run Evaluation (Optional)

```bash
python3 evaluate_v17_phase9.py
```

**Expected results:**
- Total test words: 1,468
- Exact matches: ~863 (58.8%)
- Processing time: ~1-2 minutes

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'stanza'"

**Solution:**
```bash
pip install stanza
python -c "import stanza; stanza.download('fi')"
```

### Problem: "ModuleNotFoundError: No module named 'omorfi'"

**Solution:**
```bash
pip install omorfi
pip install hfst  # Also install HFST for full functionality
```

### Problem: "Voikko not found"

**macOS Solution:**
```bash
brew install libvoikko
```

**Linux Solution:**
```bash
sudo apt-get install libvoikko1 voikko-fi
```

### Problem: Slow lemmatization

**Expected performance:**
- Single word: ~0.01-0.1 seconds
- 100 words: ~5-10 seconds
- Stanza loading: ~3-5 seconds (first time only)

**If slower:**
- Ensure models are stored on SSD
- Check available RAM (4 GB minimum)
- Close other resource-intensive applications

### Problem: Permission denied errors

**Solution:**
```bash
# Use --user flag for pip installations
pip install --user stanza omorfi voikko
```

---

## Virtual Environment (Recommended)

For isolated installation:

```bash
# Create virtual environment
python3 -m venv venv-lemmatizer

# Activate (macOS/Linux)
source venv-lemmatizer/bin/activate

# Activate (Windows)
venv-lemmatizer\Scripts\activate

# Install dependencies
pip install stanza omorfi voikko pandas tqdm

# Download Stanza model
python -c "import stanza; stanza.download('fi')"

# Test
python3 test_batch_6poems.py
```

---

## Disk Space Usage

| Component | Size | Location |
|-----------|------|----------|
| Stanza Finnish model | ~350 MB | `~/stanza_resources/fi/` |
| Omorfi models | ~134 MB | `~/.omorfi/` or pip packages |
| Voikko standard | ~10 MB | System libraries |
| Voikko Old Finnish (optional) | ~4 MB | `~/.voikko/5/5/` |
| Python packages | ~50 MB | Python site-packages |
| **Total** | **~550 MB** | |

---

## Platform-Specific Notes

### macOS

- Homebrew recommended for system dependencies
- Tested on macOS 12+ (should work on 11+)
- Apple Silicon (M1/M2) fully supported

### Linux

- Tested on Ubuntu 20.04+, Debian 10+
- RHEL/Fedora/CentOS supported
- Voikko available in most distribution repositories

### Windows

- Stanza and Omorfi work on Windows
- Voikko may require manual compilation or WSL
- Consider using Windows Subsystem for Linux (WSL) for full compatibility

---

## Uninstallation

### Remove Python Packages

```bash
pip uninstall stanza omorfi voikko pandas tqdm hfst
```

### Remove Downloaded Models

```bash
# Stanza models
rm -rf ~/stanza_resources

# Omorfi models
rm -rf ~/.omorfi

# Voikko custom dictionary (if installed)
rm -rf ~/.voikko
```

### Remove System Packages (macOS)

```bash
brew uninstall libvoikko
```

### Remove System Packages (Linux)

```bash
sudo apt-get remove libvoikko1 voikko-fi
```

---

## Version Information

**Tested versions:**
- Python: 3.8, 3.9, 3.10, 3.11
- Stanza: 1.4+
- Omorfi: 0.9.11+
- Voikko: 4.3+
- HFST: 3.16+

**Platform compatibility:**
- macOS: 11+ (Big Sur and later) ✅
- Linux: Ubuntu 20.04+, Debian 10+, RHEL 8+ ✅
- Windows: Via WSL recommended ⚠️

---

## Next Steps

After installation:

1. **Test with sample data:**
   ```bash
   python3 test_batch_6poems.py
   ```

2. **Run evaluation:**
   ```bash
   python3 evaluate_v17_phase9.py
   ```

3. **Process your own data:**
   ```bash
   python3 process_skvr_batch.py --input your_poems.csv --output results.csv
   ```

4. **Read the README:**
   See [README.md](README.md) for usage examples and API documentation

---

## Additional Resources

### Documentation
- **Stanza:** https://stanfordnlp.github.io/stanza/
- **Omorfi:** https://github.com/flammie/omorfi
- **Voikko:** https://voikko.puimula.org/
- **HFST:** https://hfst.github.io/

### Support

For issues or questions:
1. Check troubleshooting section above
2. Review [README.md](README.md) for usage documentation
3. Open an issue on GitHub (include error messages and versions)

---

**Last Updated:** 2025-11-02
**Installation Guide Version:** 2.0
**Target System:** Finnish Runosong Corpus Morphological Lemmatizer V17 Phase 9
