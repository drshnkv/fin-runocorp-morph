#!/usr/bin/env python3
"""
Interactive lemmatizer demo using refactored modules

Usage:
    python3 lemmatize_interactive.py
"""

from lemmatizer import FinnishLemmatizer

def main():
    print("=" * 80)
    print("Finnish Dialectal Lemmatizer (Refactored)")
    print("=" * 80)
    print()

    print("Loading lemmatizer...")
    lemmatizer = FinnishLemmatizer()
    print("✓ Ready!")
    print()

    print("Enter Finnish words or sentences to lemmatize.")
    print("Commands:")
    print("  - Type a word: lemmatize single word")
    print("  - Type a sentence: lemmatize with context")
    print("  - 'quit' or 'exit': exit program")
    print()

    while True:
        try:
            text = input("Finnish text > ").strip()

            if not text:
                continue

            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Tokenize
            tokens = text.split()

            if len(tokens) == 1:
                # Single word - use word-level API
                result = lemmatizer.lemmatize(tokens[0])
                print(f"\n  Word:       {result['word']}")
                print(f"  Lemma:      {result['lemma']}")
                print(f"  Method:     {result['method']}")
                print(f"  Confidence: {result['confidence']}")
                print()
            else:
                # Multiple words - use sentence-level API for better accuracy
                results = lemmatizer.lemmatize_sentence(tokens)
                print(f"\n  {'Word':<20} {'Lemma':<20} {'Method':<25} Confidence")
                print("  " + "-" * 75)
                for result in results:
                    print(f"  {result['word']:<20} {result['lemma']:<20} "
                          f"{result['method']:<25} {result.get('confidence', 'N/A')}")
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n  ⚠ Error: {e}\n")

if __name__ == '__main__':
    main()
