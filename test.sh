#!/bin/bash

# Test accuracy script for the financial data extractor
# This script runs the accuracy tests against ground truth data

# Check if a PDF file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 path/to/document.pdf [path/to/another.pdf ...]"
  echo ""
  echo "This script tests the accuracy of the data extraction against ground truth data."
  echo "Make sure you have a corresponding .json file in tests/ground_truth/ with the same name as your PDF."
  exit 1
fi

# Activate the virtual environment and run the test script
source .venv/bin/activate
for arg in "$@"; do
  if [ -d "$arg" ]; then
    # If the argument is a directory, process all PDF files in it
    find "$arg" -name '*.pdf' -print0 | while IFS= read -r -d '' pdf_file; do
        python3 tests/test_accuracy.py "$pdf_file"
    done
  elif [ -f "$arg" ]; then
    # If the argument is a file, process it directly
    python3 tests/test_accuracy.py "$arg"
  else
    echo "Warning: Argument '$arg' is not a valid file or directory. Skipping." >&2
  fi
done
