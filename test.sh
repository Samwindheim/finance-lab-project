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
for pdf_file in "$@"; do
  python3 tests/test_accuracy.py "$pdf_file"
done
