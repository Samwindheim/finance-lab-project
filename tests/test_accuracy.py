import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add the root directory to the Python path to allow importing from `src`
# This lets us import our data extraction module from anywhere in the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_extractor import extract_investor_data

def compare_investors(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    """
    Compares two investor records field by field and returns a list of differences.
    
    This function recursively compares nested dictionaries (like the 'commitment' object)
    and handles null values properly (null == null is considered a match).
    """
    errors = []
    
    # Check each field in the expected investor record
    for key in expected:
        # If the actual record is missing this field, check if it was expected to be null
        if key not in actual:
            if expected[key] is not None:
                errors.append(f"Missing key: {key}")
            continue
            
        # If this field is a nested dictionary (like 'commitment'), compare recursively
        if isinstance(expected[key], dict):
            # Recursively compare the nested dictionary and prefix errors with 'commitment.'
            errors.extend(f"commitment.{e}" for e in compare_investors(actual[key], expected[key]))
        else:
            # Compare simple values (strings, numbers, nulls)
            actual_val = actual[key]
            expected_val = expected[key]
            
            # Handle null values properly - both null should be considered equal
            # Only report a mismatch if values differ AND they're not both null
            if actual_val != expected_val and not (actual_val is None and expected_val is None):
                errors.append(f"Value mismatch for '{key}': got '{actual_val}', expected '{expected_val}'")
    
    return errors

def find_matching_investor(investor_name: str, investor_level: int, investors_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Finds an investor in the list by matching both name and investor level.
    
    This is needed because the same person might appear multiple times with different
    investor levels (e.g., as both an underwriter and a guarantor).
    
    Returns the matching investor record, or None if not found.
    """
    for investor in investors_list:
        # Match on both name AND investor level to find the exact record
        if (investor.get("name") == investor_name and 
            investor.get("investor_level") == investor_level):
            return investor
    return None

def main():
    """
    Main test function that compares actual extraction results against ground truth data.
    
    The test process:
    1. Load environment variables (needed for OpenAI API)
    2. Parse command line arguments (PDF file path)
    3. Run the data extraction on the PDF
    4. Load the corresponding ground truth JSON file
    5. Compare actual vs expected results and report differences
    """
    start_time = time.time()
    # Load environment variables from .env file (needed for OpenAI API key)
    load_dotenv()
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Test the accuracy of the data extraction against a ground truth file.")
    parser.add_argument("pdf_path", type=str, help="The file path to the PDF document to test.")
    args = parser.parse_args()

    # Build the path to the corresponding ground truth JSON file
    # Example: "ADVT_2025_08_14_Memorandum.pdf" -> "ADVT_2025_08_14_Memorandum.json"
    pdf_filename = os.path.basename(args.pdf_path)
    ground_truth_filename = os.path.splitext(pdf_filename)[0] + ".json"
    ground_truth_path = os.path.join("tests", "ground_truth", ground_truth_filename)

    # Check if the ground truth file exists
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at '{ground_truth_path}'")
        return

    print(f"--- Running Test for: {pdf_filename} ---")
    
    # Step 1: Run the actual data extraction on the PDF
    # This calls the same function that the main extraction script uses
    actual_data = extract_investor_data(args.pdf_path)
    actual_investors = actual_data.get("investors", [])

    # Step 2: Load the expected results from the ground truth JSON file
    with open(ground_truth_path, 'r') as f:
        expected_data = json.load(f)
    expected_investors = expected_data.get("investors", [])

    # Step 3: Compare the results
    print(f"Found {len(actual_investors)} investors, expected {len(expected_investors)}.")
    
    # Quick check: if the counts don't match, something is seriously wrong
    if len(actual_investors) != len(expected_investors):
        print("--- TEST FAILED: Investor count mismatch ---")
        return

    # Step 4: Detailed comparison of each investor
    total_errors = 0
    missing_investors = []
    
    # For each investor in the ground truth, find the matching investor in actual results
    for expected in expected_investors:
        expected_name = expected.get("name", "N/A")
        expected_level = expected.get("investor_level", -1)
        
        # Find the matching investor in actual results (by name + level)
        actual = find_matching_investor(expected_name, expected_level, actual_investors)
        
        if actual is None:
            # This investor is completely missing from the actual results
            missing_investors.append(f"{expected_name} (level {expected_level})")
            total_errors += 1
        else:
            # Found a match, now compare the data field by field
            errors = compare_investors(actual, expected)
            if errors:
                total_errors += len(errors)
                print(f"\n--- Mismatch found for {expected_name} (level {expected_level}) ---")
                for error in errors:
                    print(f"  - {error}")
    
    # Check for mystery investors (in actual but not in expected)
    mystery_investors_details = []
    for actual in actual_investors:
        actual_name = actual.get("name", "N/A")
        actual_level = actual.get("investor_level", -1)
        
        match = find_matching_investor(actual_name, actual_level, expected_investors)
        if match is None:
            mystery_investors_details.append(actual)
    
    # Step 5: Report the final results
    if missing_investors:
        print(f"\n--- Missing investors ---")
        for missing in missing_investors:
            print(f"  - {missing}")

    if mystery_investors_details:
        print(f"\n--- Found mystery investors (not in ground truth) ---")
        for investor in mystery_investors_details:
            # Pretty print the investor data for readability
            investor_details = json.dumps(investor, indent=4, ensure_ascii=False)
            print(f"  - {investor.get('name')} (level {investor.get('investor_level')}):")
            # Indent the JSON blob to align with the list item
            for line in investor_details.splitlines():
                print(f"    {line}")
    
    # Final verdict
    duration = time.time() - start_time
    if total_errors == 0 and not mystery_investors_details:
        print(f"\n--- TEST PASSED: All investor data matches the ground truth. ({duration:.2f}s) ---")
    else:
        report = []
        if total_errors > 0:
            report.append(f"{total_errors} mismatches")
        if mystery_investors_details:
            report.append(f"{len(mystery_investors_details)} mystery investors")
        print(f"\n--- TEST FAILED: Found {' and '.join(report)}. ({duration:.2f}s) ---")

if __name__ == "__main__":
    main()
