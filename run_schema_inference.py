import os
from schema_inference_agent import SchemaInferenceAgent
import json
from langchain_community.chat_models import ChatOpenAI

def print_test_summary(results):
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("test_passed", False))
    
    print("\n=== Test Summary ===")
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    
    print("\n=== Detailed Results ===")
    
    # Print Passing Tests
    print("\n✅ PASSING TESTS:")
    for result in results:
        if result.get("test_passed", False):
            print(f"\nTest Case ID: {result['test_case_id']}")
            print(f"Validation Success: {result['validation_success']}")
            
    # Print Failing Tests
    print("\n❌ FAILING TESTS:")
    for result in results:
        if not result.get("test_passed", False):
            print(f"\nTest Case ID: {result['test_case_id']}")
            print(f"Validation Success: {result['validation_success']}")
            
            if "schema_comparison" in result:
                print("\nDifferences found:")
                for diff in result["schema_comparison"]["differences"]:
                    print(f"\n- Path: {diff['path']}")
                    print(f"  Issue: {diff['issue']}")
                    if diff['issue'] == "type_mismatch":
                        print(f"  Inferred: {diff.get('inferred')}")
                        print(f"  Expected: {diff.get('expected')}")
                    elif diff['issue'] == "required_fields_mismatch":
                        print(f"  Missing Required Fields: {diff.get('missing', [])}")
                        print(f"  Extra Required Fields: {diff.get('extra', [])}")
                    elif diff['issue'] == "missing_property":
                        print(f"  Missing Property Definition: {diff.get('expected')}")
                    elif diff['issue'] == "extra_property":
                        print(f"  Extra Property Definition: {diff.get('inferred')}")
                    elif diff['issue'] == "missing_format":
                        print(f"  Missing String Format: {diff.get('expected_format')}")
                    elif diff['issue'] == "format_mismatch":
                        print(f"  Inferred Format: {diff.get('inferred')}")
                        print(f"  Expected Format: {diff.get('expected')}")
                    elif diff['issue'] == "missing_array_constraint":
                        print(f"  Missing Array Constraint: {diff.get('constraint')}")
                        print(f"  Expected Value: {diff.get('expected')}")
                    elif diff['issue'] == "missing_mixed_types":
                        print(f"  Missing Mixed Types Support")
                        print(f"  Expected Types: {diff.get('expected')}")
            
            print("\nInferred Schema:")
            print(json.dumps(result.get("inferred_schema", {}), indent=2))
            
            # Safely get the expected schema from the test cases file
            with open("test_cases.json", "r") as f:
                test_cases = json.load(f)
                for test_case in test_cases:
                    if test_case["id"] == result["test_case_id"]:
                        print("\nExpected Schema:")
                        print(json.dumps(test_case["expected_schema"], indent=2))
                        break

def main():
    # Get OpenAI API key from environment variable
    api_key = "your-api-key"
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Initialize the agent
    agent = SchemaInferenceAgent(api_key)

    # Process test cases
    results = agent.process_test_cases("test_cases.json")

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Schema inference completed. Results saved to inference_results.json")
    print_test_summary(results)

if __name__ == "__main__":
    main() 