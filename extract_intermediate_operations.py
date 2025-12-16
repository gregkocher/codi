# ABOUTME: Script to extract intermediate mathematical operations from multi-arith dataset using GPT API
# ABOUTME: Analyzes questions to identify operations (+, -, *, /) needed for intermediate calculations

import json

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Configuration
MODEL_NAME = "gpt-5-mini"  # You can change to "gpt-4o", "gpt-4-turbo", etc.
INPUT_FILE = "multi_arith_intermediate_results.json"
OUTPUT_FILE = "multi_arith_intermediate_operations.json"

# System prompt for extracting intermediate operations
SYSTEM_PROMPT = """You are a mathematical assistant. Given a math word problem and its intermediate numerical results, identify the mathematical operations needed to calculate those intermediate results.

You should ONLY output the basic mathematical operations: +, -, *, /

Format your response as a JSON object with:
- "operations": a list of operation symbols ("+", "-", "*", "/") that correspond to the intermediate calculations

Example 1:
Question: "Tom has 3 apples. He buys 5 more apples. Then he gives 2 apples to his friend. How many apples does Tom have now?"
Intermediate numbers: [8]
Response:
{
  "operations": ["+"]
}
Explanation: 3 + 5 = 8 (intermediate result), then 8 - 2 = 6 (final answer)

Example 2:
Question: "A store had 20 oranges. They sold 15 oranges in the morning and 3 in the afternoon. How many oranges were sold in total?"
Intermediate numbers: []
Response:
{
  "operations": []
}
Explanation: No intermediate calculation needed, just 15 + 3 = 18 (final answer)

Example 3:
Question: "There are 4 boxes with 6 toys each. John takes 8 toys. How many toys are left?"
Intermediate numbers: [24]
Response:
{
  "operations": ["*"]
}
Explanation: 4 * 6 = 24 (intermediate result), then 24 - 8 = 16 (final answer)

Important:
- Only output operations for INTERMEDIATE calculations, not the final step
- The number of operations should match the number of intermediate results
- Use only: +, -, *, /
"""


def extract_operations(question: str, intermediate_numbers: list) -> dict:
    """
    Use GPT to extract intermediate mathematical operations from a math question.

    Args:
        question: The math word problem
        intermediate_numbers: List of intermediate numerical results

    Returns:
        Dictionary containing operations list
    """
    try:
        user_content = f"""Question: {question}
Intermediate numbers: {intermediate_numbers}

What mathematical operations are needed to calculate these intermediate results?"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "operations": [],
            "error": str(e),
        }


def main():
    print(f"Loading intermediate results from {INPUT_FILE}...")

    # Load the intermediate results file
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions")
    print(f"Using model: {MODEL_NAME}\n")

    # Process each question
    results = []

    for example in tqdm(data, desc="Extracting operations"):
        idx = example["index"]
        question = example["question"]
        intermediate_numbers = example.get("intermediate_numbers", [])
        final_answer = example.get("final_answer", "")

        # Extract operations
        operation_result = extract_operations(question, intermediate_numbers)

        # Store result
        result = {
            "index": idx,
            "question": question,
            "intermediate_numbers": intermediate_numbers,
            "intermediate_operations": operation_result.get("operations", []),
            "final_answer": final_answer,
        }

        if "error" in operation_result:
            result["error"] = operation_result["error"]

        results.append(result)

        # Print first few examples
        if idx < 3:
            print(f"\n--- Example {idx + 1} ---")
            print(f"Question: {question}")
            print(f"Intermediate numbers: {intermediate_numbers}")
            print(f"Intermediate operations: {operation_result.get('operations', [])}")
            print(f"Final answer: {final_answer}")

    # Save results to JSON file
    print(f"\n\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! Processed {len(results)} questions.")
    print(f"Results saved to {OUTPUT_FILE}")

    # Print summary statistics
    total_with_operations = sum(1 for r in results if r["intermediate_operations"])
    total_operations = sum(len(r["intermediate_operations"]) for r in results)

    print("\nSummary:")
    print(f"- Questions with operations: {total_with_operations}/{len(results)}")
    print(f"- Total operations extracted: {total_operations}")

    # Count operation types
    operation_counts = {"+": 0, "-": 0, "*": 0, "/": 0}
    for r in results:
        for op in r["intermediate_operations"]:
            if op in operation_counts:
                operation_counts[op] += 1

    print("\nOperation distribution:")
    for op, count in sorted(operation_counts.items()):
        print(f"  {op}: {count}")


if __name__ == "__main__":
    main()
