"""
Script to extract intermediate calculation results from multi-arith dataset using GPT API.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from datasets import load_dataset

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
MODEL_NAME = "gpt-5-mini"  # You can change to "gpt-4o", "gpt-4-turbo", etc.
OUTPUT_FILE = "multi_arith_intermediate_results.json"

# System prompt for extracting intermediate numbers
SYSTEM_PROMPT = """You are a mathematical assistant. Given a math word problem, identify and output ONLY the intermediate numerical results that are needed to solve the problem.

Do NOT output the final answer. Only output intermediate calculations as numbers.

Format your response as a JSON object with:
- "intermediate_numbers": a list of numbers that are intermediate calculation results

Example:
Question: "Tom has 3 apples. He buys 5 more apples. Then he gives 2 apples to his friend. How many apples does Tom have now?"

Response:
{
  "intermediate_numbers": [8]
}

Note: There can be multiple intermediate results. Include all intermediate calculations, not just one.
"""


def extract_intermediate_numbers(question: str) -> dict:
    """
    Use GPT to extract intermediate calculation results from a math question.

    Args:
        question: The math word problem

    Returns:
        Dictionary containing intermediate numbers and explanation
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "intermediate_numbers": [],
            "error": str(e),
        }


def main():
    print("Loading multi-arith dataset...")
    dataset = load_dataset("ChilleD/MultiArith")
    test_set = dataset["test"]

    print(f"Dataset loaded. Total questions: {len(test_set)}")
    print(f"Using model: {MODEL_NAME}\n")

    # Process each question
    results = []

    for idx, example in enumerate(tqdm(test_set, desc="Processing questions")):
        question = example["question"]
        final_answer = example["final_ans"]

        # Extract intermediate numbers
        intermediate_result = extract_intermediate_numbers(question)

        # Store result
        result = {
            "index": idx,
            "question": question,
            "final_answer": final_answer,
            "intermediate_numbers": intermediate_result.get("intermediate_numbers", []),
        }

        if "error" in intermediate_result:
            result["error"] = intermediate_result["error"]

        results.append(result)

        # Print first few examples
        if idx < 3:
            print(f"\n--- Example {idx + 1} ---")
            print(f"Question: {question}")
            print(
                f"Intermediate numbers: {intermediate_result.get('intermediate_numbers', [])}"
            )
            print(f"Final answer: {final_answer}")

    # Save results to JSON file
    print(f"\n\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done! Processed {len(results)} questions.")
    print(f"Results saved to {OUTPUT_FILE}")

    # Print summary statistics
    total_with_intermediates = sum(1 for r in results if r["intermediate_numbers"])
    print("\nSummary:")
    print(
        f"- Questions with intermediate numbers: {total_with_intermediates}/{len(results)}"
    )
    print(
        f"- Questions without intermediate numbers: {len(results) - total_with_intermediates}"
    )


if __name__ == "__main__":
    main()
