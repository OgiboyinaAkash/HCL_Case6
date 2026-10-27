"""
test_query.py — Test any query through the PrismRoute workflow.

Usage:
  1. Pass query as command-line argument:
     python test_query.py "What is 15 percent of 500?"

  2. Interactive mode (prompts you for query in a loop):
     python test_query.py
"""

import os
import sys

# Ensure current repository root is on sys.path regardless of where script is run
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from backend.service import service


def run_test(query: str, scope_id: str = "default"):
    print("=" * 70)
    print(f"QUERY: '{query}' [Scope: {scope_id}]")
    print("=" * 70)

    result = service.process_query(query, scope_id=scope_id)
    dec = result["decision_record"]
    cost = result["cost"]

    print(f"  -> Path Taken:            {dec['path_taken']}")
    print(f"  -> Status:                {dec['status']}")
    print(f"  -> Selected Model Tier:   {dec['selected_tier']}")
    print(f"  -> Skipped Feature Layer: {dec['skipped_feature_layer']}")
    print(f"  -> Similarity:            {dec['similarity']:.4f}")
    print(f"  -> Latency:               {dec['latency_ms']:.1f} ms")
    print(f"  -> Actual Cost:           ${cost['actual_cost_usd']:.6f} (Savings: {cost['savings_pct']:.1f}%)")
    print(f"  -> Answer:                {result['answer']}")
    print("=" * 70)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        run_test(query_text)
    else:
        # Default sample if nothing passed
        print("Tip: You can pass any query as an argument, e.g.:")
        print('  python test_query.py "What is 15 percent of 500?"\n')
        print("Running default test query:")
        run_test("What is 15 percent of 500?")

        print("\nEntering interactive mode. Type any query to test (or 'exit' to quit):")
        while True:
            try:
                user_input = input("\nQuery > ").strip()
                if not user_input or user_input.lower() in ("exit", "quit", "q"):
                    print("Exiting test.")
                    break
                run_test(user_input)
            except (KeyboardInterrupt, EOFError):
                break

