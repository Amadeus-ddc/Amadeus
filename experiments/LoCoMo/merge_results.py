import os
import json
import argparse
import numpy as np
from datetime import datetime

def merge_results(results_dir):
    print(f"Aggregating results from: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        return

    all_sample_results = []
    category_scores = {}
    category_counts = {}
    total_questions = 0

    # Walk through the directory to find all sample_*.json files
    # Structure: results_dir/sample_XXX/results/sample_XXX.json
    
    found_files = 0
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.startswith("sample_") and file.endswith(".json"):
                # Check if it's in a 'results' subdirectory to confirm it's a result file
                if os.path.basename(root) == "results":
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Data should contain "qa_results"
                            if "qa_results" in data:
                                all_sample_results.append(data)
                                found_files += 1
                                
                                # Process stats for this sample
                                for q in data["qa_results"]:
                                    cat = str(q.get("category", "Unknown"))
                                    score = q.get("score", 0.0)
                                    
                                    if cat not in category_scores:
                                        category_scores[cat] = []
                                        category_counts[cat] = 0
                                    
                                    category_scores[cat].append(score)
                                    category_counts[cat] += 1
                                    total_questions += 1
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    print(f"Found {found_files} result files.")

    if found_files == 0:
        print("No valid result files found.")
        return

    # Calculate Aggregate Metrics
    aggregate_results = {"overall": {}}
    all_scores = [q["score"] for s in all_sample_results for q in s["qa_results"]]
    
    if all_scores:
        aggregate_results["overall"] = {
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores)),
            "count": len(all_scores)
        }
        
    for cat, scores in category_scores.items():
        aggregate_results[f"category_{cat}"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "count": len(scores)
        }

    # Final Summary Object
    final_summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_samples": len(all_sample_results),
        "total_questions": total_questions,
        "category_distribution": category_counts,
        "aggregate_metrics": aggregate_results,
        # "detailed_sample_summaries": all_sample_results # Optional: include all details? Maybe too big.
    }

    output_path = os.path.join(results_dir, "global_summary.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"Global summary written to: {output_path}")
    print("="*40)
    print(f"Total Samples: {len(all_sample_results)}")
    print(f"Overall Avg Score: {aggregate_results['overall'].get('mean', 0) * 100:.1f}%")
    print("Category Breakdown:")
    for cat, metrics in aggregate_results.items():
        if cat.startswith("category_"):
            cat_name = cat.replace("category_", "")
            print(f"  Category {cat_name}: {metrics['mean'] * 100:.1f}% (n={metrics['count']})")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Directory containing the run results")
    args = parser.parse_args()
    
    merge_results(args.results_dir)
