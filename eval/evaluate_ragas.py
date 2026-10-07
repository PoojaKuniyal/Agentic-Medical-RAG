"""
Decoupled 2-Step RAGAS Evaluation Script.

Modes:
  1. Generate / Update Predictions Dataset:
     Runs the 10 benchmark queries through LangGraph and saves/caches the results into eval/benchmark_predictions.csv.

  2. Evaluate Metrics:
     Loads the cached benchmark_predictions.csv and runs RAGAS metric calculations.
     This saves ~190,000 tokens by allowing evaluation to run separately or after token quota resets.

Usage:
  # Step 1: Generate predictions from LangGraph (run once)
  python eval/evaluate_ragas.py --generate

  # Step 2: Compute RAGAS scores from saved predictions
  python eval/evaluate_ragas.py --evaluate

  # Full run (both steps):
  python eval/evaluate_ragas.py --all
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset

from eval.eval_dataset import EVAL_DATASET_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTIONS_CSV = Path(__file__).parent / "benchmark_predictions.csv"
RESULTS_CSV = Path(__file__).parent / "ragas_eval_results.csv"
PLOT_PNG = Path(__file__).parent / "ragas_scores_chart.png"


def generate_predictions():
    """Step 1: Execute LangGraph on benchmark queries and cache outputs to CSV."""
    from app.graph.graph import get_graph
    langgraph_app = get_graph()

    logger.info("Executing LangGraph pipeline across %d benchmark queries...", len(EVAL_DATASET_SEED))
    eval_records = []

    for idx, item in enumerate(EVAL_DATASET_SEED, 1):
        question = item["user_input"]
        reference = item["reference"]
        
        logger.info(f"[{idx}/{len(EVAL_DATASET_SEED)}] Running LangGraph for: '{question[:50]}...'")
        
        initial_state = {
            "query": question,
            "session_id": f"eval_session_{idx}",
            "reflection_count": 0,
            "guideline_evidence": [],
            "pubmed_evidence": [],
            "ranked_evidence": [],
            "memory_context": {},
        }
        
        final_state = langgraph_app.invoke(initial_state)
        
        guideline_chunks = [g.text for g in final_state.get("guideline_evidence", [])]
        pubmed_abstracts = [p.abstract for p in final_state.get("pubmed_evidence", []) if p.abstract]
        combined_contexts = guideline_chunks + pubmed_abstracts
        
        final_response_obj = final_state.get("final_response")
        if final_response_obj and hasattr(final_response_obj, "summary"):
            answer = final_response_obj.summary
        elif isinstance(final_response_obj, dict):
            answer = final_response_obj.get("summary", "")
        else:
            answer = str(final_state.get("reasoning", ""))
            
        eval_records.append({
            "user_input": question,
            "retrieved_contexts": json.dumps(combined_contexts),  # serialized list
            "response": answer,
            "reference": reference
        })

    df_predictions = pd.DataFrame(eval_records)
    df_predictions.to_csv(PREDICTIONS_CSV, index=False)
    logger.info("Step 1 Complete! Predictions saved to: %s", PREDICTIONS_CSV)


def evaluate_metrics(batch_start: int = 0, batch_end: int = None):
    """Step 2: Run RAGAS metrics evaluation on saved predictions CSV."""
    if not PREDICTIONS_CSV.exists():
        logger.error("Predictions file %s does not exist. Run with --generate first.", PREDICTIONS_CSV)
        return

    logger.info("Loading cached predictions from %s...", PREDICTIONS_CSV)
    df_predictions = pd.read_csv(PREDICTIONS_CSV)

    if batch_end is not None:
        logger.info("Evaluating batch rows [%d:%d] out of %d total...", batch_start, batch_end, len(df_predictions))
        df_predictions = df_predictions.iloc[batch_start:batch_end].reset_index(drop=True)
    elif batch_start > 0:
        logger.info("Evaluating batch starting from row %d out of %d total...", batch_start, len(df_predictions))
        df_predictions = df_predictions.iloc[batch_start:].reset_index(drop=True)

    # Deserialize contexts back to list of strings
    df_predictions["retrieved_contexts"] = df_predictions["retrieved_contexts"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Alias columns for RAGAS legacy and modern compatibility
    df_predictions["question"] = df_predictions["user_input"]
    df_predictions["answer"] = df_predictions["response"]
    df_predictions["contexts"] = df_predictions["retrieved_contexts"]
    df_predictions["ground_truth"] = df_predictions["reference"]

    dataset = Dataset.from_pandas(df_predictions)

    # Configure Evaluator LLM & Embeddings
    import sys
    from unittest.mock import MagicMock
    sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
    sys.modules['langchain_community.llms.vertexai'] = MagicMock()

    from app.llm.factory import get_llm
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )

    raw_llm = get_llm(temperature=0.0)
    raw_embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    evaluator_llm = LangchainLLMWrapper(raw_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)
    
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    context_precision.llm = evaluator_llm
    context_recall.llm = evaluator_llm

    metrics_to_run = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=1, timeout=300, max_retries=10, max_wait=120)

    logger.info("Evaluating RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        run_config=run_config,
        raise_exceptions=False
    )

    results_df = results.to_pandas()

    if RESULTS_CSV.exists() and (batch_start > 0 or batch_end is not None):
        try:
            existing_df = pd.read_csv(RESULTS_CSV)
            # Remove any overlapping rows if re-evaluating batch
            results_df = pd.concat([existing_df, results_df], ignore_index=True).drop_duplicates(subset=["user_input"], keep="last")
        except Exception as e:
            logger.warning("Could not merge with existing RESULTS_CSV: %s", e)

    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Detailed evaluation results saved/updated in: %s", RESULTS_CSV)

    # Print overall dataset summary across all completed batches so far
    raw_metric_cols = [col for col in results_df.columns if col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
    mean_scores = results_df[raw_metric_cols].mean().dropna()

    print("\n" + "="*50)
    print("         RAGAS EVALUATION METRIC RESULTS         ")
    print("="*50)
    for metric, score in mean_scores.items():
        clean_name = metric.replace("_", " ").title()
        print(f"  • {clean_name:<22}: {score:.4f}")
    print("="*50 + "\n")

    generate_bar_chart(mean_scores, PLOT_PNG)
    logger.info("Visualization plot saved to: %s", PLOT_PNG)


def generate_bar_chart(mean_scores, save_path):
    plt.figure(figsize=(8, 5))
    metrics = [m.replace('_', ' ').title() for m in mean_scores.index]
    scores = mean_scores.values
    
    colors = ['#2b5c8f', '#4682b4', '#5c9ded', '#7eb0ee'][:len(metrics)]
    bars = plt.bar(metrics, scores, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
    
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='bold')
    plt.title("MedEvidence AI — RAGAS Benchmark Performance", fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-Step Decoupled RAGAS Evaluation Script")
    parser.add_argument("--generate", action="store_true", help="Step 1: Run LangGraph and save predictions CSV")
    parser.add_argument("--evaluate", action="store_true", help="Step 2: Run RAGAS metrics on saved predictions CSV")
    parser.add_argument("--all", action="store_true", help="Run both generation and evaluation")
    parser.add_argument("--start", type=int, default=0, help="Batch start index (0-indexed)")
    parser.add_argument("--end", type=int, default=None, help="Batch end index (exclusive)")

    args = parser.parse_args()

    # Default to --evaluate if predictions already exist, otherwise --all
    if args.generate:
        generate_predictions()
    elif args.evaluate:
        evaluate_metrics(batch_start=args.start, batch_end=args.end)
    elif args.all:
        generate_predictions()
        evaluate_metrics(batch_start=args.start, batch_end=args.end)
    else:
        if PREDICTIONS_CSV.exists():
            logger.info("Defaulting to --evaluate using cached predictions from %s", PREDICTIONS_CSV)
            evaluate_metrics(batch_start=args.start, batch_end=args.end)
        else:
            logger.info("No cached predictions found. Running full pipeline (--all)...")
            generate_predictions()
            evaluate_metrics(batch_start=args.start, batch_end=args.end)
