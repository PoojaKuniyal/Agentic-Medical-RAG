"""
Evaluation runner for RAGAS metrics.

Measures:
  1. Faithfulness (Groundedness against retrieved evidence)
  2. Answer Relevancy (Directness towards user query)
  3. Context Precision (Signal-to-noise ratio in retrieved contexts)
  4. Context Recall (Completeness of retrieved evidence against ground truth)

Outputs:
  - Markdown summary report
  - CSV breakdown per query
  - PNG visualization chart saved to eval/ragas_scores_chart.png
"""

import sys
import os
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset

# Bypass ragas legacy VertexAI imports without corrupting Pydantic v2 SecretStr
import sys
import types

if 'langchain_community.chat_models.vertexai' not in sys.modules:
    mock_module = types.ModuleType('langchain_community.chat_models.vertexai')
    class ChatVertexAI: pass
    mock_module.ChatVertexAI = ChatVertexAI
    sys.modules['langchain_community.chat_models.vertexai'] = mock_module

if 'langchain_community.llms.vertexai' not in sys.modules:
    mock_llm_module = types.ModuleType('langchain_community.llms.vertexai')
    class VertexAI: pass
    mock_llm_module.VertexAI = VertexAI
    sys.modules['langchain_community.llms.vertexai'] = mock_llm_module

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from eval.eval_dataset import EVAL_DATASET_SEED
from app.graph.graph import get_graph

langgraph_app = get_graph()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ragas_evaluation():
    logger.info("Starting RAGAS Evaluation across %d benchmark items...", len(EVAL_DATASET_SEED))
    eval_records = []

    for idx, item in enumerate(EVAL_DATASET_SEED, 1):
        question = item["user_input"]
        reference = item["reference"]
        
        logger.info(f"[{idx}/{len(EVAL_DATASET_SEED)}] Running LangGraph pipeline for query: '{question[:50]}...'")
        
        # Invoke your production multi-agent LangGraph workflow
        final_state = langgraph_app.invoke({
            "query": question,
            "session_id": f"eval_session_{idx}"
        })
        
        # Combine retrieved contexts (Guideline text chunks + PubMed abstracts)
        guideline_chunks = [g.text for g in final_state.get("guideline_evidence", [])]
        pubmed_abstracts = [p.abstract for p in final_state.get("pubmed_evidence", []) if p.abstract]
        
        combined_contexts = guideline_chunks + pubmed_abstracts
        
        # Extract response text
        final_response_obj = final_state.get("final_response")
        if final_response_obj and hasattr(final_response_obj, "summary"):
            answer = final_response_obj.summary
        elif isinstance(final_response_obj, dict):
            answer = final_response_obj.get("summary", "")
        else:
            answer = str(final_state.get("reasoning", ""))
            
        eval_records.append({
            "user_input": question,
            "retrieved_contexts": combined_contexts,
            "response": answer,
            "reference": reference
        })

    logger.info("Converting records to evaluation dataset...")
    df_eval = pd.DataFrame(eval_records)
    dataset = Dataset.from_pandas(df_eval)

    # Configure Evaluator LLM & Embeddings
    from app.llm.factory import get_llm
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    raw_llm = get_llm(temperature=0.0)
    raw_embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    evaluator_llm = LangchainLLMWrapper(raw_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)
    
    # Assign evaluator LLM and embeddings to standard metrics
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
    
    # Run RAGAS Evaluation sequentially (max_workers=1) to prevent 429 rate limit exceptions
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=1, timeout=120)

    logger.info("Evaluating RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics_to_run,
        run_config=run_config,
        raise_exceptions=False
    )

    # Convert results to DataFrame
    results_df = results.to_pandas()
    
    # Save detailed CSV
    csv_path = Path(__file__).parent / "ragas_eval_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Detailed CSV saved to: %s", csv_path)

    # Generate summary metrics, filtering out NaN columns (e.g. rate-limited metrics)
    raw_metric_cols = [col for col in results_df.columns if col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
    mean_scores = results_df[raw_metric_cols].mean().dropna()

    print("\n" + "="*50)
    print("         RAGAS EVALUATION METRIC RESULTS         ")
    print("="*50)
    for metric, score in mean_scores.items():
        clean_name = metric.replace("_", " ").title()
        print(f"  • {clean_name:<22}: {score:.4f}")
    print("="*50 + "\n")

    # Generate and Save Visualization Plot
    plot_path = Path(__file__).parent / "ragas_scores_chart.png"
    generate_bar_chart(mean_scores, plot_path)
    logger.info("Visualization plot saved to: %s", plot_path)


def generate_bar_chart(mean_scores, save_path):
    plt.figure(figsize=(8, 5))
    metrics = [m.replace('_', ' ').title() for m in mean_scores.index]
    scores = mean_scores.values
    
    colors = ['#2b5c8f', '#4682b4', '#5c9ded'][:len(metrics)]
    bars = plt.bar(metrics, scores, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
    
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='bold')
    plt.title("MedEvidence AI — RAGAS Benchmark Performance", fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate score values on top of bars
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
    run_ragas_evaluation()
