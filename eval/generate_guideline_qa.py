"""
Script to sample chunks from local Chroma vector DB collections and auto-generate 
5 Guideline QA seed items using LLM synthesis.
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
from app.rag.vectorstore import similarity_search
from app.llm.factory import get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample topics/queries to fetch representative guideline chunks from Chroma
GUIDELINE_QUERIES = [
    "glycemic targets HbA1c type 2 diabetes adults",
    "diabetic kidney disease chronic kidney disease management T2D",
    "diabetic foot risk assessment and prevention guidelines",
    "first line antihyperglycemic therapy SGLT2 inhibitor GLP1 receptor agonist",
    "finerenone chronic kidney disease type 2 diabetes recommendations"
]

def generate_guideline_eval_items():
    llm = get_llm(temperature=0.1)
    generated_items = []
    
    for query in GUIDELINE_QUERIES:
        logger.info(f"Retrieving top guideline chunk for query: {query}")
        # Search across clinical_guidelines collection
        results = similarity_search(collection_name="clinical_guidelines", query=query, n_results=1, min_score=0.0)
        if not results:
            continue
            
        chunk_dict = results[0]
        context_text = chunk_dict.get("text", "")
        source_pdf = chunk_dict.get("source_pdf", "Guideline")
        page_num = chunk_dict.get("page_number", "N/A")
        
        prompt = f"""You are an expert clinical evidence evaluator.
Based strictly on the following excerpt from a clinical guideline (Source: {source_pdf}, Page: {page_num}):

---
{context_text}
---

Generate:
1. A clear, specific clinical question ('user_input') that a physician would ask which is directly answered by this guideline chunk.
2. A concise, authoritative ground-truth answer ('reference') derived ONLY from this excerpt.

Respond in strict JSON format with keys 'user_input' and 'reference'.
Do NOT include markdown block formatting.
"""
        
        response = llm.invoke(prompt)
        content = str(response.content).strip()
        
        # Robust JSON extraction using regex
        import re
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = content
            
        try:
            qa_pair = json.loads(json_str)
            if "user_input" in qa_pair and "reference" in qa_pair:
                generated_items.append({
                    "user_input": qa_pair["user_input"].strip(),
                    "reference": qa_pair["reference"].strip()
                })
                logger.info(f"Successfully generated QA pair for {metadata.get('source_file', 'Guideline')}")
            else:
                logger.warning(f"Missing required keys in LLM output: {qa_pair}")
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}\nRaw Content: {content}")

    print("\n--- GENERATED GUIDELINE QA PAIRS ---\n")
    print(json.dumps(generated_items, indent=4))
    
    return generated_items

if __name__ == "__main__":
    generate_guideline_eval_items()
