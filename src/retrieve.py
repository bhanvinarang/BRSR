"""Reusable per-company dense retrieval for the BRSR experiment.

Example:
    python src/retrieve.py ambuja "What is total energy consumed?" --field-id F01
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Terms mirror the public BRSR table labels and units.  They are not answers
# and do not use any company-specific annotation values.
FIELD_TERMS = {
    "F01": "energy consumption total energy consumed GJ renewable non-renewable fuel electricity",
    "F02": "electricity consumption total electricity consumed GJ renewable non-renewable",
    "F03": "Scope 1 emissions greenhouse gas emissions tonnes CO2 tCO2e",
    "F04": "Scope 2 emissions greenhouse gas emissions tonnes CO2 tCO2e",
    "F05": "Scope 3 emissions greenhouse gas emissions tonnes CO2 tCO2e",
    "F06": "water withdrawn total water withdrawal kilolitres KL megalitres",
    "F07": "water discharged total water discharge effluent kilolitres KL",
    "F08": "waste generated total waste hazardous non-hazardous tonnes",
    "F09": "hazardous waste generated hazardous waste tonnes",
    "F10": "employees total employees permanent workers employee count",
    "F11": "women employees female employees women workforce employee count",
    "F12": "employee grievance mechanism grievance redressal complaints workers",
    "F13": "energy intensity optional relevant metric energy intensity ratio",
    "F14": "water intensity optional relevant metric water intensity ratio",
    "F15": "waste intensity optional relevant metric waste intensity ratio",
    "F16": "Scope 1 Scope 2 emission intensity optional relevant metric emissions intensity ratio",
}


# Wording of the public SEBI BRSR reporting template (table row labels), used only on the
# lexical (BM25) side of hybrid retrieval.  Written from the template, not from any
# annotation; the dense query keeps FIELD_TERMS above so stored query vectors stay valid.
LEXICAL_TERMS = {
    "F01": "Total energy consumption (A+B+C+D+E+F) Joules or multiples Gigajoules from renewable sources non-renewable sources electricity fuel consumption other sources",
    "F02": "Total electricity consumption (A) from renewable sources non-renewable sources Joules or multiples Gigajoules",
    "F03": "Total Scope 1 emissions Break-up of the GHG into CO2 CH4 N2O HFCs PFCs SF6 NF3 Metric tonnes of CO2 equivalent",
    "F04": "Total Scope 2 emissions Break-up of the GHG into CO2 CH4 N2O HFCs PFCs SF6 NF3 Metric tonnes of CO2 equivalent",
    "F05": "Scope 3 emissions Metric tonnes of CO2 equivalent value chain Total Scope 3",
    "F06": "Total volume of water withdrawal in kilolitres surface water groundwater third party water seawater desalinated water (i + ii + iii + iv + v)",
    "F07": "Total volume of water discharged in kilolitres surface water groundwater seawater third-parties",
    "F08": "Total waste generated in metric tonnes plastic waste e-waste bio-medical waste construction demolition battery radioactive hazardous non-hazardous waste (A+B + C + D + E + F + G + H)",
    "F09": "hazardous waste other hazardous waste generated metric tonnes total waste generated",
    "F10": "Employees and workers including differently abled permanent employees other than permanent employees Total employees (D + E) male female",
    "F11": "Employees and workers including differently abled permanent employees Female number and percentage Total employees (D + E)",
    "F12": "Whether the entity has in place a mechanism to receive and redress grievances Employees and workers grievance redressal policy",
    "F13": "Energy intensity per rupee of turnover Total energy consumption Revenue from operations intensity per rupee of turnover adjusted for Purchasing Power Parity intensity in terms of physical output optional the relevant metric may be selected by the entity",
    "F14": "Water intensity per rupee of turnover Total water consumption Revenue from operations intensity per rupee of turnover adjusted for Purchasing Power Parity intensity in terms of physical output optional the relevant metric may be selected by the entity",
    "F15": "Waste intensity per rupee of turnover Total waste generated Revenue from operations intensity per rupee of turnover adjusted for Purchasing Power Parity intensity in terms of physical output optional the relevant metric may be selected by the entity",
    "F16": "Total Scope 1 and Scope 2 emissions per rupee of turnover Total Scope 1 and Scope 2 emission Revenue from operations intensity optional the relevant metric may be selected by the entity",
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class CompanyRetriever:
    def __init__(self, index_dir="data/index", model_name="BAAI/bge-small-en-v1.5", device="cpu", load_model=True):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(model_name, device=device) if load_model else None
        self.cache = {}

    def _load_company(self, company_id):
        if company_id not in self.cache:
            index_path = self.index_dir / f"{company_id}.faiss"
            chunks_path = self.index_dir / f"{company_id}.chunks.jsonl"
            if not index_path.exists() or not chunks_path.exists():
                raise FileNotFoundError(f"No index found for {company_id}; run src/chunk_index.py first.")
            with chunks_path.open(encoding="utf-8") as stream:
                chunks = [json.loads(line) for line in stream]
            tokenized = [tokenize(chunk["text"]) for chunk in chunks]
            document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
            self.cache[company_id] = (
                faiss.read_index(str(index_path)), chunks, tokenized, document_frequency,
                sum(map(len, tokenized)) / len(tokenized),
            )
        return self.cache[company_id]

    def retrieve(self, company_id, question, k=5, unique_pages=False, candidate_k=None,
                 field_id=None, strategy="hybrid", rrf_k=60, query_embedding=None,
                 dense_weight=1.0, lexical_weight=1.5, lexical_template=True):
        """Return the highest-scoring chunks, optionally one per PDF page.

        Page-diversified retrieval is useful when a downstream prompt has a
        fixed evidence budget: otherwise several overlapping chunks from one
        page can consume all available context.

        Defaults are the selected configuration: hybrid (dense + BM25) reciprocal-rank
        fusion with the lexical side weighted 1.5, the BM25 query built from the BRSR
        template wording in LEXICAL_TERMS (when ``field_id`` is given), top-5 chunks and
        no page de-duplication.  ``strategy="dense"``, ``lexical_weight=1.0`` and
        ``lexical_template=False`` reproduce the earlier baselines.
        """
        index, chunks, tokenized, document_frequency, avg_doc_length = self._load_company(company_id)
        if k < 1:
            raise ValueError("k must be at least 1")
        search_k = candidate_k or k
        if unique_pages:
            search_k = max(search_k, k * 8)
        if strategy not in {"dense", "hybrid"}:
            raise ValueError("strategy must be 'dense' or 'hybrid'")
        if strategy == "hybrid":
            search_k = max(search_k, 50)
        expanded_query = question
        if field_id in FIELD_TERMS:
            expanded_query = f"{question} BRSR disclosure table {FIELD_TERMS[field_id]}"
        query = query_embedding
        if query is None:
            query = self.model.encode(
                [QUERY_PREFIX + expanded_query], convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).astype("float32")
        elif query.ndim == 1:
            query = query.reshape(1, -1).astype("float32")
        scores, positions = index.search(query, min(search_k, len(chunks)))
        dense_positions = [int(position) for position in positions[0] if position >= 0]
        if strategy == "hybrid":
            if lexical_template and field_id in LEXICAL_TERMS:
                query_terms = tokenize(LEXICAL_TERMS[field_id])
            else:
                query_terms = tokenize(expanded_query)
            bm25_scores = []
            document_count = len(chunks)
            for tokens in tokenized:
                counts = Counter(tokens)
                score = 0.0
                for term in query_terms:
                    frequency = counts.get(term, 0)
                    if not frequency:
                        continue
                    idf = math.log(1 + (document_count - document_frequency[term] + 0.5) /
                                   (document_frequency[term] + 0.5))
                    score += idf * frequency * 2.2 / (frequency + 1.2 * (1 - 0.75 + 0.75 * len(tokens) / avg_doc_length))
                bm25_scores.append(score)
            lexical_positions = sorted(range(document_count), key=lambda pos: bm25_scores[pos], reverse=True)[:search_k]
            dense_rank = {position: rank for rank, position in enumerate(dense_positions, 1)}
            lexical_rank = {position: rank for rank, position in enumerate(lexical_positions, 1)}
            candidate_positions = set(dense_positions) | set(lexical_positions)
            score_by_position = {
                pos: dense_weight / (rrf_k + dense_rank.get(pos, 10_000))
                + lexical_weight / (rrf_k + lexical_rank.get(pos, 10_000))
                for pos in candidate_positions
            }
            # Ties are broken by chunk position so results are deterministic.
            ranked_positions = sorted(candidate_positions, key=lambda pos: (-score_by_position[pos], pos))
        else:
            ranked_positions = dense_positions
            score_by_position = {int(position): float(score) for score, position in zip(scores[0], positions[0]) if position >= 0}

        results = []
        seen_pages = set()
        for position in ranked_positions:
            result = dict(chunks[position])
            if unique_pages and result["page"] in seen_pages:
                continue
            result["score"] = float(score_by_position[position])
            results.append(result)
            seen_pages.add(result["page"])
            if len(results) >= k:
                break
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("company_id")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--field-id", choices=sorted(FIELD_TERMS))
    parser.add_argument("--strategy", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--lexical-weight", type=float, default=1.5)
    parser.add_argument("--legacy-lexical", action="store_true",
                        help="BM25 on question + FIELD_TERMS instead of the BRSR template wording.")
    parser.add_argument("--unique-pages", action="store_true")
    args = parser.parse_args()
    retriever = CompanyRetriever(args.index_dir, args.model, args.device)
    print(json.dumps(retriever.retrieve(
        args.company_id, args.question, args.k, unique_pages=args.unique_pages,
        field_id=args.field_id, strategy=args.strategy,
        lexical_weight=args.lexical_weight, lexical_template=not args.legacy_lexical,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
