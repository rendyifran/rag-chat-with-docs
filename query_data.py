import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

CHROMA_PATH = "chroma" # Directory where the Chroma vector database is stored

PROMPT = """ 
You must answer using ONLY the context below.

If the context does NOT explicitly contain the specific facts asked about, reply with exactly:
I don't know based on the provided documents.

If the context DOES explicitly contain the answer, reply with ONE clear, natural sentence that states it.

Do NOT give vague or general answers.
Do NOT restate the question.
Do NOT guess.

Context:
{context}

Question: {question}
""" # A prompt template for the language model, instructing it to answer questions based solely on the provided context, with specific guidelines on how to respond when the answer is not explicitly present in the context, and emphasizing the need for clear and concise answers without restating the question or making guesses.


DEBUG = False # Set to True to enable debug prints and the rescue pass for bad answers
FETCH_K = 80 # Number of candidates to fetch from Chroma before intent-aware selection. We fetch more than FINAL_K to give the selector more options to choose from, especially for definition questions where we want to find strong matches with key phrases and list structures.
FINAL_K = 12 # Number of documents to include in the final context passed to the LLM. This is a balance between providing enough relevant information and not overwhelming the model with too much text, which could lead to increased latency and potential confusion. The intent-aware selector will try to choose the most relevant documents based on the detected intent of the query, and we set this to 12 to allow for a rich context while still being manageable for the model to process effectively.
MMR_LAMBDA = 0.5 # Lambda parameter for Max Marginal Relevance (MMR) search in Chroma. This parameter controls the balance between relevance and diversity in the retrieved documents. A value of 0.5 means that we give equal weight to relevance (how closely a document matches the query) and diversity (how different the documents are from each other). Adjusting this parameter can help improve the quality of the retrieved documents by ensuring that we get a good mix of relevant information without too much redundancy, which is especially important when we are fetching a larger number of candidates (FETCH_K) for the intent-aware selection step.
MAX_CONTEXT_CHARS = 4500 # Maximum number of characters to include in the context passed to the LLM. This is set to 4500 to ensure that we provide a substantial amount of relevant information from the retrieved documents while staying within typical token limits for language models (considering that 1 token is roughly 4 characters, this allows for around 1000-1100 tokens in the context). This limit helps prevent overwhelming the model with too much text, which could lead to increased latency and potential confusion, while still giving it enough information to generate accurate and informed answers based on the retrieved documents.

# Cues for intent classification and intent-aware selection
DEFINITION_CUES = [
    "was based on", "based on", "consists of", "composed of", "constructed from",
    "defined as", "variables:", "indicators:", "four variables", "five variables", "six variables",
]
CAUSAL_CUES = [
    "because", "due to", "as a result", "led to", "driven by", "explained by", "associated with",
    "suggests", "indicates", "therefore", "thus",
]
COMPARISON_CUES = [
    "compared to", "whereas", "in contrast", "difference", "similar", "higher than", "lower than",
    "original", "modified", "baseline", "version",
]
PROCEDURE_CUES = [
    "steps", "procedure", "method", "workflow", "we performed", "we used", "approach", "process",
    "first", "then", "next", "finally",
]
DESCRIPTIVE_CUES = [
    "shows", "illustrates", "figure", "table", "results", "findings", "summary", "observed",
    "we found", "the study", "this analysis",
]

def load_db():
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

# ---- Small utilities ----
def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def lexical_overlap_score(query: str, text: str) -> float:
    q_words = set(normalize_text(query).split())
    t_words = set(normalize_text(text).split())
    return float(len(q_words & t_words))

def has_any_cue(text: str, cues: List[str]) -> bool:
    t = (text or "").lower()
    return any(cue in t for cue in cues)

def has_list_structure(text: str) -> bool:
    t = (text or "")
    low = t.lower()
    return (":" in t and t.count(",") >= 2) or ("four variables" in low) or ("five variables" in low) or ("six variables" in low)

def extract_key_phrase(query: str) -> str:
    """
    Heuristic: if 'index' exists, take phrase up to 'index'.
    Otherwise return first ~6 words.
    """
    q = normalize_text(query)
    if "index" in q:
        left = q.split("index")[0].strip()
        phrase = (left + " index").strip()
        if len(phrase.split()) >= 3:
            return phrase
    words = q.split()
    return " ".join(words[:6]) if len(words) >= 6 else q

def build_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    context = "\n\n---\n\n".join([d.page_content or "" for d in docs])
    context = " ".join(context.split())
    return context[:max_chars]

def is_bad_answer(ans: str) -> bool:
    a = (ans or "").strip()
    if not a:
        return True
    if a.strip().lower() == "i don't know based on the provided documents.":
        return True
    vague_markers = [
        "related to", "at least", "generally", "some variables", "some indicators", "includes",
        "composed of at least", "variables related to", "indicators related to",
    ]
    low = a.lower()
    return any(m in low for m in vague_markers)

# ---- Query intent classification ----
@dataclass
class IntentResult:
    intent: str
    confidence: float

def classify_intent(query: str) -> IntentResult:
    q = normalize_text(query)

    def_triggers = ["compose", "composed", "consist", "based", "define", "definition", "variable", "variables", "indicator", "indicators"]
    causal_triggers = ["why", "cause", "caused", "because", "due", "driven", "lead", "effect", "impact"]
    compare_triggers = ["compare", "difference", "vs", "versus", "contrast", "similar", "original", "modified"]
    procedure_triggers = ["how to", "how do", "steps", "procedure", "workflow", "method", "process"]
    descriptive_triggers = ["summarize", "overview", "describe", "what did", "what are the results", "show", "findings"]

    scores = {
        "definition": sum(1 for t in def_triggers if t in q),
        "causal": sum(1 for t in causal_triggers if t in q),
        "comparison": sum(1 for t in compare_triggers if t in q),
        "procedure": sum(1 for t in procedure_triggers if t in q),
        "descriptive": sum(1 for t in descriptive_triggers if t in q),
    }

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    if best_score == 0:
        return IntentResult(intent="descriptive", confidence=0.4)

    conf = min(0.95, 0.55 + 0.15 * best_score)
    return IntentResult(intent=best_intent, confidence=conf)

# ---- Retrieval ----
def mmr_search(db: Chroma, query: str, k: int = FINAL_K, fetch_k: int = FETCH_K, lambda_mult: float = MMR_LAMBDA) -> List[Document]:
    return db.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

def retrieve_semantic(db: Chroma, query_text: str) -> List[Document]:
    docs = mmr_search(db, query_text, k=FINAL_K, fetch_k=FETCH_K, lambda_mult=MMR_LAMBDA)

    scored: List[Tuple[Document, float]] = []
    for d in docs:
        scored.append((d, lexical_overlap_score(query_text, d.page_content or "")))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [d for d, _ in scored[:FINAL_K]]

def lexical_fallback(db: Chroma, query_text: str, max_hits: int = 60) -> List[Document]:
    """
    Linear scan fallback. Works well for small-medium corpora.
    """
    col = db._collection
    data = col.get(include=["documents", "metadatas"])

    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    q_words = [w for w in normalize_text(query_text).split() if len(w) >= 4]
    if not q_words:
        return []

    hits: List[Tuple[Document, int]] = []
    for doc_text, meta in zip(docs, metas):
        t = normalize_text(doc_text or "")
        score = sum(1 for w in q_words if w in t)
        if score >= 3:
            hits.append((Document(page_content=doc_text or "", metadata=meta or {}), score))

    hits.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in hits[:max_hits]]

# ---- Intent-aware selection ----
def intent_aware_selector(query_text: str, intent: str, candidates: List[Document]) -> List[Document]:
    key_phrase = extract_key_phrase(query_text).lower()

    if intent == "definition":
        cues = DEFINITION_CUES
        require_list = True
    elif intent == "causal":
        cues = CAUSAL_CUES
        require_list = False
    elif intent == "comparison":
        cues = COMPARISON_CUES
        require_list = False
    elif intent == "procedure":
        cues = PROCEDURE_CUES
        require_list = False
    else:
        cues = DESCRIPTIVE_CUES
        require_list = False

    strong, medium, weak = [], [], []

    for d in candidates:
        txt = (d.page_content or "")
        low = txt.lower()

        kp_hit = key_phrase in low if len(key_phrase.split()) >= 3 else False
        cue_hit = has_any_cue(low, cues)
        list_hit = has_list_structure(txt)

        if intent == "definition":
            if kp_hit and cue_hit and list_hit:
                strong.append(d)
            elif (kp_hit and cue_hit) or (cue_hit and list_hit):
                medium.append(d)
            else:
                weak.append(d)
        else:
            if kp_hit and cue_hit:
                strong.append(d)
            elif cue_hit or kp_hit:
                medium.append(d)
            else:
                weak.append(d)

    selected = strong + medium
    if len(selected) < FINAL_K:
        selected += weak

    if intent == "definition" and require_list:
        selected.sort(key=lambda d: (not has_list_structure(d.page_content or ""),), reverse=False)

    return selected[:FINAL_K]

def debug_find_in_chroma(db: Chroma, source_contains: str, pattern: str, max_hits: int = 10) -> None:
    col = db._collection
    data = col.get(include=["documents", "metadatas"])

    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    rx = re.compile(pattern, re.IGNORECASE)

    hits = 0
    for doc_text, meta in zip(docs, metas):
        src = (meta.get("source") or "")
        if source_contains.lower() not in src.lower():
            continue

        if doc_text and rx.search(doc_text):
            hits += 1
            print("\n--- HIT", hits, "---")
            print("SOURCE:", src, "| PAGE:", meta.get("page"))
            m = rx.search(doc_text)
            start = max(0, m.start() - 200)
            end = min(len(doc_text), m.end() + 500)
            print(doc_text[start:end])
            if hits >= max_hits:
                break

    if hits == 0:
        print(f"\nNo matches found in Chroma for source containing '{source_contains}' and pattern '{pattern}'.")

# ---- Answer pipeline ----
def answer_question(db: Chroma, query_text: str) -> Tuple[str, List[str], str]:
    intent_res = classify_intent(query_text)
    intent = intent_res.intent

    semantic_docs = retrieve_semantic(db, query_text)
    lexical_docs = lexical_fallback(db, query_text, max_hits=60)

    merged: List[Document] = []
    seen = set()
    for d in (lexical_docs + semantic_docs):
        key = (d.metadata.get("source"), d.metadata.get("page"), (d.page_content or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)

    docs = intent_aware_selector(query_text, intent, merged)

    if DEBUG:
        print("\n[DEBUG] intent =", intent_res)
        print("[DEBUG] selected docs:")
        for i, d in enumerate(docs, 1):
            print(f"  [{i}] {d.metadata.get('source')} | page={d.metadata.get('page')}")

    context_text = build_context(docs)

    prompt_template = ChatPromptTemplate.from_template(PROMPT)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOllama(model="llama3.2:3b", temperature=0)
    response = model.invoke(prompt)
    answer = (response.content or "").strip()

    # Rescue only for definition-like: tighten to keyphrase + definition cues
    if is_bad_answer(answer) and intent == "definition":
        kp = extract_key_phrase(query_text).lower()
        rescue = [
            d for d in merged
            if (kp in (d.page_content or "").lower())
            and has_any_cue((d.page_content or "").lower(), DEFINITION_CUES)
        ]
        rescue = rescue[:FINAL_K] if rescue else docs
        context_text = build_context(rescue)
        prompt = prompt_template.format(context=context_text, question=query_text)
        response = model.invoke(prompt)
        answer = (response.content or "").strip()
        docs = rescue

    sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")]
    sources = dedupe_preserve_order(sources)

    return answer, sources, intent

# ---- Evaluation harness (improved) ----
@dataclass
class EvalOutcome:
    status: str  # "pass" | "partial" | "fail"
    hits: int
    needed: int
    missing: List[str]

def load_eval_cases(path: Optional[str]) -> List[Dict[str, Any]]:
    """
    JSON format:
    [
      {
        "id": "case1",
        "question_variants": ["q1", "q2", ...],
        "expected_contains": ["phrase1", "phrase2"],
        "expected_exact": null,
        "type": "definition"  // optional (we'll still compute detected intent)
      }
    ]
    """
    if not path:
        return [
            {
                "id": "infra_index_vars",
                "type": "definition",
                "question_variants": [
                    "What indicators compose the Physical Infrastructure Vulnerability Index?",
                    "The Physical Infrastructure Vulnerability Index was based on what variables?",
                    "what is the physical infrastructure vulnerability index based on",
                ],
                "expected_contains": [
                    "number of fire incidents",
                    "number of flood incidents",
                    "slum levels",
                    "population density",
                ],
                "expected_exact": None
            }
        ]

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _is_idk(answer: str) -> bool:
    return normalize_text(answer) == normalize_text("I don't know based on the provided documents.")

def eval_answer(answer: str, expected_contains: List[str], expected_exact: Optional[str]) -> EvalOutcome:
    a = (answer or "").lower()

    # If model says IDK, this is an automatic fail (but still show missing)
    missing = [p for p in expected_contains if p.lower() not in a]
    hits = len(expected_contains) - len(missing)
    needed = len(expected_contains)

    if expected_exact is not None:
        ok = a.strip() == expected_exact.strip().lower()
        return EvalOutcome(status=("pass" if ok else "fail"), hits=(1 if ok else 0), needed=1, missing=([] if ok else ["expected_exact_mismatch"]))

    if needed == 0:
        # nothing to check; treat any non-empty non-idk answer as pass
        if (answer or "").strip() and not _is_idk(answer):
            return EvalOutcome(status="pass", hits=0, needed=0, missing=[])
        return EvalOutcome(status="fail", hits=0, needed=0, missing=["no_expected_criteria"])

    if _is_idk(answer) or is_bad_answer(answer):
        return EvalOutcome(status="fail", hits=hits, needed=needed, missing=missing)

    # PASS = all phrases present
    if hits == needed:
        return EvalOutcome(status="pass", hits=hits, needed=needed, missing=[])

    # PARTIAL threshold:
    # - if <=3 expected items: require at least 1 hit for partial
    # - if >3 expected items: require at least max(2, 50% rounded down)
    if needed <= 3:
        partial_min = 1
    else:
        partial_min = max(2, needed // 2)

    if hits >= partial_min:
        return EvalOutcome(status="partial", hits=hits, needed=needed, missing=missing)

    return EvalOutcome(status="fail", hits=hits, needed=needed, missing=missing)

def run_evaluation(db: Chroma, eval_path: Optional[str], verbose: bool = False) -> None:
    cases = load_eval_cases(eval_path)

    total = 0
    pass_n = 0
    partial_n = 0
    fail_n = 0

    # Track by intent (detected)
    type_stats: Dict[str, Dict[str, int]] = {}
    # Track by case
    case_stats: List[Tuple[str, int, int, int, int]] = []  # (id, pass, partial, fail, total)

    start_all = time.time()

    for case in cases:
        cid = case.get("id", "unknown")
        variants = case.get("question_variants", [])
        expected_contains = case.get("expected_contains", [])
        expected_exact = case.get("expected_exact", None)

        c_pass = c_partial = c_fail = 0

        if verbose:
            print("\n==============================")
            print(f"CASE: {cid}")
            print("==============================")

        for q in variants:
            total += 1

            t0 = time.time()
            ans, sources, intent = answer_question(db, q)
            dt = time.time() - t0

            outcome = eval_answer(ans, expected_contains, expected_exact)

            if outcome.status == "pass":
                pass_n += 1
                c_pass += 1
            elif outcome.status == "partial":
                partial_n += 1
                c_partial += 1
            else:
                fail_n += 1
                c_fail += 1

            if intent not in type_stats:
                type_stats[intent] = {"total": 0, "pass": 0, "partial": 0, "fail": 0}
            type_stats[intent]["total"] += 1
            type_stats[intent][outcome.status] += 1

            if verbose and outcome.status != "pass":
                print(f"\nVariant: {q}")
                print(f"Detected intent: {intent} | time: {dt:.2f}s")
                print(f"Status: {outcome.status.upper()} ({outcome.hits}/{outcome.needed} hits)")
                if outcome.missing:
                    print("Missing:", outcome.missing)
                print("Answer:", ans)
                print("Sources:", sources)

        case_stats.append((cid, c_pass, c_partial, c_fail, len(variants)))

    elapsed = time.time() - start_all

    # ---- Print Report ----
    print("\n==============================")
    print("=== OVERALL PERFORMANCE ===")
    print("==============================")
    if total > 0:
        print(f"Total: {pass_n}/{total} passed ({(pass_n/total*100):.1f}%)")
        print(f"Partial: {partial_n}/{total} ({(partial_n/total*100):.1f}%)")
        print(f"Fail: {fail_n}/{total} ({(fail_n/total*100):.1f}%)")
        print(f"Pass+Partial: {(pass_n+partial_n)}/{total} ({((pass_n+partial_n)/total*100):.1f}%)")
        print(f"Elapsed: {elapsed:.1f}s")
    else:
        print("No tests run.")

    print("\n==============================")
    print("=== BY QUESTION TYPE (detected) ===")
    print("==============================")
    # consistent ordering
    for intent in ["definition", "causal", "descriptive", "comparison", "procedure"]:
        if intent not in type_stats:
            continue
        stats = type_stats[intent]
        t = stats["total"]
        p = stats["pass"]
        pa = stats["partial"]
        f = stats["fail"]
        print(f"{intent}: pass {p}/{t} ({(p/t*100):.1f}%) | partial {pa}/{t} ({(pa/t*100):.1f}%) | fail {f}/{t} ({(f/t*100):.1f}%)")

    print("\n==============================")
    print("=== BY TEST CASE ===")
    print("==============================")
    for cid, cp, cpa, cf, ct in case_stats:
        print(f"{cid}: pass {cp}/{ct} ({(cp/ct*100):.1f}%) | partial {cpa}/{ct} ({(cpa/ct*100):.1f}%) | fail {cf}/{ct} ({(cf/ct*100):.1f}%)")

    print("\nEvaluation complete.\n")

# ---- CLI ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", nargs="?", default=None, help="The query text.")

    # Debug find
    parser.add_argument("--debug_find", action="store_true", help="Scan Chroma for a regex pattern and exit.")
    parser.add_argument("--debug_source", type=str, default="LTKB8-PSA.pdf")
    parser.add_argument(
        "--debug_pattern",
        type=str,
        default=r"physical\s+infrastructure\s+vulnerability\s+index.*?(was\s+based\s+on|based\s+on|four\s+variables|variables\s*:)",
    )

    # Evaluation harness
    parser.add_argument("--eval", action="store_true", help="Run evaluation harness.")
    parser.add_argument("--eval_file", type=str, default=None, help="Path to evaluation JSON file.")
    parser.add_argument("--eval_verbose", action="store_true", help="Print failing variants with reasons.")

    args = parser.parse_args()

    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    if args.debug_find:
        debug_find_in_chroma(db, args.debug_source, args.debug_pattern, max_hits=10)
        return

    if args.eval:
        run_evaluation(db, args.eval_file, verbose=args.eval_verbose)
        return

    if not args.query_text:
        print("No query provided. Example:")
        print('  python query_data.py "What indicators compose the Physical Infrastructure Vulnerability Index?"')
        print("Or run evaluation from file:")
        print("  python query_data.py --eval --eval_file eval_cases.json")
        print("Or verbose evaluation:")
        print("  python query_data.py --eval --eval_file eval_cases.json --eval_verbose")
        return

    answer, sources, intent = answer_question(db, args.query_text)

    print("\nResponse:\n", answer)
    print("\nIntent:", intent)
    print("\nSources:", sources)

if __name__ == "__main__":
    main()