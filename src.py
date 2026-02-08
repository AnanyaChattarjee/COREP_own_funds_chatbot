import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypdf import PdfReader
import re
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
import os
from google import genai
import streamlit as st
API_KEY = st.secrets["API_KEY"]





# def extract_text_from_pdf(pdf_path):
#     """Extracts all text from a given PDF file."""
#     reader = PdfReader(pdf_path)
#     full_text = ""
#     for page in reader.pages:
#         full_text += page.extract_text() or "" 
#     return full_text


# file_path1 = 'documents/corep-own-funds-instructions.pdf'
# file_path2 = 'documents/Reporting (CRR)_06-02-2026.pdf'
# own_funds_instructions = extract_text_from_pdf(file_path1)
# reporting_crr = extract_text_from_pdf(file_path2)



# data_list = list()
# i =0
# while (i < len(own_funds_instructions)):
#     row = {'chunk_id': 'corep_'+str(int(i/1000)),'text':own_funds_instructions[i:i+1200]	,'source':'COREP_Annex_II'}
#     i=i+1000
#     data_list.append(row)

# data1=pd.DataFrame(data_list)
# def is_junk_chunk(text, min_len=200):
#     if not isinstance(text, str):
#         return True

#     t = text.strip()

#     if len(t) < min_len:
#         return True

#     no_space = re.sub(r"\s+", "", t)

#     # mostly punctuation/dots
#     if len(re.sub(r"[A-Za-z0-9]", "", no_space)) / len(no_space) > 0.85:
#         return True

#     # long dotted separator
#     if re.search(r"\.{15,}", t):
#         return True

#     # very low alphabetic content
#     alpha_chars = sum(c.isalpha() for c in t)
#     if alpha_chars / len(t) < 0.15:
#         return True

#     return False


# data1_clean = data1.copy()
# data1_clean["is_junk"] = data1_clean["text"].apply(is_junk_chunk)

# print("Before:", len(data1_clean))

# data1_clean = data1_clean[data1_clean["is_junk"] == False] \
#     .drop(columns=["is_junk"]) \
#     .reset_index(drop=True)

# print("After:", len(data1_clean))
# # Separate COREP and PRA
# corep_df = data1_clean[data1_clean["source"] == "COREP_Annex_II"].copy().reset_index(drop=True)


# # Reassign sequential chunk_ids
# corep_df["chunk_id"] = [f"corep_{i:04d}" for i in range(len(corep_df))]


# # Merge back
# data1 = pd.concat([corep_df], ignore_index=True)



# data_list_new = list()
# i =0
# while (i < len(reporting_crr)):
#     row = {'chunk_id': 'pra_'+str(int(i/1000)),'text':reporting_crr[i:i+1200]	,'source':'PRA_RULEBOOK'}
#     i=i+1000
#     data_list_new.append(row)

# data2=pd.DataFrame(data_list_new)
# def is_junk_chunk(text, min_len=200):
#     if not isinstance(text, str):
#         return True

#     t = text.strip()

#     if len(t) < min_len:
#         return True

#     no_space = re.sub(r"\s+", "", t)

#     # mostly punctuation/dots
#     if len(re.sub(r"[A-Za-z0-9]", "", no_space)) / len(no_space) > 0.85:
#         return True

#     # long dotted separator
#     if re.search(r"\.{15,}", t):
#         return True

#     # very low alphabetic content
#     alpha_chars = sum(c.isalpha() for c in t)
#     if alpha_chars / len(t) < 0.15:
#         return True

#     return False


# data2_clean = data2.copy()
# data2_clean["is_junk"] = data2_clean["text"].apply(is_junk_chunk)

# print("Before:", len(data2_clean))

# data2_clean = data2_clean[data2_clean["is_junk"] == False] \
#     .drop(columns=["is_junk"]) \
#     .reset_index(drop=True)

# print("After:", len(data2_clean))
# pra_df   = data2_clean[data2_clean["source"] == "PRA_RULEBOOK"].copy().reset_index(drop=True)

# pra_df["chunk_id"]   = [f"pra_{i:04d}" for i in range(len(pra_df))]

# # Merge back
# data2 = pd.concat([ pra_df], ignore_index=True)


# data_final = pd.concat([data1,data2],ignore_index=True)
# data_final


# bad_patterns = r"\[Deleted\]|\[ Deleted \]|Provision left blank|can be found here|\[Deleted\.\]"

# data_final = data_final[
#     ~data_final["text"].str.contains(bad_patterns, regex=True, flags=re.IGNORECASE)
# ].copy()

# data_final = data_final.reset_index(drop=True)
# keep_keywords = r"COREP|own funds|CET1|Tier 1|Tier 2|capital requirements|CRR"

# pra_useful = data_final[
#     (data_final["source"] == "PRA_RULEBOOK") &
#     (data_final["text"].str.contains(keep_keywords, regex=True, flags=re.IGNORECASE))
# ]

# corep_all = data_final[data_final["source"] == "COREP_Annex_II"]

# data_final = pd.concat([corep_all, pra_useful], ignore_index=True)
# data_final = data_final.reset_index(drop=True)



# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# embeddings = model.encode(
#     data_final["text"].tolist(),
#     show_progress_bar=True,normalize_embeddings=True
# )

# embeddings = np.array(embeddings).astype("float32")
# data_final["embedding_text"] = list(embeddings)

# for i in data_final['text']:
#     data_final['text'] = model.encode(i)


# dimension = embeddings.shape[1]
# index = faiss.IndexFlatIP(dimension)
# index.add(embeddings)

# faiss.write_index(index, "corep_faiss.index")




# metadata = data_final[["chunk_id", "source", "text"]].to_dict(orient="records")

# with open("corep_metadata.pkl", "wb") as f:
#     pickle.dump(metadata, f)



def load_retrieval_system(index_path="corep_faiss.index",
                          metadata_path="corep_metadata.pkl",
                          embed_model_name="sentence-transformers/all-MiniLM-L6-v2"):

    model = SentenceTransformer(embed_model_name)
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return model, index, metadata


model, index, metadata = load_retrieval_system()



def retrieve_chunks(query, model, index, metadata, top_k=5):

    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec).astype("float32")

    distances, indices = index.search(q_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk_data = metadata[idx]

        results.append({
            "chunk_id": chunk_data["chunk_id"],
            "source": chunk_data["source"],
            "text": chunk_data["text"],
            "score": float(distances[0][i])
        })

    return results


def definition_print(query, model, index, metadata, top_k=5):
    retrieved = retrieve_chunks(query, model, index, metadata, top_k=top_k)

    for r in retrieved:
        print(r["chunk_id"], r["source"], r["score"])
        print(r["text"][:300])
        print("-----")


def build_chunk_lookup(retrieved_chunks):
    """
    Converts retrieved_chunks list into a dictionary:
    {chunk_id: chunk_text}
    """
    return {c["chunk_id"]: c["text"] for c in retrieved_chunks}



ROW_KEYWORDS = {
    "0010": ["own funds", "total own funds"],
    "0015": ["tier 1", "tier 1 capital"],
    "0020": ["common equity tier 1", "cet1"],
    "0030": ["additional tier 1", "at1"],
    "0040": ["tier 2", "tier 2 capital"]
}


def validate_evidence(result_json, retrieved_chunks):
    """
    Checks if each populated cell has source chunks containing relevant keywords.
    Adds validation flags if evidence looks wrong.
    """
    flags = result_json.get("validation_flags", [])
    chunk_lookup = build_chunk_lookup(retrieved_chunks)

    for cell in result_json.get("populated_cells", []):
        row = cell.get("row")
        source_ids = cell.get("source_chunk_ids", [])

        if not row or row not in ROW_KEYWORDS:
            continue

        expected_keywords = ROW_KEYWORDS[row]

        # If no chunk ids at all
        if not source_ids:
            flags.append({
                "type": "missing_data",
                "message": f"Row {row} has no source_chunk_ids evidence."
            })
            continue

        # Combine all evidence text
        combined_text = ""
        missing_chunks = []

        for cid in source_ids:
            if cid in chunk_lookup:
                combined_text += " " + chunk_lookup[cid].lower()
            else:
                missing_chunks.append(cid)

        # Chunk IDs not found in retrieved set
        if missing_chunks:
            flags.append({
                "type": "warning",
                "message": f"Row {row} references chunk_ids not in retrieved context: {missing_chunks}"
            })

        # Keyword validation
        keyword_found = any(kw.lower() in combined_text for kw in expected_keywords)

        if not keyword_found:
            flags.append({
                "type": "warning",
                "message": f"Row {row} evidence may be incorrect. None of {expected_keywords} found in provided source chunks."
            })

    result_json["validation_flags"] = flags
    return result_json





def validate_evidence_keywords(result_json, retrieved_chunks):
    flags = []

    chunk_map = {c["chunk_id"]: c["text"].lower() for c in retrieved_chunks}

    row_keywords = {
        "0030": ["additional tier 1", "at1", "article 52"],
        "0040": ["tier 2", "t2", "article 62"],
        "0010": ["own funds", "total own funds", "article 72"],
        "0015": ["tier 1", "article 25"],
        "0020": ["common equity tier 1", "cet1", "article 50"]
    }

    for cell in result_json.get("populated_cells", []):
        row = cell.get("row")
        chunk_ids = cell.get("source_chunk_ids", [])

        if row not in row_keywords:
            continue

        combined_text = ""
        for cid in chunk_ids:
            combined_text += " " + chunk_map.get(cid, "")

        if not any(k in combined_text for k in row_keywords[row]):
            flags.append({
                "type": "warning",
                "message": f"Row {row} evidence may be incorrect. None of expected keywords {row_keywords[row]} found in cited source chunks."
            })

    return flags



def fix_thousand_reporting(result_json):
    """
    COREP reports values in 000 GBP (thousands).
    If values look like full GBP, convert to 000 GBP.
    """
    for cell in result_json.get("populated_cells", []):
        val = cell.get("value")
        if isinstance(val, (int, float)) and val > 1000000:
            cell["value"] = int(val / 1000)
            cell["unit"] = "000 GBP"

    for log in result_json.get("audit_log", []):
        val = log.get("value")
        if isinstance(val, (int, float)) and val > 1000000:
            log["value"] = int(val / 1000)

    return result_json





def retrieve_rowwise_context(model, index, metadata, scenario, top_k=5):
    row_queries = {
        "0020": "COREP C 01.00 row 0020 Common Equity Tier 1 capital CET1 instructions Article 50 CRR",
        "0030": "COREP C 01.00 row 0030 Additional Tier 1 capital AT1 instructions Article 52 CRR",
        "0040": "COREP C 01.00 row 0040 Tier 2 capital instructions Article 62 CRR",
        "0015": "COREP C 01.00 row 0015 Tier 1 capital instructions Tier 1 = CET1 + AT1",
        "0010": "COREP C 01.00 row 0010 Own Funds instructions Own Funds = Tier 1 + Tier 2"
    }

    retrieved_all = []

    for row, query in row_queries.items():
        chunks = retrieve_chunks(query, model, index, metadata, top_k=top_k)
        retrieved_all.extend(chunks)

    # remove duplicates
    seen = set()
    merged = []
    for c in retrieved_all:
        if c["chunk_id"] not in seen:
            merged.append(c)
            seen.add(c["chunk_id"])

    # sort by score descending
    merged = sorted(merged, key=lambda x: x["score"], reverse=True)

    return merged





MODEL_NAME = "models/gemini-3-flash-preview"
import re

ROW_EXTRACT = re.compile(r"(\d{4})")

def fix_row_codes(result_json):
    if result_json is None:
        return None

    # Fix populated_cells
    for cell in result_json.get("populated_cells", []):
        raw_row = str(cell.get("row", ""))
        match = ROW_EXTRACT.search(raw_row)
        if match:
            cell["row"] = match.group(1)

    # Fix audit_log
    for log in result_json.get("audit_log", []):
        raw_field = str(log.get("field", ""))
        match = ROW_EXTRACT.search(raw_field)
        if match:
            log["field"] = match.group(1)

    return result_json

def init_gemini_client(api_key: str):
    """
    Initializes Gemini client using a key passed at runtime.
    """
    if not api_key or api_key.strip() == "":
        raise ValueError("❌ API key is missing or empty")

    return genai.Client(api_key=api_key)



client = init_gemini_client(API_KEY)




def get_schema_template():
    return """
{
  "template": "C 01.00",
  "currency": "GBP",
  "scenario_summary": "",
  "populated_cells": [
    {
      "row": "",
      "column": "",
      "item": "",
      "value": null,
      "unit": "GBP",
      "confidence": "",
      "source_chunk_ids": []
    }
  ],
  "validation_flags": [
    {
      "type": "missing_data|inconsistency|warning",
      "message": ""
    }
  ],
  "audit_log": [
    {
      "field": "",
      "value": null,
      "justification": "",
      "source_chunk_ids": []
    }
  ]
}
"""



def build_context(retrieved_chunks):
    context = ""
    for chunk in retrieved_chunks:
        context += f"[{chunk['chunk_id']} | {chunk['source']}]\n"
        context += chunk["text"] + "\n\n"
    return context




def fix_million_values(result_json):
    """
    If values look like millions (ex: 480 instead of 480000000),
    convert them into full GBP units.
    """
    for cell in result_json.get("populated_cells", []):
        val = cell.get("value")

        if isinstance(val, (int, float)):
            # If value is too small, assume it's in millions
            if val < 1000000:
                cell["value"] = int(val * 1_000_000)

    for log in result_json.get("audit_log", []):
        val = log.get("value")
        if isinstance(val, (int, float)):
            if val < 1000000:
                log["value"] = int(val * 1_000_000)

    return result_json



def call_gemini(client, prompt, model_name=MODEL_NAME):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return response.text


def parse_json_output(llm_output):
    llm_output = re.sub(r"```json|```", "", llm_output).strip()

    try:
        return json.loads(llm_output)
    except:
        print("❌ JSON Parsing Failed. Raw output below:\n")
        print(llm_output)
        return None

def generate_corep_json(client, question, scenario, retrieved_chunks):
    schema_template = get_schema_template()

    prompt = build_prompt(
        question=question,
        scenario=scenario,
        retrieved_chunks=retrieved_chunks,
        schema_template=schema_template
    )

    llm_output = call_gemini(client, prompt)

    structured_json = parse_json_output(llm_output)

    return structured_json


def get_schema_template():
    return """
{
  "template": "C 01.00",
  "currency": "GBP",
  "scenario_summary": "",
  "populated_cells": [
    {
      "row": "0010",
      "column": "0010",
      "item": "OWN FUNDS",
      "value": null,
      "unit": "GBP",
      "confidence": "",
      "source_chunk_ids": []
    },
    {
      "row": "0015",
      "column": "0010",
      "item": "TIER 1 CAPITAL",
      "value": null,
      "unit": "GBP",
      "confidence": "",
      "source_chunk_ids": []
    },
    {
      "row": "0020",
      "column": "0010",
      "item": "COMMON EQUITY TIER 1 CAPITAL",
      "value": null,
      "unit": "GBP",
      "confidence": "",
      "source_chunk_ids": []
    },
    {
      "row": "0030",
      "column": "0010",
      "item": "ADDITIONAL TIER 1 CAPITAL",
      "value": null,
      "unit": "GBP",
      "confidence": "",
      "source_chunk_ids": []
    },
    {
      "row": "0040",
      "column": "0010",
      "item": "TIER 2 CAPITAL",
      "value": null,
      "unit": "GBP",
      "confidence": "",
      "source_chunk_ids": []
    }
  ],
  "validation_flags": [],
  "audit_log": [
    {
      "field": "0010",
      "value": null,
      "justification": "",
      "source_chunk_ids": []
    },
    {
      "field": "0015",
      "value": null,
      "justification": "",
      "source_chunk_ids": []
    },
    {
      "field": "0020",
      "value": null,
      "justification": "",
      "source_chunk_ids": []
    }
  ]
}
"""


def get_row_mapping_text():
    return """
STRICT ROW MAPPING FOR TEMPLATE C 01.00:
- 0010 = Own Funds
- 0015 = Tier 1 Capital
- 0020 = Common Equity Tier 1 (CET1)
- 0030 = Additional Tier 1 (AT1)
- 0040 = Tier 2 Capital

RULE:
Tier 1 (0015) = CET1 (0020) + AT1 (0030)
Own Funds (0010) = Tier 1 (0015) + Tier 2 (0040)
"""


def build_prompt(question, scenario, retrieved_chunks, schema_template):
    context = build_context(retrieved_chunks)

    prompt = f"""
You are a PRA COREP regulatory reporting assistant.

TASK:
Populate COREP Own Funds Template C 01.00.

STRICT RULES:
- Output MUST be valid JSON only.
- DO NOT output markdown.
- DO NOT invent new row numbers.
- Use ONLY these row codes: 0010, 0015, 0020, 0030, 0040.
- Each populated cell MUST include source_chunk_ids.
- source_chunk_ids MUST ONLY come from the retrieved context below.
- If evidence is not present in context, set value=null and add a validation flag.
- confidence MUST be exactly one of: High, Medium, Low (case-sensitive).

CALCULATION RULES:
- Tier 1 Capital (Row 0015) = CET1 (Row 0020) + AT1 (Row 0030)
- Own Funds (Row 0010) = Tier 1 (Row 0015) + Tier 2 (Row 0040)

IMPORTANT NUMERIC RULES:
- Scenario values are given in MILLIONS of GBP.
- All output values MUST be reported in THOUSANDS (000 GBP).
  Example: 540 million GBP must be written as 540000 (000 GBP units).
- Unit must ALWAYS be exactly: "000 GBP"
- In justification, always show values in 000 GBP format (not full GBP, not millions).

Scenario:
{scenario}

Question:
{question}

Retrieved regulatory context:
{context}

Return JSON strictly following this schema:
{schema_template}

IMPORTANT:
Output must start with {{ and end with }} only.
"""
    return prompt




def validate_corep_output(result_json):
    flags = []

    # Convert populated_cells into dict by row for quick lookup
    row_map = {cell["row"]: cell["value"] for cell in result_json["populated_cells"]}

    # Convert values to int if they are strings
    def to_int(x):
        if x is None:
            return None
        return int(x)

    cet1 = to_int(row_map.get("0020"))   # CET1 in 000 GBP
    at1  = to_int(row_map.get("0030"))   # AT1 in 000 GBP
    tier1 = to_int(row_map.get("0015"))  # Tier1 in 000 GBP
    tier2 = to_int(row_map.get("0040"))  # Tier2 in 000 GBP
    own_funds = to_int(row_map.get("0010"))  # Own Funds in 000 GBP

    # Rule 1: Tier 1 = CET1 + AT1
    if cet1 is not None and at1 is not None and tier1 is not None:
        expected_tier1 = cet1 + at1
        if tier1 != expected_tier1:
            flags.append({
                "type": "inconsistency",
                "message": f"Tier 1 mismatch (000 GBP): expected {expected_tier1}, got {tier1}"
            })

    # Rule 2: Own Funds = Tier 1 + Tier 2
    if tier1 is not None and tier2 is not None and own_funds is not None:
        expected_own_funds = tier1 + tier2
        if own_funds != expected_own_funds:
            flags.append({
                "type": "inconsistency",
                "message": f"Own Funds mismatch (000 GBP): expected {expected_own_funds}, got {own_funds}"
            })

    # Rule 3: Missing required rows
    required_rows = ["0010", "0015", "0020", "0030", "0040"]
    for r in required_rows:
        if r not in row_map:
            flags.append({
                "type": "missing_data",
                "message": f"Missing required row {r}"
            })

    return flags




def attach_evidence(audit_log, retrieved_chunks, snippet_len=300):
    chunk_map = {c["chunk_id"]: c["text"] for c in retrieved_chunks}

    for entry in audit_log:
        evidence = []
        for cid in entry.get("source_chunk_ids", []):
            if cid in chunk_map:
                evidence.append(chunk_map[cid][:snippet_len])
        entry["evidence_snippets"] = evidence

    return audit_log



def print_audit_log(audit_log):
    for entry in audit_log:
        print("FIELD:", entry.get("field"))
        print("VALUE:", entry.get("value"))
        print("JUSTIFICATION:", entry.get("justification"))
        print("SOURCE CHUNKS:", entry.get("source_chunk_ids"))
        if "evidence_snippets" in entry:
            print("EVIDENCE:", entry["evidence_snippets"])
        print("-----")
def merge_unique_chunks(chunks1, chunks2):
    seen = set()
    merged = []

    for c in chunks1 + chunks2:
        if c["chunk_id"] not in seen:
            merged.append(c)
            seen.add(c["chunk_id"])

    return merged


def retrieve_corep_context(question, scenario, model, index, metadata, top_k=5):
    """
    Row-wise retrieval:
    Retrieves separate evidence for each COREP row.
    Ensures Tier2 + Own Funds get proper chunks.
    """

    base_query = f"""
Scenario:
{scenario}

Question:
{question}
""".strip()

    # Row-specific retrieval queries
    row_queries = {
        "0010": "COREP Template C 01.00 Row 0010 Own Funds Article 72 CRR total own funds definition",
        "0015": "COREP Template C 01.00 Row 0015 Tier 1 capital Article 25 CRR CET1 plus AT1 definition",
        "0020": "COREP Template C 01.00 Row 0020 Common Equity Tier 1 capital Article 50 CRR deductions Article 36 CRR",
        "0030": "COREP Template C 01.00 Row 0030 Additional Tier 1 capital AT1 instruments Article 52 CRR Article 53 CRR",
        "0040": "COREP Template C 01.00 Row 0040 Tier 2 capital Article 62 CRR Tier 2 instruments definition"
    }

    retrieved_all = []

    # Retrieve for user question normally
    retrieved_main = retrieve_chunks(base_query, model, index, metadata, top_k=top_k)
    retrieved_all.extend(retrieved_main)

    # Retrieve row-wise
    for row, rq in row_queries.items():
        retrieved_row = retrieve_chunks(rq, model, index, metadata, top_k=top_k)
        retrieved_all.extend(retrieved_row)

    # Remove duplicates by chunk_id
    seen = set()
    unique_chunks = []
    for c in retrieved_all:
        if c["chunk_id"] not in seen:
            unique_chunks.append(c)
            seen.add(c["chunk_id"])

    # Sort by score descending
    unique_chunks = sorted(unique_chunks, key=lambda x: x["score"], reverse=True)

    return unique_chunks[:top_k * 5]   # return bigger context
   # keep more context


def corep_assistant(client, question, scenario, model, index, metadata, top_k=8):

    retrieved_chunks = retrieve_rowwise_context(
    model=model,
    index=index,
    metadata=metadata,
    scenario=scenario,
    top_k=5
)


    result_json = generate_corep_json(
        client=client,
        question=question,
        scenario=scenario,
        retrieved_chunks=retrieved_chunks
    )

    if result_json is None:
        return {"error": "LLM output invalid JSON"}

    result_json["validation_flags"] = validate_corep_output(result_json)

    result_json = fix_row_codes(result_json)
    result_json = fix_million_values(result_json)
    result_json = validate_evidence(result_json, retrieved_chunks)
    extra_flags = validate_evidence_keywords(result_json, retrieved_chunks)
    result_json["validation_flags"].extend(extra_flags)
    result_json = fix_thousand_reporting(result_json)



    # attach evidence snippets
    if "audit_log" in result_json:
        result_json["audit_log"] = attach_evidence(result_json["audit_log"], retrieved_chunks)

    return {
        "structured_json": result_json,
        "audit_log": result_json.get("audit_log", []),
        "retrieved_chunks": retrieved_chunks
    }




def json_to_corep_table(result_json):
    """
    Converts structured JSON output into a human-readable COREP template extract.
    Returns a Pandas DataFrame.
    """
    if result_json is None:
        return None

    populated_cells = result_json.get("populated_cells", [])

    if len(populated_cells) == 0:
        print("⚠️ No populated cells found in JSON.")
        return pd.DataFrame()

    df = pd.DataFrame(populated_cells)

    # Ensure consistent ordering of columns
    expected_cols = ["row", "column", "item", "value", "unit", "confidence", "source_chunk_ids"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = df[expected_cols]

    # Convert chunk ids list into readable string
    df["source_chunk_ids"] = df["source_chunk_ids"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    return df
def print_corep_template(df):
    if df is None or df.empty:
        print("⚠️ COREP template table is empty.")
        return

    print("\n📌 COREP Template Extract (C 01.00)\n")
    display(df)

if __name__ == "__main__":
    scenario = """
    CET1 = 540 million GBP
    AT1 = 100 million GBP
    Tier 2 = 80 million GBP
    Intangible assets deduction = 40 million GBP
    Deferred tax assets deduction = 20 million GBP
    """

    question = "How should Additional Tier 1 capital and Total Own Funds be reported in COREP template C 01.00?"

    query = f"Scenario:\n{scenario}\n\nQuestion:\n{question}"

    definition_print(query, model, index, metadata, top_k=5)

    output = corep_assistant(client, question, scenario, model, index, metadata, top_k=8)

    print(output["structured_json"]["validation_flags"])
    print_audit_log(output["audit_log"])
    print(json.dumps(output["audit_log"], indent=4))

    result_json = output["structured_json"]
    corep_df = json_to_corep_table(result_json)
    print_corep_template(corep_df)
