import os
import re
import json
import streamlit as st
from google import genai

# Optional: only needed if you want PDF support
# pip install pypdf
try:
    from pypdf import PdfReader
    HAS_PDF = True
except Exception:
    HAS_PDF = False


# ------------------------------------------------------------
# 1) SOAP hints
# ------------------------------------------------------------
def extract_soap_hints(text: str) -> dict:
    hints = {
        "has_wsdl": False,
        "has_soap_envelope": False,
        "possible_operations": [],
        "namespaces": [],
    }

    if not text:
        return hints

    t = text.lower()
    hints["has_wsdl"] = "wsdl" in t or "definitions" in t
    hints["has_soap_envelope"] = "soap:envelope" in t or "<envelope" in t

    ns = re.findall(r'xmlns:([a-zA-Z0-9_]+)=["\']([^"\']+)["\']', text)
    hints["namespaces"] = [f"{pfx}={uri}" for pfx, uri in ns][:20]

    ops = re.findall(r'operation\s+name=["\']([^"\']+)["\']', text)
    hints["possible_operations"] = list(dict.fromkeys(ops))[:20]

    return hints


# ------------------------------------------------------------
# 2) Read uploaded FedEx docs/specs into text (NO SCRAPING)
# ------------------------------------------------------------
def _read_pdf_bytes_to_text(file_bytes: bytes) -> str:
    if not HAS_PDF:
        return "[PDF upload received, but pypdf is not installed. Install pypdf to parse PDFs.]"
    try:
        reader = PdfReader(file_bytes)
        out = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if txt.strip():
                out.append(f"\n\n--- PDF PAGE {i+1} ---\n{txt}")
        return "\n".join(out).strip()
    except Exception as e:
        return f"[Failed to parse PDF: {e}]"


def read_uploaded_file_to_text(uploaded) -> str:
    """
    Supports:
      - .json (OpenAPI, sample payloads, etc.)
      - .yaml/.yml (OpenAPI)
      - .txt/.md (notes)
      - .pdf (optional, if pypdf installed)
    """
    name = uploaded.name.lower()
    raw = uploaded.getvalue()

    # PDF
    if name.endswith(".pdf"):
        return _read_pdf_bytes_to_text(raw)

    # Text-ish
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return f"[Could not decode {uploaded.name} as text.]"

    # Pretty-print JSON for readability if it is JSON
    if name.endswith(".json"):
        try:
            obj = json.loads(text)
            return json.dumps(obj, indent=2)
        except Exception:
            return text

    return text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def score_chunk(chunk: str, query: str) -> float:
    """
    Cheap relevance scoring: keyword overlap + boosts for FedEx terms
    """
    q = query.lower()
    c = chunk.lower()
    score = 0.0
    for term in set(re.findall(r"[a-zA-Z0-9_/-]{3,}", q)):
        if term in c:
            score += 1.0

    boosts = [
        "oauth", "client_credentials", "bearer", "token", "authorization",
        "ship", "shipment", "label", "tracking", "accountnumber",
        "meter", "serviceType", "packagingType", "requestedShipment",
        "/ship/v1", "/shipments", "rate limit", "429"
    ]
    for b in boosts:
        if b in c and b in q:
            score += 3.0
        elif b in c:
            score += 0.5

    return score


def build_fedex_context_from_uploads(uploads, user_input: str, top_k: int = 6) -> str:
    """
    Creates a small “context pack” by:
      - reading all uploaded docs/specs into text
      - chunking
      - ranking chunks vs user_input (SOAP text)
      - returning top_k chunks with file source labels
    """
    if not uploads:
        return ""

    all_chunks = []
    for up in uploads:
        text = read_uploaded_file_to_text(up)
        if not text.strip():
            continue
        for ch in chunk_text(text):
            all_chunks.append((up.name, ch))

    ranked = sorted(all_chunks, key=lambda x: score_chunk(x[1], user_input), reverse=True)
    picked = ranked[:top_k]

    out = []
    for fname, ch in picked:
        out.append(f"[SOURCE FILE: {fname}]\n{ch}")
    return "\n\n---\n\n".join(out)


# ------------------------------------------------------------
# 3) Gemini helper
# ------------------------------------------------------------
def gemini_generate(client, model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=[
            {
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\nUSER_INPUT:\n{user_prompt}"}],
            }
        ],
    )
    return response.text or ""


# ------------------------------------------------------------
# 4) Prompts
# ------------------------------------------------------------
DESIGN_SYSTEM_PROMPT = """
You are a migration assistant converting SOAP services to REST APIs.

Rules:
- First produce an OpenAPI 3.0 specification (YAML).
- Then describe REST endpoints, schemas, and errors.
- Provide a SOAP → REST mapping table.
- Map SOAP Faults to HTTP status codes with a JSON error body.
- If something is missing, list assumptions explicitly.
- Prefer correctness and clarity over verbosity.

Output sections (exact order):
1) Assumptions
2) REST Endpoints
3) JSON Schemas
4) Error Model
5) SOAP → REST Mapping Table
6) OpenAPI 3.0 YAML
"""

CODE_SYSTEM_PROMPT = """
You are a code generator.

Given an OpenAPI 3.0 YAML and migration notes:
- Generate ONLY a client (NO server code)
- Include TODOs for business logic
- Include example requests

Rules:
- Code must match the OpenAPI exactly
- No secrets or API keys in code
- Keep code minimal but runnable

Output sections:
1) Client Skeleton
2) Example Requests
3) Validation Notes & Tests
"""


def build_design_prompt(target_stack, soap_text, hints, rest_prefs, fedex_context):
    return f"""
Target stack:
{target_stack}

REST preferences:
{rest_prefs}

AUTHORITATIVE FEDEX REFERENCE (uploaded files; treat as source of truth):
{fedex_context}

SOAP / WSDL / XML / Code:
{soap_text}

Extracted hints:
- has_wsdl: {hints['has_wsdl']}
- has_soap_envelope: {hints['has_soap_envelope']}
- namespaces: {hints['namespaces']}
- possible_operations: {hints['possible_operations']}

Task:
Design a REST API equivalent and produce OpenAPI 3.0 YAML.
"""


def build_code_prompt(target_stack, openapi_yaml, design_output, fedex_context):
    return f"""
Target stack:
{target_stack}

AUTHORITATIVE FEDEX REFERENCE (uploaded files; treat as source of truth):
{fedex_context}

OpenAPI 3.0 YAML:
{openapi_yaml}

Design notes:
{design_output}

Generate ONLY the client code (no server).
"""


# ------------------------------------------------------------
# 5) Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="SOAP → REST Converter (Gemini)", layout="wide")
st.title("🧼➡️🌐 SOAP → REST Converter Bot")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment variables.")
    st.stop()

client = genai.Client(api_key=api_key)

with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "Gemini model",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        index=0,
    )

    target_stack = st.selectbox(
        "Target stack",
        [
            "Python requests client",
            "Node.js Axios client",
            "Java HTTP client",
            ".NET C# HttpClient",
        ],
    )

    rest_prefs = st.text_area(
        "REST preferences (optional)",
        placeholder="e.g. /v1 prefix, pagination, idempotency headers, retries",
        height=100,
    )

    st.subheader("FedEx reference (upload)")
    fedex_uploads = st.file_uploader(
        "Upload FedEx docs/specs (OpenAPI JSON/YAML, PDFs, txt)",
        type=["json", "yaml", "yml", "txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Download from the portal, then upload here. No scraping/login needed.",
    )

    if fedex_uploads and any(up.name.lower().endswith(".pdf") for up in fedex_uploads) and not HAS_PDF:
        st.info("PDF upload detected. To parse PDFs, install: pip install pypdf")


st.subheader("1) Paste SOAP artifacts")
soap_text = st.text_area(
    "SOAP XML / WSDL / SOAP client or server code",
    height=280,
    placeholder="Paste WSDL, SOAP request/response, or SOAP code here",
)

st.divider()

if st.button("🚀 Convert SOAP → REST", use_container_width=True):
    if not soap_text.strip():
        st.warning("Please paste some SOAP/WSDL content first.")
        st.stop()

    with st.spinner("Analyzing SOAP…"):
        hints = extract_soap_hints(soap_text)

    with st.spinner("Building FedEx context from uploaded docs…"):
        fedex_context = build_fedex_context_from_uploads(fedex_uploads, soap_text) if fedex_uploads else ""

    with st.spinner("Designing REST API + OpenAPI…"):
        design_prompt = build_design_prompt(target_stack, soap_text, hints, rest_prefs, fedex_context)
        design_output = gemini_generate(client, model, DESIGN_SYSTEM_PROMPT, design_prompt)

    # extract OpenAPI section (best-effort)
    openapi_yaml = design_output
    match = re.search(r"6\)\s*OpenAPI 3\.0 YAML(.*)$", design_output, re.DOTALL | re.IGNORECASE)
    if match:
        openapi_yaml = match.group(1).strip()

    with st.spinner("Generating client code…"):
        code_prompt = build_code_prompt(target_stack, openapi_yaml, design_output, fedex_context)
        code_output = gemini_generate(client, model, CODE_SYSTEM_PROMPT, code_prompt)

    st.success("Conversion complete!")

    tab1, tab2, tab3, tab4 = st.tabs(["📐 Design + OpenAPI", "💻 Client Code", "🧠 Debug", "FedEx Context"])

    with tab1:
        st.text_area("Design output", design_output, height=500)
        st.code(openapi_yaml, language="yaml")

    with tab2:
        st.text_area("Generated client code", code_output, height=600)

    with tab3:
        st.json(hints)

    with tab4:
        if fedex_context:
            st.text_area("FedEx context pack used", fedex_context, height=600)
        else:
            st.info("No FedEx files uploaded — upload docs/specs to ground the generation.")
