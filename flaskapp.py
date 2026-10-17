"""
Conversational RAG chatbot with a dynamic, per-doc_source document set.

POST /chat
{
  "doc_source": "...",
  "chat_history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "query": "..."
}

Per request:
  1. get_doc_list(doc_source)            -> list of items
  2. diff against what this doc_source had active last time
       - new item, never seen before     -> get_document(item) + embed + cache
       - new item, already in the cache  -> re-activated from cache (no download)
       - item no longer in the list      -> deactivated (kept in cache, but never retrieved)
  3. retrieve only from currently-active documents
  4. rewrite the follow-up into a standalone question using chat_history, then answer
"""
import hashlib
import json
import logging
import os
import re
import threading
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from flask import Flask, jsonify, request, send_from_directory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from places import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag")

# --------------------------------------------------------------------------
# YOUR FUNCTIONS: implement these two.
# --------------------------------------------------------------------------
def get_doc_list(doc_source):
    """Return a list of items (strings or JSON-serialisable dicts) for this doc_source."""
    
    #raise NotImplementedError("implement get_doc_list(doc_source)")
    coords = doc_source.split(",")
    lat = float(coords[0])
    lon = float(coords[1])
    places = get_nearby_places(lat, lon)
    return places


def get_document(item):
    """Return the document's full text (str) for one item from get_doc_list."""
    time.sleep(1)
    place = item
    info=(get_wikipedia_info(place['name'], lat=place['lat'], lon=place['lon']))
    if not info:
        return f"No information available about {place['name']}"
        
    print(info)
    text=get_full_wikipedia_text(info["url"])
    x= text["full_text"]
    print(x)
    return x
    
    #raise NotImplementedError("implement get_document(item)")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "granite3-dense")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters
TOP_K = int(os.getenv("TOP_K", "6"))
# Minimum cosine similarity for a chunk to be shown to the LLM. If no chunk reaches it,
# the LLM is skipped. The right value depends on your data: watch the scores logged on
# each request (or in the "retrieved" field of the response) and adjust.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.5"))
NOT_FOUND_ANSWER = "I couldn't find anything about that in the documents available for your current location."
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
DOWNLOAD_DELAY = float(os.getenv("DOWNLOAD_DELAY", "1.0"))  # seconds of pause between upstream downloads
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
# Stored vectors are only valid for the embedding model and chunking that produced them,
# so each combination gets its own folder (changing a setting starts a fresh cache).
_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", EMBED_MODEL)
CACHE_PATH = os.path.join(CACHE_DIR, f"{_slug}_c{CHUNK_SIZE}_o{CHUNK_OVERLAP}")

class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text retrieves best when inputs carry its task prefixes."""

    def embed_documents(self, texts):
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(f"search_query: {text}")


embeddings = NomicEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=NUM_CTX)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# One shared store holding every document ever ingested. Each chunk carries its
# document's key in metadata, and searches filter down to the currently-active keys.
vectorstore = InMemoryVectorStore(embeddings)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def item_key(item):
    """Stable identity for an item in get_doc_list()'s result."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and "name" in item:
        # Use only the fields that uniquely identify the place.
        # Round lat/lon so tiny floating-point differences don't create new keys.
        payload = {"name": item["name"]}
        if "lat" in item and "lon" in item:
            payload["lat"] = round(float(item["lat"]), 6)
            payload["lon"] = round(float(item["lon"]), 6)
        # Optional: also include a stable ID if your places API provides one
        # if "id" in item or "osm_id" in item:
        #     payload["id"] = item.get("id") or item.get("osm_id")
        return json.dumps(payload, sort_keys=True)
    return json.dumps(item, sort_keys=True, default=str)

def _label(item, width=40):
    """Short human-readable name for an item, used in the progress bar."""
    text = item.get("name") if isinstance(item, dict) and item.get("name") else item_key(item)
    return str(text)[:width]


def to_lc(messages):
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in messages
    ]


# --------------------------------------------------------------------------
# Global document cache: every document ever downloaded, keyed by item key.
# Never evicted by list changes, so a document that disappears and later
# returns is not downloaded or embedded again.
# --------------------------------------------------------------------------
def _digest(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _atomic_write(path, write_fn):
    """Write via a temp file + rename so a crash never leaves a half-written file."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class DocCache:
    """
    Downloaded + embedded documents, persisted to disk so they survive restarts.

    One pair of files per document in CACHE_PATH:
      <sha256(key)>.npy   float32 embeddings, one row per chunk (written first)
      <sha256(key)>.json  {"key": ..., "chunks": [...]}  (written last: its presence
                          marks the document as completely cached)
    """

    def __init__(self):
        self._keys = set()  # keys already downloaded and embedded into `vectorstore`
        self._locks = {}    # key -> Lock (prevents duplicate downloads across sources)
        self._guard = threading.Lock()
        self._download_lock = threading.Lock()  # one upstream request at a time, across all requests
        self._last_download = 0.0
        os.makedirs(CACHE_PATH, exist_ok=True)
        self._load()

    def has(self, key):
        return key in self._keys

    def _load(self):
        """Refill the vector store and key set from disk without re-downloading or re-embedding."""
        loaded = skipped = 0
        for name in sorted(os.listdir(CACHE_PATH)):
            if not name.endswith(".json"):
                continue
            stem = name[:-len(".json")]
            base = os.path.join(CACHE_PATH, stem)
            try:
                with open(base + ".json", encoding="utf-8") as f:
                    meta = json.load(f)
                key, chunks = meta["key"], meta["chunks"]
                if chunks:
                    vectors = np.load(base + ".npy")
                    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
                        raise ValueError("embedding rows don't match chunk count")
                    for i, (text, vec) in enumerate(zip(chunks, vectors)):
                        doc_id = f"{stem}:{i}"
                        vectorstore.store[doc_id] = {
                            "id": doc_id,
                            "vector": vec.tolist(),
                            "text": text,
                            "metadata": {"doc_key": key},
                        }
                self._keys.add(key)
                loaded += 1
            except Exception as e:  # unreadable entry: skip it, it will be re-fetched on demand
                log.warning("skipping unreadable cache entry %s: %s", name, e)
                skipped += 1
        log.info("disk cache %s: loaded %d documents, skipped %d", CACHE_PATH, loaded, skipped)

    def _save(self, key, chunks, ids):
        base = os.path.join(CACHE_PATH, _digest(key))
        if chunks:
            vectors = np.asarray([vectorstore.store[i]["vector"] for i in ids], dtype=np.float32)
            _atomic_write(base + ".npy", lambda f: np.save(f, vectors))
        _atomic_write(base + ".json", lambda f: f.write(json.dumps({"key": key, "chunks": chunks}).encode("utf-8")))

    def ensure(self, key, item):
        """Make sure the doc is cached. Returns True if it had to be downloaded."""
        if key in self._keys:
            return False
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if key in self._keys:
                return False
            with self._download_lock:
                wait = DOWNLOAD_DELAY - (time.monotonic() - self._last_download)
                if wait > 0:
                    time.sleep(wait)
                try:
                    text = get_document(item)
                finally:
                    self._last_download = time.monotonic()
            if not isinstance(text, str):
                raise TypeError(f"get_document must return str, got {type(text).__name__}")
            chunks = splitter.split_text(text)
            ids = [f"{_digest(key)}:{i}" for i in range(len(chunks))]
            if chunks:
                vectorstore.add_documents(
                    [Document(page_content=c, metadata={"doc_key": key}) for c in chunks], ids=ids
                )
            try:
                self._save(key, chunks, ids)
            except Exception:  # a disk problem shouldn't fail the request; the doc stays in memory
                log.exception("couldn't write %s to the disk cache", key)
            self._keys.add(key)
            return True


cache = DocCache()


# --------------------------------------------------------------------------
# Per-doc_source state: which cached docs are currently "active" (searchable)
# --------------------------------------------------------------------------
class SourceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = []      # ordered list of keys currently searchable

    def sync(self, doc_source):
        items = get_doc_list(doc_source)
        wanted = {}
        for item in items:
            wanted.setdefault(item_key(item), item)  # dedupe, keep order

        previously_active = set(self.active)
        removed = [k for k in self.active if k not in wanted]
        added = [k for k in wanted if k not in previously_active]
        to_fetch = [k for k in wanted if not cache.has(k)]

        downloaded, failed = [], []
        if to_fetch:
            desc = f"Ingesting {len(to_fetch)} new of {len(wanted)} docs"
            with tqdm(to_fetch, desc=desc, unit="doc") as bar, logging_redirect_tqdm():
                for k in bar:  # one at a time; DocCache.ensure paces the upstream requests
                    bar.set_postfix_str(_label(wanted[k]), refresh=True)  # shows what's downloading now
                    try:
                        cache.ensure(k, wanted[k])
                        downloaded.append(k)
                    except Exception as e:  # skip bad docs, retry next request
                        log.error("failed to ingest %s: %s", k, e)
                        failed.append({"item": k, "error": str(e)})

        new_active = [k for k in wanted if cache.has(k)]
        self.active = new_active

        return {
            "active_count": len(self.active),
            "added": added,
            "removed": removed,
            "downloaded": downloaded,
            "reactivated_from_cache": [k for k in added if k not in downloaded and k not in {f["item"] for f in failed}],
            "failed": failed,
        }

    def retrieve(self, query_text, k=TOP_K):
        active = set(self.active)
        if not active:
            return []
        hits = vectorstore.similarity_search_with_score(
            query_text, k=k, filter=lambda d: d.metadata["doc_key"] in active
        )
        return [{"source": d.metadata["doc_key"], "text": d.page_content, "score": float(s)} for d, s in hits]


_states = {}
_states_guard = threading.Lock()


def state_for(doc_source):
    with _states_guard:
        return _states.setdefault(doc_source, SourceState())


# --------------------------------------------------------------------------
# Conversation handling
# --------------------------------------------------------------------------
def clean_history(history, query):
    """Validate history, append the new query, merge same-role neighbours, start with 'user'."""
    msgs = []
    for m in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip():
            msgs.append({"role": m["role"], "content": m["content"].strip()})
    prior = list(msgs)
    msgs.append({"role": "user", "content": query})

    merged = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return prior, merged


def text_of(resp):
    content = resp.content
    if isinstance(content, list):
        content = "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return content.strip()


def standalone_query(prior, query):
    """Rewrite a follow-up ('what about its price?') into a self-contained search query."""
    if not prior:
        return query
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in prior[-6:])
    resp = llm.invoke([
        SystemMessage(content="Rewrite the user's latest message as a single standalone question that can be "
                              "understood without the conversation. Resolve pronouns and references. "
                              "Output only the question."),
        HumanMessage(content=f"Conversation:\n{transcript}\n\nLatest message: {query}"),
    ])
    return text_of(resp).strip('"') or query


SYSTEM_PROMPT = """You are a helpful assistant answering questions using the provided documents.

Rules:
- Base factual answers on the documents below. If they don't contain the answer, say so plainly instead of guessing.
- Use the conversation so far to interpret follow-up questions.
- When useful, mention which document a fact came from.

<documents>
{context}
</documents>"""


def build_context(chunks):
    if not chunks:
        return "(no documents available)"
    return "\n".join(f'<document source="{c["source"]}">\n{c["text"]}\n</document>' for c in chunks)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
app = Flask(__name__)


@app.get("/chatui")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.get("/map")
def localmap():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "map2.html")


@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    print(body)
    doc_source = body.get("doc_source")
    query = (body.get("query") or "").strip()
    if not doc_source or not query:
        return jsonify(error="doc_source and query are required"), 400

    prior, messages = clean_history(body.get("chat_history"), query)

    state = state_for(doc_source)
    try:
        with state.lock:  # serialize sync+retrieve per doc_source
            sync_info = state.sync(doc_source)
            search_q = standalone_query(prior, query)
            chunks = state.retrieve(search_q)
    except Exception as e:
        log.exception("sync/retrieve failed")
        return jsonify(error=f"document sync failed: {e}"), 502

    # Everything retrieved is reported; only chunks at or above MIN_SCORE reach the LLM.
    retrieved = [
        {"source": c["source"], "score": round(c["score"], 4), "used": c["score"] >= MIN_SCORE, "preview": c["text"][:300]}
        for c in chunks
    ]
    chunks = [c for c in chunks if c["score"] >= MIN_SCORE]
    log.info("retrieval for %r: scores=%s, min=%.2f, used=%d",
             search_q, [r["score"] for r in retrieved], MIN_SCORE, len(chunks))

    if not chunks:  # nothing relevant: don't let the model answer from its own memory
        return jsonify(
            answer=NOT_FOUND_ANSWER,
            standalone_query=search_q,
            sources=[],
            retrieved=retrieved,
            sync=sync_info,
        )

    try:
        resp = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT.format(context=build_context(chunks)))] + to_lc(messages)
        )
    except Exception as e:
        log.exception("LLM call failed")
        return jsonify(error=f"LLM call failed: {e}"), 502
        
    print(text_of(resp))

    return jsonify(
        answer=text_of(resp),
        standalone_query=search_q,
        sources=sorted({c["source"] for c in chunks}),
        retrieved=retrieved,
        sync=sync_info,
    )


@app.get("/health")
def health():
    return jsonify(status="ok", cached_documents=len(cache._keys))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True)