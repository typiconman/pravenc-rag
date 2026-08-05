# pravenc-rag

A RAG pipeline over [pravenc-md](https://github.com/slavonic/pravenc-md)
(Markdown of the Russian-language Orthodox Encyclopedia): an offline indexing
stage that builds a hybrid Qdrant collection, and an online query stage that
answers cross-lingual questions with verified citations. Metadata stays
Russian-canonical throughout.

**Indexing** (offline): parse (`ingest.py`) → chunk (`chunk.py`) → embed with
BGE-M3 (`embed.py`) → write to Qdrant (`store.py`), all orchestrated by the
`pravenc-index build` / `pravenc-index update` commands.

**Query** (online): embed question → Qdrant hybrid RRF → BGE rerank → hydrate
parent sections → a hosted LLM via OpenRouter (or local Ollama) → verified
citations, via `pravenc-ask`.

## Prerequisites

- Python ≥ 3.11
- [Docker](https://docs.docker.com/get-docker/) (to run Qdrant locally)
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- An [OpenRouter](https://openrouter.ai/) API key for the query stage's LLM
  (default provider), or [Ollama](https://ollama.com/) if you'd rather run the
  LLM locally — see `llm.provider` in `config.yaml`
- ~2 GB free disk space for the BGE-M3 weights, plus a few GB more for the
  reranker, plus space for the corpus (and for an Ollama model, if used locally)
- A CUDA GPU is optional but strongly recommended for indexing the full corpus
  and for reranking at query time

## Layout

```
pravenc-rag/
├── config.yaml            # paths, model, chunking, section-type map
├── docker-compose.yml     # local Qdrant
├── pyproject.toml
├── data/pravenc-md/       # the corpus (submodule, git-ignored contents)
└── scripts/
    ├── models.py          # Document / Section / Chunk / Reference / Media
    ├── ingest.py          # one .md → Document (frontmatter, sections, refs, media)
    ├── chunk.py           # Document → child chunks + citation payload
    ├── embed.py           # BGE-M3 dense + sparse
    ├── store.py           # Qdrant hybrid collection + upsert + delete-by-doc
    ├── index.py           # `pravenc-index` CLI (build, update)
    ├── audit.py           # `pravenc-audit` CLI (survey keys + headings)
    ├── hydrate.py         # query-stage: rebuild parent section text from source
    ├── retrieve.py        # query-stage: hybrid search + rerank -> LlamaIndex retriever
    ├── generate.py        # query-stage: hosted/local LLM + CitationQueryEngine + verification
    ├── app.py             # Gradio UI, served by `pravenc-ask ui`
    └── ask.py             # `pravenc-ask` CLI (q, ui, check)
```

## Setup

### 1. Fetch the corpus

The corpus is a **shallow, sparse submodule** — the pinned commit records
exactly which corpus state each index was built from (citation provenance),
while depth-1 + sparse-checkout keeps only the current article text on disk (no
history, no non-article dirs).

```bash
git submodule add --depth 1 https://github.com/slavonic/pravenc-md data/pravenc-md
git -C data/pravenc-md config core.sparseCheckout true
git -C data/pravenc-md sparse-checkout set articles
```

(If you're cloning this repo fresh and the submodule is already registered,
run `git submodule update --init --depth 1` instead.)

### 2. Start Qdrant

```bash
docker compose up -d
```

If docker is not installed, you can launch Qdrant directly, e.g.:

```bash
cd ~ && mkdir -p qdrant && cd qdrant
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz \
  | tar xz
./qdrant &        # serves on localhost:6333, storage in ./storage
```

### 3. Install the project

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

This registers the `pravenc-index`, `pravenc-audit`, and `pravenc-ask` CLI
commands (from `scripts/index.py`, `scripts/audit.py`, and `scripts/ask.py`)
in your virtualenv. `pip install -e .` works too if you don't have `uv`.

`FlagEmbedding` pulls in PyTorch; a CUDA GPU makes indexing the full corpus far
faster, but CPU works fine for smoke tests. The first run downloads the BGE-M3
weights (~2 GB) from Hugging Face, so make sure you have network access and
disk space available.

## Build the index

```bash
# 1. Survey the corpus — lists frontmatter keys and section headings
pravenc-audit run

# 2. Extend section_types in config.yaml using that output, then smoke-test
#    on a handful of files before committing to a full run
pravenc-index build --limit 50

# 3. Index everything (add --recreate to drop and rebuild the collection)
pravenc-index build
pravenc-index build --recreate
```

## Updating the index

Because you maintain the corpus, the diff is computed **upstream**, at release
time, so this side never needs corpus history.

1. In the pravenc-md repo, install `corpus-release/emit-manifest.sh` (or the
   `github-workflow.yml` as `.github/workflows/emit-manifest.yml`). On each
   tagged version it writes `updates/<tag>.txt` = `git diff --name-status`
   over `articles/` since the previous tag.
2. Bump the submodule and apply the manifest:

```bash
git submodule update --remote --depth 1 data/pravenc-md
pravenc-index update data/pravenc-md/updates/v2.4.txt   # or a downloaded manifest
git add data/pravenc-md && git commit -m "reindex against corpus v2.4"
```

`update` deletes every point of each changed/removed/renamed article first (so
shrunk or deleted sections leave nothing stale), then re-embeds and upserts the
added/modified files. Committing the moved submodule pointer dates the reindex
in your history.

## Moving the index to another machine

Embedding the full corpus is the slowest step (BGE-M3 over every article), so
if you build the index on one machine and want to query from
another, move the built Qdrant
**collection snapshot** instead of re-running `pravenc-index build`.

On the machine that built the index:

```bash
# Create a snapshot (Qdrant writes it server-side under its storage dir)
curl -X POST http://localhost:6333/collections/pravenc/snapshots

# List snapshots to get the exact filename, then download it
curl http://localhost:6333/collections/pravenc/snapshots
curl -o pravenc.snapshot \
  http://localhost:6333/collections/pravenc/snapshots/<snapshot-name>
```

Copy `pravenc.snapshot` to the target machine (`scp`, rsync, cloud storage —
whatever fits), then, with Qdrant running there (`docker compose up -d`):

```bash
curl -X PUT http://localhost:6333/collections/pravenc/snapshots/upload \
  -F "snapshot=@pravenc.snapshot"
```

This recreates the `pravenc` collection from the snapshot, vectors and payload
included — no re-embedding needed. Point `qdrant_url` in `config.yaml` at
wherever Qdrant is running (a rented GPU instance's exposed port, etc.) and
`pravenc-ask` will query it directly; the query stage only needs the embedding
model and reranker locally, plus network access to an LLM. With the default
`llm.provider: openrouter` that's just an `OPENROUTER_API_KEY` export — no
local model to move. If you use `llm.provider: ollama` instead, run Ollama on
the remote instance and set `llm.ollama_url` accordingly.

Within this repository, a zipped snapshot sits in Releases (as long as it is
under the 2G limit imposed by GitHub).

## Troubleshooting

- **`pravenc-index` / `pravenc-audit` / `pravenc-ask` command not found** — make
  sure you've activated the virtualenv (`source .venv/bin/activate`) and ran
  `uv pip install -e .` in step 3 of Setup.
- **Qdrant connection errors** — confirm the container is up with
  `docker compose ps`, and that `qdrant_url` in `config.yaml` matches the port
  in `docker-compose.yml` (default `http://localhost:6333`).
- **Empty audit output / `corpus_dir` not found** — check that the submodule
  was checked out (`ls data/pravenc-md/articles`) and that `corpus_dir` in
  `config.yaml` points at it.
- **`pravenc-ask check` reports `LLM: <VAR> is not set`** — export your
  provider's API key under the name `llm.api_key_env` in `config.yaml` points
  at (default `OPENROUTER_API_KEY`).
- **`pravenc-ask check` / `pravenc-ask models` reports an LLM catalog error at
  `api_base`** — confirm the key is valid and `llm.api_base` in `config.yaml`
  is reachable (default `https://openrouter.ai/api/v1`); model slugs on
  OpenRouter churn, so re-run `pravenc-ask models` if `llm.model` or an entry
  in `llm.models` comes back "not in catalog."
- **`pravenc-ask check` reports Ollama FAILED** (only relevant with
  `llm.provider: ollama`) — confirm the Ollama daemon is running
  (`ollama list`) and that `llm.ollama_url` in `config.yaml` matches where it's
  listening (default `http://localhost:11434`); pull the configured model with
  `ollama pull <model>` if it's missing.
- **`ValueError: ... require users to upgrade torch to at least v2.6`** — this
  repo pins `torch>=2.6` in `pyproject.toml`; if you still hit it, your
  environment has a stale `torch` installed outside of `pip install -e .`
  (common on prebuilt ML container images) — reinstall with `uv pip install -e
  .` (or `pip install -e .`) to pick up the pin.
- **`RuntimeError: operator torchvision::nms does not exist`** after upgrading
  torch — a leftover `torchvision`, built against the old `torch`, is now ABI-
  mismatched. This repo never uses `torchvision` (only `transformers`
  opportunistically imports it if present); the fix is `pip uninstall -y
  torchvision`, not trying to match versions.

## Notes

- The filename stem is the article id and equals the number in `source_url`; it
  is the citation key and the cross-reference target key.
- **Parent text is not stored in Qdrant.** Payloads hold vectors, citation
  metadata, and the child text only; `hydrate.ParentHydrator` rebuilds the full
  section text from the source file at query time (cached per document). This
  avoids duplicating section text across every chunk — the dominant index-size
  cost on a corpus this large. It assumes the on-disk article matches what was
  indexed, which the update workflow keeps true.
- Reference sections (`Источники`, `Литература`) are tagged, not dropped — the
  query side can filter them in or out via the `section_type` payload index.
- Resolved cross-references (`refs_resolved`) form a see-also graph by article
  id; unresolved ones are kept separately for later title-based resolution.
- `/char/` glyph images become a `⟨glyph⟩` placeholder in the text so the gap is
  visible rather than a silent join, with the source URL kept in `media`.
- Re-indexing is idempotent: chunk ids map to stable Qdrant point UUIDs, so a
  rebuild overwrites points instead of duplicating them.

## Query stage

Generation runs against a **hosted LLM** by default (`llm.provider:
openrouter` in `config.yaml`) rather than a local model — export an
[OpenRouter](https://openrouter.ai/) API key under the variable named in
`llm.api_key_env` (default `OPENROUTER_API_KEY`):

```bash
export OPENROUTER_API_KEY=sk-or-...
```

`llm.model` is the active model (a slug like `qwen/qwen3.6-27b`); `llm.models`
is a candidate list used by the UI's model picker and by `pravenc-ask
compare`. OpenRouter's catalog changes weekly, so treat the defaults in
`config.yaml` as a starting point — `pravenc-ask models` lists what's actually
live for your key.

If you'd rather run the LLM locally, set `llm.provider: ollama` and pull a
model first:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:12b        # gemma3:4b (3.3 GB) is far faster on CPU
```

Check everything is wired up, then ask:

```bash
pravenc-ask check                        # Qdrant points, LLM provider, corpus
pravenc-ask models --filter qwen         # live model catalog for your provider
pravenc-ask q "Кто такой Алексий, человек Божий?"
pravenc-ask q "Who was Alexius, Man of God?" --language en --model google/gemma-4-31b-it
pravenc-ask compare "Заменил ли канон кондак?"   # same question across llm.models
pravenc-ask ui                           # http://localhost:7860
```

### How the query path works

1. **Encode** the question with BGE-M3 (dense + sparse). Cross-lingual, so an
   English question retrieves Russian passages.
2. **Hybrid search** in Qdrant: dense and sparse prefetch, fused server-side
   with RRF, filtered to exclude bibliography sections by default.
3. **Rerank** the top candidates with BGE-reranker-v2-m3.
4. **Hydrate** each surviving chunk to its full parent section from the corpus
   (nothing but child text lives in the index), deduplicating chunks that share
   a section.
5. **Generate** with the configured LLM (hosted via OpenRouter by default, or
   local Ollama) through `CitationQueryEngine`, which numbers the sources and
   asks for `[N]` markers.
6. **Verify** every bracketed marker the model emits against the sources
   actually retrieved. Anything that isn't a valid in-range `[N]` — an
   out-of-range number, `[New Source]`, `[Author, pages]` — is stripped and
   reported; an answer with no valid citations is flagged.

### Notes

- **Answer language** is auto-detected from the question and can be forced.
  Article titles stay in Cyrillic inside citations even in English answers —
  the Russian title is the citation anchor and must stay verifiable against the
  printed edition.
- **Model choice matters for citation discipline.** Smaller/looser models
  fabricate citation-shaped brackets (author names, page ranges, multi-number
  markers) more often than they invent plain out-of-range `[N]`s; `pravenc-ask
  compare` and the `suspect` flag (3+ fabricated markers, or none at all) help
  spot this per model.
- **`num_ctx` matters for Ollama.** Its default context is much smaller than
  most models' nominal window and will silently truncate retrieved sections,
  which looks like the model ignoring its sources. `llm.num_ctx` is passed
  through explicitly; for the OpenRouter path it's used as a context-window
  hint rather than an enforced truncation point.
- **Reranking is the CPU bottleneck** — one cross-encoder pass per candidate.
  Lower `retrieval.rerank_candidates`, or set `use_reranker: false`, if queries
  feel slow. Both are also toggles in the UI. `pravenc-ask compare` re-runs
  retrieval (and reranking) once per model, so keep the candidate list short or
  turn the reranker off while comparing.
- The index stores `doc_id` + `section_idx`, not section text, so **the corpus
  checkout must match the commit the index was built from** or hydration
  misaligns. Run `pravenc-index update` whenever you bump the submodule.