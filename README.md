# pravenc-rag — indexing stage

Offline pipeline that turns the [pravenc-md](https://github.com/slavonic/pravenc-md)
corpus (Markdown of the Russian-language Orthodox Encyclopedia) into a hybrid Qdrant
collection ready for cross-lingual retrieval. Metadata stays Russian-canonical;
the query/generation stage lives in a separate step.

Flow: `parse (ingest.py) → chunk (chunk.py) → embed BGE-M3 (embed.py) → Qdrant (store.py)`,
orchestrated by `pravenc-index build` / `pravenc-index update`.

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
    ├── hydrate.py         # query-stage: rebuild parent section text from source
    └── audit.py           # `pravenc-audit` CLI (survey keys + headings)
```

## Setup

The corpus is a **shallow, sparse submodule** — the pinned commit records
exactly which corpus state each index was built from (citation provenance),
while depth-1 + sparse-checkout keeps only the current article text on disk (no
history, no non-article dirs).

```bash
# 1. Corpus as a shallow, sparse submodule
git submodule add --depth 1 https://github.com/slavonic/pravenc-md data/pravenc-md
git -C data/pravenc-md config core.sparseCheckout true
git -C data/pravenc-md sparse-checkout set articles

# 2. Qdrant
docker compose up -d

# 3. Install (uv recommended; pip works too)
uv venv && source .venv/bin/activate
uv pip install -e .
```

`FlagEmbedding` pulls in PyTorch; a CUDA GPU makes indexing the full corpus far
faster, but CPU works for smoke tests. First run downloads the BGE-M3 weights
(~2 GB) from Hugging Face.

## Build

```bash
# Survey the corpus first — extend section_types in config.yaml from this output
pravenc-audit run

# Smoke-test on a handful of files, then index everything
pravenc-index build --limit 50
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
