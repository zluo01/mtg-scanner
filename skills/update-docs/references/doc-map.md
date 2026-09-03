# Which document covers what

| Change | Document | Section |
|--------|----------|---------|
| How to install, run, configure, back up | `README.md` | Quick start, Configuration, Keeping it running |
| Architecture, pipeline, key decisions, repo structure | `docs/overview.md` | Approach, Key Technical Decisions, Project Structure |
| Scryfall pipeline, bulk export format, image download | `docs/phase1-embedding-index.md` | Components §1, Validation |
| Embedding index build, refresh, metadata columns | `docs/phase1-embedding-index.md` | Components §3, Validation (refresh entries) |
| Card detector training and synthetic data | `docs/phase2-card-detection.md` | Components, Experiments |
| Detection + rectification + identification end to end | `docs/phase3-hybrid-pipeline.md` | |
| Encoder comparisons, fine-tuning attempts, OCR removal | `docs/phase4-refinements.md` | Experiments |
| Web app: screens, sheets, filters, search, theme | `docs/phase5-interface.md` | Components §1 (Frontend), design notes |
| Server: routes, stores, SQL, library rule, imports | `docs/phase5-interface.md` | Components §2 (Server), Configuration (REST API, Scan decision) |
| Shared API types | `docs/phase5-interface.md` | Components §3; the source of truth is `app/shared/api.ts` |
| Container, Makefile, workspace layout | `docs/phase5-interface.md` | Configuration (Container), Checklist (Packaging) |
| Measurements: latency, memory, parity, round trips | `docs/phase5-interface.md` | Experiments |
| Verification method for UI work | `docs/phase5-interface.md` | Experiment 4, design notes |
| Cloud deployment plan and stack | `docs/phase6-deployment.md` | Whole document (currently historical; rewrite when phase 6 starts) |
| Training-side scripts and their defaults | `docs/phase1-embedding-index.md`, `docs/phase2-card-detection.md`, `README.md` (Building the index and models) | |

Test counts quoted in `docs/phase5-interface.md` (Validation and the
Server checklist) change whenever tests are added; update both places.
