---
name: update-docs
description: Keep the design record in docs/ in step with a code change; which document to touch, and how experiments and superseded work are written up
---

# Update the documentation

Every change that alters behaviour, an interface, a number the docs quote,
or a decision, is documented in the same piece of work. The docs are the
project's memory; a reader of a phase document should be able to tell what
exists today and why.

## Steps

1. Find the documents the change touches with `references/doc-map.md`.
   Most application changes land in `docs/phase5-interface.md`; anything
   about the index or Scryfall pipeline in `docs/phase1-embedding-index.md`;
   architecture-level statements in `docs/overview.md`; setup and usage in
   `README.md`.
2. Correct every sentence that is no longer true. Search the docs for the
   old behaviour's key words (a component name, an endpoint, a count) so
   nothing stale survives.
3. If the work involved a measurement or a comparison, add an experiment:
   a numbered `### Experiment N` under the phase's Experiments section
   with `#### Hypothesis`, `#### Setup`, `#### Results` (a table when
   there are numbers) and `#### Verdict`. Real numbers only, from runs you
   made.
4. Update the phase checklist. New work is a `- [x]` line that says what
   was done and how it was verified. Work that was replaced is marked
   `- [~] ... -- Superseded: <why>`, never deleted; the record of dead
   ends is part of the value.
5. Keep quoted numbers current: test counts, index size, timings, memory.
   Take them from the run you just did, not from memory.
6. Read the edited sections once more as a stranger would. Prose, short
   sentences, no abbreviations the reader has not seen defined.

## Style

- British spelling in prose (colour, behaviour); code identifiers as they
  are.
- Tables for parallel facts, prose for reasoning.
- Say what was verified and how; say plainly what was not.
