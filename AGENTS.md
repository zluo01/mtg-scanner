# Working on MTG Scanner

Instructions for any model, agent or harness working in this repository.
They are tool-neutral: follow them with whatever editor, shell and test
runner you have. Repeatable procedures live in `skills/`, one directory
per skill with a `SKILL.md` (name and description up top, then the steps)
and optional `references/`; load the one that matches the task:

- `skills/update-docs`: keep `docs/` in step with a change.
- `skills/verify-ui`: prove a frontend change against a real server in
  headless Chromium.
- `skills/refresh-index`: update the card index for new sets.

## What this is

A personal Magic: The Gathering card scanner. Detection runs in the phone's
browser, identification and the library run in one Node process, and the
reference index is built offline in Python. Read `README.md` first, then
`docs/overview.md`; the phase documents under `docs/` record every design
decision and experiment and are kept current.

## Layout

- `app/` is one pnpm workspace: `web/` (SolidJS, Vite, Tailwind), `server/`
  (Hono on Node 24, TypeScript run natively, no build step), `shared/`
  (API types). One `node_modules`, one lockfile, one `tsconfig.json`, one
  Biome config. Use pnpm, never npm.
- `training/` is Python, run from the `learning` conda environment with
  `python -s` (a user-level numpy would otherwise shadow the one faiss was
  built against).
- All workflows are Makefile targets at the root: `make check`, `make dev`,
  `make start`, `make parity`, `make docker-build`.

## Rules that are not negotiable

1. **Verify before you claim.** Run `make check` (typecheck, lint, all
   tests). For anything the user sees, build and start the server against
   real data and look at it, with the headless-Chromium tool in
   `app/scripts/screenshot.mjs` on both phone and desktop sizes and both
   themes. Report what you actually observed, including what you could not
   verify. Never describe a result you did not run.
2. **Read the current state before asserting anything** about what exists,
   what a file contains, or what the last run showed.
3. **Edit files with file-editing tools**, never with `sed`, Python or
   shell heredocs. Running the formatter afterwards is fine.
4. **Keep the stack minimal.** No frameworks, routers, query libraries,
   component kits, or TLS work. The app runs over plain http on a LAN and
   is designed for it. If something needs a new dependency, say why in a
   sentence and expect to be asked.
5. **Rules live in the data layer.** If the database can express an
   invariant, it does (the one-row-per-printing rule is a unique index and
   an upsert). Application code handles only what the database cannot.
   The client displays what the server returns.
6. **No migration framework.** Schema changes go into `server/src/sql/`
   and are applied to existing databases directly, in the open path or by
   a one-off statement, and noted in the phase 5 document.
7. **All SQL lives in `app/server/src/sql/*.sql`**, one statement per
   file, named parameters, loaded by `sql.ts`. Do not build SQL from
   strings.
8. **Nothing is stored until the user confirms.** A scan is identified
   first; the card and its photo are written only on Add.
9. **Update the documentation with the change**, in the same piece of
   work, using `skills/update-docs`. Experiments get a hypothesis, setup,
   results and verdict; superseded work is marked, not deleted.
10. **Commit only when asked**, and only what was asked for.

## Conventions

- Tabs, single quotes, 110-column lines; Biome enforces them.
- Tests use `node --test`; server tests run on in-memory SQLite and need
  no data directory.
- Copy in the interface is sentence case, plain verbs, and says what
  happens ("Add", "Discard"), never "Submit".
- The reference to a card is always its Scryfall id; set code and
  collector number are labels derived from it.
- Scratch files, screenshots and logs go outside the repository.
