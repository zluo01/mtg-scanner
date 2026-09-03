---
name: verify-ui
description: Prove a frontend change works by running the real server and driving headless Chromium at phone and desktop sizes, in both themes, with real scan scenes
---

# Verify a UI change

Screenshots and console output from a real browser against a real server
are the evidence for any change the user sees. Nothing in this skill needs
a display.

## Setup

1. Build the frontend: from `app/`, `pnpm build`.
2. Start a review server on a scratch data directory, not the user's:
   copy or symlink the index and models into `<scratch>/reviewdata/index`
   and `<scratch>/reviewdata/models`, then from `app/`:
   `DATA_DIR=<scratch>/reviewdata PORT=3100 HOST=0.0.0.0 node server/src/index.ts`
   in the background, and wait for `/api/health` to answer.
3. Test photos: `training/scripts/tools/make_test_scenes.py` composes
   phone-like scenes (single card, tilted card, binder page, empty
   background) from Scryfall images.

## Driving the browser

`app/scripts/screenshot.mjs` drives the installed Chromium over the
DevTools protocol with a fresh profile each run (a reused profile would
serve a stale build through the service worker):

```
node app/scripts/screenshot.mjs <url> <out.png> [--dark|--light]
     [--width 430] [--height 932] [--wait ms]
     [--click "css"]... [--eval "js"]... [--file "css" "/path"]... [--sleep ms]...
```

Steps run in order, then the screenshot is taken. Only `console.warn` and
`console.error` from the page are relayed, so log probe results with
`console.warn`. Use 430x932 for the phone and 1280x900 for desktop, and
run both themes for anything with colour.

## What to check, in order

- The state before: counts from `/api/health` or `/api/library`.
- The screen itself: open it, look at the PNG, read it critically.
- The state after: the same counts, so nothing was written or removed
  that should not have been.
- Behaviour that a screenshot cannot show: dispatch synthetic events from
  `--eval` (cancellable `TouchEvent`s for gestures, `Event('resize')` on
  `visualViewport` for the keyboard) and log what the code decided.
  Headless Chromium does not answer CDP gesture synthesis, so real swipes
  and wheel input cannot be exercised; say so when that is the gap.
- A stray headless browser from a hung run holds nothing important; find
  it by its `--headless=new` argument and kill it.

Record what was verified, with numbers, in the phase 5 document.
