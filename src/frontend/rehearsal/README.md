# Rehearsal app

**Status: working prototype (research_pilot).** A speaker consents, records answers to five interview questions, and gets a per-answer report from the backend's `POST /analyze`.

- Screens: consent → recording → processing → report (`src/`).
- Backend failures are shown as errors with a retry button (processing screen, and per answer in the report). The app never displays placeholder or invented scores (`src/analyze.js` holds the shared request helper).
- Known gaps, fixed in later phases: it still asks for a self-label per answer and has a "Delete my data" button that does not delete anything, because the backend stores nothing. See `PROJECT_STATUS.md`.

## Setup

```bash
npm install
npm run dev      # development server (Vite)
npm run build    # production build check
```

The app reads `VITE_BACKEND_URL` from a `.env` file at the **repository root** (git-ignored; no `.env.example` yet). If it is unset the app uses `http://localhost:8000`. A `VITE_BACKEND_URL` set in the shell overrides the file.

Vite starts on port 5173 and moves to the next free port (5174, 5175) when several apps run at once; the backend's CORS allow-list (`CORS_ALLOWED_ORIGINS`) defaults to those three origins.

To see the error state, start it against a port with nothing listening:

```bash
VITE_BACKEND_URL=http://localhost:9 npm run dev
```

Project overview and status: [`../../../PROJECT_STATUS.md`](../../../PROJECT_STATUS.md).
