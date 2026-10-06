# Evaluator portal

**Status: prototype, not connected.** Intended for neurotypical evaluators to label clips; part of the future gap-data collection.

- Login is a password compared in the page code (`src/LoginScreen.jsx`) — not real authentication.
- Clips are placeholders (no audio source), and submitted labels are posted as JSON to `/score`, which expects an audio upload, so the request fails and nothing is saved.
- It is out of scope for the current milestone. See `PROJECT_STATUS.md`.

## Setup

```bash
npm install
npm run dev      # development server (Vite)
npm run build    # production build check
```

The app reads `VITE_BACKEND_URL` from a `.env` file at the **repository root** (git-ignored; no `.env.example` yet). If it is unset the app uses `http://localhost:8000`. A `VITE_BACKEND_URL` set in the shell overrides the file.

Project overview and status: [`../../../PROJECT_STATUS.md`](../../../PROJECT_STATUS.md).
