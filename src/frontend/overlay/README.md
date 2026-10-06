# Evaluator overlay

**Status: prototype, not connected.** A card meant to show live content-quality information during an interview by listening on the backend's `/stream` WebSocket.

- It only receives messages; it never sends audio, and the backend's `/stream` is a stub that returns constants (to be disabled in this milestone).
- `mock-server.js` at the repository root can feed it random demo values for UI work only. Those values come from no model.
- It is out of scope for the current milestone. See `PROJECT_STATUS.md`.

## Setup

```bash
npm install
npm run dev      # development server (Vite)
npm run build    # production build check
```

The app reads `VITE_BACKEND_URL` from a `.env` file at the **repository root** (git-ignored; no `.env.example` yet). If it is unset the app uses `http://localhost:8000`. A `VITE_BACKEND_URL` set in the shell overrides the file.

Project overview and status: [`../../../PROJECT_STATUS.md`](../../../PROJECT_STATUS.md).
