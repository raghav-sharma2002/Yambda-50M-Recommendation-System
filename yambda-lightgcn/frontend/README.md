# Frontend

React + TypeScript + Vite + Tailwind demo that calls the FastAPI service
(`GET /health`, `GET /recommend/{user_id}?k=10`). Yambda ships no song/artist metadata,
so the UI shows anonymised `item_id`s with scores — an honest reflection of the data.

Drop the existing frontend source here. Dev server: `npm run dev` (http://localhost:5173, CORS-allowed by the API).
