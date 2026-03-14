"""
Vercel entrypoint for NovaAI.

Thin wrapper that exposes the FastAPI `app` from `app.main`
at the module level so Vercel's Python runtime can import it.
"""

from app.main import app

