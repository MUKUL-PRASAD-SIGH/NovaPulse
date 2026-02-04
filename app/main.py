"""Nova Intelligence Agent - Main Application Entry Point.

Voice-powered multi-agent news intelligence system using Amazon Nova.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.api.routes import router


# Create FastAPI app
app = FastAPI(
    title="Nova Intelligence Agent",
    description="AI-powered news intelligence with voice interface",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api", tags=["Intelligence"])

# Serve frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("[Nova] Intelligence Agent starting...")
    print("[API] Available at: http://localhost:8000/api")
    print("[Web] Frontend at: http://localhost:8000")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
