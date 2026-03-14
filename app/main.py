"""Nova Intelligence Agent - Main Application Entry Point.

v3: Graph-orchestrated intelligence with memory and continuous monitoring.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
import os
import jwt
import datetime
from dotenv import load_dotenv
load_dotenv()

from app.api.routes import router
from app.api.auth_db import (
    init_auth_db, check_user_exists, create_pending_otp,
    verify_otp_and_register_or_login, get_user_by_email,
    store_refresh_token, validate_and_revoke_refresh_token,
    cleanup_expired_tokens, register_oauth_user
)
from app.api.email_service import send_otp_email
import asyncio
import httpx
from fastapi.responses import RedirectResponse
import urllib.parse

# v3: WebSocket for continuous mode
try:
    from app.api.ws import ws_manager, continuous_engine
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

# v3: Graph runner for continuous engine
try:
    from app.graph.nova_graph import run_graph
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

# Setup Logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Nova Intelligence Engine v3",
    description="AI-powered news intelligence with voice interface — v3 Graph Engine",
    version="3.0.0"
)

# Background tasks
async def background_cleanup():
    """Runs every hour to delete expired OTPs and refresh tokens from the db."""
    while True:
        try:
            await cleanup_expired_tokens()
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")
        await asyncio.sleep(3600)  # Sleep for 1 hour



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


# ── WebSocket endpoint for continuous intelligence ──
if WS_AVAILABLE:
    @app.websocket("/ws/intelligence")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time intelligence updates.
        
        Connect: ws://localhost:8000/ws/intelligence
        Receives: JSON messages with type field
        """
        await ws_manager.connect(websocket)
        try:
            # Send initial connection confirmation
            await ws_manager.send_to(websocket, {
                "type": "connected",
                "message": "Nova Intelligence WebSocket connected",
                "clients": ws_manager.client_count,
            })

            while True:
                # Keep connection alive and handle client messages
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await ws_manager.send_to(websocket, {"type": "pong"})
                elif msg_type == "status":
                    monitors = continuous_engine.get_active_monitors() if continuous_engine else []
                    await ws_manager.send_to(websocket, {
                        "type": "status",
                        "active_monitors": monitors,
                        "clients": ws_manager.client_count,
                    })

        except WebSocketDisconnect:
            logger.info("Client disconnected normally")
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            ws_manager.disconnect(websocket)
            try:
                await websocket.close()
            except Exception:
                pass

# ======== Auth Endpoints ========

class SendOTPRequest(BaseModel):
    mode: str       # "register" | "login"
    email: str
    username: str = ""
    password: str = ""

class VerifyOTPRequest(BaseModel):
    email: str
    otp: str

@app.post("/api/auth/send-otp")
async def send_otp(req: SendOTPRequest):
    email = req.email.strip().lower()
    username = req.username.strip()
    
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")
    
    # Send an OTP based on the mode requested by frontend
    if req.mode == "register":
        # Check if already exists
        exists = await check_user_exists(username, email)
        if exists == "username_exists":
            raise HTTPException(status_code=400, detail="Username already exists")
        if exists == "email_exists":
            raise HTTPException(status_code=400, detail="Email already registered")
            
        otp = await create_pending_otp(email, "REGISTER", username, req.password)
        # Mock Email Service
        await send_otp_email(email, otp, username=username)
        return {"status": "success", "message": "OTP sent to your email."}
        
    elif req.mode == "login":
        exists = await get_user_by_email(email)
        if not exists:
            raise HTTPException(status_code=404, detail="Email not found. Please register.")
            
        otp = await create_pending_otp(email, "LOGIN")
        await send_otp_email(email, otp, username=exists)
        return {"status": "success", "message": "OTP sent to your email."}
        
    else:
        raise HTTPException(status_code=400, detail="Invalid mode.")

@app.post("/api/auth/verify-otp")
async def verify_otp(req: VerifyOTPRequest):
    email = req.email.strip().lower()
    otp = req.otp.strip()
    
    result = await verify_otp_and_register_or_login(email, otp)
    if result["status"] == "error":
        raise HTTPException(status_code=401, detail=result["message"])
        
    username = result["username"]
    
    # Generate Short-Lived Access JWT (e.g. 1 hour)
    access_token = jwt.encode(
        {"sub": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, 
        "novasecret", algorithm="HS256"
    )
    
    # Generate Long-Lived Refresh JWT (e.g. 7 days)
    refresh_exp = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    refresh_token = jwt.encode(
        {"sub": username, "type": "refresh", "exp": refresh_exp}, 
        "nova_refresh_secret", algorithm="HS256"
    )
    
    # Store refresh token state in DB for revocation tracking
    await store_refresh_token(refresh_token, username, refresh_exp)

    return {
        "status": "success", 
        "token": access_token, 
        "refresh_token": refresh_token,
        "username": username
    }

class RefreshTokenRequest(BaseModel):
    refresh_token: str

@app.post("/api/auth/refresh")
async def refresh_access_token(req: RefreshTokenRequest):
    try:
        # Check signature & expiry
        payload = jwt.decode(req.refresh_token, "nova_refresh_secret", algorithms=["HS256"])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Validate against database (ensures it wasn't revoked or already used)
    username = await validate_and_revoke_refresh_token(req.refresh_token)
    if not username:
        raise HTTPException(status_code=401, detail="Refresh token revoked or used")

    # Issue NEW Access Token
    access_token = jwt.encode(
        {"sub": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, 
        "novasecret", algorithm="HS256"
    )
    
    # Issue NEW Refresh Token (Rolling refresh token strategy)
    refresh_exp = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    new_refresh_token = jwt.encode(
        {"sub": username, "type": "refresh", "exp": refresh_exp}, 
        "nova_refresh_secret", algorithm="HS256"
    )
    
    await store_refresh_token(new_refresh_token, username, refresh_exp)

    return {
        "status": "success",
        "token": access_token,
        "refresh_token": new_refresh_token
    }

# ======== OAuth Endpoints ========
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your-google-client-id")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "your-google-client-secret")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI",
    "http://localhost:8000/api/auth/google/callback"
)

@app.get("/api/auth/google/login")
async def google_login():
    url = f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id={GOOGLE_CLIENT_ID}&redirect_uri={urllib.parse.quote(GOOGLE_REDIRECT_URI)}&scope=openid%20email%20profile"
    return RedirectResponse(url)

@app.get("/api/auth/google/callback")
async def google_callback(code: str):
    token_url = "https://oauth2.googleapis.com/token"
    async with httpx.AsyncClient() as client:
        res = await client.post(token_url, data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code"
        })
        token_data = res.json()
        if "access_token" not in token_data:
            raise HTTPException(status_code=400, detail="Failed to get access token from Google")
            
        user_info_res = await client.get("https://www.googleapis.com/oauth2/v2/userinfo", headers={"Authorization": f"Bearer {token_data['access_token']}"})
        user_info = user_info_res.json()
        
    email = user_info.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Google authentication did not return an email.")
        
    base_name = user_info.get("name", email.split("@")[0])
    
    # Register or login OAuth user handles collisions gracefully
    username = await register_oauth_user(email, base_name)
    
    # Generate tokens
    access_token = jwt.encode(
        {"sub": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, 
        "novasecret", algorithm="HS256"
    )
    refresh_exp = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    refresh_token = jwt.encode(
        {"sub": username, "type": "refresh", "exp": refresh_exp}, 
        "nova_refresh_secret", algorithm="HS256"
    )
    await store_refresh_token(refresh_token, username, refresh_exp)

    # Fast redirect injection handling via query args intercepted heavily by modern SPAs.
    return RedirectResponse(url=f"/?token={access_token}&refresh={refresh_token}&username={urllib.parse.quote(username)}")
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    await init_auth_db()
    asyncio.create_task(background_cleanup())

    print("=" * 60)
    print("  Nova Intelligence Agent v3.0")
    print("=" * 60)
    print(f"  Engine:    {'LangGraph' if GRAPH_AVAILABLE else 'v2-executor'}")
    print(f"  WebSocket: {'Active' if WS_AVAILABLE else 'Unavailable'}")
    print(f"  API:       http://localhost:8000/api")
    print(f"  Frontend:  http://localhost:8000")
    print(f"  WS:        ws://localhost:8000/ws/intelligence")
    print("=" * 60)

    # Wire up continuous engine to graph runner
    if WS_AVAILABLE and GRAPH_AVAILABLE:
        continuous_engine.set_graph_runner(run_graph)
        print("  [Continuous] Graph runner connected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
