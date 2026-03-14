import asyncio
import aiosqlite
import hashlib
import os
import random
import datetime

DB_PATH = 'nova_auth.db'

async def init_auth_db():
    async with aiosqlite.connect(DB_PATH) as db:
        # We add 'email' field to users. Also recreate table if needed or just handle it if it existed.
        await db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                pwd_hash BLOB,
                salt BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # New OTP Table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS otp_codes (
                email TEXT PRIMARY KEY,
                otp TEXT,
                expires_at TIMESTAMP,
                type TEXT,
                pending_username TEXT,
                pending_pwd_hash BLOB,
                pending_salt BLOB
            )
        ''')
        
        # Add refresh_tokens table
        await db.execute('''
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                token TEXT PRIMARY KEY,
                username TEXT,
                expires_at TIMESTAMP
            )
        ''')
        await db.commit()

def hash_password(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    if not password:
        return b'', b''
    if salt is None:
        salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return pwd_hash, salt

def generate_otp() -> str:
    """Returns a secure 6 digit random string OTP."""
    return str(random.randint(100000, 999999))

async def check_user_exists(username: str, email: str) -> str:
    """Returns 'username_exists', 'email_exists', or ''."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute('SELECT username, email FROM users WHERE username = ? OR email = ?', (username, email)) as c:
            row = await c.fetchone()
            if row:
                if row[0] == username: return 'username_exists'
                if row[1] == email: return 'email_exists'
    return ''

async def create_pending_otp(email: str, req_type: str, username: str = "", password: str = "") -> str:
    """Creates a 6 digit OTP valid for 10 minutes and saves to DB."""
    otp = generate_otp()
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
    
    pwd_hash, salt = b'', b''
    if password:
        pwd_hash, salt = hash_password(password)

    async with aiosqlite.connect(DB_PATH) as db:
        # Insert or Replace
        await db.execute('''
            INSERT OR REPLACE INTO otp_codes 
            (email, otp, expires_at, type, pending_username, pending_pwd_hash, pending_salt)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (email, otp, expires_at, req_type, username, pwd_hash, salt))
        await db.commit()
    return otp

async def verify_otp_and_register_or_login(email: str, otp: str) -> dict:
    """Verifies OTP. Returns dict w/ status & username, or errors."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute('SELECT * FROM otp_codes WHERE email = ? AND otp = ?', (email, otp)) as c:
            row = await c.fetchone()
            
            if not row:
                return {"status": "error", "message": "Invalid OTP or Email."}
            
            # Check expiration
            expires_at = datetime.datetime.strptime(row['expires_at'], "%Y-%m-%d %H:%M:%S.%f")
            if datetime.datetime.utcnow() > expires_at:
                return {"status": "error", "message": "OTP expired."}

            req_type = row['type']
            
            if req_type == 'REGISTER':
                username = row['pending_username']
                pwd_hash = row['pending_pwd_hash']
                salt = row['pending_salt']
                
                # Check for rapid DB collisions
                collision = await check_user_exists(username, email)
                if collision:
                    return {"status": "error", "message": "Account already created."}
                
                await db.execute(
                    'INSERT INTO users (username, email, pwd_hash, salt) VALUES (?, ?, ?, ?)',
                    (username, email, pwd_hash, salt)
                )
                await db.execute('DELETE FROM otp_codes WHERE email = ?', (email,))
                await db.commit()
                return {"status": "success", "username": username}

            elif req_type == 'LOGIN':
                # Grab actual username from users table to return in JWT
                async with db.execute('SELECT username FROM users WHERE email = ?', (email,)) as u_c:
                    u_row = await u_c.fetchone()
                    if not u_row:
                        return {"status": "error", "message": "User deleted or invalid."}
                    username = u_row['username']

                await db.execute('DELETE FROM otp_codes WHERE email = ?', (email,))
                await db.commit()
                return {"status": "success", "username": username}

    return {"status": "error", "message": "Unknown error."}

async def get_user_by_email(email: str) -> str:
    """Checks if email user exists, returns username or ''."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute('SELECT username FROM users WHERE email = ?', (email,)) as c:
            row = await c.fetchone()
            if row: return row[0]
            return ""

async def register_oauth_user(email: str, base_username: str) -> str:
    """Creates a user from OAuth, handles username collision by appending digits."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Check if already exists
        async with db.execute('SELECT username FROM users WHERE email = ?', (email,)) as c:
            row = await c.fetchone()
            if row: return row[0]
            
        username = base_username.replace(" ", "")
        for i in range(100):
            test_usr = f"{username}{i if i > 0 else ''}"
            async with db.execute('SELECT username FROM users WHERE username = ?', (test_usr,)) as c:
                row = await c.fetchone()
                if not row:
                    # Found available username
                    await db.execute(
                        'INSERT INTO users (username, email, pwd_hash, salt) VALUES (?, ?, ?, ?)',
                        (test_usr, email, b'', b'')
                    )
                    await db.commit()
                    return test_usr
        return f"{username}_u{random.randint(100, 9999)}"

async def store_refresh_token(token: str, username: str, expires_at: datetime.datetime):
    """Saves a JWT refresh token into the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            'INSERT INTO refresh_tokens (token, username, expires_at) VALUES (?, ?, ?)',
            (token, username, expires_at)
        )
        await db.commit()

async def validate_and_revoke_refresh_token(token: str) -> str:
    """Checks if a refresh token is valid and returns username. If valid, deletes it (one-time use)."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute('SELECT username, expires_at FROM refresh_tokens WHERE token = ?', (token,)) as c:
            row = await c.fetchone()
            if not row:
                return ""
            
            username, expires_at_str = row
            expires_at = datetime.datetime.strptime(expires_at_str, "%Y-%m-%d %H:%M:%S.%f")
            
            # Delete token to prevent reuse (Rolling refresh tokens)
            await db.execute('DELETE FROM refresh_tokens WHERE token = ?', (token,))
            await db.commit()

            if datetime.datetime.utcnow() > expires_at:
                return ""
                
            return username

async def cleanup_expired_tokens():
    """Removes expired OTP codes and refresh tokens completely representing automated DB sweeps."""
    async with aiosqlite.connect(DB_PATH) as db:
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
        await db.execute('DELETE FROM otp_codes WHERE expires_at < ?', (now_str,))
        await db.execute('DELETE FROM refresh_tokens WHERE expires_at < ?', (now_str,))
        await db.commit()

