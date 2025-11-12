# Authentication System Architecture

## 🔐 Complete Authentication Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER AUTHENTICATION FLOW                         │
└─────────────────────────────────────────────────────────────────────────┘

1. REGISTRATION
═══════════════════════════════════════════════════════════════════════════
   Frontend                          Backend
      │                                 │
      ├─ POST /auth/signup ───────────>│
      │  {email, password}              │
      │                                 │ ✓ Validate email
      │                                 │ ✓ Hash password
      │                                 │ ✓ Create user in DB
      │                                 │
      │<─────── {user, tokens} ────────┤
      │                                 │


2. LOGIN
═══════════════════════════════════════════════════════════════════════════
   Frontend                          Backend
      │                                 │
      ├─ POST /auth/login ────────────>│
      │  {email, password}              │
      │                                 │ ✓ Find user by email
      │                                 │ ✓ Verify password
      │                                 │ ✓ Generate JWT tokens
      │                                 │
      │<─────── {user, tokens} ────────┤
      │  access_token (30 min)          │
      │  refresh_token (7 days)         │
      │                                 │


3. AUTHENTICATED REQUEST
═══════════════════════════════════════════════════════════════════════════
   Frontend                          Backend
      │                                 │
      ├─ POST /api/v1/protected/... ──>│
      │  Authorization: Bearer <token>  │
      │  {contract_code}                │
      │                                 │ ✓ Verify token signature
      │                                 │ ✓ Extract user_id
      │                                 │ ✓ Check token expiration
      │                                 │ ✓ Find user by id
      │                                 │ ✓ Process request
      │                                 │ ✓ Save analysis to DB
      │                                 │
      │<──── {analysis_result} ────────┤
      │                                 │


4. TOKEN REFRESH
═══════════════════════════════════════════════════════════════════════════
   Frontend                          Backend
      │                                 │
      ├─ POST /auth/refresh ──────────>│
      │  {refresh_token}                │
      │                                 │ ✓ Verify refresh token
      │                                 │ ✓ Generate new access token
      │                                 │
      │<──── {new_access_token} ──────┤
      │                                 │


5. LOGOUT
═══════════════════════════════════════════════════════════════════════════
   Frontend                          Backend
      │                                 │
      ├─ DELETE localStorage token ─────│
      │                                 │
      ├─ POST /auth/logout ───────────>│
      │                                 │ (Optional logging)
      │                                 │
      │<─────── {success} ─────────────┤
      │                                 │

```

## 📊 Database Relationships

```
USERS Table
══════════════════════════════════════════════════════════════
  PK │ id            (INT)
     │ email         (VARCHAR) UNIQUE
     │ password_hash (VARCHAR)
     │ created_at    (DATETIME)
     │ updated_at    (DATETIME)
     │
     └──┐
        │ (1:N Relationship)
        │
        ├──► ANALYSES Table
             ══════════════════════════════════════════════════════════════
             PK │ id              (VARCHAR/UUID)
             FK │ user_id         (INT) ─── References: USERS.id
                │ contract_code   (TEXT)
                │ results         (JSON)
                │ risk_score      (INT 0-100)
                │ timestamp       (DATETIME)
```

## 🔑 JWT Token Structure

```
Access Token (30 minutes)
═══════════════════════════════════════════════════════════════
Header:    { "alg": "HS256", "typ": "JWT" }
Payload:   { "sub": "user@example.com", "user_id": 1, "exp": ... }
Signature: HMACSHA256(header.payload, SECRET_KEY)

Format: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWI...


Refresh Token (7 days)
═══════════════════════════════════════════════════════════════
Header:    { "alg": "HS256", "typ": "JWT" }
Payload:   { "sub": "user@example.com", "user_id": 1, "type": "refresh", "exp": ... }
Signature: HMACSHA256(header.payload, SECRET_KEY)
```

## 🛡️ Security Layers

```
┌──────────────────────────────────────────────────────────────┐
│                      SECURITY ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Layer 1: Password Security                                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • Bcrypt hashing (salted)                              │ │
│  │ • Never store plain text                               │ │
│  │ • Secure comparison on verification                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Layer 2: Token Security                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • JWT with HS256 algorithm                             │ │
│  │ • Configurable expiration                              │ │
│  │ • Signature verification                               │ │
│  │ • Access token (short-lived)                           │ │
│  │ • Refresh token (long-lived)                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Layer 3: Authentication Layer                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • HTTP Bearer token verification                       │ │
│  │ • Token extraction from header                         │ │
│  │ • User lookup from token claims                        │ │
│  │ • Route-level protection                               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Layer 4: Authorization Layer                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • User isolation (only see own data)                   │ │
│  │ • Foreign key enforcement                              │ │
│  │ • Resource ownership verification                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Layer 5: API Security                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ • CORS configuration                                   │ │
│  │ • Input validation (Pydantic)                          │ │
│  │ • Output encoding                                      │ │
│  │ • Error handling without disclosure                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## 📡 API Endpoint Tree

```
HTTP API
│
├── /auth (Public)
│   ├── POST /signup ..................... Register new user
│   ├── POST /login ...................... Authenticate user
│   ├── POST /logout ..................... Log out user
│   └── POST /refresh .................... Refresh access token
│
├── /api/v1/protected (Protected - Requires Auth Token)
│   ├── POST /analyze .................... Analyze contract (save to history)
│   ├── GET /analyses .................... Get analysis history
│   ├── GET /analyses/{id} .............. Get analysis details
│   └── DELETE /analyses/{id} ........... Delete analysis
│
└── /api/v1 (Public - Existing)
    ├── POST /analyze .................... Analyze contract (no history)
    ├── GET /status ...................... System status
    └── GET /health ...................... Health check
```

## 🔄 Token Lifecycle

```
Token Generation (Login/Signup)
═════════════════════════════════════════════════════════════
    │
    ├─ Access Token (30 min) ───► Use for API requests
    │     │
    │     └─► After 30 min
    │         │
    │         ├─ Token expires
    │         ├─ API returns 401 Unauthorized
    │         │
    │         ✓ Use refresh token to get new access token
    │
    ├─ Refresh Token (7 days) ───► Store securely
          │
          └─► After 7 days
              │
              ├─ Token expires
              │
              ✓ User must login again


Typical Session Flow:
═════════════════════════════════════════════════════════════
Day 1 (Hour 0)      → User logs in
                    → Receives: access_token (30 min), refresh_token (7 days)
                    
Day 1 (Hour 0:30)   → Access token expires
                    → Use refresh token to get new access token
                    
Day 1 (Hour 1)      → New access token expires
                    → Use same refresh token again
                    
...

Day 7               → Refresh token expires
                    → User must login again
```

## 💾 Request/Response Examples

### Registration Request
```
POST /auth/signup HTTP/1.1
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

### Registration Response (201 Created)
```
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "created_at": "2025-11-12T10:00:00"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Authenticated Request
```
POST /api/v1/protected/analyze HTTP/1.1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "contract_code": "pragma solidity ^0.8.0; contract Test { }"
}
```

### Authenticated Response
```
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": 1,
  "vulnerabilities": [...],
  "optimizations": [...],
  "risk_score": 0.45,
  "analysis_time": 2.5,
  "timestamp": "2025-11-12T10:05:00"
}
```

## ❌ Error Responses

```
400 Bad Request - Email Already Registered
{
  "detail": "Email already registered"
}

401 Unauthorized - Invalid Credentials
{
  "detail": "Invalid email or password"
}

401 Unauthorized - Invalid Token
{
  "detail": "Invalid authentication credentials"
}

404 Not Found - Analysis Not Found
{
  "detail": "Analysis not found"
}

500 Internal Server Error
{
  "detail": "An unexpected error occurred"
}
```

---

**This document visualizes the complete authentication system architecture.**
