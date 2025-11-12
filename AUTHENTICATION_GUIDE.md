# Authentication Implementation Guide

## Overview

The backend now includes complete JWT-based authentication system with user registration, login, and protected routes for contract analysis.

## Backend Features Implemented

### 1. Database Models
- **User Model**: Stores user credentials with hashed passwords
  - `id`: Primary key
  - `email`: Unique email address
  - `password_hash`: Bcrypt hashed password
  - `created_at`: Account creation timestamp
  - `analyses`: Relationship to user's analyses

- **Analysis Model**: Stores contract analysis history
  - `id`: Unique analysis identifier (UUID)
  - `user_id`: Foreign key to user
  - `contract_code`: Original smart contract code
  - `results`: JSON field with vulnerabilities and recommendations
  - `risk_score`: Risk score percentage (0-100)
  - `timestamp`: Analysis timestamp

### 2. Authentication Endpoints

#### Sign Up
```
POST /auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}

Response:
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

#### Login
```
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}

Response: Same as signup
```

#### Logout
```
POST /auth/logout

Response:
{
  "message": "Successfully logged out"
}
```

#### Refresh Token
```
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

Response:
{
  "access_token": "new_token",
  "refresh_token": "same_or_new_token",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 3. Protected Analysis Endpoints

#### Analyze Contract (Authenticated)
```
POST /api/v1/protected/analyze
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "contract_code": "pragma solidity ^0.8.0; contract Test { ... }"
}

Response:
{
  "analysis_id": "uuid-string",
  "user_id": 1,
  "vulnerabilities": [...],
  "optimizations": [...],
  "risk_score": 0.45,
  "analysis_time": 2.5,
  "timestamp": "2025-11-12T10:05:00"
}
```

#### Get Analysis History
```
GET /api/v1/protected/analyses?skip=0&limit=20
Authorization: Bearer <access_token>

Response:
{
  "total": 15,
  "skip": 0,
  "limit": 20,
  "analyses": [
    {
      "id": "uuid",
      "timestamp": "2025-11-12T10:05:00",
      "risk_score": 45,
      "vulnerability_count": 3,
      "results": {...}
    }
  ]
}
```

#### Get Analysis Detail
```
GET /api/v1/protected/analyses/{analysis_id}
Authorization: Bearer <access_token>

Response:
{
  "id": "uuid",
  "user_id": 1,
  "timestamp": "2025-11-12T10:05:00",
  "risk_score": 45,
  "contract_code": "pragma solidity...",
  "results": {...}
}
```

#### Delete Analysis
```
DELETE /api/v1/protected/analyses/{analysis_id}
Authorization: Bearer <access_token>

Response:
{
  "message": "Analysis deleted successfully"
}
```

## Security Features

1. **Password Hashing**: Uses bcrypt for secure password storage
2. **JWT Tokens**: Stateless authentication with expiring tokens
3. **Access/Refresh Tokens**: Separate short-lived and long-lived tokens
4. **HTTP Bearer Authentication**: Standard token authorization header
5. **Database Constraints**: Unique email constraint, foreign key relationships

## Configuration

### Environment Variables
```bash
# Set these in your .env file
SECRET_KEY=your-secret-key-here-change-in-production
DATABASE_URL=sqlite:///./smart_contract_analyzer.db
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Default Settings
- Access token expiration: 30 minutes
- Refresh token expiration: 7 days
- Database: SQLite (can be changed to PostgreSQL)

## Installation & Setup

### 1. Install Dependencies
```bash
pip install sqlalchemy python-jose passlib python-multipart python-dotenv
```

### 2. Initialize Database
The database tables are automatically created when the application starts.

### 3. Start Backend
```bash
python -m api.main
```

The API will be available at `http://localhost:8000`

## Usage Workflow

### Frontend Integration

1. **Sign Up User**
   ```javascript
   const response = await fetch('http://localhost:8000/auth/signup', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       email: 'user@example.com',
       password: 'password123'
     })
   });
   const data = await response.json();
   localStorage.setItem('access_token', data.access_token);
   localStorage.setItem('refresh_token', data.refresh_token);
   ```

2. **Make Authenticated Requests**
   ```javascript
   const token = localStorage.getItem('access_token');
   const response = await fetch('http://localhost:8000/api/v1/protected/analyze', {
     method: 'POST',
     headers: {
       'Authorization': `Bearer ${token}`,
       'Content-Type': 'application/json'
     },
     body: JSON.stringify({
       contract_code: 'pragma solidity...'
     })
   });
   ```

3. **Handle Token Expiration**
   ```javascript
   // When access token expires, use refresh token
   const refreshToken = localStorage.getItem('refresh_token');
   const response = await fetch('http://localhost:8000/auth/refresh', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ refresh_token: refreshToken })
   });
   const data = await response.json();
   localStorage.setItem('access_token', data.access_token);
   ```

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Analyses Table
```sql
CREATE TABLE analyses (
    id VARCHAR(36) PRIMARY KEY,
    user_id INTEGER NOT NULL,
    contract_code TEXT NOT NULL,
    results JSON NOT NULL,
    risk_score INTEGER DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

## Testing

### Test Sign Up
```bash
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

### Test Login
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

### Test Protected Endpoint
```bash
curl -X GET http://localhost:8000/api/v1/protected/analyses \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Error Handling

- `400 Bad Request`: Email already registered
- `401 Unauthorized`: Invalid credentials or token
- `404 Not Found`: Analysis not found
- `500 Internal Server Error`: Server-side error

## Next Steps

1. **Frontend Login/Signup Pages**: Create React components for authentication UI
2. **Token Refresh Logic**: Implement automatic token refresh in frontend
3. **Protected Routes**: Add route guards in frontend to require authentication
4. **User Profile**: Add user profile management endpoints
5. **Rate Limiting**: Implement rate limiting per user
6. **Audit Logging**: Add audit trails for user actions

## Security Checklist for Production

- [ ] Change `SECRET_KEY` to a strong random string
- [ ] Set `DEBUG=False` in production
- [ ] Use HTTPS/SSL certificates
- [ ] Configure CORS properly (don't use `*`)
- [ ] Use PostgreSQL instead of SQLite
- [ ] Implement rate limiting
- [ ] Add email verification for sign ups
- [ ] Add password reset functionality
- [ ] Implement 2FA (Two-Factor Authentication)
- [ ] Add API key management
- [ ] Set up monitoring and alerting
