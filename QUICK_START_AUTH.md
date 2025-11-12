# Quick Start: User Authentication System

## What's New?

Your backend now has complete user authentication with JWT tokens and analysis history!

## 🚀 Quick Setup

### 1. Install Dependencies (if not already done)
```bash
cd "C:\Users\AADHITHAN\Downloads\Project 2"
pip install sqlalchemy python-jose passlib python-multipart python-dotenv
```

### 2. Create .env File
```bash
# Create a file named .env in your project root with:
SECRET_KEY=your-super-secret-key-change-in-production
DATABASE_URL=sqlite:///./smart_contract_analyzer.db
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 3. Start Backend
```bash
python -m api.main
```

The database will be automatically created with the users and analyses tables.

## 📋 Test It Out

### 1. Sign Up a User
```bash
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass123"}'
```

**Response** (save the `access_token`):
```json
{
  "user": {
    "id": 1,
    "email": "test@example.com",
    "created_at": "2025-11-12T10:00:00"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### 2. Analyze a Contract (Authenticated)
Replace `YOUR_TOKEN` with the `access_token` from above:

```bash
curl -X POST http://localhost:8000/api/v1/protected/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"contract_code":"pragma solidity ^0.8.0; contract Test {}"}'
```

### 3. View Analysis History
```bash
curl -X GET "http://localhost:8000/api/v1/protected/analyses" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. Get Specific Analysis
Replace `ANALYSIS_ID` with ID from response above:

```bash
curl -X GET "http://localhost:8000/api/v1/protected/analyses/ANALYSIS_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 📁 New Files

| File | Purpose |
|------|---------|
| `api/database.py` | Database configuration and session management |
| `api/models/database_models.py` | SQLAlchemy User and Analysis models |
| `api/models/auth_schemas.py` | Pydantic request/response schemas |
| `api/routers/auth.py` | Authentication endpoints (signup, login, logout) |
| `api/routers/authenticated_analysis.py` | Protected analysis endpoints |
| `api/middleware/auth.py` | Token verification and user extraction |
| `AUTHENTICATION_GUIDE.md` | Complete API documentation |
| `BACKEND_AUTH_SUMMARY.md` | Implementation summary |

## 🔑 Key Features

✅ **User Registration** - Create accounts with email and password  
✅ **JWT Authentication** - Secure token-based auth  
✅ **Analysis History** - Save and retrieve past analyses  
✅ **Token Refresh** - Keep users logged in longer  
✅ **Data Isolation** - Users can only see their own data  
✅ **Automatic Database** - Tables created on first startup  

## 📊 Database Schema

### Users Table
```
id           | email (unique) | password_hash | created_at
```

### Analyses Table
```
id        | user_id | contract_code | results (JSON) | risk_score | timestamp
```

## 🛡️ Security

- ✅ Passwords hashed with bcrypt
- ✅ JWT tokens with expiration
- ✅ HTTP Bearer authentication
- ✅ User data isolation
- ✅ Proper error handling

## 📚 Full Documentation

See these files for complete details:
- `AUTHENTICATION_GUIDE.md` - Complete API docs and examples
- `BACKEND_AUTH_SUMMARY.md` - Implementation details

## 🎯 Next Steps

1. **Build Frontend Login Pages**
   - Create signup form component
   - Create login form component
   - Store tokens in localStorage

2. **Update Frontend API Calls**
   - Add Authorization header with token
   - Implement token refresh logic
   - Create protected routes

3. **Enhance Features**
   - Email verification
   - Password reset
   - User profile management
   - 2FA support

## ⚠️ Important Notes

1. **Secret Key**: Change `SECRET_KEY` in production!
2. **Database**: Switch to PostgreSQL for production (from SQLite)
3. **CORS**: Update `allow_origins` for your domain
4. **HTTPS**: Always use HTTPS in production

## 🆘 Troubleshooting

**Error: "Email already registered"**
- This email already has an account
- Try login instead or use a different email

**Error: "Invalid authentication credentials"**
- Token is invalid or expired
- Get a new token by logging in again

**Error: "User not found"**
- User was deleted or doesn't exist
- Create a new account

## 📞 Support

Refer to `AUTHENTICATION_GUIDE.md` for:
- API endpoint documentation
- Error codes and meanings
- Frontend integration examples
- Security best practices

---

**Status**: ✅ Authentication system fully implemented and ready to use!
