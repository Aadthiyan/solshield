# 🚀 Deployment Guide: Render & Vercel

This guide will help you host the **Backend on Render** and the **Frontend on Vercel**.

## 📋 Prerequisites
- A GitHub repository with this project code pushed to it.
- Accounts on [Render.com](https://render.com) and [Vercel.com](https://vercel.com).

---

## 🏗️ Phase 1: Deploy Backend to Render

1.  **Log in to Render** and go to your Dashboard.
2.  Click **New +** -> **Blueprint** (recommended) or **Web Service**.
3.  **Connect your GitHub repository**.

### Option A: Using Blueprint (Recommended)
1.  Render will detect the `render.yaml` file in your repository.
2.  It will ask you to approve the creation of a **Web Service** (API) and a **PostgreSQL Database**.
3.  Click **Apply**. Render will start building your backend and database.
    *   *Note: This might take a few minutes.*
4.  Once deployed, copy your **Service URL** (e.g., `https://solshield-api.onrender.com`).

### Option B: Manual Setup
1.  Create a **PostgreSQL** database first. Copy the `Internal Connection URL`.
2.  Create a **Web Service**.
    *   **Root Directory**: `.` (leave empty).
    *   **Runtime**: Python 3.
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
    *   **Environment Variables**:
        *   `DATABASE_URL`: Paste the database internal connection URL.
        *   `PYTHON_VERSION`: `3.11.9`
        *   `SECRET_KEY`: Generate a random string.

---

## 🎨 Phase 2: Deploy Frontend to Vercel

1.  **Log in to Vercel**.
2.  Click **Add New...** -> **Project**.
3.  **Import your GitHub repository**.
4.  **Configure Project**:
    *   **Root Directory**: Click `Edit` and select `frontend`. **(Crucial Step!)**
    *   **Framework Preset**: Next.js (should detect automatically).
    *   **Environment Variables**:
        *   Key: `BACKEND_URL`
        *   Value: Your Render Backend URL (e.g., `https://solshield-api.onrender.com`). **Must include `https://` and NO trailing slash.**
5.  Click **Deploy**.

Vercel will build and deploy your frontend. Once done, copy your **Frontend Domain** (e.g., `https://solshield-frontend.vercel.app`).

---

## 🔗 Phase 3: Connect & Secure

Now that both are online, you need to tell the Backend to accept requests from your Vercel Frontend (CORS).

1.  Go back to **Render Dashboard** -> **solshield-api** -> **Environment**.
2.  Edit the `ALLOWED_ORIGINS` variable (or add it if missing).
3.  **Value**: `https://solshield-frontend.vercel.app` (Your actual Vercel domain).
    *   *Note: You can separate multiple domains with a comma if you want to keep localhost working: `https://solshield-frontend.vercel.app,http://localhost:3000`*
4.  **Save Changes**. Render will automatically restart the backend.

---

## ✅ Verification

1.  Open your Vercel App URL.
2.  Try to **Sign Up** or **Login**.
3.  If successful, your entire stack is live and communicating! 

### Troubleshooting
- **Frontend Error**: "Backend connection failed" -> Check Vercel Env Var `BACKEND_URL`. Check if Render backend is "Live".
- **CORS Error**: Check Render Env Var `ALLOWED_ORIGINS`. Ensure no trailing slashes in URLs.
