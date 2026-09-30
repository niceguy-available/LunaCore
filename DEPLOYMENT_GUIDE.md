# 🚀 Lună — Full-Stack Deployment Guide

This guide explains how to deploy the **Lună** Lunar Image Registration Platform so that it is live and accessible to judges and users worldwide.

---

## ⚡ Option 1: Instant Public Access (Instant Tunnel — 30 Seconds)

If you want to share the live application with anyone right now while running locally on your computer:

### Step 1: Start Backend & Frontend
Make sure both servers are running:
* **Backend:** `cd backend && py -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
* **Frontend:** `cd frontend && npm run dev`

### Step 2: Launch Public HTTPS Tunnel
Open a new terminal and run:
```bash
npx localtunnel --port 3000
```
or with Cloudflare Tunnels:
```bash
npx untun@latest tunnel http://localhost:3000
```
This generates an instant, secure public URL (e.g. `https://luna-isro-demo.loca.lt`) accessible from any smartphone, laptop, or judge's workstation worldwide!

---

## 🌐 Option 2: Permanent 24/7 Cloud Deployment (Vercel + Render)

For a permanent, production-ready deployment with 0 server maintenance:

```
┌────────────────────────┐         ┌────────────────────────┐
│   Vercel (Frontend)    │  ────►  │    Render (Backend)    │
│  Next.js 14 App Router │  HTTPS  │ FastAPI + OpenCV Engine│
│ luna-app.vercel.app    │         │ luna-api.onrender.com  │
└────────────────────────┘         └────────────────────────┘
```

### Part A: Deploy the Backend (Render.com)

1. Push your repository to **GitHub** (e.g. `https://github.com/your-username/luna-isro`).
2. Go to [https://render.com](https://render.com) and log in.
3. Click **New +** → **Web Service**.
4. Connect your GitHub repository.
5. Configure the service settings:
   * **Name:** `luna-backend`
   * **Root Directory:** `backend`
   * **Runtime:** `Python 3` (or `Docker` using the included `Dockerfile`)
   * **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   * **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   * **Instance Type:** `Free`
6. Click **Deploy Web Service**.
7. Copy your backend URL (e.g. `https://luna-backend.onrender.com`).

---

### Part B: Deploy the Frontend (Vercel)

1. Go to [https://vercel.com](https://vercel.com) and log in.
2. Click **Add New...** → **Project**.
3. Import your GitHub repository.
4. Configure the project settings:
   * **Framework Preset:** `Next.js`
   * **Root Directory:** Click Edit and select `frontend`.
5. Under **Environment Variables**, add:
   * **Key:** `NEXT_PUBLIC_API_URL`
   * **Value:** `https://luna-backend.onrender.com` *(your Render backend URL from Part A)*
6. Click **Deploy**.

Vercel will build and assign you a global domain (e.g., `https://luna-isro.vercel.app`).

---

## 🐳 Option 3: Single-Container Docker Deployment

To deploy anywhere that supports Docker (AWS ECS, Google Cloud Run, DigitalOcean, Hugging Face Spaces):

```bash
# Build the backend container
cd backend
docker build -t luna-backend .

# Run container on port 8000
docker run -d -p 8000:8000 --name luna-api luna-backend
```
