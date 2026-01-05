# 🚀 Free Deployment Guide for DocMind RAG

Deploy your DocMind RAG application for **$0/month** (only pay for OpenAI API usage).

## Architecture

```
┌─────────────────────────────────────────────┐
│           FREE DEMO ARCHITECTURE            │
├─────────────────────────────────────────────┤
│                                             │
│   Vercel (FREE)      Render (FREE)          │
│   ┌─────────┐        ┌─────────────┐        │
│   │Frontend │───────►│  Backend    │        │
│   │ React   │        │  FastAPI    │        │
│   └─────────┘        └──────┬──────┘        │
│                             │               │
│         ┌───────────────────┼───────────┐   │
│         ▼                   ▼           │   │
│   ┌───────────┐      ┌───────────┐      │   │
│   │ Qdrant    │      │ Upstash   │      │   │
│   │ Cloud     │      │ Redis     │      │   │
│   │ (FREE)    │      │ (FREE)    │      │   │
│   └───────────┘      └───────────┘      │   │
│                                             │
│   Total Cost: $0 (+ OpenAI API usage)       │
└─────────────────────────────────────────────┘
```

## Step 1: Set Up Qdrant Cloud (Free Vector Database)

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Sign up for free account
3. Create a new cluster:
   - Name: `docmind-rag`
   - Cloud: AWS
   - Region: Choose closest to you
   - Plan: **Free** (1GB storage)
4. Copy your credentials:
   - **Cluster URL**: `https://xxx-xxx.aws.cloud.qdrant.io`
   - **API Key**: Click "API Keys" → Create new key

## Step 2: Set Up Upstash Redis (Free Queue)

1. Go to [upstash.com](https://upstash.com)
2. Sign up for free account
3. Create a new Redis database:
   - Name: `docmind-redis`
   - Region: Choose closest to you
   - Plan: **Free** (10K commands/day)
4. Copy your **Redis URL** (format: `rediss://default:xxx@xxx.upstash.io:6379`)

## Step 3: Deploy Backend to Render (Free)

1. Go to [render.com](https://render.com)
2. Sign up and connect your GitHub account
3. Click **"New"** → **"Web Service"**
4. Connect your `docmind-rag` repository
5. Configure:
   - **Name**: `docmind-backend`
   - **Root Directory**: `backend`
   - **Runtime**: Docker
   - **Plan**: **Free**
6. Add Environment Variables:

| Key | Value |
|-----|-------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `QDRANT_URL` | Your Qdrant Cloud URL |
| `QDRANT_API_KEY` | Your Qdrant API key |
| `REDIS_URL` | Your Upstash Redis URL |

7. Click **"Create Web Service"**
8. Wait for deployment (~5-10 min)
9. Copy your backend URL: `https://docmind-backend.onrender.com`

## Step 4: Deploy Frontend to Vercel (Free)

1. Go to [vercel.com](https://vercel.com)
2. Sign up and connect your GitHub account
3. Click **"Add New..."** → **"Project"**
4. Import your `docmind-rag` repository
5. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
6. Add Environment Variables:

| Key | Value |
|-----|-------|
| `VITE_API_URL` | `https://docmind-backend.onrender.com` |

7. Click **"Deploy"**
8. Your app is live at: `https://docmind-rag.vercel.app`

## Step 5: Update Frontend API URL

Edit `frontend/src/App.tsx` or create `.env.production`:

```env
VITE_API_URL=https://docmind-backend.onrender.com
```

## 🎉 Done!

Your app is now live at:
- **Frontend**: `https://your-project.vercel.app`
- **Backend API**: `https://docmind-backend.onrender.com`
- **API Docs**: `https://docmind-backend.onrender.com/docs`

## Free Tier Limits

| Service | Free Limit | What Happens When Exceeded |
|---------|------------|---------------------------|
| **Render** | 750 hrs/month | Service sleeps (cold start) |
| **Vercel** | Unlimited static | - |
| **Qdrant Cloud** | 1GB storage | Upgrade required |
| **Upstash Redis** | 10K commands/day | Upgrade required |
| **OpenAI API** | Pay per use | ~$0.002/1K tokens |

## Troubleshooting

### Backend sleeps on Render Free Tier
- Free tier services sleep after 15 min of inactivity
- First request after sleep takes ~30 seconds (cold start)
- Solution: Use [UptimeRobot](https://uptimerobot.com) to ping every 14 minutes

### CORS Errors
Add your Vercel URL to backend CORS:

```python
# backend/app/main.py
origins = [
    "http://localhost:3000",
    "https://your-project.vercel.app",  # Add this
]
```

### Redis Connection Issues
- Ensure you're using `rediss://` (with double 's') for Upstash
- Check if Redis URL includes the password

## Alternative: Railway (Even Simpler)

If you prefer one-click deploy:

1. Click: [![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/xxx)
2. Add your `OPENAI_API_KEY`
3. Done!

Railway gives $5 free credit/month which is enough for demo usage.

