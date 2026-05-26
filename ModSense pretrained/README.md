# ModSense

A web UI for the NUSMods Disqus sentiment analysis notebook.  
The **Disqus API key lives only on the server** — it is never sent to the browser.

---

## Project structure

```
modsense/
├── app.py               ← Flask backend (all logic, API key here)
├── templates/
│   └── index.html       ← Frontend (pure HTML/CSS/JS)
├── sentiment_model.keras ← Pretrained model used for predictions
├── requirements.txt
├── Procfile             ← For Heroku / Railway / Render
├── .env.example         ← Copy → .env and fill in your key
└── .gitignore
```

---

## Pretrained model

This version does not train a model at startup. It loads **`sentiment_model.keras`**
from the project root and uses it directly for predictions.

---

## Setting your API key

### Locally

```bash
cp .env.example .env
# Open .env and replace the placeholder with your real Disqus PUBLIC key
```

The app reads `DISQUS_API_KEY` from the environment via `python-dotenv`.  
The key **never** appears in any HTTP response or HTML page.

### On a deployment platform

Set the environment variable in the platform's dashboard — never in code:

| Platform | Where to set env vars |
|---|---|
| **Railway** | Project → Variables tab |
| **Render** | Service → Environment tab |
| **Heroku** | Settings → Config Vars |
| **Fly.io** | `fly secrets set DISQUS_API_KEY=...` |

---

## Running locally

```bash
# 1. Create and activate a clean environment.
# Do not use Anaconda base/Python 3.13 for this app.
conda env create -f environment.yml
conda activate modsense

# 2. Set your key
cp .env.example .env
# edit .env

# 3. Start the server
python app.py
# → http://localhost:5000
```

On first start the server will:
1. Load `sentiment_model.keras`
2. Fetch all NUSMods reviews from Disqus (takes several minutes)
3. Show "Ready" in the status bar once complete

The browser polls `/api/status` every 2.5 seconds and enables the search box automatically.

---

## Deploying to Railway (recommended, free tier available)

```bash
# 1. Push to a GitHub repo
git init && git add . && git commit -m "init"
gh repo create nusmods-sentiment --public --push

# 2. Go to https://railway.app → New Project → Deploy from GitHub
# 3. Select your repo
# 4. In Variables tab: add DISQUS_API_KEY = <your key>
# 5. Railway auto-detects the Procfile and deploys
```

> **Note**: Make sure `sentiment_model.keras` is included in the deployment or mounted as a persistent file.

---

## API endpoints (internal, for reference)

| Endpoint | Description |
|---|---|
| `GET /` | Serves the UI |
| `GET /api/status` | Returns `ready`, `loading`, `progress`, `error` |
| `GET /api/modules` | Returns list of all module codes |
| `GET /api/analyze/<code>` | Returns sentiment for a module |
| `POST /api/reload` | Re-fetches Disqus data |
