# Stock Prediction System

Predict stock **up/down direction** with **separate scale-native models**
(day / natural week / natural month). The stack is
**FastAPI (API + train worker) + React (Ant Design)**, with PostgreSQL for job
and artifact metadata.

## Architecture

```
frontend/   React + Vite + Ant Design (UI)
backend/    FastAPI API + async train worker + ML
db          PostgreSQL
```

Training runs in a dedicated worker process. Artifacts are saved to
`models_store/{market}/{ticker}/{day|week|month}/` (`model.pt`, `scaler.joblib`, `meta.json`).

模型结构、特征工程与训练/推理流程见：[docs/model-architecture.md](docs/model-architecture.md)。

## Docker Compose — 开发环境一键启动（推荐）

```bash
./bin/start_dev.sh
# 或: docker compose -f docker-compose.dev.yml up --build -d
```

停止：

```bash
./bin/stop_dev.sh
```

| 地址 | 说明 |
|------|------|
| http://localhost:${DEV_UI_PORT:-8080} | 前端（Vite 热加载，端口见 `.env` 的 `DEV_UI_PORT`） |
| http://localhost:8000/docs | 后端 API 文档（uvicorn 热加载） |

> HMR 要求：浏览器访问的端口必须与 `DEV_UI_PORT` 一致，否则改代码不会即时刷新。本机若 `5173` 已被占用，请保持 `DEV_UI_PORT=8080`。

| Service | 热加载 |
|---------|--------|
| `api` | `uvicorn --reload` |
| `frontend` | Vite HMR |
| `worker` | `watchmedo` 监听 `backend/app/**/*.py` |

## Docker Compose — 生产

```bash
./bin/start_prod.sh
# 或: docker compose up --build -d
```

停止：

```bash
./bin/stop_prod.sh
```

- UI: http://localhost:8080（nginx 静态资源 + 反代 API）

## 本地裸机开发（可选）

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# terminal 1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# terminal 2
watchmedo auto-restart --directory=./app --pattern='*.py' --recursive -- \
  python -m app.worker --poll-interval 2
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Main API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/home` | Home dashboard quotes + series |
| GET | `/api/suggest?q=` | Ticker autocomplete |
| GET | `/api/models` | List ready models |
| POST | `/api/train` | Enqueue training `{ "ticker": "600519" }` |
| GET | `/api/train/{job_id}` | Training job status |
| GET | `/api/predict/{ticker}` | Day / natural-week / natural-month direction + history |
| GET | `/api/tickers` | Ticker catalog |

## Disclaimer

Educational use only. USE AT YOUR OWN RISK. No warranty for trading results.
