import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
API_GATEWAY_URL = os.environ.get("API_GATEWAY_URL", "http://127.0.0.1:5000")

app = FastAPI(title="Web UI Service")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _jinja_url_for(name: str, **path_params):
    if name == "static":
        filename = path_params.get("filename", "")
        return f"/static/{filename}" if filename else "/static/"
    return f"/{name}"


templates.env.globals["url_for"] = _jinja_url_for


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "web-ui"}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/forecast")
async def forecast_page(request: Request):
    return templates.TemplateResponse("forecast.html", {"request": request})


@app.get("/meta-learning")
async def meta_learning_page(request: Request):
    return templates.TemplateResponse("meta-learning.html", {"request": request})


@app.get("/advanced")
async def advanced_page(request: Request):
    return templates.TemplateResponse("advanced.html", {"request": request})


@app.get("/causal")
async def causal_page(request: Request):
    return templates.TemplateResponse("causal.html", {"request": request})


@app.get("/analytics")
async def analytics_page(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})


@app.get("/branches")
async def branches_page(request: Request):
    return templates.TemplateResponse("branches.html", {"request": request})


@app.get("/explainability")
async def explainability_page(request: Request):
    return templates.TemplateResponse("explainability.html", {"request": request})


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_api(path: str, request: Request):
    target_url = f"{API_GATEWAY_URL.rstrip('/')}/api/{path}"
    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream = await client.request(
                request.method,
                target_url,
                params=request.query_params,
                content=body,
                headers=headers,
            )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={"success": False, "error": f"API gateway unavailable: {exc}"},
        )

    excluded_headers = {"content-encoding", "transfer-encoding", "connection"}
    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in excluded_headers
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )
