from apps.api_gateway.main import app


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("apps.api_gateway.main:app", host="0.0.0.0", port=port)
