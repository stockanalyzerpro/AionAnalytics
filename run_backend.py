import warnings, os, uvicorn

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")
warnings.filterwarnings("ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"pandas_ta\.__init__")
os.environ.setdefault("PYTHONWARNINGS", "ignore:::pandas_ta.__init__")

if __name__ == "__main__":
    # Configure host/port via environment variables if needed
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    print(f"🚀 Launching StockAnalyzerPro backend on http://{host}:{port}")
    uvicorn.run("backend.backend_service:app", host=host, port=port, reload=False)
