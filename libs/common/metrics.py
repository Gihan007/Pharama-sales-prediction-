import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import PlainTextResponse


def install_metrics(app, service_name: str):
    started_at = time.time()
    request_counts = defaultdict(int)
    request_duration_sum = defaultdict(float)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        labels = (request.method, path, str(response.status_code))
        request_counts[labels] += 1
        request_duration_sum[labels] += duration

        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        lines = [
            "# HELP app_up Application health status.",
            "# TYPE app_up gauge",
            f'app_up{{service="{service_name}"}} 1',
            "# HELP app_uptime_seconds Application uptime in seconds.",
            "# TYPE app_uptime_seconds gauge",
            f'app_uptime_seconds{{service="{service_name}"}} {time.time() - started_at:.3f}',
            "# HELP app_http_requests_total Total HTTP requests.",
            "# TYPE app_http_requests_total counter",
        ]

        for (method, path, status), count in sorted(request_counts.items()):
            lines.append(
                'app_http_requests_total{'
                f'service="{service_name}",method="{method}",path="{path}",status="{status}"'
                f"}} {count}"
            )

        lines.extend(
            [
                "# HELP app_http_request_duration_seconds_sum Total HTTP request duration.",
                "# TYPE app_http_request_duration_seconds_sum counter",
            ]
        )
        for (method, path, status), duration in sorted(request_duration_sum.items()):
            lines.append(
                'app_http_request_duration_seconds_sum{'
                f'service="{service_name}",method="{method}",path="{path}",status="{status}"'
                f"}} {duration:.6f}"
            )

        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")
