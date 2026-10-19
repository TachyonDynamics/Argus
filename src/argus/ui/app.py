"""Admin web UI: server-rendered pages, no build step, nothing loaded from the internet."""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from argus.config import Config
from argus.core.models import now_ns
from argus.diagnostics.engine import explain
from argus.integrity.service import IntegrityService
from argus.proxy.tracker import Tracker
from argus.storage.db import Database
from argus.ui.auth import GuardMiddleware
from argus.ui.present import one_line_why, present, present_finding, present_live, waterfall
from argus.ui.present_integrity import present_integrity

_HERE = Path(__file__).parent
PAGE_SIZE = 50
_CURSOR = re.compile(r"^(\d{1,20})_([0-9a-f]{32})$")


def parse_cursor(value: str | None) -> tuple[int, str] | None:
    m = _CURSOR.match(value or "")
    return (int(m.group(1)), m.group(2)) if m else None


def create_admin_app(cfg: Config, db: Database, tracker: Tracker, token: str,
                     integrity: IntegrityService) -> Starlette:
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    proxy_url = f"http://{cfg.proxy_host}:{cfg.proxy_port}"

    def nav_context(active: str) -> dict[str, object]:
        return {"upstream": cfg.upstream_url, "active_nav": active,
                "integrity_issues": db.get_integrity_warning_count()}

    async def requests_page(request: Request) -> Response:
        q = request.query_params
        problems = q.get("problems") == "1"
        cursor = parse_cursor(q.get("before"))
        rows = await asyncio.to_thread(db.list_requests, PAGE_SIZE + 1, cursor, problems)
        more = len(rows) > PAGE_SIZE
        rows = rows[:PAGE_SIZE]
        next_cursor = f"{rows[-1]['started_wall_ns']}_{rows[-1]['id']}" if more else None
        total = await asyncio.to_thread(db.count_requests)
        running = sum(1 for t in tracker.snapshot() if t.is_inference)
        return templates.TemplateResponse(request, "requests.html", {
            **nav_context("requests"),
            "rows": [present(r) for r in rows],
            "problems": problems,
            "cursor": cursor is not None,
            "next_cursor": next_cursor,
            "auto_refresh": cursor is None and q.get("pause") != "1",
            "paused": q.get("pause") == "1",
            "running": running,
            "total": total,
            "proxy_url": proxy_url,
        })

    async def now_page(request: Request) -> Response:
        n = now_ns()
        live = sorted(
            (present_live(t, n) for t in tracker.snapshot() if t.is_inference),
            key=lambda c: c["elapsed_ns"], reverse=True,
        )
        recent = []
        for row in await asyncio.to_thread(db.list_requests, 10):
            baseline_tps = baseline_n = None
            if row.get("model"):
                baseline_tps, baseline_n = await asyncio.to_thread(
                    db.baseline_tokens_per_second, row["model"])
            diagnosis = explain(row, baseline_tps=baseline_tps, baseline_n=baseline_n,
                               upstream_url=cfg.upstream_url)
            p = present(row)
            recent.append({**p, "why": one_line_why(row, diagnosis)})
        return templates.TemplateResponse(request, "now.html", {
            **nav_context("now"),
            "live": live,
            "recent": recent,
            "proxy_url": proxy_url,
        })

    async def request_detail_page(request: Request) -> Response:
        row = await asyncio.to_thread(db.get_request, request.path_params["id"])
        if row is None:
            return templates.TemplateResponse(request, "not_found.html",
                                              nav_context("requests"), status_code=404)
        baseline_tps = baseline_n = None
        if row.get("model"):
            baseline_tps, baseline_n = await asyncio.to_thread(
                db.baseline_tokens_per_second, row["model"])
        diagnosis = explain(row, baseline_tps=baseline_tps, baseline_n=baseline_n,
                           upstream_url=cfg.upstream_url)
        return templates.TemplateResponse(request, "detail.html", {
            **nav_context("requests"),
            "row": present(row),
            "waterfall": waterfall(diagnosis.breakdown),
            "primary": present_finding(diagnosis.primary) if diagnosis.primary else None,
            "also": [present_finding(f) for f in diagnosis.also],
        })

    async def integrity_page(request: Request) -> Response:
        q = request.query_params
        result = await asyncio.to_thread(integrity.latest)
        view = present_integrity(result, scanning=integrity.scanning, now_wall_ns=time.time_ns(),
                                 msg=q.get("msg"),
                                 msg_count=int(q["count"]) if "count" in q else None)
        return templates.TemplateResponse(request, "integrity.html",
                                          {**nav_context("integrity"), **view})

    def _redirect(msg: str, count: int | None = None) -> Response:
        url = f"/integrity?msg={msg}" + (f"&count={count}" if count is not None else "")
        return RedirectResponse(url, status_code=303)

    async def integrity_scan(request: Request) -> Response:
        force = request.path_params.get("mode") == "force"
        result = await asyncio.to_thread(integrity.scan, force=force)
        return _redirect("busy" if result is None else "scanned")

    async def integrity_approve(request: Request) -> Response:
        form = await request.form()
        ok = await asyncio.to_thread(integrity.approve, str(form.get("name", "")),
                                     str(form.get("digest", "")))
        return _redirect("approved" if ok else "approve_failed")

    async def integrity_approve_all(request: Request) -> Response:
        count = await asyncio.to_thread(integrity.approve_all_new)
        return _redirect("nothing_to_approve" if count == 0 else "approved_all", count)

    async def integrity_forget(request: Request) -> Response:
        form = await request.form()
        ok = await asyncio.to_thread(integrity.forget, str(form.get("name", "")))
        return _redirect("forgotten" if ok else "forget_failed")

    return Starlette(
        routes=[
            Route("/", requests_page),
            Route("/now", now_page),
            Route("/requests", requests_page),
            Route("/requests/{id}", request_detail_page),
            Route("/integrity", integrity_page),
            Route("/integrity/scan", integrity_scan, methods=["POST"]),
            Route("/integrity/scan/{mode}", integrity_scan, methods=["POST"]),
            Route("/integrity/approve", integrity_approve, methods=["POST"]),
            Route("/integrity/approve-all", integrity_approve_all, methods=["POST"]),
            Route("/integrity/forget", integrity_forget, methods=["POST"]),
            Mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static"),
        ],
        middleware=[Middleware(GuardMiddleware, token=token, port=cfg.admin_port)],
    )