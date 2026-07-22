"""FastAPI application for the existing-embedding Eyck vertical slice."""

from __future__ import annotations

import hmac
import inspect
import json
import secrets
from collections.abc import Callable, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .discovery import ProjectSpec, discover_projects
from .models import (
    AnnotationDetail,
    AnnotationIndex,
    AnnotationPoints,
    AnnotationSummary,
    ExportResponse,
    FeatureDescriptor,
    FeaturesPayload,
    LabelDescriptor,
    MembershipDraftPut,
    MembershipPayload,
    NamedInput,
    RestartResponse,
)
from .project import EyckProject, ProjectError, RevisionConflict

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class RequestLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, send: Send) -> None:
        payload = json.dumps({"detail": "request body too large"}).encode("utf-8")
        await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]})
        await send({"type": "http.response.body", "body": payload})


def _project_or_404(projects: dict[str, EyckProject], project_id: str) -> EyckProject:
    try:
        return projects[project_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc


def create_app(
    roots_or_specs: Sequence[str | Path] | dict[str, ProjectSpec],
    *,
    allowed_hosts: Sequence[str] = ("127.0.0.1", "localhost", "testserver"),
    allowed_origins: Sequence[str] = ("http://127.0.0.1:8000", "http://localhost:8000", "http://testserver"),
    max_request_bytes: int = 64 * 1024 * 1024,
    restart_callback: Callable[[], Any] | None = None,
    proxy_prefix: str = "",
) -> FastAPI:
    specs = roots_or_specs if isinstance(roots_or_specs, dict) else discover_projects(list(roots_or_specs))
    projects = {project_id: EyckProject(spec) for project_id, spec in specs.items()}
    csrf_token = secrets.token_urlsafe(32)
    origin_set = {origin.rstrip("/") for origin in allowed_origins}
    normalized_prefix = "/" + proxy_prefix.strip("/") if proxy_prefix.strip("/") else ""

    app = FastAPI(title="Eyck", version="1", docs_url=None, redoc_url=None)
    app.state.projects = projects
    app.state.csrf_token = csrf_token
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    app.add_middleware(RequestLimitMiddleware, max_bytes=max_request_bytes)

    @app.middleware("http")
    async def mutation_security(request: Request, call_next: Callable[..., Any]):
        if request.method in UNSAFE_METHODS:
            origin = request.headers.get("origin", "").rstrip("/")
            if not origin or origin not in origin_set:
                return JSONResponse(status_code=403, content={"detail": "untrusted request origin"})
            supplied = request.headers.get("x-eyck-csrf", "")
            if not hmac.compare_digest(supplied, csrf_token):
                return JSONResponse(status_code=403, content={"detail": "invalid CSRF token"})
        return await call_next(request)

    @app.exception_handler(RevisionConflict)
    async def revision_error(_request: Request, exc: RevisionConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ProjectError)
    async def project_error(_request: Request, exc: ProjectError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    api_root = "/api/v1/annotations"
    public_api_root = f"{normalized_prefix}{api_root}"

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(api_root, response_model=AnnotationIndex)
    def project_index() -> AnnotationIndex:
        return AnnotationIndex(
            projects=[
                AnnotationSummary(
                    project_id=projects[key].spec.id,
                    title=projects[key].spec.title,
                    description=projects[key].spec.description,
                    n_observations=projects[key].n_obs,
                    n_features=projects[key].n_vars,
                )
                for key in sorted(projects)
            ],
            csrf_token=csrf_token,
            restart_available=restart_callback is not None,
        )

    @app.get(f"{api_root}/{{project_id}}", response_model=AnnotationDetail)
    def project_detail(project_id: str) -> AnnotationDetail:
        project = _project_or_404(projects, project_id)
        base = f"{public_api_root}/{project_id}"
        first_embedding = project.spec.embeddings[0]
        return AnnotationDetail(
            project_id=project.spec.id,
            title=project.spec.title,
            description=project.spec.description,
            dataset_id=project.spec.dataset_id,
            source_owner=project.spec.source_owner,
            n_observations=project.n_obs,
            n_features=project.n_vars,
            expression_layer=project.spec.expression_layer,
            labels=[
                LabelDescriptor(
                    label_id=label.id,
                    display_name=label.name,
                    parents=label.parent_ids,
                    description=label.description,
                    ontology_ids=label.ontology_ids,
                )
                for label in project.spec.labels.labels
            ],
            metadata_columns=list(project.spec.metadata_columns),
            embeddings=[NamedInput(id=item.id, obsm=item.obsm, dimensions=project.input_dimensions[item.id]) for item in project.spec.embeddings],
            modalities=[NamedInput(id=item.id, obsm=item.obsm, dimensions=project.input_dimensions[item.id]) for item in project.spec.modalities],
            identities=project.identities,
            points_url=f"{base}/points?embedding={first_embedding.id}",
            features_url=f"{base}/features",
            memberships_url=f"{base}/memberships",
            export_url=f"{base}/export",
            csrf_token=csrf_token,
            restart_available=restart_callback is not None,
        )

    @app.get(f"{api_root}/{{project_id}}/points", response_model=AnnotationPoints)
    def points(project_id: str, embedding: str = Query(...)) -> AnnotationPoints:
        return _project_or_404(projects, project_id).annotation_points(embedding)

    @app.get(f"{api_root}/{{project_id}}/features", response_model=FeaturesPayload)
    def feature_search(project_id: str, q: str = "", limit: int = Query(30, ge=1, le=100)) -> FeaturesPayload:
        hits = _project_or_404(projects, project_id).feature_search(q, limit).features
        return FeaturesPayload(
            features=[
                FeatureDescriptor(
                    feature_index=hit.index,
                    feature_id=hit.id,
                    feature_symbol=_project_or_404(projects, project_id).feature_symbols[hit.index],
                )
                for hit in hits
            ]
        )

    @app.get(f"{api_root}/{{project_id}}/features/{{feature_index}}/values")
    def feature_values(project_id: str, feature_index: int) -> dict[str, list[float | None]]:
        return {"values": _project_or_404(projects, project_id).feature_values_by_index(feature_index)}

    @app.get(f"{api_root}/{{project_id}}/memberships", response_model=MembershipPayload)
    def memberships(project_id: str) -> MembershipPayload:
        document = _project_or_404(projects, project_id).current_memberships()
        return MembershipPayload(
            revision=document.revision,
            origin=document.source,
            rows=document.rows,
            support_observation_ids=sorted({row.observation_id for row in document.rows}),
        )

    @app.put(f"{api_root}/{{project_id}}/memberships", response_model=MembershipPayload)
    def put_memberships(
        project_id: str,
        body: MembershipDraftPut,
        if_match: str = Header(..., alias="If-Match"),
    ) -> MembershipPayload:
        project = _project_or_404(projects, project_id)
        unknown_support = set(body.support_observation_ids) - set(project.observation_ids)
        if unknown_support:
            raise ProjectError(f"unknown support observation IDs: {sorted(unknown_support)}")
        document = project.put_memberships(if_match.removeprefix("W/").strip('"'), body.rows)
        return MembershipPayload(
            revision=document.revision,
            origin=document.source,
            rows=document.rows,
            support_observation_ids=sorted(set(body.support_observation_ids) | {row.observation_id for row in document.rows}),
        )

    @app.post(f"{api_root}/{{project_id}}/export", response_model=ExportResponse)
    def export(project_id: str, if_match: str = Header(..., alias="If-Match")) -> ExportResponse:
        result = _project_or_404(projects, project_id).export(
            if_match.removeprefix("W/").strip('"')
        )
        return result.model_copy(
            update={"report_url": f"{public_api_root}/{project_id}/export/report"}
        )

    @app.get(f"{api_root}/{{project_id}}/export/report")
    def export_report(project_id: str) -> dict[str, Any]:
        return _project_or_404(projects, project_id).export_report()

    @app.post("/api/v1/restart", response_model=RestartResponse)
    async def restart() -> RestartResponse:
        if restart_callback is None:
            raise HTTPException(status_code=409, detail="restart is disabled")
        result = restart_callback()
        if inspect.isawaitable(result):
            await result
        return RestartResponse(restarting=True)

    built_assets = files("eyck").joinpath("web", "assets")
    if not built_assets.joinpath("index.html").is_file():
        raise RuntimeError("Eyck frontend assets are not installed")
    asset_root = built_assets

    @app.get("/assets/polyptich-ui.css", include_in_schema=False)
    def polyptich_stylesheet() -> FileResponse:
        resource = files("polyptich.www").joinpath("static", "polyptich-ui.css")
        if not resource.is_file():
            raise HTTPException(status_code=404, detail="Polyptich UI stylesheet is unavailable")
        return FileResponse(str(resource))

    app.mount("/assets", StaticFiles(directory=str(asset_root)), name="assets")
    index_path = asset_root.joinpath("index.html")

    def frontend_html() -> HTMLResponse:
        html = index_path.read_text(encoding="utf-8").replace(
            "__BASE_PATH__", normalized_prefix
        )
        if normalized_prefix:
            html = html.replace('"/assets/', f'"{normalized_prefix}/assets/')
        return HTMLResponse(html)

    @app.get("/", include_in_schema=False)
    def frontend_index() -> HTMLResponse:
        return frontend_html()

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend_route(frontend_path: str) -> HTMLResponse:
        if frontend_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        return frontend_html()

    return app
