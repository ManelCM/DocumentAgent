"""
FastAPI REST endpoint for DocumentAgent.

Start:
    uvicorn src.document_agent.api:app --reload --port 8000

Endpoints
---------
POST   /process          Upload a PDF → returns {job_id}
GET    /jobs/{job_id}    Poll status + full output when done
GET    /jobs/{job_id}/markdown   Download .md export
GET    /jobs/{job_id}/tables     Download tables as .xlsx
GET    /jobs/{job_id}/full_text  Plain extracted text
DELETE /jobs/{job_id}    Remove job from memory
GET    /health           Liveness check
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

app = FastAPI(
    title="DocumentAgent API",
    description="Multimodal PDF → structured JSON, Markdown, tables",
    version="1.0.0",
)

# ── In-memory job store ────────────────────────────────────────────────────────
# For production swap with Redis / a DB. Jobs expire after JOB_TTL_S seconds.

JOB_TTL_S = int(os.getenv("DOCAGENT_JOB_TTL", "3600"))  # 1 hour default

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _job_get(job_id: str) -> Dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job


def _job_set(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _expire_jobs() -> None:
    """Remove jobs older than JOB_TTL_S. Called lazily on each request."""
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items()
                   if now - j.get("created_at", now) > JOB_TTL_S]
        for jid in expired:
            _cleanup_job_files(_jobs.pop(jid))


def _cleanup_job_files(job: Dict) -> None:
    for key in ("pdf_path", "md_path", "xlsx_path"):
        p = job.get(key)
        if p:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
    tables_dir = job.get("tables_dir")
    if tables_dir:
        import shutil
        try:
            shutil.rmtree(tables_dir, ignore_errors=True)
        except Exception:
            pass


# ── Pipeline runner (runs in background thread) ────────────────────────────────

def _run_pipeline(job_id: str, pdf_path: str, max_pages: int) -> None:
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    _job_set(job_id, status="running", started_at=time.time())

    try:
        from .graph import build_graph
        from .exporters.markdown import to_markdown
        from .exporters.tables import export_tables, export_tables_excel
        from .langfuse_exporter import export as lf_export

        app_graph = build_graph().compile()
        result = app_graph.invoke({
            "input_path": pdf_path,
            "status": "init",
            "max_pages": max_pages,
        })

        output  = result.get("output", {})
        trace   = result.get("_trace", {})
        run_id  = result.get("run_id", job_id)

        # Markdown
        tmp_dir   = Path(pdf_path).parent
        md_path   = tmp_dir / f"{job_id}.md"
        md_path.write_text(to_markdown(output), encoding="utf-8")

        # Tables
        tables_dir = tmp_dir / f"{job_id}_tables"
        csvs  = export_tables(output, tables_dir)
        xlsx  = export_tables_excel(output, tmp_dir / f"{job_id}_tables.xlsx")

        # Langfuse (opt-in)
        try:
            lf_export(trace, output, pdf_path, run_id)
        except Exception:
            pass

        _job_set(
            job_id,
            status="done",
            finished_at=time.time(),
            output=output,
            md_path=str(md_path),
            xlsx_path=str(xlsx) if xlsx else None,
            tables_dir=str(tables_dir),
            n_csv_tables=len(csvs),
            run_id=run_id,
        )

    except Exception as exc:
        _job_set(job_id, status="error", error=str(exc), finished_at=time.time())


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "jobs": len(_jobs)}


@app.post("/process", status_code=202)
async def process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to process"),
    max_pages: int = Form(default=0, description="Max pages to process (0 = all)"),
):
    """Upload a PDF and start processing. Returns a job_id to poll."""
    _expire_jobs()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    job_id = str(uuid.uuid4())

    # Save upload to a temp file (kept until job TTL expires)
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf",
        prefix=f"docagent_{job_id}_",
    )
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id":     job_id,
            "status":     "queued",
            "filename":   file.filename,
            "pdf_path":   tmp.name,
            "created_at": time.time(),
            "max_pages":  max_pages,
        }

    background_tasks.add_task(_run_pipeline, job_id, tmp.name, max_pages)

    return {"job_id": job_id, "status": "queued", "filename": file.filename}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Poll job status. Returns full output when status == 'done'."""
    _expire_jobs()
    job = _job_get(job_id)

    response: Dict[str, Any] = {
        "job_id":    job["job_id"],
        "status":    job["status"],
        "filename":  job.get("filename"),
        "created_at": job.get("created_at"),
    }

    if job["status"] == "done":
        out = job.get("output", {})
        response["summary"]    = out.get("summary", {})
        response["warnings"]   = out.get("warnings", [])
        response["run_id"]     = job.get("run_id")
        response["finished_at"] = job.get("finished_at")
        response["exports"] = {
            "markdown":  f"/jobs/{job_id}/markdown",
            "excel":     f"/jobs/{job_id}/tables" if job.get("xlsx_path") else None,
            "full_text": f"/jobs/{job_id}/full_text",
        }
        response["blocks"] = out.get("blocks", [])

    elif job["status"] == "error":
        response["error"] = job.get("error")

    return JSONResponse(content=response)


@app.get("/jobs/{job_id}/markdown")
def get_markdown(job_id: str):
    """Download the Markdown export."""
    job = _job_get(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']!r}")
    md_path = job.get("md_path")
    if not md_path or not Path(md_path).exists():
        raise HTTPException(status_code=404, detail="Markdown file not found")
    stem = Path(job.get("filename", "document")).stem
    return FileResponse(md_path, media_type="text/markdown",
                        filename=f"{stem}.md")


@app.get("/jobs/{job_id}/tables")
def get_tables(job_id: str):
    """Download all tables as an Excel workbook."""
    job = _job_get(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']!r}")
    xlsx_path = job.get("xlsx_path")
    if not xlsx_path or not Path(xlsx_path).exists():
        raise HTTPException(status_code=404, detail="No tables extracted or Excel not available")
    stem = Path(job.get("filename", "document")).stem
    return FileResponse(xlsx_path,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        filename=f"{stem}_tables.xlsx")


@app.get("/jobs/{job_id}/full_text", response_class=PlainTextResponse)
def get_full_text(job_id: str):
    """Return the plain extracted full text."""
    job = _job_get(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']!r}")
    return job.get("output", {}).get("full_text", "")


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    """Remove a job and its temporary files."""
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if job:
        _cleanup_job_files(job)
