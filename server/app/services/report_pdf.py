"""Forensic report as a real PDF (M10 FE-3).

The console already had an "Export report" button, but it handed off to the
browser's print dialog: the operator picks a printer, finds "Save as PDF",
chooses a destination. That is three decisions and a dialog between wanting a
report and having one, and nothing about it can be automated — a case cannot
be emailed, attached, or archived by anything but a human with a mouse.

This builds the document server-side instead, so `GET /cases/{id}/report.pdf`
is a file. ReportLab rather than a headless browser because the alternative is
shipping a 300 MB Chromium to render a page the server would also have to be
able to reach; and rather than a client-side canvas capture because that
rasterises everything, losing selectable text — which for a document meant for
journalism or legal use is most of the point.

The cost is that the layout lives here rather than being reused from the React
report. That is the right trade for an archival document: it renders the same
in five years regardless of what the console looks like then.

Editorial rule for everything below: the report states what was measured and
what it means, including when that is "nothing conclusive". A forensic document
that overstates its own certainty is worse than no document.
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm

INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
RULE = colors.HexColor("#cbd5e1")
BAND = colors.HexColor("#f1f5f9")

VERDICT_COLOUR = {
    "authentic":    colors.HexColor("#047857"),
    "manipulated":  colors.HexColor("#b91c1c"),
    "inconclusive": colors.HexColor("#b45309"),
    "processing":   colors.HexColor("#0369a1"),
    "failed":       colors.HexColor("#475569"),
}

VERDICT_MEANING = {
    "authentic": "No evidence of synthetic generation or manipulation was found "
                 "by the models that ran. This is not proof of authenticity.",
    "manipulated": "The ensemble found evidence consistent with synthetic "
                   "generation or manipulation.",
    "inconclusive": "The analysis did not produce a usable signal. This is not "
                    "a finding of authenticity — it means the question was not "
                    "answered.",
    "processing": "Analysis had not finished when this report was generated.",
    "failed": "Analysis did not complete. No conclusion can be drawn.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────

def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=17, leading=21, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontName="Helvetica",
                              fontSize=8.5, leading=12, textColor=MUTED),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=10.5, leading=13, textColor=INK,
                             spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["Normal"], fontName="Helvetica",
                               fontSize=9, leading=13, textColor=INK),
        "small": ParagraphStyle("sm", parent=base["Normal"], fontName="Helvetica",
                                fontSize=7.8, leading=11, textColor=MUTED),
        "mono": ParagraphStyle("m", parent=base["Normal"], fontName="Courier",
                               fontSize=8, leading=11, textColor=INK),
        "caption": ParagraphStyle("c", parent=base["Normal"], fontName="Helvetica-Oblique",
                                  fontSize=7.8, leading=11, textColor=MUTED),
    }


def _esc(value) -> str:
    """Paragraph text is parsed as mini-HTML, so anything user-supplied — a
    case title, a filename — has to be escaped or an stray '&' aborts the
    render."""
    return (str(value if value is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _local_file(url: str | None, directory: Path) -> Path | None:
    """Resolve a served URL back to the file on disk.

    The stored URLs are absolute (http://host/uploads/x.jpg) but the bytes are
    local. Fetching our own HTTP endpoint to build a PDF would be absurd, and
    would break the moment BASE_URL points somewhere this process cannot reach.
    Only the basename is used, so a URL cannot walk out of the directory.
    """
    if not url:
        return None
    name = Path(url.split("?")[0]).name
    if not name or name in (".", ".."):
        return None
    candidate = directory / name
    return candidate if candidate.is_file() else None


def _fit(path: Path, max_w: float, max_h: float) -> Image | None:
    """An Image flowable scaled to fit, preserving aspect ratio."""
    try:
        iw, ih = ImageReader(str(path)).getSize()
        if not iw or not ih:
            return None
        scale = min(max_w / iw, max_h / ih)
        return Image(str(path), width=iw * scale, height=ih * scale)
    except Exception:
        # A corrupt or unreadable image must not take the whole report down;
        # the section simply renders without it.
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Page furniture
# ─────────────────────────────────────────────────────────────────────────────

def _decorate(canvas, doc, case_id: str, generated: str):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 6 * mm, "DEEP TRUTH · FORENSIC ANALYSIS REPORT")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 6 * mm, case_id)

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 4 * mm, PAGE_W - MARGIN, PAGE_H - MARGIN + 4 * mm)
    canvas.line(MARGIN, MARGIN - 4 * mm, PAGE_W - MARGIN, MARGIN - 4 * mm)

    canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN, MARGIN - 8 * mm, f"Generated {generated}")
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 8 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────

def _kv_table(rows: list[tuple[str, str]], st: dict, width: float) -> Table:
    data = [[Paragraph(k, st["small"]), Paragraph(_esc(v), st["body"])] for k, v in rows]
    t = Table(data, colWidths=[width * 0.32, width * 0.68])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
    ]))
    return t


def _verdict_block(case: dict, summary: dict | None, st: dict, width: float) -> Table:
    status = case.get("status", "processing")
    colour = VERDICT_COLOUR.get(status, MUTED)
    risk = case.get("risk_score") or 0.0

    conf = None
    if summary:
        conf = summary.get("confidence")

    left = [
        Paragraph(f'<font color="{colour.hexval()}"><b>{status.upper()}</b></font>',
                  ParagraphStyle("v", parent=st["title"], fontSize=20, leading=23)),
        Spacer(1, 2),
        Paragraph(VERDICT_MEANING.get(status, ""), st["small"]),
    ]
    right = [
        Paragraph("SYNTHETIC LIKELIHOOD", st["small"]),
        Paragraph(f"<b>{risk:.1f}%</b>",
                  ParagraphStyle("r", parent=st["title"], fontSize=20, leading=23)),
        Paragraph(
            "Engine confidence "
            + (f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else "—"),
            st["small"]),
    ]

    t = Table([[left, right]], colWidths=[width * 0.62, width * 0.38])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colour),
    ]))
    return t


def _checkpoint_table(rows: list[dict], st: dict, width: float) -> Table | None:
    """Every model that voted, with its own score.

    Included in full rather than summarised because the fused number is a
    weighted opinion, and a reader assessing this report needs to see the
    spread behind it — six models at 0.5 and six at 0.9 average the same as
    twelve at 0.7 and mean something very different.
    """
    if not rows:
        return None

    data = [[Paragraph("<b>Model</b>", st["small"]),
             Paragraph("<b>Trained on</b>", st["small"]),
             Paragraph("<b>P(synthetic)</b>", st["small"]),
             Paragraph("<b>Reading</b>", st["small"])]]

    for r in sorted(rows, key=lambda x: -(x.get("confidence") or 0)):
        d = r.get("details") or {}
        data.append([
            Paragraph(_esc(r.get("model_name")), st["mono"]),
            Paragraph(_esc(d.get("label_text") or r.get("model_name")), st["body"]),
            Paragraph(f"{(r.get('confidence') or 0):.2f}%", st["mono"]),
            Paragraph(_esc(r.get("label", "")), st["small"]),
        ])

    t = Table(data, colWidths=[width * 0.26, width * 0.36, width * 0.18, width * 0.20],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
    ]))
    return t


def _artifact_section(summary: dict | None, st: dict, width: float) -> list:
    """The map, plus the sentence that stops it being over-read.

    `localised` decides the wording. A Grad-CAM map always exists; whether it
    is a statement about a *region* depends on whether the relevance actually
    concentrates. Printing a diffuse map under the heading "manipulated region"
    would put a claim in a legal document that the data does not support.
    """
    am = (summary or {}).get("artifact_map")
    if not am or not am.get("url"):
        return []

    path = _local_file(am.get("url"), ARTIFACT_DIR)
    if path is None:
        return []

    img = _fit(path, width, 95 * mm)
    if img is None:
        return []

    localised = am.get("localised") is True
    pct = am.get("concentration")
    pct_txt = f"{pct * 100:.0f}%" if isinstance(pct, (int, float)) else "—"

    if localised:
        caveat = (f"Relevance is concentrated: the strongest tenth of the frame "
                  f"holds {pct_txt} of the total. The highlighted regions carry "
                  f"most of what drove this verdict.")
    else:
        caveat = (f"Relevance is distributed rather than localised (strongest "
                  f"tenth holds {pct_txt}). This is what wholly synthetic media "
                  f"looks like — the signal is texture across the whole frame, "
                  f"not one edited region. Read this as model attention, not as "
                  f"a marked-up area.")

    out = [
        Paragraph("Artifact map", st["h2"]),
        img,
        Spacer(1, 3),
        Paragraph(caveat, st["small"]),
    ]

    profile = am.get("temporal_profile")
    if profile:
        out.append(Spacer(1, 3))
        out.append(Paragraph(
            "Per-segment relevance is shown on the contact sheet. Temporal "
            "localisation has not been validated against a clip with a known "
            "edit timestamp and should not be relied on; the spatial heat is "
            "the supported finding.", st["small"]))

    out.append(Spacer(1, 3))
    out.append(Paragraph(
        f"Method: {_esc(am.get('method', 'grad-cam'))}. "
        f"Fused from {len(am.get('contributors') or [])} model(s), each weighted "
        f"by how strongly it called the media synthetic. This is a saliency "
        f"approximation at {am.get('grid', 14)}×{am.get('grid', 14)} "
        f"resolution, not a segmentation, and it explains the model rather than "
        f"the image.", st["caption"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_report(case: dict, results: list[dict], job: dict | None = None) -> bytes:
    """Render one case to PDF bytes.

    `case` is the raw DB row; `results` the analysis_results rows with
    `details` already decoded.
    """
    st = _styles()
    width = PAGE_W - 2 * MARGIN
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    case_id = case.get("case_id", "—")

    summary = None
    checkpoints: list[dict] = []
    secondary: list[dict] = []
    for r in results:
        d = r.get("details") or {}
        tier = d.get("tier")
        if tier == "summary":
            summary = d
        elif tier == "checkpoint":
            checkpoints.append(r)
        elif tier == "secondary":
            secondary.append(r)

    story: list = []

    # ── Identity ────────────────────────────────────────────────────────────
    story.append(Paragraph(_esc(case.get("title") or "Untitled case"), st["title"]))
    story.append(Paragraph(
        f"Case {_esc(case_id)} &middot; {_esc(case.get('media_type', '')).upper()} "
        f"&middot; opened {_esc(case.get('created_at'))}", st["sub"]))
    story.append(Spacer(1, 8))

    # ── Verdict ─────────────────────────────────────────────────────────────
    story.append(_verdict_block(case, summary, st, width))
    story.append(Spacer(1, 4))

    if summary and (summary.get("confidence") or 0) <= 0:
        story.append(Paragraph(
            "<b>No engine contributed a usable signal.</b> A zero-confidence "
            "result is the pipeline's way of saying it has nothing to report, "
            "and it is never treated as a finding either way.", st["small"]))
        story.append(Spacer(1, 4))

    if summary and summary.get("rationale"):
        story.append(Paragraph("How this was decided", st["h2"]))
        story.append(Paragraph(_esc(summary["rationale"]), st["body"]))

    # ── Evidence ────────────────────────────────────────────────────────────
    evidence = _local_file(case.get("file_url"), UPLOAD_DIR)
    if evidence and evidence.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp",
                                                ".bmp", ".tiff"):
        img = _fit(evidence, width * 0.55, 70 * mm)
        if img is not None:
            story.append(Paragraph("Evidence", st["h2"]))
            story.append(img)
            story.append(Spacer(1, 2))
            story.append(Paragraph(_esc(case.get("file_name") or ""), st["caption"]))

    # ── Artifact map ────────────────────────────────────────────────────────
    story.extend(_artifact_section(summary, st, width))

    # ── Model breakdown ─────────────────────────────────────────────────────
    table = _checkpoint_table(checkpoints, st, width)
    if table is not None:
        story.append(Paragraph("Model breakdown", st["h2"]))
        story.append(table)
        story.append(Spacer(1, 3))
        story.append(Paragraph(
            "Each model was fine-tuned on a different corpus and votes "
            "independently. The headline figure is their weighted fusion, not "
            "an average.", st["caption"]))

    skipped = (summary or {}).get("skipped") or []
    if skipped:
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Not every model ran. " + "; ".join(
                f"{_esc(s.get('slug'))}: {_esc(s.get('reason'))}" for s in skipped
            ), st["small"]))

    # ── Supplementary signals ───────────────────────────────────────────────
    if secondary:
        story.append(Paragraph("Supplementary analysis", st["h2"]))
        for r in secondary:
            d = r.get("details") or {}
            note = d.get("note") or d.get("rationale") or ""
            story.append(Paragraph(
                f"<b>{_esc(r.get('model_name'))}</b> — {_esc(note)}", st["small"]))
        story.append(Spacer(1, 2))
        story.append(Paragraph(
            "Supplementary signals are recorded as evidence but never override "
            "the primary verdict.", st["caption"]))

    # ── Provenance ──────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Chain of custody", st["h2"]))

    size = case.get("file_size")
    rows = [
        ("Case ID", case_id),
        ("Internal ID", case.get("id")),
        ("File name", case.get("file_name") or "—"),
        ("File size", f"{size:,} bytes" if isinstance(size, int) else "—"),
        ("Modality", (case.get("media_type") or "").upper()),
        ("Submitted by", case.get("user_id") or "—"),
        ("Opened", case.get("created_at")),
        ("Last updated", case.get("updated_at")),
    ]
    if job:
        if job.get("contentHash"):
            rows.append(("Content SHA-256", job["contentHash"]))
        if job.get("sourceUrl"):
            rows.append(("Source URL", job["sourceUrl"]))
        if job.get("worker"):
            rows.append(("Analysed on", job["worker"]))
        if job.get("cacheHit"):
            rows.append(("Result origin",
                         "Replayed from cache — identical content had been "
                         "analysed before"))
    story.append(_kv_table(rows, st, width))

    if summary:
        story.append(Paragraph("Technical detail", st["h2"]))
        tech = [
            ("Model version", summary.get("model_version") or "—"),
            ("Fusion policy", summary.get("policy") or "—"),
            ("Decision threshold", str(summary.get("threshold", "—"))),
            ("Modality analysed", summary.get("modality") or "—"),
        ]
        if summary.get("face_detected") is not None:
            tech.append(("Face detected", "yes" if summary["face_detected"] else "no"))
        story.append(_kv_table(tech, st, width))

    # ── Limits ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Limitations of this analysis", st["h2"]))
    for line in (
        "Detection relies on high-frequency artefacts. Heavy recompression, as "
        "applied by messaging apps and social platforms, can blur those "
        "artefacts and reduce accuracy.",
        "An <b>authentic</b> result means no evidence of manipulation was found "
        "by the models that ran. It is not proof that none exists.",
        "An <b>inconclusive</b> result means the question was not answered, and "
        "must not be read as either finding.",
        "Models are fine-tuned on specific corpora and are less reliable on "
        "material unlike their training data.",
        "This report records an automated analysis. It is a technical finding, "
        "not an expert opinion, and does not by itself establish provenance.",
    ):
        story.append(Paragraph(f"&bull; {line}", st["small"]))
        story.append(Spacer(1, 2))

    # ── Build ───────────────────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"Deep Truth forensic report — {case_id}",
        author="Deep Truth", subject="Synthetic media analysis",
    )
    frame = Frame(MARGIN, MARGIN, width, PAGE_H - 2 * MARGIN, id="body")
    doc.addPageTemplates([PageTemplate(
        id="report", frames=[frame],
        onPage=lambda c, d: _decorate(c, d, case_id, generated),
    )])
    doc.build(story)
    return buffer.getvalue()
