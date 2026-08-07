"""Top-level orchestrator.

Pipeline.analyze(path)         direct synchronous call
make_queue(pipeline)           wraps the pipeline behind a Task abstraction

For each call:
  1. ffprobe inspection -> MediaInfo (kind, duration, ...)
  2. compute media_key from file bytes
  3. for each modality wanted: preprocessor.run() then inferencer.predict()
  4. assemble AnalyzeResult
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .demux import probe, MediaInfo
from .registry import Registry
from .storage import CacheStore, CacheRecord, media_key_from_file
from .task_queue import InMemoryQueue, Task
from .inferencers.base import InferenceResult

log = logging.getLogger(__name__)


@dataclass
class AnalyzeResult:
    media_key: str
    media_info: dict
    inferences: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(self, registry: Registry | None = None,
                 store: CacheStore | None = None):
        self.store = store or CacheStore()
        self.registry = registry or Registry(store=self.store)

    def analyze(self, path: str | Path, *,
                media_kind_hint: str | None = None,
                modalities: list[str] | None = None,
                threshold: float = 0.5,
                force: bool = False) -> AnalyzeResult:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        info = probe(path)
        kind = media_kind_hint or info.kind
        log.info(f"analyze: {path.name}  kind={kind}  "
                 f"duration={info.duration_seconds}  "
                 f"has_audio={info.has_audio}")

        media_key = media_key_from_file(path)
        rec = self.store.load(media_key) or CacheRecord(
            media_key=media_key,
            source_path=str(path),
            media_kind=kind,
            duration_seconds=info.duration_seconds,
        )
        self.store.save(rec)

        wanted = modalities or self._default_modalities(info)
        result = AnalyzeResult(media_key=media_key,
                               media_info=_info_to_dict(info))

        for modality in wanted:
            pre = self.registry.preprocessor_for(modality)
            inf = self.registry.inferencer_for(modality)
            if not pre or not inf:
                result.warnings.append(f"no handler for modality '{modality}'")
                continue

            try:
                preprocessed = pre.run(path, media_key, rec, force=force)
            except NotImplementedError as e:
                result.warnings.append(f"{modality} preprocessor not ready: {e}")
                continue
            except Exception as e:
                result.warnings.append(f"{modality} preprocessing failed: {e}")
                log.exception(f"{modality} preprocessing failed")
                continue

            try:
                ir: InferenceResult = inf.predict(media_key, preprocessed,
                                                  threshold=threshold)
                result.inferences[modality] = _ir_to_dict(ir)
            except NotImplementedError as e:
                result.warnings.append(f"{modality} inferencer not ready: {e}")
            except Exception as e:
                result.warnings.append(f"{modality} inference failed: {e}")
                log.exception(f"{modality} inference failed")

        return result

    def _default_modalities(self, info: MediaInfo) -> list[str]:
        # When the audio model lands, change the video branch to ["video", "audio"]
        # for files where info.has_audio. The plumbing is already there.
        if info.kind == "video":
            return ["video"]
        if info.kind == "audio":
            return ["audio"]
        if info.kind == "image":
            return ["image"]
        return []


def _info_to_dict(info: MediaInfo) -> dict:
    d = asdict(info)
    d["path"] = str(d["path"])
    return d


def _ir_to_dict(ir: InferenceResult) -> dict:
    return {
        "modality":    ir.modality,
        "trust_score": ir.trust_score,
        "verdict":     ir.verdict,
        "confidence":  ir.confidence,
        "per_model":   ir.per_model,
        "rationale":   ir.rationale,
        **ir.extra,
    }


def make_queue(pipeline: Pipeline, async_workers: int = 0) -> InMemoryQueue:
    """Wrap a Pipeline behind the Task interface. async_workers=0 runs inline."""
    def runner(task: Task):
        task.stage = "analyzing"
        opts = task.options or {}
        result = pipeline.analyze(
            task.media_path,
            media_kind_hint=task.media_kind_hint,
            modalities=opts.get("modalities"),
            threshold=opts.get("threshold", 0.5),
            force=opts.get("force", False),
        )
        task.result = asdict(result)

    return InMemoryQueue(runner=runner, async_workers=async_workers)
