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

    # probe, preprocess, then infer for each modality; failures become warnings
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

        # same bytes always map to the same cache folder
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

    # a video with sound gets both a video and an audio pass
    def _default_modalities(self, info: MediaInfo) -> list[str]:
        if info.kind == "video":
            return ["video", "audio"] if info.has_audio else ["video"]
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
