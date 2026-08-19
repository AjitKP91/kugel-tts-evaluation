from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RivaConfig:
    """Retained for symmetry with the multi-provider harness. Unused for
    provider=kugel — the Riva TTSClient path (eval/tts/client.py) is dormant."""

    use_ssl: bool = True
    auth_token_env: str = "AICORE_BEARER_TOKEN"

    @property
    def auth_token(self) -> str:
        token = os.environ.get(self.auth_token_env, "")
        if not token:
            raise EnvironmentError(
                f"Set ${self.auth_token_env} with a valid Bearer token"
            )
        return token


@dataclass
class KugelTTSConfig:
    """Configuration for Kugel-TTS via the KugelAudio HTTP + WebSocket API.

    Auth is a Bearer API key read from the ${api_key_env} environment variable
    (default KUGELAUDIO_API_KEY); no secret is stored in the YAML. The same key
    is passed as the `api_key` query parameter on the WebSocket URL.

    Kugel is REST + WebSocket only (no gRPC). The batch REST endpoint returns
    raw PCM16 LE audio; the WebSocket endpoint streams base64 PCM16 chunks.
    """

    api_key_env: str = "KUGELAUDIO_API_KEY"
    model_id: str = "kugel-3"
    voice_id: int | str = 1071
    language: str | None = None          # ISO 639-1; auto-detected if None
    cfg_scale: float = 2.0               # 1.2–2.5, higher = more expressive
    temperature: float = 0.4             # 0.0–1.0
    speed: float = 1.0                   # 0.8–1.2, pitch-preserving
    normalize: bool = True
    max_new_tokens: int = 2048
    sample_rate: int = 24000             # native Kugel rate; 8k/16k/22.05k/44.1k also allowed
    rest_endpoint: str = "https://api.kugelaudio.com/v1/tts/generate"
    ws_endpoint: str = "wss://api.kugelaudio.com/ws/tts"
    voices_endpoint: str = "https://api.kugelaudio.com/v1/voices"
    request_timeout_s: int = 120
    # Optional matched-speaker reference set for Test 2.4 (MCD/PESQ/STOI).
    # Populated by scripts/clone_reference_voice.py: it clones a single-speaker
    # corpus (LJSpeech) into a Kugel voice and writes the ground-truth reference
    # WAVs + a manifest here. When set, Test 2.4 synthesizes with the cloned
    # voice (below) and compares against these matched-speaker references
    # instead of self-skipping.
    reference_set_dir: str | None = None
    # voice_id of the cloned reference speaker (returned by POST /v1/voices).
    # Used by Test 2.4 only; the main suite still uses `voice_id` above.
    reference_voice_id: int | str | None = None

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise EnvironmentError(
                f"Set ${self.api_key_env} with a valid KugelAudio API key"
            )
        return key


@dataclass
class TTSConfig:
    # Provider switch: "kugel" (KugelAudio, default here) or "riva" (Magpie on
    # AI Core — dormant path retained for symmetry). The factory in
    # eval/tts/client_factory.py dispatches on this.
    provider: str = "kugel"
    # ── Riva/Magpie fields (used only when provider=riva) ────────────────
    grpc_uri: str = ""
    model_name: str = ""
    voice_name: str = ""
    rest_endpoint: str = ""
    language_code: str = "en-US"
    auth_header: str = "Authorization"
    request_timeout_s: int = 60
    sample_rate: int = 22050
    max_sequence_tokens: int = 400
    # ── Kugel sub-block (used when provider=kugel) ───────────────────────
    kugel: KugelTTSConfig | None = None


@dataclass
class EvalConfig:
    tts_concurrency_levels: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    bootstrap_n: int = 1000
    results_dir: str = "results/"
    log_level: str = "INFO"
    data_dir: str = "eval/data/"


@dataclass
class Config:
    tts: TTSConfig
    evaluation: EvalConfig
    riva: RivaConfig | None = None

    @property
    def results_path(self) -> Path:
        return Path(self.evaluation.results_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.evaluation.data_dir)


_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _resolve_env(value):
    """Recursively expand ${ENV_VAR} substrings in strings/lists/dicts.

    Unset env vars are left as the literal ${NAME} so the caller notices the
    misconfiguration rather than getting a silent empty string.
    """
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    path = Path(path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Pop the optional Kugel sub-block so we can build it separately and pass
    # the rest into TTSConfig unchanged. Env-var expansion runs on the kugel
    # block only (the sole place ${...} substitution is used today).
    tts_raw = dict(raw["tts"])
    kugel_raw = tts_raw.pop("kugel", None)
    kugel_cfg = KugelTTSConfig(**_resolve_env(kugel_raw)) if kugel_raw else None
    tts_cfg = TTSConfig(**tts_raw, kugel=kugel_cfg)

    eval_cfg = EvalConfig(**raw.get("evaluation", {}))

    # Provider-scoped fallback for results_dir: when the operator hasn't passed
    # --results-dir and the YAML still carries the bare default, route output to
    # results/<provider>/ so back-to-back runs of different providers don't
    # collide. The canonical path remains results/run-<DD-MM-YY>/<provider>/
    # set by start_eval.sh via --results-dir; this branch is for ad-hoc runs.
    if eval_cfg.results_dir.rstrip("/") == "results":
        eval_cfg.results_dir = f"results/{tts_cfg.provider}"

    riva_cfg = RivaConfig(**raw["riva"]) if raw.get("riva") else None

    return Config(tts=tts_cfg, evaluation=eval_cfg, riva=riva_cfg)
