"""Provider-aware TTS client factory.

Test modules and Phase 0 construct the TTS client through this factory rather
than instantiating a client class directly. Dispatch keys off
config.tts.provider (set in eval/config.yaml).

Lazy imports keep each provider's heavy deps off the critical path for the
other — riva.client isn't needed for a Kugel run, and websocket-client isn't
needed for a Riva run.
"""
from __future__ import annotations

from eval.config import Config


def build_tts_client(config: Config):
    """Return a TTS client matching config.tts.provider.

    Returns an `eval.tts.kugel_client.KugelTTSClient` (KugelAudio) or an
    `eval.tts.client.TTSClient` (Magpie on AI Core — dormant path). Both expose
    the same public surface — `synthesize_batch`, `synthesize_stream`,
    `synthesize_batch_rest`, `synthesize_stream_rest`, `save_synthesis`,
    `bytes_to_wav` — so test modules need not branch.
    """
    provider = (getattr(config.tts, "provider", None) or "kugel").lower()
    if provider == "kugel":
        from eval.tts.kugel_client import KugelTTSClient
        return KugelTTSClient(config)
    if provider == "riva":
        from eval.tts.client import TTSClient
        return TTSClient(config)
    raise ValueError(
        f"Unknown tts.provider {provider!r}; expected 'kugel' or 'riva'"
    )
