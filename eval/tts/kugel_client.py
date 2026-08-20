"""Kugel-TTS client (KugelAudio HTTP batch/stream + WebSocket streaming).

Mirrors eval.tts.client.TTSClient so the test modules and Phase 0 smoke test
work unchanged when config.tts.provider == "kugel".

Transports:
  synthesize_batch          → POST /v1/tts/generate            (raw PCM16 body)
  synthesize_batch_rest     → POST /v1/tts/generate            (same; alias)
  synthesize_stream         → wss://…/ws/tts                   (WebSocket, base64 PCM chunks)
  synthesize_stream_rest    → POST /v1/tts/generate (stream=True, chunked read)

KugelAudio is REST + WebSocket only — there is no gRPC surface. To keep the
existing "grpc vs rest" test axis meaningful, `synthesize_batch` and
`synthesize_stream` map to the WebSocket-eligible path and the REST variants
map to plain HTTP. Audio is raw PCM16 LE at cfg.sample_rate (native 24 kHz).
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

from eval.config import Config
from eval.utils import retry_with_backoff

logger = logging.getLogger("eval.tts.kugel_client")


def _is_permanent_kugel_error(exc: Exception) -> bool:
    """Return True for errors that will never succeed on retry.

    KugelAudio surfaces permanent failures as HTTP 4xx: 404 (invalid
    voice_id/model_id), 400 (empty text, text > 10k chars, bad params), 401/403
    (bad API key). Transient failures (429 rate-limit, 5xx, network) fall
    through to the backoff loop.
    """
    status = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
    if status in (400, 401, 403, 404, 413, 422):
        return True
    msg = str(exc).lower()
    permanent_markers = (
        "at least one non-whitespace",
        "limited to 10,000 characters",
        "invalid voice",
        "invalid model",
        "unauthorized",
    )
    return any(m in msg for m in permanent_markers)


class KugelTTSClient:
    """Drop-in replacement for TTSClient when provider=kugel."""

    def __init__(self, config: Config):
        if config.tts.kugel is None:
            raise RuntimeError(
                "tts.kugel block missing in config — set provider=riva or fill it in"
            )
        self.config = config
        self.tts_cfg = config.tts
        self.cfg = config.tts.kugel
        # Mirror the Magpie attribute name so test modules that reach for
        # `tts_client.sample_rate` keep working unchanged.
        self.sample_rate = self.cfg.sample_rate

    # ------------------------------------------------------------------
    # Request/response helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def _body(self, text: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "text": text,
            "model_id": self.cfg.model_id,
            "voice_id": self.cfg.voice_id,
            "cfg_scale": self.cfg.cfg_scale,
            "temperature": self.cfg.temperature,
            "speed": self.cfg.speed,
            "normalize": self.cfg.normalize,
            "max_new_tokens": self.cfg.max_new_tokens,
            "sample_rate": self.cfg.sample_rate,
        }
        if self.cfg.language:
            body["language"] = self.cfg.language
        return body

    def _duration(self, audio_bytes: bytes) -> float:
        # PCM16 mono: 2 bytes/sample.
        return len(audio_bytes) / (self.sample_rate * 2)

    # ------------------------------------------------------------------
    # Voice cloning — POST /v1/voices (multipart)
    # ------------------------------------------------------------------

    @retry_with_backoff(reraise_if=_is_permanent_kugel_error)
    def clone_voice(
        self,
        reference_paths: list[str | Path],
        name: str,
        sex: str = "female",
        description: str = "",
        category: str = "conversational",
    ) -> dict[str, Any]:
        """Create a cloned voice from one or more clean reference recordings.

        Multipart upload to POST /v1/voices: a `metadata` JSON part plus one
        `files` part per reference clip. Returns the parsed JSON response, which
        includes the new `voice_id`. Used by scripts/clone_reference_voice.py to
        build the Test 2.4 matched-speaker reference; not part of the scored
        test suite itself.
        """
        metadata = json.dumps(
            {"name": name, "sex": sex, "description": description, "category": category}
        )
        files = [("metadata", (None, metadata, "application/json"))]
        opened = []
        try:
            for p in reference_paths:
                fh = open(p, "rb")
                opened.append(fh)
                files.append(("files", (Path(p).name, fh, "application/octet-stream")))
            resp = requests.post(
                self.cfg.voices_endpoint,
                files=files,
                headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                timeout=self.cfg.request_timeout_s,
            )
            if not resp.ok:
                # Surface the API's error body — a bare raise_for_status() hides
                # the reason for a 400 (bad category, unsupported audio, etc.).
                body = resp.text[:1000]
                raise requests.HTTPError(
                    f"{resp.status_code} from POST {self.cfg.voices_endpoint}: {body}",
                    response=resp,
                )
            return resp.json()
        finally:
            for fh in opened:
                fh.close()

    # ------------------------------------------------------------------
    # Batch — POST /v1/tts/generate (raw PCM16 body)
    # ------------------------------------------------------------------

    @retry_with_backoff(reraise_if=_is_permanent_kugel_error)
    def synthesize_batch(self, text: str, voice_id: int | str | None = None) -> dict[str, Any]:
        body = self._body(text)
        if voice_id is not None:
            body["voice_id"] = voice_id
        start = time.perf_counter()
        resp = requests.post(
            self.cfg.rest_endpoint,
            json=body,
            headers=self._headers(),
            timeout=self.cfg.request_timeout_s,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()

        audio_bytes = resp.content
        audio_duration = self._duration(audio_bytes)
        return {
            "audio_bytes": audio_bytes,
            "audio_duration": audio_duration,
            "elapsed_s": elapsed,
            "rtf": elapsed / audio_duration if audio_duration > 0 else None,
            "interface": "rest",
            "mode": "batch",
            "http_status": resp.status_code,
            "char_count": len(text),
        }

    # KugelAudio has a single HTTP batch endpoint, so the REST-labelled batch
    # call is identical to synthesize_batch. Kept as a separate method so
    # callers that explicitly ask for interface="rest" get the same contract.
    synthesize_batch_rest = synthesize_batch

    # ------------------------------------------------------------------
    # WebSocket streaming — wss://…/ws/tts
    # ------------------------------------------------------------------

    @retry_with_backoff(reraise_if=_is_permanent_kugel_error)
    def synthesize_stream(self, text: str) -> dict[str, Any]:
        """Stream via the KugelAudio WebSocket API.

        Sends one JSON request, collects base64 PCM16 chunks, and stops on the
        `final` message. Returns the same dict shape as the Riva gRPC streamer
        so Test 2.5 (latency) records TTFB uniformly.
        """
        try:
            from websocket import create_connection
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "websocket-client is required for Kugel streaming — "
                "pip install websocket-client"
            ) from e

        url = f"{self.cfg.ws_endpoint}?api_key={self.cfg.api_key}"
        req = {
            "text": text,
            "model_id": self.cfg.model_id,
            "voice_id": self.cfg.voice_id,
            "cfg_scale": self.cfg.cfg_scale,
            "speed": self.cfg.speed,
        }
        if self.cfg.language:
            req["language"] = self.cfg.language

        start = time.perf_counter()
        first_chunk_time: float | None = None
        chunks: list[bytes] = []
        chunk_times: list[float] = []
        final_meta: dict[str, Any] | None = None

        ws = create_connection(url, timeout=self.cfg.request_timeout_s)
        try:
            ws.send(json.dumps(req))
            while True:
                raw = ws.recv()
                if raw is None or raw == "":
                    break
                # Kugel sends JSON text frames; be defensive about binary.
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="ignore")
                msg = json.loads(raw)

                if msg.get("audio"):
                    recv_t = time.perf_counter()
                    if first_chunk_time is None:
                        first_chunk_time = recv_t
                    chunks.append(base64.b64decode(msg["audio"]))
                    chunk_times.append(recv_t)

                if msg.get("type") == "final" or "final" in msg or msg.get("done"):
                    final_meta = msg
                    break
        finally:
            ws.close()

        total = time.perf_counter() - start
        audio_bytes = b"".join(chunks)
        duration = self._duration(audio_bytes)
        return {
            "audio_bytes": audio_bytes,
            "audio_duration": duration,
            "elapsed_s": total,
            "ttfb": (first_chunk_time - start) if first_chunk_time else None,
            "rtf": total / duration if duration > 0 else None,
            "n_chunks": len(chunks),
            "chunk_times": chunk_times,
            "interface": "websocket",
            "mode": "streaming",
            "final_meta": final_meta,
        }

    # ------------------------------------------------------------------
    # REST streaming — POST /v1/tts/generate with chunked read
    # ------------------------------------------------------------------

    @retry_with_backoff(reraise_if=_is_permanent_kugel_error)
    def synthesize_stream_rest(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        resp = requests.post(
            self.cfg.rest_endpoint,
            json=self._body(text),
            headers=self._headers(),
            timeout=self.cfg.request_timeout_s,
            stream=True,
        )

        first_chunk_time: float | None = None
        chunks: list[bytes] = []
        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            chunks.append(chunk)

        total = time.perf_counter() - start
        resp.raise_for_status()

        audio_bytes = b"".join(chunks)
        duration = self._duration(audio_bytes)
        return {
            "audio_bytes": audio_bytes,
            "audio_duration": duration,
            "elapsed_s": total,
            "ttfb": (first_chunk_time - start) if first_chunk_time else None,
            "rtf": total / duration if duration > 0 else None,
            "interface": "rest",
            "mode": "streaming",
            "http_status": resp.status_code,
        }

    # ------------------------------------------------------------------
    # Helpers — identical contract to TTSClient.{save_synthesis,bytes_to_wav}
    # ------------------------------------------------------------------

    def save_synthesis(
        self,
        text: str,
        output_path: str | Path,
        interface: str = "rest",
        voice_id: int | str | None = None,
    ) -> dict[str, Any]:
        if interface == "websocket":
            result = self.synthesize_stream(text)
        else:
            result = self.synthesize_batch(text, voice_id=voice_id)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_array = np.frombuffer(result["audio_bytes"], dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        sf.write(str(output_path), audio_float, self.sample_rate)

        result["output_path"] = str(output_path)
        return result

    def bytes_to_wav(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        return audio_float, self.sample_rate
