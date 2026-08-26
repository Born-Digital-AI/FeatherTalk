from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime
from collections import deque
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websockets
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from dihuman_run import DiHumanProcessor, FRAME_LEN, SAMPLE_RATE


ROOT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = ROOT_DIR / "audio"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = ROOT_DIR / "data"
FEATHER_HUBERT_SCRIPT = ROOT_DIR / "data_utils" / "feather_hubert" / "feather_hubert.py"
FEATHER_HUBERT_CHECKPOINT = (
    ROOT_DIR / "assets" / "featherhubert" / "feather_hubert_188_latest_99.pth"
)
FEATHER_HUBERT_ONNX = ROOT_DIR / "assets" / "featherhubert" / "feather_hubert.onnx"
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
REALTIME_VOICE_WS_URL = "wss://internal-test.borndigital.ai/realtime-voice-control/ws"
STREAM_MAX_FRAMES = 180
PLAYBACK_CHUNK_FRAMES = 4


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FeatherTalk</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #e7e9ee;
      background: #0f1115;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      background: #0f1115;
    }
    main {
      display: grid;
      grid-template-columns: 304px minmax(0, 1fr);
      min-height: 100vh;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.1;
      font-weight: 700;
      letter-spacing: 0;
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 18px;
      min-height: 100vh;
      padding: 22px 18px;
      border-right: 1px solid #2b313b;
      background: #171b22;
      overflow-y: auto;
    }
    .brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    form {
      display: grid;
      gap: 14px;
    }
    label {
      display: grid;
      gap: 8px;
      color: #c4cad4;
      font-size: 13px;
      font-weight: 600;
    }
    select,
    input[type="tel"],
    input[type="file"],
    button,
    textarea {
      width: 100%;
      border: 1px solid #384150;
      border-radius: 6px;
      font: inherit;
      font-size: 14px;
    }
    select,
    input[type="tel"],
    input[type="file"],
    button {
      min-height: 38px;
      padding: 8px 9px;
      background: #10141a;
      color: #eef1f6;
    }
    button {
      cursor: pointer;
      border-color: #2f8a66;
      background: #2f8a66;
      color: #ffffff;
      font-weight: 700;
    }
    button:disabled {
      cursor: not-allowed;
      border-color: #343b47;
      background: #252b34;
      color: #737c8c;
    }
    textarea {
      min-height: 112px;
      resize: vertical;
      padding: 10px;
      color: #d9dee7;
      background: #10141a;
    }
    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      padding: 4px;
      border: 1px solid #303744;
      border-radius: 8px;
      background: #10141a;
    }
    .tab {
      min-height: 34px;
      padding: 7px 10px;
      border: 0;
      border-radius: 5px;
      background: transparent;
      color: #a7afbd;
    }
    .tab.active {
      background: #26313d;
      color: #ffffff;
    }
    .panel { display: none; }
    .panel.active { display: block; }
    .stage {
      display: grid;
      place-items: center;
      min-width: 0;
      min-height: 100vh;
      padding: 24px;
      background: #0b0d11;
    }
    .stage-inner {
      display: grid;
      place-items: center;
      position: relative;
      width: 100%;
      height: 100%;
      min-height: 0;
      border: 1px solid #222833;
      border-radius: 8px;
      background: #050608;
      overflow: hidden;
    }
    video,
    #streamImage {
      display: none;
      width: 100%;
      height: 100%;
      max-width: 100%;
      max-height: calc(100vh - 48px);
      background: #000000;
      object-fit: contain;
    }
    .stage-loader {
      display: none;
      position: absolute;
      inset: 0;
      place-items: center;
      background: rgba(5, 6, 8, 0.86);
      color: #d9dee7;
      z-index: 2;
    }
    .stage-loader.active {
      display: grid;
    }
    .loader-content {
      display: grid;
      justify-items: center;
      gap: 14px;
    }
    .loader-spinner {
      width: 42px;
      height: 42px;
      border: 3px solid #2b313b;
      border-top-color: #2f8a66;
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }
    .loader-text {
      color: #aeb6c5;
      font-size: 14px;
      font-weight: 600;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    @media (max-width: 720px) {
      body { overflow: auto; }
      main {
        grid-template-columns: 1fr;
      }
      .sidebar {
        min-height: auto;
        border-right: 0;
        border-bottom: 1px solid #2b313b;
      }
      .stage {
        min-height: 58vh;
        padding: 12px;
      }
      video,
      #streamImage {
        max-height: 58vh;
      }
    }
  </style>
</head>
<body>
  <main>
    <aside class="sidebar">
      <div class="brand">
        <h1>FeatherTalk</h1>
      </div>
      <div class="tabs">
        <button class="tab active" type="button" data-panel="realtimePanel">Realtime</button>
        <button class="tab" type="button" data-panel="offlinePanel">Offline</button>
      </div>
      <section id="realtimePanel" class="panel active">
        <form id="streamForm">
          <label>
            Avatar
            <select id="streamAvatar" name="streamAvatar"></select>
          </label>
          <label>
            Phone number
            <input id="phoneNumber" name="phoneNumber" type="tel" autocomplete="tel" placeholder="+420...">
          </label>
          <button id="startStream" type="button">Start streaming</button>
          <button id="stopStream" type="button" disabled>Stop stream</button>
          <button id="startConversation" type="button" disabled>Start conversation</button>
          <button id="stopConversation" type="button" disabled>Stop conversation</button>
          <label>
            Realtime status
            <textarea id="streamStatus" readonly>Start streaming to show the idle avatar.</textarea>
          </label>
        </form>
      </section>
      <section id="offlinePanel" class="panel">
        <form id="form">
          <label>
            Avatar
            <select id="avatar" name="avatar"></select>
          </label>
          <label>
            Audio
            <input id="audio" name="audio" type="file" accept=".mp3,.wav,audio/mpeg,audio/wav">
          </label>
          <button id="generate" type="submit" disabled>Generate</button>
          <label>
            Status
            <textarea id="status" readonly>Upload an .mp3 or .wav file.</textarea>
          </label>
        </form>
      </section>
    </aside>
    <section class="stage">
      <div class="stage-inner">
        <img id="streamImage" alt="Realtime avatar stream">
        <video id="video" controls></video>
        <div id="stageLoader" class="stage-loader" aria-live="polite" aria-busy="true">
          <div class="loader-content">
            <div class="loader-spinner"></div>
            <div id="stageLoaderText" class="loader-text">Preparing avatar...</div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((item) => item.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.panel).classList.add("active");
        if (tab.dataset.panel === "realtimePanel") {
          video.style.display = "none";
          if (streamWs) streamImage.style.display = "block";
        }
        if (tab.dataset.panel === "offlinePanel") streamImage.style.display = "none";
      });
    });

    const avatar = document.querySelector("#avatar");
    const streamAvatar = document.querySelector("#streamAvatar");
    const audio = document.querySelector("#audio");
    const generate = document.querySelector("#generate");
    const form = document.querySelector("#form");
    const statusBox = document.querySelector("#status");
    const video = document.querySelector("#video");
    const phoneNumber = document.querySelector("#phoneNumber");
    const startStream = document.querySelector("#startStream");
    const stopStream = document.querySelector("#stopStream");
    const startConversation = document.querySelector("#startConversation");
    const stopConversation = document.querySelector("#stopConversation");
    const streamStatus = document.querySelector("#streamStatus");
    const streamImage = document.querySelector("#streamImage");
    const stageLoader = document.querySelector("#stageLoader");
    const stageLoaderText = document.querySelector("#stageLoaderText");

    let uploadedAudioPath = null;
    let streamWs = null;
    let conversationWs = null;
    let mediaStream = null;
    let audioContext = null;
    let microphoneNode = null;
    let microphoneSilentGain = null;
    let audioPlayerContext = null;
    let nextPlayTime = 0;
    const playbackLeadSeconds = 0.25;

    function setStatus(value) {
      statusBox.value = value;
    }

    function setStreamStatus(value) {
      streamStatus.value = value;
    }

    function showStageLoader(value) {
      stageLoaderText.textContent = value;
      stageLoader.classList.add("active");
    }

    function hideStageLoader() {
      stageLoader.classList.remove("active");
    }

    async function loadAvatars() {
      const response = await fetch("/api/avatars");
      const data = await response.json();
      avatar.replaceChildren();
      streamAvatar.replaceChildren();
      for (const name of data.avatars) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        avatar.appendChild(option.cloneNode(true));
        streamAvatar.appendChild(option);
      }
      if (!data.avatars.length) {
        setStatus("No avatars found in checkpoints/.");
        setStreamStatus("No avatars found in checkpoints/.");
      }
    }

    function wsUrl(path) {
      const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${scheme}//${window.location.host}${path}`;
    }

    function base64ToInt16(value) {
      const raw = atob(value);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      return new Int16Array(bytes.buffer);
    }

    function playPcm16(base64Audio) {
      const pcm = base64ToInt16(base64Audio);
      if (!pcm.length) return;
      audioPlayerContext = audioPlayerContext || new AudioContext({ sampleRate: 16000 });
      if (audioPlayerContext.state === "suspended") audioPlayerContext.resume();
      const buffer = audioPlayerContext.createBuffer(1, pcm.length, 16000);
      const channel = buffer.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) channel[i] = Math.max(-1, Math.min(1, pcm[i] / 32768));
      const source = audioPlayerContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioPlayerContext.destination);
      const now = audioPlayerContext.currentTime;
      nextPlayTime = Math.max(nextPlayTime, now + playbackLeadSeconds);
      source.start(nextPlayTime);
      nextPlayTime += buffer.duration;
    }

    startStream.addEventListener("click", () => {
      if (streamWs) return;
      audioPlayerContext = audioPlayerContext || new AudioContext({ sampleRate: 16000 });
      if (audioPlayerContext.state === "suspended") audioPlayerContext.resume();
      nextPlayTime = audioPlayerContext.currentTime;
      streamImage.style.display = "none";
      video.style.display = "none";
      showStageLoader("Preparing avatar stream...");
      setStreamStatus("Starting avatar stream...");
      streamWs = new WebSocket(wsUrl(`/ws/stream?avatar=${encodeURIComponent(streamAvatar.value)}`));
      streamWs.addEventListener("open", () => {
        startStream.disabled = true;
        stopStream.disabled = false;
        startConversation.disabled = false;
        setStreamStatus("Streaming idle avatar.");
      });
      streamWs.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "frame") {
          streamImage.src = message.image;
          streamImage.style.display = "block";
          hideStageLoader();
          if (message.audio) playPcm16(message.audio);
        } else if (message.type === "audio") {
          playPcm16(message.audio);
        } else if (message.type === "status") {
          setStreamStatus(message.message);
        } else if (message.type === "error") {
          hideStageLoader();
          setStreamStatus(message.message);
        }
      });
      streamWs.addEventListener("close", () => {
        streamWs = null;
        startStream.disabled = false;
        stopStream.disabled = true;
        startConversation.disabled = true;
        stopConversation.disabled = true;
        hideStageLoader();
        setStreamStatus("Stream stopped.");
      });
    });

    stopStream.addEventListener("click", () => {
      stopConversation.click();
      if (streamWs) streamWs.close();
    });

    async function startMicrophone() {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(mediaStream);
      microphoneNode = audioContext.createScriptProcessor(2048, 1, 1);
      microphoneNode.onaudioprocess = (event) => {
        if (!conversationWs || conversationWs.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          const sample = Math.max(-1, Math.min(1, input[i]));
          pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
        }
        const bytes = new Uint8Array(pcm.buffer);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        conversationWs.send(JSON.stringify({
          type: "audio",
          audio: btoa(binary),
          sample_rate: audioContext.sampleRate
        }));
      };
      source.connect(microphoneNode);
      microphoneSilentGain = audioContext.createGain();
      microphoneSilentGain.gain.value = 0;
      microphoneNode.connect(microphoneSilentGain);
      microphoneSilentGain.connect(audioContext.destination);
    }

    function stopMicrophone() {
      if (microphoneNode) microphoneNode.disconnect();
      if (microphoneSilentGain) microphoneSilentGain.disconnect();
      if (audioContext) audioContext.close();
      if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
      microphoneNode = null;
      microphoneSilentGain = null;
      audioContext = null;
      mediaStream = null;
    }

    startConversation.addEventListener("click", async () => {
      if (!streamWs) {
        setStreamStatus("Start streaming before starting a conversation.");
        return;
      }
      setStreamStatus("Connecting realtime voice...");
      conversationWs = new WebSocket(wsUrl("/ws/conversation"));
      conversationWs.addEventListener("open", async () => {
        conversationWs.send(JSON.stringify({
          type: "start",
          phone_number: phoneNumber.value,
          avatar: streamAvatar.value,
          conversation_id: `feathertalk-${Date.now()}`
        }));
        await startMicrophone();
        startConversation.disabled = true;
        stopConversation.disabled = false;
        setStreamStatus("Conversation started.");
      });
      conversationWs.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "status") setStreamStatus(message.message);
        if (message.type === "error") setStreamStatus(message.message);
      });
      conversationWs.addEventListener("close", () => {
        stopMicrophone();
        conversationWs = null;
        startConversation.disabled = !streamWs;
        stopConversation.disabled = true;
        setStreamStatus(streamWs ? "Conversation stopped. Idle streaming continues." : "Conversation stopped.");
      });
    });

    stopConversation.addEventListener("click", () => {
      stopMicrophone();
      if (conversationWs) {
        conversationWs.send(JSON.stringify({ type: "stop" }));
        conversationWs.close();
      }
    });

    audio.addEventListener("change", async () => {
      generate.disabled = true;
      uploadedAudioPath = null;
      video.style.display = "none";
      hideStageLoader();

      if (!audio.files.length) {
        setStatus("Upload an .mp3 or .wav file.");
        return;
      }

      const body = new FormData();
      body.append("audio", audio.files[0]);
      setStatus("Uploading audio...");

      const response = await fetch("/api/upload", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) {
        setStatus(data.detail || "Upload failed.");
        return;
      }

      uploadedAudioPath = data.audio_path;
      generate.disabled = false;
      setStatus(data.message);
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!uploadedAudioPath) return;

      generate.disabled = true;
      video.style.display = "none";
      streamImage.style.display = "none";
      showStageLoader("Generating avatar video...");
      setStatus("Generating...");

      const body = new FormData();
      body.append("avatar", avatar.value);
      body.append("audio_path", uploadedAudioPath);

      const response = await fetch("/api/generate", { method: "POST", body });
      const data = await response.json();
      setStatus(data.log || data.detail || "Generation finished.");
      generate.disabled = false;

      if (response.ok && data.video_url) {
        streamImage.style.display = "none";
        video.src = data.video_url;
        video.style.display = "block";
        hideStageLoader();
      } else {
        hideStageLoader();
      }
    });

    loadAvatars().catch((error) => setStatus(error.toString()));
  </script>
</body>
</html>
"""


def _safe_filename(filename: str) -> str:
    safe = Path(filename).name.strip().replace(" ", "_")
    return "".join(char for char in safe if char.isalnum() or char in "._-")


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(details or f"Command failed with exit code {completed.returncode}")


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR))


def list_avatars() -> list[str]:
    if not CHECKPOINTS_DIR.exists():
        return []
    return sorted(path.name for path in CHECKPOINTS_DIR.iterdir() if path.is_dir())


def generate_video(avatar: str, uploaded_audio_path: str) -> tuple[str, Path | None]:
    audio_path = Path(uploaded_audio_path)
    if not audio_path.is_absolute():
        audio_path = ROOT_DIR / audio_path
    if not audio_path.exists():
        return f"Audio file does not exist: {audio_path}", None
    if audio_path.parent.resolve() != AUDIO_DIR.resolve():
        return "Audio file must be inside the audio directory.", None

    checkpoint_path = CHECKPOINTS_DIR / avatar / "last.pth"
    dataset_path = DATA_DIR / avatar
    if not checkpoint_path.exists():
        return f"Missing checkpoint: {_relative(checkpoint_path)}", None
    if not dataset_path.exists():
        return f"Missing dataset: {_relative(dataset_path)}", None

    OUTPUTS_DIR.mkdir(exist_ok=True)
    audio_feat_path = AUDIO_DIR / f"{audio_path.stem}_hu.npy"
    log_lines: list[str] = []

    try:
        if audio_feat_path.exists():
            log_lines.append(f"Features already exist, skipping: {_relative(audio_feat_path)}")
        else:
            feature_command = [
                sys.executable,
                _relative(FEATHER_HUBERT_SCRIPT),
                "--wav",
                _relative(audio_path),
                "--checkpoint",
                _relative(FEATHER_HUBERT_CHECKPOINT),
                "--out",
                _relative(audio_feat_path),
            ]
            log_lines.append("Creating audio features...")
            _run_command(feature_command)
            log_lines.append(f"Features created: {_relative(audio_feat_path)}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = OUTPUTS_DIR / f"{avatar}_{timestamp}.mp4"
        final_video_path = save_path.with_name(f"{save_path.stem}_with_audio{save_path.suffix}")
        inference_command = [
            sys.executable,
            "inference.py",
            "--asr",
            "hubert",
            "--unet",
            "original",
            "--dataset",
            _relative(dataset_path),
            "--audio_feat",
            _relative(audio_feat_path),
            "--checkpoint",
            _relative(checkpoint_path),
            "--save_path",
            _relative(save_path),
            "--audio_wav",
            _relative(audio_path),
        ]

        log_lines.append("Running inference...")
        _run_command(inference_command)

        if final_video_path.exists():
            log_lines.append(f"Video created: {_relative(final_video_path)}")
            return "\n".join(log_lines), final_video_path

        if save_path.exists():
            log_lines.append(
                "Video with audio was not created, returning the video without the audio suffix. "
                "Check that ffmpeg is available."
            )
            return "\n".join(log_lines), save_path

        log_lines.append(f"Inference finished, but output was not found: {_relative(final_video_path)}")
        return "\n".join(log_lines), None
    except Exception as exc:
        log_lines.append(f"Error: {exc}")
        return "\n".join(log_lines), None


def ensure_streaming_unet(avatar: str) -> Path:
    checkpoint_path = CHECKPOINTS_DIR / avatar / "last.pth"
    dataset_path = DATA_DIR / avatar
    onnx_path = dataset_path / "unet.onnx"
    if onnx_path.exists():
        return onnx_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {_relative(checkpoint_path)}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {_relative(dataset_path)}")

    _run_command(
        [
            sys.executable,
            "pth2onnx.py",
            "--checkpoint",
            _relative(checkpoint_path),
            "--onnx_path",
            _relative(onnx_path),
            "--asr",
            "hubert",
            "--unet",
            "original",
        ]
    )
    return onnx_path


def preload_onnxruntime_cuda() -> None:
    try:
        import torch  # noqa: F401
        import onnxruntime
    except ImportError:
        return
    if hasattr(onnxruntime, "preload_dlls"):
        with contextlib.suppress(Exception):
            onnxruntime.preload_dlls(directory=None)


def ensure_feather_hubert_onnx() -> Path:
    if FEATHER_HUBERT_ONNX.exists():
        return FEATHER_HUBERT_ONNX
    if not FEATHER_HUBERT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing FeatherHuBERT checkpoint: {_relative(FEATHER_HUBERT_CHECKPOINT)}")

    _run_command(
        [
            sys.executable,
            "FeatherTalk-CPP/tools/export_models.py",
            "--feather-checkpoint",
            _relative(FEATHER_HUBERT_CHECKPOINT),
            "--unet-checkpoint",
            _relative(next_available_checkpoint()),
            "--output-dir",
            _relative(FEATHER_HUBERT_ONNX.parent),
        ]
    )
    if not FEATHER_HUBERT_ONNX.exists():
        raise FileNotFoundError(f"FeatherHuBERT ONNX export did not create {_relative(FEATHER_HUBERT_ONNX)}")
    return FEATHER_HUBERT_ONNX


def next_available_checkpoint() -> Path:
    for avatar in list_avatars():
        checkpoint_path = CHECKPOINTS_DIR / avatar / "last.pth"
        if checkpoint_path.exists():
            return checkpoint_path
    raise FileNotFoundError("No avatar checkpoint found for ONNX export.")


def _jpeg_data_url(img: np.ndarray, max_width: int = 720) -> str:
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * scale)))
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode stream frame.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _pcm16_to_base64(audio: np.ndarray) -> str:
    return base64.b64encode(audio.astype(np.int16).tobytes()).decode("ascii")


def _resample_pcm16_bytes(audio_bytes: bytes, source_rate: int, target_rate: int = SAMPLE_RATE) -> bytes:
    if not audio_bytes or source_rate == target_rate:
        return audio_bytes
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio.size == 0:
        return b""

    try:
        from scipy.signal import resample_poly

        common = np.gcd(source_rate, target_rate)
        resampled = resample_poly(
            audio.astype(np.float32),
            target_rate // common,
            source_rate // common,
        )
    except Exception:
        source_x = np.arange(audio.size, dtype=np.float32)
        target_size = max(1, int(round(audio.size * target_rate / source_rate)))
        target_x = np.linspace(0, max(0, audio.size - 1), target_size, dtype=np.float32)
        resampled = np.interp(target_x, source_x, audio.astype(np.float32))
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16).tobytes()


def _pcm16_base64_to_wav_bytes(encoded_audio: str, source_rate: int) -> bytes:
    pcm16 = base64.b64decode(encoded_audio)
    pcm16 = _resample_pcm16_bytes(pcm16, source_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm16)
    return buffer.getvalue()


def _decode_audio_payload(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, list):
        return np.array(value, dtype=np.int16).tobytes()
    if isinstance(value, str):
        try:
            return base64.b64decode(value)
        except Exception:
            return b""
    return b""


def _extract_sample_rate(data: dict[str, Any]) -> int:
    for key in ("sample_rate_hz", "sample_rate", "rate"):
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return SAMPLE_RATE


def _extract_realtime_audio_chunk(message: Any) -> tuple[bytes, int]:
    if isinstance(message, bytes):
        return message, SAMPLE_RATE
    if not isinstance(message, str):
        return b"", SAMPLE_RATE
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return b"", SAMPLE_RATE

    if data.get("type") == "audio_chunk":
        return _decode_audio_payload(data.get("data")), _extract_sample_rate(data)

    audio = _extract_audio_from_realtime_message(message)
    return audio, _extract_sample_rate(data)


def _extract_audio_from_realtime_message(message: Any) -> bytes:
    if isinstance(message, bytes):
        return message
    if not isinstance(message, str):
        return b""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return b""

    candidates = [
        data.get("audio"),
        data.get("audio_delta"),
        data.get("delta"),
        data.get("chunk"),
        data.get("pcm"),
        data.get("data") if data.get("type") == "audio_chunk" else None,
    ]
    output = data.get("output")
    if isinstance(output, dict):
        candidates.extend([output.get("audio"), output.get("audio_delta"), output.get("delta")])
    media = data.get("media")
    if isinstance(media, dict):
        candidates.extend([media.get("payload"), media.get("audio"), media.get("delta")])
    for candidate in candidates:
        audio = _decode_audio_payload(candidate)
        if audio:
            return audio
    return _find_audio_payload(data)


def _find_audio_payload(value: Any) -> bytes:
    if isinstance(value, dict):
        for key in ("audio", "audio_delta", "delta", "chunk", "pcm", "payload"):
            audio = _decode_audio_payload(value.get(key))
            if audio:
                return audio
        for child in value.values():
            audio = _find_audio_payload(child)
            if audio:
                return audio
    elif isinstance(value, list):
        for child in value:
            audio = _find_audio_payload(child)
            if audio:
                return audio
    return b""


class AvatarStream:
    def __init__(self, avatar: str):
        preload_onnxruntime_cuda()
        ensure_streaming_unet(avatar)
        feather_hubert_onnx = ensure_feather_hubert_onnx()
        self.avatar = avatar
        self.processor = DiHumanProcessor(
            _relative(DATA_DIR / avatar),
            asr="hubert",
            encoder_onnx=_relative(feather_hubert_onnx),
            max_frames=STREAM_MAX_FRAMES,
        )
        self.processor.warm_up()
        self.audio_chunks: deque[np.ndarray] = deque()

    def provider_summary(self) -> str:
        providers = []
        for session in (self.processor.ort_unet, self.processor.ort_ae):
            if session is not None:
                providers.append("/".join(session.get_providers()))
        return "; ".join(providers) or "unknown"

    def enqueue_audio(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        for offset in range(0, audio.shape[0], FRAME_LEN):
            chunk = audio[offset : offset + FRAME_LEN]
            if chunk.shape[0] < FRAME_LEN:
                chunk = np.pad(chunk, (0, FRAME_LEN - chunk.shape[0]))
            self.audio_chunks.append(chunk.astype(np.int16))

    def queued_ms(self) -> int:
        return round(len(self.audio_chunks) * FRAME_LEN / SAMPLE_RATE * 1000)

    def next_frame(self) -> tuple[np.ndarray | None, np.ndarray]:
        if self.audio_chunks:
            audio_frame = self.audio_chunks.popleft()
        else:
            audio_frame = np.zeros([FRAME_LEN], dtype=np.int16)
        img, playing_audio, check_img = self.processor.process(audio_frame)
        if not check_img or img is None:
            return None, playing_audio
        return img, playing_audio

    def next_packet(self, chunks: int = PLAYBACK_CHUNK_FRAMES) -> tuple[np.ndarray | None, np.ndarray]:
        image: np.ndarray | None = None
        audio_frames = []
        for _ in range(max(1, chunks)):
            next_image, playing_audio = self.next_frame()
            if next_image is not None:
                image = next_image
            audio_frames.append(playing_audio)
        return image, np.concatenate(audio_frames)


current_stream: AvatarStream | None = None
current_stream_lock = asyncio.Lock()


app = FastAPI(title="FeatherTalk")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/avatars")
def avatars() -> dict[str, list[str]]:
    return {"avatars": list_avatars()}


@app.post("/api/upload")
def upload_audio(audio: UploadFile = File(...)) -> dict[str, str]:
    filename = _safe_filename(audio.filename or "")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .mp3 and .wav files are supported.")
    if not filename:
        raise HTTPException(status_code=400, detail="The uploaded file does not have a usable name.")

    AUDIO_DIR.mkdir(exist_ok=True)
    target_path = AUDIO_DIR / filename
    with target_path.open("wb") as target_file:
        shutil.copyfileobj(audio.file, target_file)

    return {
        "audio_path": _relative(target_path),
        "message": f"Audio uploaded: {_relative(target_path)}",
    }


@app.post("/api/generate")
def generate(avatar: str = Form(...), audio_path: str = Form(...)) -> dict[str, str | None]:
    if avatar not in list_avatars():
        raise HTTPException(status_code=400, detail="Selected avatar does not exist.")

    log, video_path = generate_video(avatar, audio_path)
    if video_path is None:
        return {"log": log, "video_url": None}

    return {
        "log": log,
        "video_url": f"/outputs/{video_path.name}",
    }


@app.get("/outputs/{filename}")
def output_file(filename: str) -> FileResponse:
    safe_filename = _safe_filename(filename)
    if safe_filename != filename:
        raise HTTPException(status_code=404, detail="Output not found.")

    path = OUTPUTS_DIR / safe_filename
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="Output not found.")

    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.websocket("/ws/stream")
async def stream_avatar(websocket: WebSocket, avatar: str) -> None:
    global current_stream

    await websocket.accept()
    if avatar not in list_avatars():
        await websocket.send_json({"type": "error", "message": "Selected avatar does not exist."})
        await websocket.close()
        return

    try:
        await websocket.send_json({"type": "status", "message": "Preparing avatar stream..."})
        async with current_stream_lock:
            current_stream = await asyncio.to_thread(AvatarStream, avatar)
            stream = current_stream
        await websocket.send_json(
            {
                "type": "status",
                "message": f"Streaming idle avatar. ONNX providers: {stream.provider_summary()}",
            }
        )

        stream_tick_interval = FRAME_LEN * PLAYBACK_CHUNK_FRAMES / SAMPLE_RATE
        next_tick_at = asyncio.get_running_loop().time()
        max_lag_ms = 0.0
        tick_count = 0
        while True:
            started = asyncio.get_running_loop().time()
            img, playing_audio = await asyncio.to_thread(stream.next_packet)
            if np.any(playing_audio):
                await websocket.send_json({"type": "audio", "audio": _pcm16_to_base64(playing_audio)})
            if img is not None:
                message: dict[str, str] = {"type": "frame", "image": _jpeg_data_url(img)}
                await websocket.send_json(message)
            elapsed = asyncio.get_running_loop().time() - started
            tick_count += 1
            next_tick_at += stream_tick_interval
            lag_ms = max(0.0, (asyncio.get_running_loop().time() - next_tick_at) * 1000)
            max_lag_ms = max(max_lag_ms, lag_ms)
            if tick_count % 125 == 0 and max_lag_ms > 100:
                await websocket.send_json(
                    {
                        "type": "status",
                        "message": (
                            f"Streaming, but backend accumulated up to {max_lag_ms:.0f} ms lag recently. "
                            f"Audio queue: {stream.queued_ms()} ms."
                        ),
                    }
                )
                max_lag_ms = 0.0
                next_tick_at = asyncio.get_running_loop().time()
            await asyncio.sleep(max(0.001, next_tick_at - asyncio.get_running_loop().time()))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        async with current_stream_lock:
            if current_stream is not None and current_stream.avatar == avatar:
                current_stream = None


@app.websocket("/ws/conversation")
async def conversation_bridge(websocket: WebSocket) -> None:
    await websocket.accept()

    async with current_stream_lock:
        stream = current_stream
    if stream is None:
        await websocket.send_json({"type": "error", "message": "Start avatar streaming first."})
        await websocket.close()
        return

    try:
        async with websockets.connect(REALTIME_VOICE_WS_URL) as realtime_ws:
            await websocket.send_json({"type": "status", "message": "Connected to realtime voice."})
            started = False
            last_audio_status_at = 0.0

            async def browser_to_realtime() -> None:
                nonlocal started
                while True:
                    raw = await websocket.receive_text()
                    data = json.loads(raw)
                    event_type = data.get("type")
                    if event_type == "start":
                        payload = {
                            "event": "start_session",
                            "phone_number": data.get("phone_number", ""),
                            "config_lookup_key": data.get("phone_number", ""),
                            "conversation_id": data.get("conversation_id", ""),
                            "call_id": data.get("conversation_id", ""),
                            "avatar": data.get("avatar", ""),
                        }
                        await realtime_ws.send(json.dumps(payload))
                        started = True
                    elif event_type == "audio":
                        if not started:
                            continue
                        try:
                            source_rate = int(data.get("sample_rate") or SAMPLE_RATE)
                            wav_bytes = _pcm16_base64_to_wav_bytes(data.get("audio", ""), source_rate)
                        except Exception as exc:
                            await websocket.send_json({"type": "error", "message": f"Invalid microphone audio: {exc}"})
                            continue
                        await realtime_ws.send(wav_bytes)
                    elif event_type == "stop":
                        await realtime_ws.send(json.dumps({"event": "stop_session"}))
                        return

            async def realtime_to_browser() -> None:
                nonlocal last_audio_status_at
                async for message in realtime_ws:
                    audio, sample_rate = _extract_realtime_audio_chunk(message)
                    if audio:
                        audio = _resample_pcm16_bytes(audio, sample_rate)
                        stream.enqueue_audio(audio)
                        now = asyncio.get_running_loop().time()
                        if now - last_audio_status_at >= 1.0:
                            last_audio_status_at = now
                            await websocket.send_json(
                                {
                                    "type": "status",
                                    "message": f"Realtime audio queue: {stream.queued_ms()} ms.",
                                }
                            )
                        continue
                    if isinstance(message, str):
                        try:
                            event = json.loads(message)
                        except json.JSONDecodeError:
                            event = {}
                        if event.get("type") == "error":
                            error = event.get("message") or event.get("error") or "Realtime voice error."
                            await websocket.send_json({"type": "error", "message": str(error)})
                        elif event.get("type") == "session_updated":
                            await websocket.send_json({"type": "status", "message": "Realtime voice session ready."})
                        elif event.get("type") == "audio_end":
                            await websocket.send_json({"type": "status", "message": "Realtime audio finished."})
                        else:
                            await websocket.send_json({"type": "status", "message": "Realtime voice event received."})

            done, pending = await asyncio.wait(
                {
                    asyncio.create_task(browser_to_realtime()),
                    asyncio.create_task(realtime_to_browser()),
                },
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
