from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websockets
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dihuman_run import DiHumanProcessor, FEATHER_REALTIME_HISTORY_SAMPLES, FRAME_LEN, SAMPLE_RATE
from realtime_media import SpeechProducer, idle_dependency_signature, load_silent_clip, prepare_inferred_idle_loop, read_idle_video


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
PLAYBACK_CHUNK_FRAMES = 4
STREAM_FPS = SAMPLE_RATE / (FRAME_LEN * PLAYBACK_CHUNK_FRAMES)
AUDIO_TAIL_TIMEOUT = 0.5  # Fallback for servers without audio_end; tolerate chunk jitter.
STREAM_PREPARATION_STATUS_INTERVAL = 5.0
REALTIME_SOURCE_SAMPLE_RATE = 24000
REALTIME_BATCH_FRAMES = 8  # 320 ms per encoder call, emitted at 25 FPS.
REALTIME_LOOKAHEAD_FRAMES = 8  # 320 ms of actual future waveform.
logger = logging.getLogger("uvicorn.error")

# Stream ownership remains exclusive until its producer has drained. Keep the
# expensive avatar resources alive when that stream ends, including on reload.
_PREPARED_AVATARS: dict[tuple[str, bool], tuple[tuple, dict[str, Any]]] = {}
_AVATAR_PREPARATION_LOCK = threading.Lock()
_SPEECH_MODEL_LOCK = threading.Lock()
_SPEECH_PRELOAD_LOCK = threading.Lock()
_SPEECH_PRELOAD_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avatar-model-load")


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
    .stage-error {
      display: none;
      position: absolute;
      inset: 0;
      place-items: center;
      padding: 32px;
      background: rgba(5, 6, 8, 0.94);
      z-index: 3;
    }
    .stage-error.active { display: grid; }
    .stage-error-content { max-width: 560px; text-align: center; }
    .stage-error h2 { color: #f0aaa5; }
    .stage-error p { line-height: 1.6; overflow-wrap: anywhere; }
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
          <button id="startStream" type="button" disabled>Start streaming</button>
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
        <video id="idleVideo" muted playsinline loop aria-label="Idle avatar"></video>
        <canvas id="streamImage" aria-label="Realtime avatar stream"></canvas>
        <video id="video" controls></video>
        <div id="stageError" class="stage-error" role="alert">
          <div class="stage-error-content">
            <h2>Realtime streaming cannot start</h2>
            <p id="stageErrorText"></p>
          </div>
        </div>
        <div id="stageLoader" class="stage-loader" aria-live="polite" aria-busy="true">
          <div class="loader-content">
            <div class="loader-spinner"></div>
            <div id="stageLoaderText" class="loader-text">Preparing avatar...</div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script src="/static/realtime-player.js"></script>
  <script>
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((item) => item.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.panel).classList.add("active");
        if (tab.dataset.panel === "realtimePanel") {
          video.style.display = "none";
          if (realtimePlayer) realtimePlayer.setVisible(true);
          if (streamFailed) stageError.classList.add("active");
        }
        if (tab.dataset.panel === "offlinePanel") {
          streamImage.style.display = "none";
          if (realtimePlayer) realtimePlayer.setVisible(false);
          stageError.classList.remove("active");
        }
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
    const idleVideo = document.querySelector("#idleVideo");
    const stageLoader = document.querySelector("#stageLoader");
    const stageLoaderText = document.querySelector("#stageLoaderText");
    const stageError = document.querySelector("#stageError");
    const stageErrorText = document.querySelector("#stageErrorText");

    let uploadedAudioPath = null;
    let streamWs = null;
    let streamReady = false;
    let streamFailed = false;
    let streamStopRequested = false;
    let conversationWs = null;
    let mediaStream = null;
    let audioContext = null;
    let microphoneNode = null;
    let microphoneSilentGain = null;
    let audioPlayerContext = null;
    let realtimePlayer = null;
    let serverReady = false;
    let idleReady = false;
    let currentFrame = null;
    let currentDirection = 1;

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

    function showStreamError(value) {
      streamFailed = true;
      streamReady = false;
      if (realtimePlayer) realtimePlayer.stop();
      hideStageLoader();
      stageErrorText.textContent = value;
      stageError.classList.add("active");
      setStreamStatus(value);
      updateRealtimeButtons();
    }

    function clearPlayback() {
      if (realtimePlayer) realtimePlayer.stop();
      realtimePlayer = null;
      idleReady = false;
      serverReady = false;
      if (audioPlayerContext) {
        audioPlayerContext.close();
        audioPlayerContext = null;
      }
    }

    function hasPhoneNumber() {
      return phoneNumber.value.trim().length > 0;
    }

    function updateRealtimeButtons() {
      const phoneNumberFilled = hasPhoneNumber();
      startStream.disabled = !phoneNumberFilled || !!streamWs;
      startConversation.disabled = !phoneNumberFilled || !streamReady || !!conversationWs;
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

    startStream.addEventListener("click", () => {
      if (streamWs) return;
      clearPlayback();
      streamReady = false;
      streamFailed = false;
      streamStopRequested = false;
      stageError.classList.remove("active");
      audioPlayerContext = audioPlayerContext || new AudioContext({ sampleRate: 16000 });
      if (audioPlayerContext.state === "suspended") audioPlayerContext.resume();
      realtimePlayer = new AvatarPlayer({
        canvas: streamImage, idleVideo, context: audioPlayerContext,
        onReady: () => {
          idleReady = true;
          streamReady = serverReady && idleReady;
          hideStageLoader();
          updateRealtimeButtons();
          if (streamWs && streamWs.readyState === WebSocket.OPEN) streamWs.send(JSON.stringify({ type: "ready" }));
        },
        onError: showStreamError,
        onPlayed: (frame) => {
          currentFrame = frame.frame_index;
          currentDirection = frame.direction;
          if (streamWs && streamWs.readyState === WebSocket.OPEN) {
            streamWs.send(JSON.stringify({ type: "played", frame_id: frame.frame_id, generation: frame.generation }));
          }
        },
        onIdle: (generation) => {
          if (streamWs && streamWs.readyState === WebSocket.OPEN) streamWs.send(JSON.stringify({ type: "idle", generation }));
        },
      });
      realtimePlayer.setVisible(document.querySelector("#realtimePanel").classList.contains("active"));
      streamImage.style.display = "none";
      video.style.display = "none";
      showStageLoader("Preparing avatar stream...");
      setStreamStatus("Starting avatar stream...");
      streamWs = new WebSocket(wsUrl(`/ws/stream?avatar=${encodeURIComponent(streamAvatar.value)}`));
      updateRealtimeButtons();
      streamWs.addEventListener("open", () => {
        updateRealtimeButtons();
        stopStream.disabled = false;
        setStreamStatus("Preparing avatar stream...");
      });
      streamWs.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "stream_init") {
          realtimePlayer.configure(message);
        } else if (message.type === "prepare_speech") {
          const cursor = realtimePlayer.reserveSpeech(message);
          if (cursor) streamWs.send(JSON.stringify(cursor));
        } else if (message.type === "frame") {
          realtimePlayer.enqueue(message);
        } else if (message.type === "reset") {
          realtimePlayer.resetSpeech(message.generation);
        } else if (message.type === "metrics") {
          setStreamStatus(`Streaming. Encoder: ${message.encoder_ms.toFixed(0)} ms/reply; render: ${message.render_ms.toFixed(1)} ms/frame; PNG patch: ${message.encode_ms.toFixed(1)} ms/frame; queue: ${message.queued_frames} frames.`);
        } else if (message.type === "status") {
          if (message.ready) {
            serverReady = true;
            streamReady = serverReady && idleReady;
            updateRealtimeButtons();
          }
          setStreamStatus(message.message);
          if (!serverReady) stageLoaderText.textContent = message.message;
        } else if (message.type === "error") {
          showStreamError(message.message);
        }
      });
      streamWs.addEventListener("error", () => {
        showStreamError("Cannot connect to the avatar stream. Check the server and try again.");
      });
      streamWs.addEventListener("close", (event) => {
        streamWs = null;
        streamReady = false;
        clearPlayback();
        if (conversationWs) stopConversation.click();
        updateRealtimeButtons();
        stopStream.disabled = true;
        stopConversation.disabled = true;
        hideStageLoader();
        if (!streamFailed) {
          if (streamStopRequested) {
            setStreamStatus("Stream stopped.");
          } else {
            const reason = event.reason ? `: ${event.reason}` : "";
            showStreamError(`Avatar stream disconnected unexpectedly (WebSocket ${event.code}${reason}). Check the backend log for preparation errors or a server restart.`);
          }
        }
      });
    });

    stopStream.addEventListener("click", () => {
      streamStopRequested = true;
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
      if (!hasPhoneNumber()) {
        updateRealtimeButtons();
        return;
      }
      setStreamStatus("Connecting realtime voice...");
      conversationWs = new WebSocket(wsUrl("/ws/conversation"));
      updateRealtimeButtons();
      conversationWs.addEventListener("open", async () => {
        conversationWs.send(JSON.stringify({
          type: "start",
          phone_number: phoneNumber.value,
          avatar: streamAvatar.value,
          conversation_id: `feathertalk-${Date.now()}`
        }));
        await startMicrophone();
        updateRealtimeButtons();
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
        updateRealtimeButtons();
        stopConversation.disabled = true;
        if (!streamFailed) {
          setStreamStatus(streamWs ? "Conversation stopped. Idle streaming continues." : "Conversation stopped.");
        }
      });
    });

    stopConversation.addEventListener("click", () => {
      stopMicrophone();
      if (conversationWs) {
        conversationWs.send(JSON.stringify({ type: "stop" }));
        conversationWs.close();
      }
    });

    phoneNumber.addEventListener("input", updateRealtimeButtons);

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
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {_relative(checkpoint_path)}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {_relative(dataset_path)}")
    if onnx_path.exists() and onnx_path.stat().st_mtime_ns >= checkpoint_path.stat().st_mtime_ns:
        return onnx_path

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
        import onnxruntime
    except ImportError:
        return
    if hasattr(onnxruntime, "preload_dlls"):
        with contextlib.suppress(Exception):
            onnxruntime.preload_dlls(directory="")
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


def _frame_data_url(img: np.ndarray) -> str:
    # Preserve the training image's resolution and pixels on the wire.
    ok, encoded = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        raise RuntimeError("Failed to encode stream frame.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


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
        if isinstance(value, (int, str)) and not isinstance(value, bool):
            try:
                rate = int(value)
                if rate > 0:
                    return rate
            except ValueError:
                pass
    for key in ("output", "media", "audio", "data"):
        child = data.get(key)
        if isinstance(child, dict):
            rate = _extract_sample_rate(child)
            if rate:
                return rate
    return 0


def _extract_realtime_audio_chunk(message: Any, default_rate: int = REALTIME_SOURCE_SAMPLE_RATE) -> tuple[bytes, int]:
    if isinstance(message, bytes):
        return message, default_rate
    if not isinstance(message, str):
        return b"", SAMPLE_RATE
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return b"", SAMPLE_RATE

    if data.get("type") == "audio_chunk":
        return _decode_audio_payload(data.get("data")), _extract_sample_rate(data) or default_rate

    audio = _extract_audio_from_realtime_message(message)
    return audio, _extract_sample_rate(data) or default_rate


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
    def __init__(self, avatar: str, *, offline_quality: bool = True, cache_silence_frames: bool = True):
        self.avatar = avatar
        self.session_id = uuid.uuid4().hex
        self.silence_path = DATA_DIR / avatar / "silence.mp4"
        self.idle_path = None
        self.timings = {"encoder_ms": 0.0, "render_ms": 0.0, "encode_ms": 0.0, "frames": 0, "patch_bytes": 0}
        self.offline_quality = offline_quality
        self.local_idle = not cache_silence_frames
        self._prepare_avatar(avatar, offline_quality, cache_silence_frames)
        self.audio_lock = threading.Lock()
        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.prepared_audio = deque()
        self.audio_end_offsets = deque()
        self.resampler = None
        self.source_rate = SAMPLE_RATE
        self.batch_frames = REALTIME_BATCH_FRAMES
        self.lookahead_frames = REALTIME_LOOKAHEAD_FRAMES
        self.utterance_active = False
        self.audio_generation = 0
        self.processor_audio_generation = 0
        self.audio_ended = False
        self.last_audio_at = 0.0
        self.speaking = False
        self.current_frame = 0
        self.current_direction = 1
        self.render_batch_frames = 1
        if self.local_idle and getattr(self.processor, "_models_initialized", False):
            requested_batch = max(1, min(8, int(os.environ.get("FEATHER_REALTIME_UNET_BATCH", "1"))))
            if requested_batch > 1 and self.processor.supports_realtime_batch():
                self.render_batch_frames = requested_batch

    def _idle_dependencies(self):
        return [
            self.silence_path, DATA_DIR / self.avatar, CHECKPOINTS_DIR / self.avatar / "last.pth",
            FEATHER_HUBERT_CHECKPOINT if self.offline_quality else FEATHER_HUBERT_ONNX,
        ]

    def _prepare_avatar(self, avatar, offline_quality, cache_silence_frames):
        if not self.local_idle:
            self._load_avatar(avatar, offline_quality, cache_silence_frames)
            return
        started = time.perf_counter()
        with _AVATAR_PREPARATION_LOCK:
            key = (str((DATA_DIR / avatar).resolve()), offline_quality)
            signature = idle_dependency_signature(self._idle_dependencies())
            cached = _PREPARED_AVATARS.get(key)
            if cached is not None and cached[0] == signature:
                resources = cached[1]
                for name in ("processor", "silence_frame_count", "silence_size", "idle_path"):
                    setattr(self, name, resources[name])
                self.background_cache_key = resources["background_cache_key"]
                self.silent_frames = []
                self.timings["render_ms"] = resources["render_ms"]
                self.processor.reset_realtime_audio()
                self.processor.frame_picker.index, self.processor.frame_picker.step = 0, 1
                self.processor.last_frame_index = 0
                logger.info("Avatar cache hit: avatar=%s pid=%s ready in %.3fs (models and idle reused)",
                            avatar, os.getpid(), time.perf_counter() - started)
                return
            logger.info("Avatar cache miss: avatar=%s pid=%s preparing resources", avatar, os.getpid())
            self._load_avatar(avatar, offline_quality, cache_silence_frames)
            signature = idle_dependency_signature(self._idle_dependencies())
            resources = {name: getattr(self, name) for name in
                         ("processor", "silence_frame_count", "silence_size", "idle_path")}
            self.background_cache_key = hashlib.sha256(
                repr(("silence-video-base-v1", signature)).encode()).hexdigest()[:24]
            resources["background_cache_key"] = self.background_cache_key
            resources["render_ms"] = self.timings["render_ms"]
            _PREPARED_AVATARS[key] = (signature, resources)
            logger.info("Avatar %s prepared in %.2fs", avatar, time.perf_counter() - started)

    def _load_avatar(self, avatar, offline_quality, cache_silence_frames):
        silence_path = DATA_DIR / avatar / "silence.mp4"
        if not silence_path.is_file():
            raise FileNotFoundError(
                f"Upload a silent avatar clip to {_relative(silence_path)} before starting realtime streaming. "
                "Use the same frames, order and 25 FPS as the avatar training video."
            )
        # Decode and validate the source before loading expensive model sessions.
        started = time.perf_counter()
        clip = load_silent_clip(silence_path, STREAM_FPS, cache=not self.local_idle)
        self.silence_frame_count = len(clip.frames)
        self.silence_size = clip.size
        self.silent_frames = [_frame_data_url(image) for image in clip.frames] if cache_silence_frames else []
        logger.info("Silent clip ready: %s frames in %.2fs", self.silence_frame_count, time.perf_counter() - started)

        # Initialize models after validation; local idle is rendered once below.
        encoder_options = (
            {"feather_hubert_checkpoint": _relative(FEATHER_HUBERT_CHECKPOINT)}
            if offline_quality else {"encoder_onnx": _relative(FEATHER_HUBERT_ONNX)}
        )
        self.processor = DiHumanProcessor(
            _relative(DATA_DIR / avatar), asr="hubert", **encoder_options,
            include_last_frame=True, defer_models=True, lazy_assets=self.local_idle,
        )
        if self.silence_frame_count != len(self.processor.full_body_img_list):
            raise ValueError(
                f"silence.mp4 has {self.silence_frame_count} frames, but the avatar has "
                f"{len(self.processor.full_body_img_list)} image/landmark pairs. "
                "Upload a silence clip matching the training frames exactly."
            )
        if self.silence_size != self.processor.full_body_img_list[0].shape[:2]:
            raise ValueError("silence.mp4 dimensions must match the avatar training frames. Upload a matching clip.")
        if self.local_idle:
            from frame_store import prepare_source_frames
            try:
                # Decode/compress once before playback. Random access to cached
                # PNGs avoids per-frame MP4 seeks, especially on the reverse leg.
                source = prepare_source_frames(silence_path, clip.frames, OUTPUTS_DIR / "realtime-cache")
                self.processor.full_body_img_list = source
                self.processor.source_image_paths = source.paths
                for name in ("_realtime_input_cache", "_realtime_gpu_inputs"):
                    getattr(self.processor, name, {}).clear()
                logger.info("Idle preparation uses bounded decoded frame caches; %s native frames are streamed/spooled",
                            self.silence_frame_count)
                self.processor.reset_realtime_audio()
                self.idle_path = prepare_inferred_idle_loop(
                    self.processor, STREAM_FPS, OUTPUTS_DIR / "realtime-cache",
                    dependencies=self._idle_dependencies(), offline_quality=offline_quality,
                    source_frames=source,
                    initialize_models=self._initialize_idle_models,
                )
            finally:
                clip.frames.close()
            self.processor.reset_realtime_audio()
        logger.info("Idle ready: avatar=%s source=%s speech_models=%s", avatar,
                    "silent offline inference on silence.mp4" if self.local_idle else "silence.mp4",
                    "loaded" if getattr(self.processor, "_models_initialized", False) else "background preload")

    def _initialize_idle_models(self):
        preload_onnxruntime_cuda()
        ensure_streaming_unet(self.avatar)
        if not self.offline_quality:
            ensure_feather_hubert_onnx()
        self.processor.initialize_models()
        self.processor.enable_realtime_io_binding()

    def start_speech_preload(self):
        """Load already exported models in the background, never execute a model."""
        if getattr(self.processor, "_models_initialized", False):
            return
        with _SPEECH_PRELOAD_LOCK:
            future = getattr(self.processor, "_preload_future", None)
            if future is None or (future.done() and future.exception() is not None):
                self.processor._preload_future = _SPEECH_PRELOAD_POOL.submit(self._preload_speech_models)

    def _preload_speech_models(self):
        try:
            return self._ensure_speech_models(allow_export=False)
        except Exception:
            logger.exception("Background speech model loading failed for avatar %s; will retry on audio", self.avatar)
            raise

    def _ensure_speech_models(self, *, allow_export=True) -> bool:
        with _SPEECH_MODEL_LOCK:
            return self._load_speech_models(allow_export=allow_export)

    def _load_speech_models(self, *, allow_export) -> bool:
        if getattr(self.processor, "_models_initialized", False):
            return True
        if not allow_export:
            unet = DATA_DIR / self.avatar / "unet.onnx"
            checkpoint = CHECKPOINTS_DIR / self.avatar / "last.pth"
            if (not unet.is_file() or (checkpoint.is_file() and unet.stat().st_mtime_ns < checkpoint.stat().st_mtime_ns)
                    or (not self.offline_quality and not FEATHER_HUBERT_ONNX.is_file())):
                logger.info("Speech preload deferred: missing/stale ONNX for %s; export waits for audio", self.avatar)
                return False
        started = time.perf_counter()
        signature_before_loading = idle_dependency_signature(self._idle_dependencies())
        logger.info("Loading speech models: avatar=%s reason=%s (no warmup inference)", self.avatar,
                    "received audio" if allow_export else "background preload")
        preload_onnxruntime_cuda()
        if allow_export:
            ensure_streaming_unet(self.avatar)
            if not self.offline_quality:
                ensure_feather_hubert_onnx()
        self.processor.initialize_models()
        # Deferred export creates/updates unet.onnx after the idle cache entry
        # was published. Keep our own export from invalidating the next session.
        with _AVATAR_PREPARATION_LOCK:
            key = (str((DATA_DIR / self.avatar).resolve()), self.offline_quality)
            cached = _PREPARED_AVATARS.get(key)
            if (cached is not None and cached[0] == signature_before_loading
                    and cached[1]["processor"] is self.processor):
                _PREPARED_AVATARS[key] = (idle_dependency_signature(self._idle_dependencies()), cached[1])
        if self.local_idle:
            self.processor.enable_realtime_io_binding()
            for index in range(min(8, self.silence_frame_count)):
                self.processor.prepare_frame_input(index)
            requested_batch = max(1, min(8, int(os.environ.get("FEATHER_REALTIME_UNET_BATCH", "1"))))
            if requested_batch > 1 and self.processor.supports_realtime_batch():
                self.render_batch_frames = requested_batch
        logger.info("Speech models ready: avatar=%s %.2fs providers=%s", self.avatar,
                    time.perf_counter() - started, self.provider_summary())
        return True

    def provider_summary(self) -> str:
        providers = []
        for session in (getattr(self.processor, "ort_unet", None), getattr(self.processor, "ort_ae", None)):
            if session is not None:
                providers.append("/".join(session.get_providers()))
        if getattr(self.processor, "feather_model", None) is not None:
            providers.append(f"FeatherHuBERT torch/{self.processor.feather_device}")
        return "; ".join(providers) or "deferred until first audio"

    def _flush_resampler(self) -> None:
        """Called with audio_lock held; preserve the filter's final samples."""
        if self.resampler is not None:
            tail = self.resampler.resample_chunk(np.empty(0, np.float32), last=True)
            tail = np.clip(np.rint(tail), -32768, 32767).astype(np.int16)
            self.audio_buffer = np.concatenate([self.audio_buffer, tail])
            self.resampler = None

    def enqueue_audio(self, audio_bytes: bytes, source_rate: int = SAMPLE_RATE) -> None:
        if not audio_bytes:
            return
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        with self.audio_lock:
            if source_rate <= 0:
                raise ValueError("Audio sample rate must be positive.")
            if source_rate != self.source_rate:
                self._flush_resampler()
            if source_rate != SAMPLE_RATE:
                if self.resampler is None:
                    import soxr

                    self.resampler = soxr.ResampleStream(source_rate, SAMPLE_RATE, 1, dtype="float32")
                audio = self.resampler.resample_chunk(audio.astype(np.float32))
                audio = np.clip(np.rint(audio), -32768, 32767).astype(np.int16)
            self.source_rate = source_rate
            self.audio_buffer = np.concatenate([self.audio_buffer, audio])
            self.audio_ended = False
            self.last_audio_at = time.monotonic()

    def queued_ms(self) -> int:
        with self.audio_lock:
            samples = len(self.audio_buffer) + sum(len(audio) for audio, _ in self.prepared_audio)
            if self.resampler is not None:
                samples += self.resampler.delay()
            return round(samples / SAMPLE_RATE * 1000)

    def finish_audio(self) -> None:
        with self.audio_lock:
            self._flush_resampler()
            # Relative offsets in the unprepared PCM queue survive another
            # reply arriving before the previous one has finished playing.
            offset = len(self.audio_buffer)
            if not self.audio_end_offsets or self.audio_end_offsets[-1] != offset:
                self.audio_end_offsets.append(offset)
            self.audio_ended = True

    def clear_audio(self) -> None:
        with self.audio_lock:
            self.audio_buffer = np.empty(0, dtype=np.int16)
            self.prepared_audio.clear()
            self.audio_end_offsets.clear()
            self.resampler = None
            self.audio_generation += 1
            self.audio_ended = True

    def has_prepared_audio(self) -> bool:
        with self.audio_lock:
            return bool(self.prepared_audio)

    def prepare_audio(self) -> None:
        """Only the background producer calls this; never block media delivery."""
        samples = FRAME_LEN * PLAYBACK_CHUNK_FRAMES
        batch = None
        with self.audio_lock:
            while not self.prepared_audio and self.audio_end_offsets and self.audio_end_offsets[0] == 0:
                self.audio_end_offsets.popleft()
                self.utterance_active = False
            flush_tail = self.audio_ended or time.monotonic() - self.last_audio_at >= AUDIO_TAIL_TIMEOUT
            if flush_tail:
                self._flush_resampler()
            available = self.audio_end_offsets[0] if self.audio_end_offsets else len(self.audio_buffer)
            flush_batch = flush_tail or bool(self.audio_end_offsets)
            needed = (self.batch_frames + self.lookahead_frames) * samples + 80
            ready = bool(available and flush_batch) if self.offline_quality else (
                available >= needed or (available and flush_batch)
            )
            if not self.prepared_audio and ready:
                count = (available + samples - 1) // samples
                if not self.offline_quality:
                    count = min(self.batch_frames, count)
                batch = self.audio_buffer[:available if self.offline_quality else min(available, needed)].copy()
                raw_batch = batch
                consumed = min(available, count * samples)
                self.audio_buffer = self.audio_buffer[consumed:]
                self.audio_end_offsets = deque(offset - consumed for offset in self.audio_end_offsets)
                batch = np.pad(batch, (0, max(0, count * samples - len(batch))))
                generation = self.audio_generation

        if batch is not None:
            self._ensure_speech_models()
            logger.info("Speech inference: avatar=%s samples=%s offline_quality=%s",
                        self.avatar, len(raw_batch), self.offline_quality)
            encoding_started = time.perf_counter()
            if not self.utterance_active or generation != self.processor_audio_generation:
                self.processor.reset_realtime_audio()
                self.utterance_active = True
                self.processor_audio_generation = generation
            if self.offline_quality or np.any(batch):
                windows = (self.processor.prepare_offline_features(raw_batch) if self.offline_quality
                           else self.processor.prepare_realtime_features(batch, count))
            else:
                # Keep silent intervals in audio history without running a model.
                history = np.concatenate([
                    self.processor.feather_audio_buffer,
                    batch[:count * samples].astype(np.float32) / 32768,
                ])
                self.processor.feather_audio_buffer = history[-FEATHER_REALTIME_HISTORY_SAMPLES:]
                windows = [None] * count
            self.timings["encoder_ms"] = (time.perf_counter() - encoding_started) * 1000
            with self.audio_lock:
                if generation == self.audio_generation:
                    self.prepared_audio.extend((batch[i * samples:(i + 1) * samples], windows[i]) for i in range(count))

    def _next_image(self, *, prepare: bool = True) -> tuple[np.ndarray | str | None, np.ndarray, bool]:
        if prepare:
            self.prepare_audio()
        samples = FRAME_LEN * PLAYBACK_CHUNK_FRAMES
        with self.audio_lock:
            flush_tail = self.audio_ended or time.monotonic() - self.last_audio_at >= AUDIO_TAIL_TIMEOUT
            if self.prepared_audio:
                audio, features = self.prepared_audio.popleft()
            else:
                audio, features = np.zeros(samples, dtype=np.int16), None
            at_boundary = self.audio_end_offsets and self.audio_end_offsets[0] == 0
            ended = not self.prepared_audio and (at_boundary or (not len(self.audio_buffer) and flush_tail))

        picker = self.processor.frame_picker
        self.current_frame, self.current_direction = picker.index, picker.step
        speaking = bool(np.any(audio))
        if speaking or (self.offline_quality and features is not None):
            rendering_started = time.perf_counter()
            image = self.processor.process_realtime(audio, features)
            self.timings["render_ms"] = (time.perf_counter() - rendering_started) * 1000
        else:
            if not self.silent_frames:
                # A cancel can win the race between producer readiness and pop.
                return None, audio, bool(ended)
            image = self.silent_frames[self.current_frame]
        self.speaking = speaking
        if ended:
            self.utterance_active = False
            if not speaking:
                self.processor.reset_realtime_audio()
        picker.advance()
        return image, audio, bool(ended)

    def next_packet(self) -> tuple[str, np.ndarray]:
        image, audio, _ = self._next_image()
        if image is None:
            raise RuntimeError("Idle frames are played locally; render only prepared speech packets.")
        return _frame_data_url(image) if isinstance(image, np.ndarray) else image, audio

    def next_patch_packet(self, *, prepare: bool = True) -> dict[str, Any] | None:
        generation = self.audio_generation
        image, audio, ended = self._next_image(prepare=prepare)
        if image is None:
            return None
        return self._encode_patch_packet(image, audio, ended, generation, self.current_frame, self.current_direction)

    def _encode_patch_packet(self, image, audio, ended, generation, frame_index, direction) -> dict[str, Any]:
        if isinstance(image, str):
            image = cv2.imdecode(np.frombuffer(base64.b64decode(image.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        xmin, ymin, xmax, ymax = self.processor.bbox_list[frame_index]
        # NumPy clamps slices to image bounds; reflect the same region on canvas.
        height, width = image.shape[:2]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(width, xmax), min(height, ymax)
        encoding_started = time.perf_counter()
        patch_image = _frame_data_url(image[ymin:ymax, xmin:xmax])
        self.timings["encode_ms"] = (time.perf_counter() - encoding_started) * 1000
        self.timings["frames"] += 1
        self.timings["patch_bytes"] += len(patch_image)
        return {
            "type": "frame", "image": patch_image, "audio": _pcm16_to_base64(audio),
            "duration": 1 / STREAM_FPS, "frame_index": frame_index,
            "direction": direction, "speaking": bool(np.any(audio)),
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "background": f"/api/realtime/{self.session_id}/background/{frame_index}",
            "generation": generation, "utterance_end": ended,
        }

    def next_patch_packets(self, count: int, *, prepare: bool = False) -> list[dict[str, Any]]:
        if prepare:
            self.prepare_audio()
        if count <= 1 or not self.offline_quality:
            packet = self.next_patch_packet(prepare=False)
            return [packet] if packet is not None else []
        with self.audio_lock:
            count = min(count, len(self.prepared_audio))
            if not count:
                return []
            generation = self.audio_generation
            queued = [self.prepared_audio.popleft() for _ in range(count)]
            flush_tail = self.audio_ended or time.monotonic() - self.last_audio_at >= AUDIO_TAIL_TIMEOUT
            boundary = self.audio_end_offsets and self.audio_end_offsets[0] == 0
            ended = not self.prepared_audio and (boundary or (not len(self.audio_buffer) and flush_tail))
        positions = []
        picker = self.processor.frame_picker
        for _ in queued:
            positions.append((picker.index, picker.step))
            picker.advance()
        started = time.perf_counter()
        images = self.processor.process_realtime_batch([(index, features) for (index, _), (_, features) in zip(positions, queued)])
        self.timings["render_ms"] = (time.perf_counter() - started) * 1000 / count
        packets = [self._encode_patch_packet(image, audio, bool(ended and i == count - 1), generation, index, direction)
                   for i, (image, (audio, _), (index, direction)) in enumerate(zip(images, queued, positions))]
        self.current_frame, self.current_direction = positions[-1]
        self.speaking = bool(np.any(queued[-1][0]))
        if ended:
            self.utterance_active = False
            if not self.speaking:
                self.processor.reset_realtime_audio()
        return packets


current_stream: AvatarStream | None = None
current_stream_lock = asyncio.Lock()


app = FastAPI(title="FeatherTalk")
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")


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


def _realtime_session(session_id: str) -> AvatarStream:
    stream = current_stream
    if stream is None or stream.session_id != session_id:
        raise HTTPException(status_code=404, detail="Realtime session has ended.")
    return stream


@app.get("/api/realtime/{session_id}/idle")
def realtime_idle(session_id: str) -> Response:
    stream = _realtime_session(session_id)
    if stream.idle_path is None:
        raise HTTPException(status_code=404, detail="Idle video is not ready.")
    video = read_idle_video(stream.idle_path)
    logger.info("Serving idle video: avatar=%s pid=%s bytes=%s", stream.avatar, os.getpid(), len(video))
    return Response(video, media_type="video/webm", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/realtime/idle/{cache_key}")
def realtime_shared_idle(cache_key: str, request: Request):
    """One immutable asset for all users, with native partial video playback."""
    if not re.fullmatch(r"idle-inferred-[0-9a-f]{24}\.webm", cache_key):
        raise HTTPException(status_code=404, detail="Unknown idle cache key.")
    path = OUTPUTS_DIR / "realtime-cache" / cache_key
    video = None
    if path.is_file():
        size = path.stat().st_size
    else:
        try:
            video = read_idle_video(path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Idle video is not prepared.")
        size = len(video)
    headers = {"Cache-Control": "public, max-age=31536000, immutable",
               "ETag": f'"{cache_key}"', "Accept-Ranges": "bytes"}
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    start, end, status = 0, size - 1, 200
    byte_range = request.headers.get("range")
    if request.headers.get("if-range", headers["ETag"]) != headers["ETag"]:
        byte_range = None
    if byte_range:
        try:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", byte_range)
            if match is None or not any(match.groups()):
                raise ValueError("Invalid byte range")
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            else:
                if int(right) <= 0:
                    raise ValueError("Invalid suffix range")
                start = max(0, size - int(right))
            if not 0 <= start <= end < size:
                raise ValueError("Unsatisfiable byte range")
        except ValueError:
            return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(max(0, end - start + 1))

    def chunks():
        if video is not None:
            for offset in range(start, end + 1, 64 * 1024):
                yield video[offset:min(offset + 64 * 1024, end + 1)]
        else:
            with path.open("rb") as file:
                file.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = file.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

    return StreamingResponse(chunks(), status_code=status, media_type="video/webm", headers=headers)


@app.get("/api/realtime/{session_id}/background/{index}")
def realtime_background(session_id: str, index: int) -> Response:
    stream = _realtime_session(session_id)
    if not 0 <= index < len(stream.processor.full_body_img_list):
        raise HTTPException(status_code=404, detail="Unknown background frame.")
    paths = getattr(stream.processor, "source_image_paths", [])
    headers = {"Cache-Control": "private, max-age=86400, immutable"}
    if paths:
        # Original JPEG/PNG bytes: no additional lossy image encoding.
        return FileResponse(paths[index], headers=headers)
    # Custom/in-memory processors can still provide their own frame sequence.
    image = stream.processor.full_body_img_list[index]
    encoded = _frame_data_url(image).split(",", 1)[1]
    return Response(base64.b64decode(encoded), media_type="image/png", headers=headers)


async def _prepare_avatar_stream(websocket: WebSocket, avatar: str):
    """Keep startup traffic flowing while waiting for blocking model work."""
    started = time.monotonic()
    preparation = asyncio.create_task(asyncio.to_thread(AvatarStream, avatar, cache_silence_frames=False))
    try:
        while True:
            done, _ = await asyncio.wait({preparation}, timeout=STREAM_PREPARATION_STATUS_INTERVAL)
            if done:
                return preparation.result()
            elapsed = time.monotonic() - started
            logger.info("Stream preparation still running: avatar=%s elapsed=%.1fs", avatar, elapsed)
            await websocket.send_json({"type": "status", "message":
                                       f"Preparing avatar stream ({elapsed:.0f}s). Loading models / generating cached idle video..."})
    finally:
        # A disconnect cannot cancel a thread/GPU call. Keep ownership until
        # it finishes; the completed cache remains available for the next start.
        if not preparation.done():
            try:
                await asyncio.shield(preparation)
            except Exception:
                logger.exception("Avatar preparation failed after connection ended: avatar=%s", avatar)


@app.websocket("/ws/stream")
async def stream_avatar(websocket: WebSocket, avatar: str) -> None:
    global current_stream

    await websocket.accept()
    if avatar not in list_avatars():
        await websocket.send_json({"type": "error", "message": "Selected avatar does not exist."})
        await websocket.close()
        return

    stream = None
    stream_initialized = False
    producer = None
    tasks = []
    try:
        await websocket.send_json({"type": "status", "message": "Preparing avatar stream..."})
        async with current_stream_lock:
            if current_stream is not None:
                raise RuntimeError("Another avatar stream is active. Stop it before starting a new stream.")
            current_stream = await _prepare_avatar_stream(websocket, avatar)
            stream = current_stream
        if stream.idle_path is None:
            raise RuntimeError("Inferred idle video was not prepared.")
        await websocket.send_json({
            "type": "stream_init", "idle_url": f"/api/realtime/{stream.session_id}/idle",
            "idle_stream_url": f"/api/realtime/idle/{stream.idle_path.name}",
            "idle_cache_key": stream.idle_path.name,
            "background_cache_key": getattr(stream, "background_cache_key", stream.session_id),
            "background_base": f"/api/realtime/{stream.session_id}/background/",
            "frame_count": stream.silence_frame_count, "fps": STREAM_FPS,
            "width": stream.silence_size[1], "height": stream.silence_size[0],
            "generation": stream.audio_generation,
        })
        stream_initialized = True
        stream.start_speech_preload()
        await websocket.send_json(
            {
                "type": "status",
                "message": (
                    "Streaming idle avatar. Offline quality: replies play after audio_end "
                    f"(or an audio timeout). ONNX providers: {stream.provider_summary()}"
                ),
                "ready": True,
            }
        )

        client_ready = asyncio.Event()
        client_idle = asyncio.Event()
        delivery_slots = asyncio.Semaphore(24)
        outstanding = set()
        cursor_requests = {}
        send_lock = asyncio.Lock()

        async def send_message(message) -> None:
            async with send_lock:
                await websocket.send_json(message)

        async def select_cursor(generation: int):
            while not client_idle.is_set():
                if generation != stream.audio_generation:
                    return None
                await asyncio.sleep(0.02)
            if generation != stream.audio_generation:
                return None
            client_idle.clear()
            request_id = uuid.uuid4().hex
            future = asyncio.get_running_loop().create_future()
            cursor_requests[request_id] = future
            try:
                lead = max(0.8, stream.timings["render_ms"] * 12 / 1000 + 0.4)
                await send_message({"type": "prepare_speech", "request_id": request_id,
                                    "generation": generation, "lead_seconds": lead})
                while generation == stream.audio_generation:
                    try:
                        return await asyncio.wait_for(asyncio.shield(future), 0.04)
                    except asyncio.TimeoutError:
                        continue
                return None
            finally:
                cursor_requests.pop(request_id, None)
                if not future.done():
                    future.cancel()

        producer = SpeechProducer(stream, select_cursor=select_cursor)

        async def receive_controls() -> None:
            while True:
                message = await websocket.receive_json()
                if message.get("type") == "ready":
                    client_ready.set()
                    client_idle.set()
                if message.get("type") == "idle" and message.get("generation") == stream.audio_generation:
                    client_idle.set()
                if message.get("type") == "speech_cursor" and message.get("generation") == stream.audio_generation:
                    future = cursor_requests.get(message.get("request_id"))
                    index, direction, start = message.get("frame_index"), message.get("direction"), message.get("playback_start")
                    if (future is not None and not future.done() and isinstance(index, int)
                            and 0 <= index < stream.silence_frame_count and direction in (-1, 1)
                            and isinstance(start, (int, float)) and np.isfinite(start) and start >= 0):
                        future.set_result({"frame_index": index, "direction": direction, "playback_start": start})
                if message.get("type") == "played":
                    key = (message.get("generation"), message.get("frame_id"))
                    if key in outstanding:
                        outstanding.remove(key)
                        delivery_slots.release()
                if message.get("type") == "ping":
                    continue
                if message.get("type") == "stop":
                    return

        async def send_speech() -> None:
            await client_ready.wait()
            producer.start()
            sent_generation = stream.audio_generation
            sent_frames = 0
            warmup = []
            async def sync_generation() -> None:
                nonlocal sent_generation
                if sent_generation == stream.audio_generation:
                    return
                sent_generation = stream.audio_generation
                warmup.clear()
                producer.discard_stale()
                for _ in outstanding:
                    delivery_slots.release()
                outstanding.clear()
                await send_message({"type": "reset", "generation": sent_generation})

            while True:
                await sync_generation()
                try:
                    packet = await asyncio.wait_for(producer.get(), 0.04)
                except asyncio.TimeoutError:
                    continue
                if packet["generation"] != stream.audio_generation:
                    continue
                # Prime the producer buffer once per utterance. Short replies can
                # begin as soon as their last frame is ready.
                warmup.append(packet)
                if len(warmup) < 6 and not packet["utterance_end"]:
                    continue
                batch = warmup[:]
                warmup.clear()
                for frame in batch:
                    if frame["generation"] != stream.audio_generation:
                        break
                    acquired = False
                    while not acquired and frame["generation"] == stream.audio_generation:
                        try:
                            await asyncio.wait_for(delivery_slots.acquire(), 0.04)
                            acquired = True
                        except asyncio.TimeoutError:
                            await sync_generation()
                    if not acquired:
                        break
                    if frame["generation"] != stream.audio_generation:
                        delivery_slots.release()
                        break
                    frame["frame_id"] = sent_frames
                    outstanding.add((frame["generation"], frame["frame_id"]))
                    # The browser's audio clock controls playback. Fill the
                    # bounded delivery window as frames become ready; pacing
                    # here adds transport/scheduler latency to every 40 ms
                    # block and eventually drains the client's jitter buffer.
                    await send_message(frame)
                    sent_frames += 1
                    if sent_frames % 125 == 0:
                        await send_message({"type": "metrics", **stream.timings,
                                                   "queued_frames": producer.queue.qsize(),
                                                   "io_binding": bool(getattr(stream.processor, "realtime_io_binding", False)),
                                                   "batch_frames": stream.render_batch_frames,
                                                   "audio_queue_ms": stream.queued_ms()})

        tasks = [asyncio.create_task(receive_controls()), asyncio.create_task(send_speech())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    except WebSocketDisconnect as exc:
        logger.warning("Avatar stream disconnected: avatar=%s stage=%s code=%s reason=%s", avatar,
                       "streaming" if stream_initialized else "startup", exc.code, exc.reason)
    except Exception as exc:
        logger.exception("Avatar stream failed: avatar=%s stage=%s", avatar,
                         "streaming" if stream_initialized else "startup")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if producer:
            await producer.close()
        async with current_stream_lock:
            if current_stream is stream:
                current_stream = None
        with contextlib.suppress(Exception):
            await websocket.close()


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
                received_sample_rate = REALTIME_SOURCE_SAMPLE_RATE
                async for message in realtime_ws:
                    audio, sample_rate = _extract_realtime_audio_chunk(message, received_sample_rate)
                    if audio:
                        received_sample_rate = sample_rate
                        stream.enqueue_audio(audio, source_rate=sample_rate)
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
                        received_sample_rate = _extract_sample_rate(event) or received_sample_rate
                        if event.get("type") == "error":
                            error = event.get("message") or event.get("error") or "Realtime voice error."
                            await websocket.send_json({"type": "error", "message": str(error)})
                        elif event.get("type") == "session_updated":
                            await websocket.send_json({"type": "status", "message": "Realtime voice session ready."})
                        elif event.get("type") == "audio_end":
                            stream.finish_audio()
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
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        stream.clear_audio()
        with contextlib.suppress(Exception):
            await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
