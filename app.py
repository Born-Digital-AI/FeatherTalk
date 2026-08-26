from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse


ROOT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = ROOT_DIR / "audio"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = ROOT_DIR / "data"
FEATHER_HUBERT_SCRIPT = ROOT_DIR / "data_utils" / "feather_hubert" / "feather_hubert.py"
FEATHER_HUBERT_CHECKPOINT = (
    ROOT_DIR / "assets" / "featherhubert" / "feather_hubert_188_latest_99.pth"
)
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav"}


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FeatherTalk</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #15181e;
      background: #f5f6f8;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; }
    main {
      width: min(920px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 48px 0;
    }
    h1 {
      margin: 0 0 28px;
      font-size: 32px;
      line-height: 1.1;
      font-weight: 700;
      letter-spacing: 0;
    }
    form {
      display: grid;
      gap: 18px;
      padding: 24px;
      border: 1px solid #d9dde5;
      border-radius: 8px;
      background: #ffffff;
    }
    label {
      display: grid;
      gap: 8px;
      font-size: 14px;
      font-weight: 600;
    }
    select,
    input[type="file"],
    button,
    textarea {
      width: 100%;
      border: 1px solid #c5cad4;
      border-radius: 6px;
      font: inherit;
      font-size: 15px;
    }
    select,
    input[type="file"],
    button {
      min-height: 42px;
      padding: 8px 10px;
      background: #ffffff;
    }
    button {
      cursor: pointer;
      border-color: #176b52;
      background: #176b52;
      color: #ffffff;
      font-weight: 700;
    }
    button:disabled {
      cursor: not-allowed;
      border-color: #b7bcc7;
      background: #d8dce3;
      color: #6f7785;
    }
    textarea {
      min-height: 150px;
      resize: vertical;
      padding: 12px;
      color: #252a33;
      background: #f9fafb;
    }
    video {
      display: none;
      width: min(100%, 720px);
      max-height: calc(100vh - 340px);
      margin-top: 20px;
      margin-left: auto;
      margin-right: auto;
      border-radius: 8px;
      background: #000000;
      object-fit: contain;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    @media (max-width: 720px) {
      main { width: min(100vw - 24px, 920px); padding: 24px 0; }
      form { padding: 16px; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <h1>FeatherTalk</h1>
    <form id="form">
      <div class="row">
        <label>
          Avatar
          <select id="avatar" name="avatar"></select>
        </label>
        <label>
          Audio
          <input id="audio" name="audio" type="file" accept=".mp3,.wav,audio/mpeg,audio/wav">
        </label>
      </div>
      <button id="generate" type="submit" disabled>Generate</button>
      <label>
        Status
        <textarea id="status" readonly>Upload an .mp3 or .wav file.</textarea>
      </label>
    </form>
    <video id="video" controls></video>
  </main>
  <script>
    const avatar = document.querySelector("#avatar");
    const audio = document.querySelector("#audio");
    const generate = document.querySelector("#generate");
    const form = document.querySelector("#form");
    const statusBox = document.querySelector("#status");
    const video = document.querySelector("#video");

    let uploadedAudioPath = null;

    function setStatus(value) {
      statusBox.value = value;
    }

    async function loadAvatars() {
      const response = await fetch("/api/avatars");
      const data = await response.json();
      avatar.replaceChildren();
      for (const name of data.avatars) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        avatar.appendChild(option);
      }
      if (!data.avatars.length) {
        setStatus("No avatars found in checkpoints/.");
      }
    }

    audio.addEventListener("change", async () => {
      generate.disabled = true;
      uploadedAudioPath = null;
      video.style.display = "none";

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
      setStatus("Generating...");

      const body = new FormData();
      body.append("avatar", avatar.value);
      body.append("audio_path", uploadedAudioPath);

      const response = await fetch("/api/generate", { method: "POST", body });
      const data = await response.json();
      setStatus(data.log || data.detail || "Generation finished.");
      generate.disabled = false;

      if (response.ok && data.video_url) {
        video.src = data.video_url;
        video.style.display = "block";
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
