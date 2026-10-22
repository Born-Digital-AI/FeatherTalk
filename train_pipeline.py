#!/usr/bin/env python3
"""Background training pipeline: TBDub → FeatherTalk training → email notification."""
from __future__ import annotations

import argparse
import os
import shutil
import smtplib
import subprocess
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TBDUB_DIR = Path("/opt/digital-human/TBDub")
TBDUB_SCRIPT = TBDUB_DIR / "scripts" / "tbdub.sh"
TBDUB_AUDIO = TBDUB_DIR / "data" / "driving_audio.wav"
UPLOADS_DIR = ROOT_DIR / "uploads"


def _load_dotenv() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _send_email(to: str, avatar_name: str, app_url: str, *, success: bool, error: str = "") -> None:
    server = os.environ.get("SMTP_SERVER", "")
    port_str = os.environ.get("SMTP_PORT", "587")
    sender = os.environ.get("SMTP_SENDER", "")
    password = os.environ.get("SMTP_PWD", "")

    if not server or not sender:
        print("SMTP not configured; skipping email notification.", file=sys.stderr)
        return

    try:
        port = int(port_str)
    except ValueError:
        port = 587

    if success:
        subject = f"FeatherTalk: Avatar '{avatar_name}' is ready"
        body_html = f"""\
<html><body style="font-family:sans-serif;color:#1a1a1a;background:#f5f5f5;padding:32px;">
<div style="max-width:560px;margin:0 auto;background:#ffffff;border-radius:8px;padding:40px;">
  <h2 style="color:#2f8a66;margin-top:0;">Avatar ready: {avatar_name}</h2>
  <p>Your avatar <strong>{avatar_name}</strong> has been trained successfully and is ready to use in FeatherTalk.</p>
  <p style="margin-top:32px;">
    <a href="{app_url}" style="display:inline-block;padding:12px 28px;background:#2f8a66;color:#ffffff;text-decoration:none;border-radius:6px;font-weight:700;">Open FeatherTalk</a>
  </p>
</div>
</body></html>"""
    else:
        subject = f"FeatherTalk: Avatar '{avatar_name}' training failed"
        safe_error = error.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body_html = f"""\
<html><body style="font-family:sans-serif;color:#1a1a1a;background:#f5f5f5;padding:32px;">
<div style="max-width:560px;margin:0 auto;background:#ffffff;border-radius:8px;padding:40px;">
  <h2 style="color:#c0392b;margin-top:0;">Training failed: {avatar_name}</h2>
  <p>Training of avatar <strong>{avatar_name}</strong> failed with the following error:</p>
  <pre style="background:#f8f8f8;border:1px solid #ddd;padding:12px;border-radius:4px;overflow-x:auto;white-space:pre-wrap;">{safe_error}</pre>
  <p style="margin-top:32px;">
    <a href="{app_url}" style="display:inline-block;padding:12px 28px;background:#2f8a66;color:#ffffff;text-decoration:none;border-radius:6px;font-weight:700;">Open FeatherTalk</a>
  </p>
</div>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    try:
        if port == 465:
            with smtplib.SMTP_SSL(server, port, timeout=30) as smtp:
                smtp.login(sender, password)
                smtp.sendmail(sender, [to], msg.as_string())
        else:
            with smtplib.SMTP(server, port, timeout=30) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(sender, password)
                smtp.sendmail(sender, [to], msg.as_string())
        print(f"Email notification sent to {to}")
    except Exception as exc:
        print(f"Failed to send email to {to}: {exc}", file=sys.stderr)


def _run(cmd: list[str], cwd: Path, log_file, prefix: str = "", env: dict | None = None) -> None:
    cmd_str = " ".join(str(c) for c in cmd)
    header = f"{prefix}$ {cmd_str}"
    print(header, file=log_file, flush=True)
    print(header, flush=True)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
        sys.stdout.write(prefix + line)
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command exited with code {proc.returncode}: {cmd_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FeatherTalk avatar training pipeline")
    parser.add_argument("--avatar", required=True, help="Avatar name")
    parser.add_argument("--video", required=True, help="Path to the silent training video")
    parser.add_argument("--email", required=True, help="Notification email address")
    parser.add_argument("--app-url", default="https://feathertalk.ultimate.borndigital.ai/",
                        help="FeatherTalk URL for email button")
    args = parser.parse_args()

    _load_dotenv()

    avatar_name = args.avatar
    silent_video = Path(args.video)
    app_url = os.environ.get("APP_URL", args.app_url)

    conda_base = Path(os.environ.get("CONDA_BASE", "/home/ultimate/miniconda3"))
    tbdub_conda_env = os.environ.get("TBDUB_CONDA_ENV", "tbdub")
    tbdub_bin = conda_base / "envs" / tbdub_conda_env / "bin"
    tbdub_env = {**os.environ, "PATH": f"{tbdub_bin}:{os.environ.get('PATH', '')}"}
    for _var in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        tbdub_env.pop(_var, None)

    UPLOADS_DIR.mkdir(exist_ok=True)
    log_path = UPLOADS_DIR / f"{avatar_name}_train.log"
    tbdub_output = UPLOADS_DIR / f"{avatar_name}_tbdub_out.mp4"
    data_dir = ROOT_DIR / "data" / avatar_name
    data_dir.mkdir(parents=True, exist_ok=True)

    error_msg = ""
    success = False

    tag = f"[training:{avatar_name}] "

    def log_print(msg: str) -> None:
        print(msg, file=log, flush=True)
        print(tag + msg, flush=True)

    with open(log_path, "w", buffering=1) as log:
        try:
            log_print(f"=== FeatherTalk training pipeline: {avatar_name} ===")

            log_print(f"[1/3] TBDub: {silent_video.name} -> {tbdub_output.name}")
            if not TBDUB_SCRIPT.is_file():
                raise FileNotFoundError(f"TBDub script not found: {TBDUB_SCRIPT}")
            if not TBDUB_AUDIO.is_file():
                raise FileNotFoundError(f"TBDub driving audio not found: {TBDUB_AUDIO}")
            _run(
                ["bash", TBDUB_SCRIPT, str(silent_video), str(TBDUB_AUDIO), str(tbdub_output)],
                cwd=TBDUB_DIR,
                log_file=log,
                prefix=tag,
                env=tbdub_env,
            )

            log_print(f"[2/3] Moving files to data/{avatar_name}/")
            shutil.move(str(tbdub_output), str(data_dir / "train.mp4"))
            shutil.move(str(silent_video), str(data_dir / "silence.mp4"))
            log_print(f"  silence.mp4 -> data/{avatar_name}/silence.mp4")
            log_print(f"  train.mp4   -> data/{avatar_name}/train.mp4")

            log_print(f"[3/3] Training avatar: {avatar_name}")
            _run(["bash", "scripts/train_avatar.sh", avatar_name], cwd=ROOT_DIR, log_file=log, prefix=tag)

            log_print("=== Pipeline complete ===")
            success = True

        except Exception as exc:
            error_msg = str(exc)
            log_print(f"=== PIPELINE FAILED: {exc} ===")
            if tbdub_output.exists():
                tbdub_output.unlink(missing_ok=True)

    _send_email(args.email, avatar_name, app_url, success=success, error=error_msg)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
