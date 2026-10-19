/* Native idle playback and lossless speech patches on one audio timeline. */
globalThis.AvatarPlayer = class AvatarPlayer {
  static speechStartFrames = 12;
  static idleBlobs = new Map();
  static backgroundImages = new Map();

  static decodePacket(data) {
    if (typeof data === "string") return JSON.parse(data);
    if (!(data instanceof ArrayBuffer) || data.byteLength < 4) throw new Error("Invalid binary packet");
    const headerSize = new DataView(data).getUint32(0);
    const imageStart = 4 + headerSize;
    if (!headerSize || imageStart > data.byteLength) throw new Error("Truncated packet header");
    const frame = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array(data, 4, headerSize)));
    const imageSize = frame.png_bytes;
    if (frame.type !== "frame" || !Number.isSafeInteger(imageSize) || imageSize <= 0 ||
        imageSize > data.byteLength - imageStart || (data.byteLength - imageStart - imageSize) % 2) {
      throw new Error("Invalid packet media lengths");
    }
    frame.image = new Blob([new Uint8Array(data, imageStart, imageSize)], { type: "image/png" });
    frame.audio = new Uint8Array(data, imageStart + imageSize);
    return frame;
  }

  static idleStore(mode, key, blob) {
    if (typeof indexedDB === "undefined") return Promise.resolve(null);
    return new Promise(resolve => {
      let database, timer, finished = false;
      const finish = value => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        database?.close();
        resolve(value);
      };
      // Storage failure/blocking must not prevent playback. Only the most
      // recently downloaded idle clip is persisted, bounding disk usage.
      timer = setTimeout(() => finish(null), 500);
      try {
        const request = indexedDB.open("feathertalk-idle", 1);
        request.onupgradeneeded = () => request.result.createObjectStore("clips");
        request.onerror = request.onblocked = () => finish(null);
        request.onsuccess = () => {
          database = request.result;
          if (finished) { database.close(); return; }
          try {
            const transaction = database.transaction("clips", mode === "read" ? "readonly" : "readwrite");
            const store = transaction.objectStore("clips");
            let result = null;
            if (mode === "read") {
              const read = store.get(key);
              read.onsuccess = () => { result = read.result || null; };
            } else {
              store.clear();
              store.put(blob, key);
            }
            transaction.oncomplete = () => finish(result);
            transaction.onerror = transaction.onabort = () => finish(null);
          } catch (_) { finish(null); }
        };
      } catch (_) { finish(null); }
    });
  }

  static loadIdleBlob(config) {
    const key = config.idle_cache_key || config.idle_url;
    let loading = this.idleBlobs.get(key);
    if (!loading) {
      loading = (async () => {
        const stored = await this.idleStore("read", key);
        if (stored) return stored;
        // A stopped player can leave this download running for the next start.
        const response = await fetch(config.idle_url);
        if (!response.ok) throw new Error("Cannot load the local idle video.");
        const blob = await response.blob();
        void this.idleStore("write", key, blob);
        return blob;
      })().catch(error => {
        if (this.idleBlobs.get(key) === loading) this.idleBlobs.delete(key);
        throw error;
      });
    }
    this.idleBlobs.delete(key);
    this.idleBlobs.set(key, loading);
    while (this.idleBlobs.size > 2) this.idleBlobs.delete(this.idleBlobs.keys().next().value);
    return loading;
  }

  constructor({ canvas, idleVideo, context, onReady, onError, onPlayed, onIdle }) {
    Object.assign(this, { canvas, idleVideo, context, onReady, onError, onPlayed, onIdle });
    this.drawing = canvas.getContext("2d");
    this.backgrounds = new Map();
    this.sources = new Set();
    this.pending = [];
    this.scheduled = [];
    this.decodeChain = Promise.resolve();
    this.generation = 0;
    this.epoch = 0;
    this.visible = true;
    this.stopped = false;
    this.speaking = false;
    this.nextPlayTime = 0;
    this.lastDisplayed = null;
    this.returningIdle = false;
    this.handoff = 0;
    this.abort = new AbortController();
  }

  setVisible(visible) {
    this.visible = visible;
    this.canvas.style.display = visible && this.config ? "block" : "none";
    this.idleVideo.style.display = "none";
  }

  async configure(config) {
    const epoch = this.epoch;
    try {
      this.config = config;
      this.generation = config.generation;
      this.canvas.width = config.width;
      this.canvas.height = config.height;
      if (config.idle_stream_url) {
        // Native HTTP range playback can start after the first buffered frames;
        // every user shares this asset URL and the browser's normal HTTP cache.
        this.idleVideo.src = config.idle_stream_url;
        this.idleVideo.preload = "auto";
      } else {
        const blob = await AvatarPlayer.loadIdleBlob(config);
        if (epoch !== this.epoch || this.stopped) return;
        this.idleObjectUrl = URL.createObjectURL(blob);
        this.idleVideo.src = this.idleObjectUrl;
      }
      this.idleVideo.loop = true;
      this.idleVideo.muted = true;
      this.idleVideo.playsInline = true;
      if (this.idleVideo.readyState < 2) {
        await new Promise((resolve, reject) => {
          const cleanup = () => {
            this.idleVideo.removeEventListener("loadeddata", loaded);
            this.idleVideo.removeEventListener("error", failed);
          };
          const loaded = () => { cleanup(); resolve(); };
          const failed = () => { cleanup(); reject(new Error("Cannot decode idle video. Use a browser supporting VP9/WebM.")); };
          this.idleVideo.addEventListener("loadeddata", loaded);
          this.idleVideo.addEventListener("error", failed);
        });
      }
      if (epoch !== this.epoch || this.stopped) return;
      await this.idleVideo.play();
      if (epoch !== this.epoch || this.stopped) return;
      this.drawIdle();
      if (typeof this.idleVideo.requestVideoFrameCallback === "function") {
        const onFrame = () => {
          if (this.stopped) return;
          this.idleFrameDirty = true;
          this.videoFrameCallback = this.idleVideo.requestVideoFrameCallback(onFrame);
        };
        this.videoFrameCallback = this.idleVideo.requestVideoFrameCallback(onFrame);
      }
      this.setVisible(this.visible);
      this.onReady();
      this.raf = requestAnimationFrame(() => this.tick());
    } catch (error) {
      if (!this.stopped && epoch === this.epoch) this.onError(error.message);
    }
  }

  async decodeImage(url) {
    const image = new Image();
    const objectUrl = typeof url === "string" ? null : URL.createObjectURL(url);
    try {
      image.src = objectUrl || url;
      await image.decode();
      return image;
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    }
  }

  background(url) {
    const key = this.config?.background_cache_key
      ? `${this.config.background_cache_key}:${url.slice(url.lastIndexOf("/") + 1)}` : url;
    let image = AvatarPlayer.backgroundImages.get(key) || this.backgrounds.get(url);
    if (!image) image = this.decodeImage(url);
    AvatarPlayer.backgroundImages.delete(key);
    AvatarPlayer.backgroundImages.set(key, image);
    image.catch(() => {
      if (AvatarPlayer.backgroundImages.get(key) === image) AvatarPlayer.backgroundImages.delete(key);
      this.backgrounds.delete(url);
    });
    while (AvatarPlayer.backgroundImages.size > 64) {
      AvatarPlayer.backgroundImages.delete(AvatarPlayer.backgroundImages.keys().next().value);
    }
    this.backgrounds.delete(url);
    this.backgrounds.set(url, image);
    while (this.backgrounds.size > 32) this.backgrounds.delete(this.backgrounds.keys().next().value);
    return image;
  }

  prefetch(frame, count = 8) {
    if (!this.config) return;
    let index = frame.frame_index, direction = frame.direction;
    for (let step = 0; step < count; step++) {
      index += direction;
      if (index >= this.config.frame_count - 1) direction = -1;
      if (index <= 0) direction = 1;
      this.background(this.config.background_base + index).catch(() => {});
    }
  }

  pcmBuffer(encoded) {
    let bytes = encoded;
    if (typeof encoded === "string") {
      const raw = atob(encoded);
      bytes = new Uint8Array(raw.length);
      for (let index = 0; index < raw.length; index++) bytes[index] = raw.charCodeAt(index);
    }
    const pcm = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const samples = bytes.byteLength / 2;
    const buffer = this.context.createBuffer(1, samples, 16000);
    const channel = buffer.getChannelData(0);
    for (let index = 0; index < samples; index++) channel[index] = pcm.getInt16(index * 2, true) / 32768;
    return buffer;
  }

  reserveSpeech(message) {
    if (this.stopped || !this.config || message.generation !== this.generation) return null;
    const cycle = 2 * this.config.frame_count - 2;
    const origin = this.idleVideo.currentTime;
    // Choose an actual frame boundary, rather than a point inside a frame.
    const future = Math.ceil((origin + message.lead_seconds) * this.config.fps);
    const sequence = future % cycle;
    const cursor = {
      type: "speech_cursor", request_id: message.request_id, generation: this.generation,
      frame_index: sequence < this.config.frame_count ? sequence : cycle - sequence,
      direction: sequence < this.config.frame_count - 1 ? 1 : -1,
      playback_start: this.context.currentTime + future / this.config.fps - origin,
    };
    this.speechReservation = {
      predecessor: (sequence + cycle - 1) % cycle,
      distance: Math.max(0, (future - 1) / this.config.fps - origin),
      lastMediaTime: origin, progress: 0, holding: false,
    };
    this.background(this.config.background_base + cursor.frame_index).catch(() => {});
    this.prefetch(cursor, AvatarPlayer.speechStartFrames - 1);
    return cursor;
  }

  prepareSpeechHandoff() {
    const reservation = this.speechReservation;
    if (!reservation || reservation.holding) return;
    const cycle = 2 * this.config.frame_count - 2;
    let elapsed = this.idleVideo.currentTime - reservation.lastMediaTime;
    if (elapsed < 0) elapsed += cycle / this.config.fps;
    reservation.progress += elapsed;
    reservation.lastMediaTime = this.idleVideo.currentTime;
    if (reservation.progress + 1e-6 < reservation.distance) return;
    // Native video progress owns the handoff. If preparation runs late, keep
    // the exact predecessor visible rather than letting idle outrun speech.
    reservation.holding = true;
    this.idleVideo.pause();
    const current = Math.floor(this.idleVideo.currentTime * this.config.fps) % cycle;
    if (current !== reservation.predecessor) {
      this.idleVideo.currentTime = (reservation.predecessor + 0.05) / this.config.fps;
    }
  }

  enqueue(frame) {
    if (this.stopped || frame.generation !== this.generation) return;
    const epoch = this.epoch;
    // Decode in parallel, then commit in packet order. Audio never starts while
    // its corresponding image is still waiting for decode.
    const ready = Promise.all([this.background(frame.background), this.decodeImage(frame.image)]);
    this.prefetch(frame);
    // Attach a rejection handler immediately, even if an earlier decode is slow.
    ready.catch(() => {});
    this.decodeChain = this.decodeChain.then(async () => {
      const [background, patch] = await ready;
      if (epoch !== this.epoch || this.stopped || frame.generation !== this.generation) return;
      this.pending.push({ ...frame, backgroundImage: background, patchImage: patch, audioBuffer: this.pcmBuffer(frame.audio) });
      this.schedule();
    }).catch(error => {
      if (!this.stopped && epoch === this.epoch) this.onError(`Cannot prepare speech playback: ${error.message}`);
    });
  }

  schedule() {
    if (!this.pending.length) return;
    if (!this.speaking) {
      if (this.pending.length < AvatarPlayer.speechStartFrames && !this.pending.some(frame => frame.utterance_end)) return;
      if (this.speechReservation && (!this.speechReservation.holding || this.idleVideo.seeking || this.idleVideo.readyState < 2)) return;
    }
    const now = this.context.currentTime;
    // Apply a safety lead only at startup or after actual starvation. Moving
    // a still-future deadline to now + 10 ms inserts audible gaps between PCM
    // blocks as the initial buffer drains during a long utterance.
    if (!this.speaking) {
      const reserved = this.pending[0].playback_start;
      this.nextPlayTime = Math.max(this.nextPlayTime, reserved ?? now + 0.12, now + 0.01);
    } else if (this.nextPlayTime <= now) {
      this.nextPlayTime = now + 0.01;
    }
    this.speaking = true;
    for (const frame of this.pending.splice(0)) {
      const source = this.context.createBufferSource();
      source.buffer = frame.audioBuffer;
      source.connect(this.context.destination);
      source.onended = () => this.sources.delete(source);
      this.sources.add(source);
      frame.playTime = this.nextPlayTime;
      frame.endTime = frame.playTime + frame.audioBuffer.duration;
      source.start(frame.playTime);
      this.scheduled.push(frame);
      this.nextPlayTime = frame.endTime;
    }
  }

  drawIdle(force = true) {
    if (!this.config || this.idleVideo.readyState < 2 || this.idleVideo.seeking) return;
    const phase = Math.floor(this.idleVideo.currentTime * this.config.fps);
    if (!force) {
      if (this.videoFrameCallback != null ? !this.idleFrameDirty : this.lastIdlePhase === phase) return;
    }
    this.idleFrameDirty = false;
    this.lastIdlePhase = phase;
    this.drawing.drawImage(this.idleVideo, 0, 0, this.canvas.width, this.canvas.height);
    const cycle = 2 * this.config.frame_count - 2;
    const sequence = Math.floor(this.idleVideo.currentTime * this.config.fps) % cycle;
    this.canvas.dataset.frame = String(sequence < this.config.frame_count ? sequence : cycle - sequence);
    this.canvas.dataset.direction = String(sequence < this.config.frame_count - 1 ? 1 : -1);
    this.canvas.dataset.mode = "idle";
  }

  prepareIdleSeek(afterFrame) {
    if (this.idleSeek?.afterFrame === afterFrame && !this.idleSeek.cancelled) return this.idleSeek;
    this.cancelSeek?.();
    this.idleVideo.pause();
    const cycle = 2 * this.config.frame_count - 2;
    const phase = afterFrame.direction > 0 ? afterFrame.frame_index : cycle - afterFrame.frame_index;
    // The ping-pong phase advances once, including at either endpoint.
    const target = (((phase + 1) % cycle) + 0.05) / this.config.fps;
    const state = { afterFrame, cancelled: false };
    this.idleSeek = state;
    state.promise = new Promise((resolve, reject) => {
      const cleanup = () => {
        this.idleVideo.removeEventListener("seeked", seeked);
        this.idleVideo.removeEventListener("error", failed);
        if (this.cancelSeek === cancel) this.cancelSeek = null;
      };
      const seeked = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error("Cannot seek idle video.")); };
      const cancel = () => { state.cancelled = true; cleanup(); resolve(); };
      this.cancelSeek = cancel;
      this.idleVideo.addEventListener("seeked", seeked);
      this.idleVideo.addEventListener("error", failed);
      this.idleVideo.currentTime = target;
      if (!this.idleVideo.seeking && this.idleVideo.readyState >= 2) seeked();
    });
    state.promise.catch(() => {});
    return state;
  }

  async showIdle(afterFrame = null) {
    const handoff = ++this.handoff;
    if (this.stopped || !this.config) { this.cancelSeek?.(); return; }
    this.returningIdle = true;
    try {
      if (afterFrame) {
        const seek = this.prepareIdleSeek(afterFrame);
        await seek.promise;
        if (seek.cancelled) return;
      } else {
        this.cancelSeek?.();
      }
      if (handoff !== this.handoff || this.stopped) return;
      this.idleSeek = null;
      // Keep the last speaking image visible throughout the asynchronous seek.
      this.drawIdle();
      this.lastDisplayed = null;
      this.speaking = false;
      this.returningIdle = false;
      this.setVisible(this.visible);
      await this.idleVideo.play();
      if (handoff === this.handoff && !this.stopped) this.onIdle?.(this.generation);
    } catch (error) {
      if (handoff === this.handoff && !this.stopped) {
        this.returningIdle = false;
        this.onError(error.message);
      }
    }
  }

  tick() {
    if (this.stopped) return;
    const now = this.context.currentTime;
    if (!this.lastDisplayed && !this.returningIdle && this.speechReservation) {
      this.prepareSpeechHandoff();
      this.schedule();
    }
    let due = null;
    while (this.scheduled.length && this.scheduled[0].playTime <= now) {
      due = this.scheduled.shift();
      this.onPlayed(due); // Releases the server's bounded delivery window.
    }
    if (due) {
      this.speechReservation = null;
      if (this.returningIdle) {
        this.cancelSeek?.();
        this.idleSeek = null;
      }
      this.handoff++;
      this.returningIdle = false;
      this.idleVideo.pause();
      this.speaking = true;
      this.lastDisplayed = due;
      this.setVisible(this.visible);
      this.drawing.drawImage(due.backgroundImage, 0, 0, this.canvas.width, this.canvas.height);
      this.drawing.drawImage(due.patchImage, ...due.bbox);
      this.canvas.dataset.frame = String(due.frame_index);
      this.canvas.dataset.direction = String(due.direction);
      this.canvas.dataset.mode = "speaking";
    }
    if (this.lastDisplayed && !this.returningIdle) {
      const finalFrame = this.lastDisplayed.utterance_end
        ? this.lastDisplayed : this.scheduled.find(frame => frame.utterance_end);
      // Decode the next idle frame while speech audio is still playing.
      if (finalFrame) this.prepareIdleSeek(finalFrame);
    }
    if (!this.lastDisplayed && !this.returningIdle) this.drawIdle(false);
    if (this.lastDisplayed?.utterance_end && now >= this.lastDisplayed.endTime && !this.scheduled.length && this.speaking && !this.returningIdle) {
      this.showIdle(this.lastDisplayed);
    }
    this.raf = requestAnimationFrame(() => this.tick());
  }

  resetSpeech(generation) {
    this.generation = generation;
    this.epoch++;
    this.pending = [];
    this.speechReservation = null;
    this.scheduled = [];
    this.decodeChain = Promise.resolve();
    for (const source of this.sources) {
      try { source.stop(); } catch (_) { /* Source may already have ended. */ }
    }
    this.sources.clear();
    this.nextPlayTime = 0;
    this.showIdle(this.lastDisplayed);
  }

  stop() {
    this.stopped = true;
    this.abort.abort();
    this.resetSpeech(this.generation + 1);
    cancelAnimationFrame(this.raf);
    if (this.videoFrameCallback != null) {
      this.idleVideo.cancelVideoFrameCallback(this.videoFrameCallback);
      this.videoFrameCallback = null;
    }
    this.idleVideo.pause();
    this.idleVideo.removeAttribute("src");
    this.idleVideo.load();
    if (this.idleObjectUrl) URL.revokeObjectURL(this.idleObjectUrl);
    this.backgrounds.clear();
    this.canvas.style.display = "none";
    this.idleVideo.style.display = "none";
  }
};
