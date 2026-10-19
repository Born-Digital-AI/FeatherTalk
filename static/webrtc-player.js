/* One native video element owns decoded video, audio and their synchronization. */
globalThis.NativeAvatarPlayer = class NativeAvatarPlayer {
  constructor({ video, onReady, onError, onStatus, send }) {
    Object.assign(this, { video, onReady, onError, onStatus, send });
    this.stopped = false;
    this.pendingIce = [];
    this.visible = true;
    this.ready = false;
    this.video.loop = false;
    this.video.muted = false;
    this.video.autoplay = true;
    this.video.playsInline = true;
    this.loaded = () => {
      if (!this.stopped && !this.ready && !this.video.paused && this.video.readyState >= 2 &&
          this.media?.getVideoTracks().length && this.media?.getAudioTracks().length) {
        this.ready = true;
        this.video.controls = false;
        this.onReady();
      }
    };
    this.video.addEventListener("playing", this.loaded);
    this.video.addEventListener("loadeddata", this.loaded);
  }

  setVisible(visible) {
    this.visible = visible;
    this.video.style.display = visible ? "block" : "none";
  }

  async handle(message) {
    if (this.stopped) return;
    if (message.type === "media_init") {
      if (this.peer) throw new Error("Media connection is already configured");
      this.media = new MediaStream();
      this.peer = new RTCPeerConnection({ iceServers: message.ice_servers || [] });
      this.video.srcObject = this.media;
      this.peer.ontrack = event => {
        if (this.stopped) return;
        this.media.addTrack(event.track);
        this.video.play().then(this.loaded).catch(() => {
          if (!this.stopped) {
            this.video.controls = true;
            this.onStatus?.("Tap Play to start the avatar and audio.");
          }
        });
      };
      this.peer.onicecandidate = event => {
        if (!this.stopped && event.candidate) this.send({ type: "ice", candidate: event.candidate.toJSON() });
      };
      this.peer.onconnectionstatechange = () => {
        if (!this.stopped && this.peer.connectionState === "failed") {
          const ice = this.peer.iceConnectionState;
          this.onError(ice === "failed"
            ? "Avatar media connection failed (ICE). Check server UDP reachability and STUN/TURN configuration."
            : `Avatar media connection failed (ICE: ${ice}).`);
        }
      };
    } else if (message.type === "offer") {
      await this.peer.setRemoteDescription({ type: "offer", sdp: message.sdp });
      for (const candidate of this.pendingIce.splice(0)) {
        if (this.stopped) return;
        await this.peer.addIceCandidate(candidate);
      }
      if (this.stopped) return;
      for (const transceiver of this.peer.getTransceivers()) transceiver.direction = "recvonly";
      const answer = await this.peer.createAnswer();
      if (this.stopped) return;
      await this.peer.setLocalDescription(answer);
      if (!this.stopped) this.send({ type: "answer", sdp: this.peer.localDescription.sdp });
    } else if (message.type === "ice") {
      if (!this.peer?.remoteDescription) this.pendingIce.push(message.candidate);
      else await this.peer.addIceCandidate(message.candidate);
    }
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this.video.removeEventListener("playing", this.loaded);
    this.video.removeEventListener("loadeddata", this.loaded);
    this.peer?.close();
    for (const track of this.media?.getTracks() || []) track.stop();
    this.pendingIce = [];
    this.video.pause();
    this.video.srcObject = null;
    this.video.removeAttribute("src");
    this.video.load();
    this.video.style.display = "none";
  }
};
