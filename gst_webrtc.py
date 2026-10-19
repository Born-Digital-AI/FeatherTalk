"""GStreamer WebRTC transport for already encoded H.264 and matched PCM."""
from __future__ import annotations

import json
import logging
import contextlib
import ctypes
import ctypes.util
import ipaddress
import os
import threading
import time
from urllib.parse import parse_qsl, quote, urlsplit

logger = logging.getLogger("uvicorn.error")


def _gobject_pointer(obj):
    """Extract the native peer pointer without wrapping any of its properties."""
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(obj.__gpointer__, None)
    if not pointer:
        raise RuntimeError("WebRTC peer has no native GObject pointer")
    return pointer


def _gobject_api():
    library = ctypes.util.find_library("gobject-2.0")
    if library is None:
        raise RuntimeError("GObject library is required for native ICE configuration")
    api = ctypes.CDLL(library)
    # These varargs functions are always invoked with exactly one property.
    api.g_object_get.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    api.g_object_get.restype = None
    api.g_object_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_void_p]
    api.g_object_set.restype = None
    api.g_object_unref.argtypes = [ctypes.c_void_p]
    api.g_object_unref.restype = None
    return api


def configure_ice_ports(peer, minimum=40000, maximum=40100):
    """Avoid GI sinking the floating default ICE agent owned by webrtcbin.

    GStreamer 1.24 creates that agent without sinking its floating reference.
    A Python get_property wrapper can consume it, then free the agent when the
    wrapper goes out of scope. Native g_object_get adds an ordinary reference;
    release only that temporary reference after setting the port range.
    """
    if not 1 <= minimum <= maximum <= 65535:
        raise ValueError("Invalid ICE UDP port range")
    api = _gobject_api()
    agent = ctypes.c_void_p()
    logger.info("WebRTC ICE startup: acquiring native agent")
    api.g_object_get(_gobject_pointer(peer), b"ice-agent", ctypes.byref(agent), None)
    if not agent.value:
        raise RuntimeError("WebRTC ICE agent was not created")
    try:
        logger.info("WebRTC ICE startup: configuring UDP ports %s-%s", minimum, maximum)
        api.g_object_set(agent, b"max-rtp-port", maximum, None)
        api.g_object_set(agent, b"min-rtp-port", minimum, None)
    finally:
        api.g_object_unref(agent)
    logger.info("WebRTC ICE startup: native agent configured")


def load_gstreamer():
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstWebRTC", "1.0")
        gi.require_version("GstSdp", "1.0")
        from gi.repository import Gst, GstWebRTC, GstSdp
        Gst.init(None)
        for name in ("webrtcbin", "appsrc", "h264parse", "rtph264pay", "opusenc", "rtpopuspay", "nice"):
            if name != "nice" and Gst.ElementFactory.find(name) is None:
                raise RuntimeError(f"Missing GStreamer element: {name}")
        return Gst, GstWebRTC, GstSdp
    except ImportError as exc:
        raise RuntimeError("WebRTC requires the Linux media image with GStreamer and PyGObject.") from exc


def _shared_turn_config():
    """Build browser and GStreamer configurations from a shared TURN account."""
    raw = os.environ.get("TURN_SERVER_URLS", "").strip()
    if not raw:
        return None, []
    username = os.environ.get("TURN_SERVER_USERNAME", "")
    password = os.environ.get("TURN_SERVER_PASSWORD", "")
    if not username or not password:
        raise ValueError("TURN_SERVER_URLS requires TURN_SERVER_USERNAME and TURN_SERVER_PASSWORD")
    try:
        urls = json.loads(raw) if raw.startswith("[") else [raw]
        if not isinstance(urls, list) or not urls:
            raise ValueError()
        browser_urls, server_urls = [], []
        for url in urls:
            scheme, location = url.split(":", 1)
            uri = urlsplit(scheme + "://" + location.removeprefix("//"))
            if (scheme not in ("turn", "turns") or not uri.hostname or
                    uri.username is not None or uri.password is not None or
                    uri.path or uri.fragment or (uri.port is not None and uri.port == 0) or
                    any(key != "transport" or value not in ("udp", "tcp")
                        for key, value in parse_qsl(uri.query, keep_blank_values=True))):
                raise ValueError()
            suffix = uri.netloc + ("?" + uri.query if uri.query else "")
            browser_urls.append(f"{scheme}:{suffix}")
            server_urls.append(f"{scheme}://{quote(username, safe='')}:{quote(password, safe='')}@{suffix}")
    except (ValueError, TypeError, AttributeError):
        # Do not echo URLs that may accidentally contain credentials.
        raise ValueError("TURN_SERVER_URLS must be a TURN URL or JSON array of TURN URLs without credentials") from None
    return dict(urls=browser_urls, username=username, credential=password), server_urls


def ice_config():
    """Separate public browser ICE settings from server URI configuration."""
    servers = json.loads(os.environ.get("FEATHER_WEBRTC_ICE_SERVERS", "[]"))
    if not isinstance(servers, list):
        raise ValueError("FEATHER_WEBRTC_ICE_SERVERS must be a JSON array")
    shared, _ = _shared_turn_config()
    if shared is not None and shared not in servers:
        servers.append(shared)
    return servers


def turn_server_config():
    servers = json.loads(os.environ.get("FEATHER_WEBRTC_TURN_SERVERS", "[]"))
    if not isinstance(servers, list) or any(not isinstance(uri, str) for uri in servers):
        raise ValueError("FEATHER_WEBRTC_TURN_SERVERS must be a JSON array of URI strings")
    _, shared = _shared_turn_config()
    return list(dict.fromkeys([*servers, *shared]))


class BrowserIceResolver:
    """Resolve browser mDNS on asyncio before passing candidates to GStreamer.

    This avoids system NSS/Avahi dependencies inside the container, and the
    premature/double promise reply in GStreamer 1.24's asynchronous DNS path.
    One instance belongs to one signaling session, including its UDP sockets.
    """

    def __init__(self, transport):
        self.transport = transport
        self.protocol = None
        self.addresses = {}
        self.closed = False
        self.candidates = 0

    async def apply_answer(self, sdp):
        if not isinstance(sdp, str) or not 1 <= len(sdp) <= 128 * 1024:
            raise ValueError("Invalid WebRTC answer")
        lines, candidates, index = [], [], -1
        for line in sdp.splitlines():
            if line.startswith("m="):
                index += 1
            if line.startswith("a=candidate:"):
                candidates.append(dict(candidate=line[2:], sdpMLineIndex=index))
            elif line != "a=end-of-candidates":
                lines.append(line)
        self.transport.accept_answer("\r\n".join(lines) + "\r\n")
        for candidate in candidates:
            await self.add(candidate)

    async def add(self, candidate):
        if self.closed or not candidate:
            return
        line, index = candidate.get("candidate"), candidate.get("sdpMLineIndex")
        if line == "":
            return  # End-of-generation marker, not a candidate to resolve.
        if (not isinstance(line, str) or len(line) > 8192 or
                not isinstance(index, int) or not 0 <= index < 8):
            raise ValueError("Invalid ICE candidate")
        fields = line.split()
        if len(fields) < 8 or not fields[0].startswith("candidate:") or fields[6] != "typ":
            raise ValueError("Malformed ICE candidate")
        self.candidates += 1
        if self.candidates > 64:
            raise ValueError("Too many browser ICE candidates")
        hostname = fields[4]
        if hostname.lower().endswith(".local"):
            from aioice.mdns import create_mdns_protocol, is_mdns_hostname
            if not is_mdns_hostname(hostname):
                raise ValueError("Invalid ICE mDNS hostname")
            if hostname not in self.addresses:
                try:
                    if self.protocol is None:
                        self.protocol = await create_mdns_protocol()
                    address = await self.protocol.resolve(hostname, timeout=3)
                except OSError as exc:
                    logger.warning("WebRTC mDNS unavailable: hostname=%s reason=%s; skipping candidate", hostname, exc)
                    return
                if address is None:
                    logger.warning("WebRTC mDNS resolution failed: hostname=%s; skipping candidate", hostname)
                    return  # Other adapters / server-reflexive candidates may still work.
                self.addresses[hostname] = str(ipaddress.ip_address(address))
                logger.info("WebRTC mDNS resolved: hostname=%s address=%s", hostname, self.addresses[hostname])
            fields[4] = self.addresses[hostname]
            candidate = dict(candidate, candidate=" ".join(fields))
        if not self.closed:
            self.transport.add_ice(candidate)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.protocol is not None:
            await self.protocol.close()


class GstTransport:
    def __init__(self, settings, notify):
        self.Gst, self.WebRTC, self.Sdp = load_gstreamer()
        Gst = self.Gst
        self.settings = settings
        self.notify = notify
        self.keyframe_requested = threading.Event()
        self.pending_ice = []
        self.remote_set = False
        self.answer_received = False
        self.ice_lock = threading.RLock()
        self.closed = False
        self.last_error = None
        self.timestamp_origin = 0
        logger.info("GStreamer startup: creating WebRTC pipeline")
        self.pipeline = Gst.parse_launch(
            'webrtcbin name=peer bundle-policy=max-bundle '
            'appsrc name=video is-live=true format=time block=false max-buffers=3 '
            '! h264parse config-interval=-1 '
            '! rtph264pay name=pay pt=96 config-interval=-1 '
            '! application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 '
            '! peer. '
            'appsrc name=audio is-live=true format=time block=false max-buffers=3 '
            '! audioconvert ! audioresample '
            '! audio/x-raw,rate=48000,channels=1 '
            '! opusenc frame-size=20 bitrate=64000 dtx=true '
            '! rtpopuspay name=audio_pay pt=97 '
            '! application/x-rtp,media=audio,encoding-name=OPUS,payload=97,clock-rate=48000 '
            '! peer.'
        )
        self.peer = self.pipeline.get_by_name("peer")
        logger.info("GStreamer startup: pipeline created; configuring transport")
        try:
            self._configure()
        except BaseException:
            self.close()
            raise

    def _configure(self):
        Gst = self.Gst
        settings = self.settings
        self.video = self.pipeline.get_by_name("video")
        self.audio = self.pipeline.get_by_name("audio")
        for index in range(2):
            logger.info("GStreamer startup: configuring transceiver %s", index)
            transceiver = self.peer.emit("get-transceiver", index)
            if transceiver is not None:
                transceiver.set_property("direction", self.WebRTC.WebRTCRTPTransceiverDirection.SENDONLY)
        configure_ice_ports(self.peer)
        self.video.set_property("caps", Gst.Caps.from_string(
            f"video/x-h264,stream-format=byte-stream,alignment=au,width={settings.width},"
            f"height={settings.height},framerate=25/1,profile=constrained-baseline"))
        self.audio.set_property("caps", Gst.Caps.from_string("audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1"))
        stun = os.environ.get("FEATHER_WEBRTC_STUN_SERVER")
        if stun:
            self.peer.set_property("stun-server", stun)
        for uri in turn_server_config():
            if not self.peer.emit("add-turn-server", uri):
                raise ValueError("Invalid server TURN configuration")
        self.peer.connect("on-ice-candidate", self._on_ice)
        self.peer.connect("notify::connection-state", self._on_connection)
        self.peer.connect("notify::ice-connection-state", self._on_ice_state)
        self.peer.connect("notify::ice-gathering-state", self._on_ice_state)
        self.video.get_static_pad("src").add_probe(Gst.PadProbeType.EVENT_UPSTREAM, self._on_video_event)
        bus = self.pipeline.get_bus()
        bus.set_sync_handler(self._on_bus, None)
        logger.info("GStreamer startup: setting pipeline to PLAYING")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.close()
            raise RuntimeError("Could not start GStreamer WebRTC pipeline")
        logger.info("GStreamer startup: pipeline started")

    def _on_ice(self, peer, mline, candidate):
        self._log_ice_candidate("server", candidate)
        self.notify(dict(type="ice", candidate=dict(candidate=candidate, sdpMLineIndex=mline)))

    @staticmethod
    def _log_ice_candidate(origin, candidate):
        # Log routing information only, without SDP, ICE passwords or TURN credentials.
        fields = candidate.split()
        if len(fields) >= 8 and fields[6] == "typ":
            logger.info("WebRTC ICE candidate: origin=%s type=%s protocol=%s address=%s port=%s",
                        origin, fields[7], fields[2], fields[4], fields[5])

    def _on_connection(self, peer, prop):
        state = peer.get_property("connection-state").value_nick
        logger.info("WebRTC connection state: %s", state)
        self.notify(dict(type="connection", state=state))

    def _on_ice_state(self, peer, prop):
        logger.info("WebRTC %s: %s", prop.name, peer.get_property(prop.name).value_nick)

    def _on_video_event(self, pad, info):
        event = info.get_event()
        structure = event.get_structure() if event else None
        if structure and structure.get_name() == "GstForceKeyUnit":
            self.keyframe_requested.set()
        return self.Gst.PadProbeReturn.OK

    def _on_bus(self, bus, message, unused):
        if message.type == self.Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            self.last_error = error.message
            # Never expose credentials or arbitrary pipeline debug text to the UI.
            self.notify(dict(type="error", message=f"WebRTC media pipeline failed: {error.message}"))
        return self.Gst.BusSyncReply.DROP

    def _checked(self, promise, action):
        if self.closed:
            return False
        reply = promise.get_reply()
        if promise.wait() != self.Gst.PromiseResult.REPLIED or (reply is not None and reply.has_field("error")):
            self.notify(dict(type="error", message=f"Could not {action} WebRTC description."))
            return False
        return True

    def offer(self):
        def on_local(promise, sdp_text, *unused):
            if self._checked(promise, "apply"):
                logger.info("WebRTC signaling: local offer applied; sending to browser")
                self.notify(dict(type="offer", sdp=sdp_text))

        def on_offer(promise, *unused):
            if not self._checked(promise, "create"):
                return
            # Keep the reply alive while reading/copying its boxed SDP value.
            # Chained get_reply().get_value() can free the owner too early in GI.
            reply = promise.get_reply()
            offer = reply.get_value("offer")
            if offer is None:
                self.notify(dict(type="error", message="WebRTC offer is missing."))
                return
            local_offer = offer.copy()
            sdp_text = local_offer.sdp.as_text()
            local = self.Gst.Promise.new_with_change_func(on_local, sdp_text, None)
            self.peer.emit("set-local-description", local_offer, local)

        promise = self.Gst.Promise.new_with_change_func(on_offer, None, None)
        self.peer.emit("create-offer", None, promise)

    def accept_answer(self, sdp):
        if self.answer_received:
            raise ValueError("WebRTC answer was already accepted")
        if not isinstance(sdp, str) or not 1 <= len(sdp) <= 128 * 1024:
            raise ValueError("Invalid WebRTC answer")
        for line in sdp.splitlines():
            if line.startswith(("m=", "a=recvonly", "a=inactive", "a=sendrecv", "a=sendonly")):
                logger.info("WebRTC browser media: %s", line)
        result, message = self.Sdp.SDPMessage.new()
        if self.Sdp.sdp_message_parse_buffer(sdp.encode(), message) != self.Sdp.SDPResult.OK:
            raise ValueError("Malformed WebRTC SDP")
        remote = self.WebRTC.WebRTCSessionDescription.new(self.WebRTC.WebRTCSDPType.ANSWER, message)
        self.answer_received = True

        def on_remote(promise, *unused):
            if not self._checked(promise, "accept"):
                return
            logger.info("WebRTC signaling: browser answer applied")
            with self.ice_lock:
                self.remote_set = True
                pending, self.pending_ice = self.pending_ice, []
            for candidate in pending:
                self.add_ice(candidate)

        promise = self.Gst.Promise.new_with_change_func(on_remote, None, None)
        self.peer.emit("set-remote-description", remote, promise)

    def add_ice(self, candidate):
        if not candidate:
            return
        line = candidate.get("candidate")
        index = candidate.get("sdpMLineIndex")
        if not isinstance(line, str) or len(line) > 8192 or not isinstance(index, int) or not 0 <= index < 8:
            raise ValueError("Invalid ICE candidate")
        self._log_ice_candidate("browser", line)
        with self.ice_lock:
            if not self.remote_set:
                if len(self.pending_ice) >= 64:
                    raise ValueError("Too many pending ICE candidates")
                self.pending_ice.append(candidate)
            else:
                # GStreamer 1.24 can reply twice to the full variant when DNS
                # fails. BrowserIceResolver already resolves mDNS before here.
                self.peer.emit("add-ice-candidate", index, line)

    def prime(self, unit):
        """Publish codec caps before answering SDP; no inference or re-encoding."""
        self.push(unit, bytes(1280), 0)
        deadline = time.monotonic() + 5
        pads = [self.pipeline.get_by_name(name).get_static_pad("src") for name in ("pay", "audio_pay")]
        while not all(pad.get_current_caps() is not None for pad in pads):
            if self.last_error or self.closed or time.monotonic() > deadline:
                raise RuntimeError("WebRTC media codec negotiation failed")
            time.sleep(0.01)

    def begin(self):
        clock = self.pipeline.get_clock()
        self.timestamp_origin = clock.get_time() - self.pipeline.get_base_time()

    def push(self, unit, pcm, tick, stream=None, generation=None):
        # Nonblocking appsrc + single producer: validate both queue slots before
        # enqueue, and hold only a short lock shared with clear_audio().
        with stream.audio_lock if stream is not None else contextlib.nullcontext():
            if stream is not None and stream.audio_generation != generation:
                return False
            if self.closed:
                return False
            sources = [self.audio] + ([self.video] if unit else [])
            if any(source.get_property("current-level-buffers") >= 3 for source in sources):
                return False
            self._push_buffers(unit, pcm, tick)
            return True

    def _push_buffers(self, unit, pcm, tick):
        Gst = self.Gst
        pts = self.timestamp_origin + tick * Gst.SECOND // self.settings.fps
        duration = Gst.SECOND // self.settings.fps
        for source, data in ((self.video, unit.data if unit else None), (self.audio, pcm)):
            if data is None:
                continue
            buffer = Gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)
            buffer.pts = buffer.dts = pts
            buffer.duration = duration
            if source is self.video and not unit.keyframe:
                buffer.set_flags(Gst.BufferFlags.DELTA_UNIT)
            if source.emit("push-buffer", buffer) != Gst.FlowReturn.OK and not self.closed:
                raise RuntimeError("WebRTC media transport stopped accepting frames")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pipeline.set_state(self.Gst.State.NULL)
