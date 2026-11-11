#!/usr/bin/env python3
"""Live video via Baichuan Preview (AES). Saves elementary stream to /tmp/video.*"""
import sys, time, struct
sys.path.insert(0, "/home/z/my-project/scripts")
from bc_final import Client, preview_xml_compact

def preview_xml(handle, stream_name):
    if stream_name:
        st = f"<streamType>{stream_name}</streamType>"
    else:
        st = ""
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            f'<body><Preview version="1.1"><channelId>0</channelId><handle>{handle}</handle>'
            f'{st}</Preview></body>').encode()

def is_video_magic(b):
    return len(b) >= 4 and 48 <= b[0] <= 57 and b[1] in (48, 49) and b[2] == 100 and b[3] == 99

def is_audio_magic(b):
    return len(b) >= 4 and 48 <= b[0] <= 57 and b[1] in (48, 53) and b[2] == 119 and b[3] == 98

def is_info_magic(b):
    return b[:4] in (b"1001", b"1002")

def magic_at(buf, off=0):
    return len(buf) >= off + 4 and (is_video_magic(buf[off:off+4]) or is_audio_magic(buf[off:off+4]) or is_info_magic(buf[off:off+4]))

def search_prefix(buf, hdr, codec):
    region = buf[hdr:hdr + 4096]
    i = 0
    while i + 5 <= len(region):
        if region[i:i+4] == b"\x00\x00\x00\x01":
            nb = region[i+4]
            if codec.upper().startswith("H26"):
                t = (nb >> 1) & 0x3F
                if (t <= 9 or 16 <= t <= 21 or 32 <= t <= 40) and (nb & 1) == 0 and (region[i+5] & 7) >= 1:
                    return i
            else:
                if (nb & 0x80) == 0 and (nb & 0x1F) in (1, 5, 6, 7, 8, 9):
                    return i
        i += 1
    return 0

class Depacketizer:
    def __init__(self):
        self.buf = bytearray()
        self.frames = []
        self.info = None
        self.filler = 0

    def feed(self, data, starts):
        if starts and self.buf:
            self.filler += len(self.buf)
            self.buf = bytearray()
        self.buf += data
        self.drain()

    def consume(self, n):
        del self.buf[:n]
        for pad in range(0, 8):
            if magic_at(self.buf, pad):
                if pad:
                    self.filler += pad
                    del self.buf[:pad]
                return

    def drain(self):
        buf = self.buf
        while True:
            if len(buf) < 8:
                return
            if is_video_magic(buf):
                hdr = 32 if buf[1] == 48 else 24
                if len(buf) < hdr + 4:
                    return
                size = struct.unpack_from("<I", buf, 8)[0]
                if size > (1 << 24):
                    del buf[:1]
                    continue
                stated = struct.unpack_from("<I", buf, 12)[0]
                prefix = 24 + stated - hdr
                if not (0 <= prefix <= 4096 and len(buf) >= hdr + prefix + 4
                        and buf[hdr+prefix:hdr+prefix+4] == b"\x00\x00\x00\x01"):
                    prefix = search_prefix(buf, hdr, bytes(buf[4:8]).decode("ascii", "replace"))
                if len(buf) < hdr + prefix + size:
                    return
                data = bytes(buf[hdr+prefix:hdr+prefix+size])
                kind = "I" if buf[1] == 48 else "P"
                codec = bytes(buf[4:8]).rstrip(b"\x00").decode("ascii", "replace")
                self.frames.append((kind, codec, data))
                if len(self.frames) <= 3 or len(self.frames) % 50 == 0:
                    print(f"  [frame {len(self.frames)}] {kind} {codec} {size}B", flush=True)
                self.consume(hdr + prefix + size)
                continue
            if is_audio_magic(buf):
                size = struct.unpack_from("<H", buf, 4)[0]
                if len(buf) < 8 + size:
                    return
                self.consume(8 + size)
                continue
            if is_info_magic(buf):
                n = struct.unpack_from("<I", buf, 4)[0]
                if n < 8 or n > (1 << 16):
                    del buf[:1]
                    continue
                if len(buf) < n:
                    return
                w = struct.unpack_from("<I", buf, 8)[0]
                h = struct.unpack_from("<I", buf, 12)[0]
                fps = buf[17] if n > 17 else 0
                self.info = (w, h, fps)
                print(f"  [info] {w}x{h}@{fps}", flush=True)
                self.consume(n)
                continue
            return

def try_stream(c, stream_name, stream_id, handle, seconds=15, max_bytes=10_000_000):
    print(f"--- PREVIEW {stream_name} handle={handle} ---", flush=True)
    c.send_xml(3, preview_xml(handle, stream_name), stream_id=stream_id)
    dp = Depacketizer()
    last_ping = time.time()
    acked = False
    deadline = time.time() + seconds
    while time.time() < deadline:
        vbytes = sum(len(d) for _, _, d in dp.frames)
        if vbytes > max_bytes:
            break
        if time.time() - last_ping > 8:
            c.ping()
            last_ping = time.time()
        m = c.read_message(timeout=5)
        if m is None:
            continue
        msg_id, status, xml, payload = m
        if msg_id == "CLOSED":
            print("  connection closed")
            break
        if msg_id == 3:
            if xml and b"<Preview" in xml:
                print("  XML reply:", xml[:200])
            if not payload and status != 200:
                print(f"  DENIED status={status}")
                break
            if payload:
                if not acked:
                    acked = True
                    print("  media flowing", flush=True)
                starts = bool(xml and b"binaryData>1<" in xml)
                dp.feed(payload, starts)
    # stop stream (release session)
    try:
        c.send_xml(3, preview_xml(handle, ""), stream_id=stream_id)
    except Exception:
        pass
    v = [f for f in dp.frames if f[0] in "IP"]
    print(f"  result: {len(v)} video frames, info={dp.info}, filler={dp.filler}B")
    return dp

def main():
    c = Client()
    try:
        for name, sid in (("mainStream", 0), ("subStream", 1), ("externStream", 4)):
            dp = try_stream(c, name, sid, handle=1, seconds=15)
            if dp.frames:
                v = [f for f in dp.frames if f[0] in "IP"]
                if v:
                    codec = v[0][1]
                    ext = "h265" if "265" in codec.upper() else "h264"
                    path = f"/tmp/video.{ext}"
                    with open(path, "wb") as f:
                        for kind, cdc, data in v:
                            if not data.startswith(b"\x00\x00\x00\x01"):
                                f.write(b"\x00\x00\x00\x01")
                            f.write(data)
                    print(f"VIDEO OK: {name} codec={codec} frames={len(v)} -> {path}")
                    return
    finally:
        c.close()
    print("NO VIDEO")

if __name__ == "__main__":
    main()
