#!/usr/bin/env python3
"""Final Reolink Baichuan client (AES post-login): Snap for JPEG + Preview for video."""
import sys, time, struct, re
from ConnectionLayer import (BaichuanSession, recv_frame, parse_header, bc_crypt,
                             derive_aes_key, aes_encrypt, aes_decrypt, MAGIC,
                             build_get_nonce_payload, build_header, next_mess_id,
                             CMD_LOGIN, ENCRYPT_ADV)
from datetime import datetime

HOST, PORT = "93.100.247.132", 9000
USER, PASS = "user", "qwerty"

CLASS_MODERN = 0x6414
CLASS_ZERO = 0x0000

def md5(s):
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()

def aes_key_upper(nonce, password):
    # reostream/neolink: first 16 chars of UPPERCASE hex MD5(nonce + "-" + password)
    return md5(f"{nonce}-{password}").upper()[:16].encode("ascii")

def probe_cipher():
    """Send GetNonce on a throwaway connection and read the negotiated cipher from the reply header."""
    import socket
    from ConnectionLayer import open_session, close_session, recv_frame as rf, parse_header as ph, \
        build_get_nonce_payload, build_header, next_mess_id, CMD_LOGIN, ENCRYPT_ADV
    s = open_session(HOST, PORT)
    try:
        payload = build_get_nonce_payload()
        header = build_header(cmd_id=CMD_LOGIN, payload_len=len(payload), message_class=0x6514,
                              ch_id=0xFA, mess_id=next_mess_id(s), encrypt=ENCRYPT_ADV)
        s.sock.sendall(header + payload)
        hdr, body = rf(s.sock)
        if not hdr:
            print("probe: no reply")
            return None
        enc, dirb = hdr[16], hdr[17]
        print(f"probe: nonce reply header bytes16/17 = {enc:#04x}/{dirb:#04x}")
        if dirb == 0xDD:
            if enc == 0x00:
                return "none"
            if enc == 0x01:
                return "bc"
            return "aes"
        return "aes"
    finally:
        close_session(s)

def try_decrypt_xml(key, ch_id, blob):
    for d in (lambda b: aes_decrypt(key, b), lambda b: bc_crypt(ch_id, b), lambda b: b):
        try:
            out = d(blob)
        except Exception:
            continue
        if out.lstrip(b"\x00 \t\r\n").startswith(b"<"):
            return out
    return None

class Client:
    def __init__(self):
        self.bc = BaichuanSession(HOST, PORT)
        self.bc.__enter__()
        self.bc.connect(USER, PASS)
        self.nonce = self.bc.session.nonce
        self.key = aes_key_upper(self.nonce, PASS)
        self.sock = self.bc.session.sock
        self.counter = 0
        print(f"login OK; nonce={self.nonce}; aes_key={self.key!r}")

    def close(self):
        try:
            self.bc.__exit__(None, None, None)
        except Exception:
            pass

    def next_counter(self):
        self.counter += 1
        return self.counter

    def _header(self, msg_id, body_len, stream_id, payload_off=0):
        return (MAGIC.to_bytes(4, "little")
                + struct.pack("<I", msg_id)
                + struct.pack("<I", body_len)
                + bytes([0, stream_id, self.next_counter(), 0])
                + b"\x00\x00"
                + struct.pack("<H", CLASS_MODERN)
                + struct.pack("<I", payload_off))

    def send_xml(self, msg_id, xml_bytes, stream_id=0):
        sealed = aes_encrypt(self.key, xml_bytes)
        self.sock.sendall(self._header(msg_id, len(sealed), stream_id) + sealed)

    def send_parts(self, msg_id, ext_xml, body_xml, stream_id=0):
        ext = aes_encrypt(self.key, ext_xml)
        body = aes_encrypt(self.key, body_xml)
        self.sock.sendall(self._header(msg_id, len(ext) + len(body), stream_id, payload_off=len(ext)) + ext + body)

    def ping(self):
        self.sock.sendall(self._header(93, 0, 0))

    def read_message(self, timeout=10):
        """Returns (msg_id, status, xml|None, payload bytes) or None on timeout, or ('CLOSED',0,None,None)."""
        self.sock.settimeout(timeout)
        try:
            hdr, body = recv_frame(self.sock)
        except Exception as e:
            if "timed out" in str(e):
                return None
            print("read error:", e)
            return ("CLOSED", 0, None, None)
        if not hdr:
            return ("CLOSED", 0, None, None)
        f = parse_header(hdr)
        status = f["status_code"]
        if not body:
            return (f["cmd_id"], status, None, b"")
        poff = f["payload_offset"] or 0
        if f["message_class"] == CLASS_ZERO and poff == 0 and f["cmd_id"] == 3:
            return (3, status, None, body)
        xml_part = body[:poff] if poff > 0 else body
        payload = body[poff:] if poff > 0 else b""
        xml = try_decrypt_xml(self.key, f["ch_id"], xml_part) if xml_part else None
        if payload and xml:
            m = re.search(rb"<encryptLen>(\d+)</encryptLen>", xml)
            if m:
                n = min(int(m.group(1)), len(payload))
                try:
                    payload = aes_decrypt(self.key, payload[:n]) + payload[n:]
                except Exception:
                    pass
        return (f["cmd_id"], status, xml, payload)

CAMERA_XML_HDR = '<?xml version="1.0" encoding="UTF-8" ?>\n'

def channel_xml(ch=0):
    return (CAMERA_XML_HDR + f'<Extension version="1.1">\n<channelId>{ch}</channelId>\n</Extension>\n').encode()

def snap_xml(ch=0, stream="main"):
    return (CAMERA_XML_HDR +
            f'<body>\n<Snap version="1.1">\n<channelId>{ch}</channelId>\n'
            f'<logicChannel>{ch}</logicChannel>\n<time>0</time>\n<fullFrame>0</fullFrame>\n'
            f'<streamType>{stream}</streamType>\n</Snap>\n</body>\n').encode()

def preview_xml_compact(handle, stream_name):
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            f'<body><Preview version="1.1"><channelId>0</channelId><handle>{handle}</handle>'
            f'<streamType>{stream_name}</streamType></Preview></body>').encode()

def do_snap(c, stream="main", wait=45):
    print(f"--- SNAP request (stream={stream}) ---", flush=True)
    c.send_parts(109, channel_xml(), snap_xml(0, stream))
    size = None
    fname = None
    data = b""
    deadline = time.time() + wait
    last_ping = time.time()
    while time.time() < deadline:
        if time.time() - last_ping > 8:
            c.ping()
            last_ping = time.time()
        m = c.read_message(timeout=8)
        if m is None:
            continue
        msg_id, status, xml, payload = m
        if msg_id == "CLOSED":
            print("connection closed")
            return None
        if msg_id == 109:
            if xml and b"pictureSize" in xml:
                mm = re.search(rb"<pictureSize>(\d+)</pictureSize>", xml)
                fn = re.search(rb"<fileName>([^<]+)</fileName>", xml)
                if mm:
                    size = int(mm.group(1))
                    fname = fn.group(1) if fn else b"?".decode()
                    print(f"snap reply: size={size} file={fname}", flush=True)
            if payload:
                data += payload
                print(f"  +{len(payload)}B (total {len(data)}/{size})", flush=True)
                if size and len(data) >= size:
                    return data[:size]
        elif msg_id == 93:
            pass
        else:
            xl = (xml[:120] if xml else b"")
            print(f"  msg {msg_id} status {status} payload={len(payload)}B xml={xl}", flush=True)
    print("snap timeout")
    return None

def main():
    print("=== cipher probe ===")
    cipher = probe_cipher()
    print("negotiated cipher:", cipher)
    now = datetime.now()
    c = Client()
    result = now.strftime("%Y_%m_%d__%H_%M_%S")
    try:
        jpg = do_snap(c, "main")
        if jpg is None:
            jpg = do_snap(c, "sub")
        if jpg:
            with open(f"{result}.jpg", "wb") as f:
                f.write(jpg)
            print(f"SAVED {result}.jpg", len(jpg), "bytes")
            print("jpeg magic ok:", jpg[:3] == b"\xff\xd8\xff")
            return
        print("SNAP FAILED")
    finally:
        c.close()

if __name__ == "__main__":
    main()
