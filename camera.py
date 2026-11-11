# Получение JPEG-снимка с камеры Reolink по протоколу Baichuan.
# Логика взята из bc_final.py (Snap по AES после логина); отличия:
#   * адрес и учётные данные передаются параметрами (из переменных окружения),
#   * вместо print — logging, ключ AES в лог не пишется,
#   * проба шифра (probe_cipher) убрана: её результат нигде не использовался,
#   * резервный запрос «sub»-потока убран: маска рассчитана на кадр основного потока,
#   * несколько попыток подряд.
import logging
import re
import struct
import time

from ConnectionLayer import (BaichuanSession, MAGIC, aes_decrypt, aes_encrypt,
                             bc_crypt, parse_header, recv_frame)
import hashlib

log = logging.getLogger("camera")

CLASS_MODERN = 0x6414
CLASS_ZERO = 0x0000
CAMERA_XML_HDR = '<?xml version="1.0" encoding="UTF-8" ?>\n'


def aes_key_upper(nonce, password):
    # первые 16 символов HEX-MD5(nonce + "-" + password) в ВЕРХНЕМ регистре
    return hashlib.md5(f"{nonce}-{password}".encode()).hexdigest().upper()[:16].encode("ascii")


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
    def __init__(self, host, port, user, password):
        self.bc = BaichuanSession(host, port)
        self.bc.__enter__()
        try:
            self.bc.connect(user, password)
            self.nonce = self.bc.session.nonce
            self.key = aes_key_upper(self.nonce, password)
            self.sock = self.bc.session.sock
            self.counter = 0
        except Exception:
            self.close()
            raise
        log.debug("login OK")

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

    def send_parts(self, msg_id, ext_xml, body_xml, stream_id=0):
        ext = aes_encrypt(self.key, ext_xml)
        body = aes_encrypt(self.key, body_xml)
        self.sock.sendall(self._header(msg_id, len(ext) + len(body), stream_id,
                                       payload_off=len(ext)) + ext + body)

    def ping(self):
        self.sock.sendall(self._header(93, 0, 0))

    def read_message(self, timeout=10):
        """(msg_id, status, xml|None, payload) | None по таймауту | ('CLOSED', 0, None, None)."""
        self.sock.settimeout(timeout)
        try:
            hdr, body = recv_frame(self.sock)
        except Exception as e:
            if "timed out" in str(e):
                return None
            log.warning("ошибка чтения: %s", e)
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


def channel_xml(ch=0):
    return (CAMERA_XML_HDR + f'<Extension version="1.1">\n<channelId>{ch}</channelId>\n</Extension>\n').encode()


def snap_xml(ch=0, stream="main"):
    return (CAMERA_XML_HDR +
            f'<body>\n<Snap version="1.1">\n<channelId>{ch}</channelId>\n'
            f'<logicChannel>{ch}</logicChannel>\n<time>0</time>\n<fullFrame>0</fullFrame>\n'
            f'<streamType>{stream}</streamType>\n</Snap>\n</body>\n').encode()


def do_snap(c, stream="main", wait=45):
    log.debug("запрос снимка (stream=%s)", stream)
    c.send_parts(109, channel_xml(), snap_xml(0, stream))
    size = None
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
            log.warning("камера закрыла соединение")
            return None
        if msg_id == 109:
            if xml and b"pictureSize" in xml:
                mm = re.search(rb"<pictureSize>(\d+)</pictureSize>", xml)
                if mm:
                    size = int(mm.group(1))
                    log.debug("размер снимка: %d байт", size)
            if payload:
                data += payload
                if size and len(data) >= size:
                    return data[:size]
        elif msg_id != 93:
            log.debug("сообщение %s status=%s payload=%dB", msg_id, status, len(payload))
    log.warning("таймаут ожидания снимка")
    return None


def grab_snapshot(host, port, user, password, attempts=2, pause=3.0):
    """JPEG-байты снимка основного потока или None, если не удалось."""
    for i in range(1, attempts + 1):
        c = None
        try:
            c = Client(host, port, user, password)
            jpg = do_snap(c, "main")
            if jpg and jpg[:3] == b"\xff\xd8\xff":
                return jpg
            log.warning("попытка %d/%d: снимок не получен или не JPEG", i, attempts)
        except Exception as e:
            log.warning("попытка %d/%d: ошибка камеры: %s", i, attempts, e)
        finally:
            if c:
                c.close()
        if i < attempts:
            time.sleep(pause)
    return None
