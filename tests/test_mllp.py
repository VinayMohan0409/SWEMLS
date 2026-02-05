# tests/unit/test_mllp.py

import socket
import time

import pytest

import aki_service.mllp as m

from aki_service.mllp import MLLPDecoder, frame_mllp


def test_mllp_decoder_splits_multiple_frames():
    d = MLLPDecoder()
    msg1 = b"MSH|^~\\&|A\rPID|1||123\r"
    msg2 = b"MSH|^~\\&|B\rPID|1||456\r"
    buf = frame_mllp(msg1) + frame_mllp(msg2)

    frames = d.feed(buf)
    assert [f.hl7 for f in frames] == [msg1, msg2]


def test_mllp_decoder_handles_partial_reads():
    d = MLLPDecoder()
    msg = b"MSH|^~\\&|A\rPID|1||123\r"
    framed = frame_mllp(msg)
    part1 = framed[:5]
    part2 = framed[5:]

    assert d.feed(part1) == []
    frames = d.feed(part2)
    assert len(frames) == 1
    assert frames[0].hl7 == msg

########## ADDITIONS ##########


def test_decoder_single_frame():
    # Decodes one frame
    d = m.MLLPDecoder()
    payload = b"MSH|x\r"
    data = m.MLLP_START_OF_BLOCK + payload + m.MLLP_END_SEQUENCE
    frames = d.feed(data)
    assert [f.hl7 for f in frames] == [payload]


def test_decoder_multiple_frames_one_feed():
    # Decodes two frames back to back
    d = m.MLLPDecoder()
    p1 = b"A"
    p2 = b"B"
    data = m.frame_mllp(p1) + m.frame_mllp(p2)
    frames = d.feed(data)
    assert [f.hl7 for f in frames] == [p1, p2]


def test_decoder_split_across_feeds():
    # Handles partial feed then completes
    d = m.MLLPDecoder()
    payload = b"HELLO"
    framed = m.frame_mllp(payload)

    a = framed[:3]
    b = framed[3:]

    assert d.feed(a) == []
    frames = d.feed(b)
    assert [f.hl7 for f in frames] == [payload]


def test_decoder_drops_garbage_before_start():
    # Drops junk bytes before start marker
    d = m.MLLPDecoder()
    payload = b"X"
    data = b"junkjunk" + m.frame_mllp(payload)
    frames = d.feed(data)
    assert [f.hl7 for f in frames] == [payload]


def test_decoder_no_start_clears_buffer():
    # No start marker clears buffer
    d = m.MLLPDecoder()
    assert d.feed(b"garbage") == []
    assert d._buf == bytearray()


def test_decoder_desync_wrong_end_sequence_recovers():
    # Wrong end sequence forces rescan
    d = m.MLLPDecoder()
    payload = b"OK"

    bad = m.MLLP_START_OF_BLOCK + payload + m.MLLP_END_OF_BLOCK + b"X"
    good = m.frame_mllp(payload)

    frames = d.feed(bad + good)
    assert [f.hl7 for f in frames] == [payload]


def test_decoder_empty_payload_ignored():
    # Empty payload yields no frame
    d = m.MLLPDecoder()
    data = m.MLLP_START_OF_BLOCK + b"" + m.MLLP_END_SEQUENCE
    frames = d.feed(data)
    assert frames == []


def test_decoder_max_frame_bytes_enforced():
    # Buffer limit triggers decode error
    d = m.MLLPDecoder(max_frame_bytes=5)
    with pytest.raises(m.MLLPDecodeError):
        d.feed(b"123456")


def test_decoder_reset_clears_state():
    # Reset clears buffer
    d = m.MLLPDecoder()
    d.feed(m.MLLP_START_OF_BLOCK + b"A")
    assert len(d._buf) > 0
    d.reset()
    assert d._buf == bytearray()


def test_build_hl7_ack_with_timestamp():
    # Uses provided timestamp
    ack = m.build_hl7_ack(ack_code="AA", timestamp_hl7="20250101123059")
    s = ack.decode("ascii")
    assert "MSH|^~\\&" in s
    assert "|20250101123059||ACK" in s
    assert "MSA|AA" in s


def test_build_hl7_ack_default_timestamp(monkeypatch):
    # Uses gmtime based timestamp
    monkeypatch.setattr(time, "strftime", lambda *_a, **_k: "19990102030405", raising=True)
    ack = m.build_hl7_ack(ack_code="AE")
    s = ack.decode("ascii")
    assert "|19990102030405||ACK" in s
    assert "MSA|AE" in s


def test_frame_mllp_wraps_markers():
    # Adds start and end sequence
    payload = b"DATA"
    framed = m.frame_mllp(payload)
    assert framed.startswith(m.MLLP_START_OF_BLOCK)
    assert framed.endswith(m.MLLP_END_SEQUENCE)
    assert framed[1:-2] == payload


class FakeSocket:
    def __init__(self, recv_chunks):
        self._chunks = list(recv_chunks)
        self.sent = []
        self.timeout = None

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def settimeout(self, t):
        self.timeout = t

    def recv(self, n):
        if not self._chunks:
            return b""
        item = self._chunks.pop(0)
        if item == "timeout":
            raise socket.timeout()
        return item

    def sendall(self, data):
        self.sent.append(data)


class StopEvent:
    def __init__(self):
        self.flag = False

    def is_set(self):
        return self.flag


def test_client_run_sends_ack_for_frame(monkeypatch):
    # Reads one frame then replies with ack
    payload = b"MSH|x\rPID|y\r"
    framed_in = m.frame_mllp(payload)

    sock = FakeSocket([framed_in, b""])
    stop = StopEvent()

    def fake_create_connection(addr, timeout):
        return sock

    def fake_sleep(_s):
        stop.flag = True

    monkeypatch.setattr(socket, "create_connection", fake_create_connection, raising=True)
    monkeypatch.setattr(time, "sleep", fake_sleep, raising=True)
    monkeypatch.setattr(time, "strftime", lambda *_a, **_k: "20000101000000", raising=True)

    calls = []

    def handler(hl7_bytes):
        calls.append(hl7_bytes)
        return "AA"

    c = m.MLLPClient(("h", 1), reconnect_backoff_s=0.0, socket_timeout_s=0.01)
    c.run(handler, stop_event=stop)

    assert calls == [payload]
    assert len(sock.sent) == 1

    out = sock.sent[0]
    assert out.startswith(m.MLLP_START_OF_BLOCK)
    assert out.endswith(m.MLLP_END_SEQUENCE)

    ack_payload = out[1:-2]
    s = ack_payload.decode("ascii")
    assert "MSA|AA" in s
    assert "|20000101000000||ACK" in s


def test_client_run_handler_exception_sends_ae(monkeypatch):
    # Handler crash results in AE ack
    payload = b"X"
    framed_in = m.frame_mllp(payload)

    sock = FakeSocket([framed_in, b""])
    stop = StopEvent()

    def fake_create_connection(addr, timeout):
        return sock

    def fake_sleep(_s):
        stop.flag = True

    monkeypatch.setattr(socket, "create_connection", fake_create_connection, raising=True)
    monkeypatch.setattr(time, "sleep", fake_sleep, raising=True)
    monkeypatch.setattr(time, "strftime", lambda *_a, **_k: "20000101000000", raising=True)

    def handler(_hl7_bytes):
        raise RuntimeError("boom")

    c = m.MLLPClient(("h", 1), reconnect_backoff_s=0.0, socket_timeout_s=0.01)
    c.run(handler, stop_event=stop)

    ack_payload = sock.sent[0][1:-2]
    s = ack_payload.decode("ascii")
    assert "MSA|AE" in s


def test_client_run_timeout_does_not_reconnect(monkeypatch):
    # socket timeout loops without reconnect
    payload = b"Z"
    framed_in = m.frame_mllp(payload)

    sock = FakeSocket(["timeout", framed_in, b""])
    stop = StopEvent()

    create_calls = {"n": 0}

    def fake_create_connection(addr, timeout):
        create_calls["n"] += 1
        return sock

    def fake_sleep(_s):
        stop.flag = True

    monkeypatch.setattr(socket, "create_connection", fake_create_connection, raising=True)
    monkeypatch.setattr(time, "sleep", fake_sleep, raising=True)
    monkeypatch.setattr(time, "strftime", lambda *_a, **_k: "20000101000000", raising=True)

    def handler(_hl7_bytes):
        return "AA"

    c = m.MLLPClient(("h", 1), reconnect_backoff_s=0.0, socket_timeout_s=0.01)
    c.run(handler, stop_event=stop)

    assert create_calls["n"] == 1
    assert len(sock.sent) == 1


def test_client_run_decoder_error_reconnects_then_stops(monkeypatch):
    # Decode error triggers sleep then retry loop
    stop = StopEvent()
    create_calls = {"n": 0}

    class BadDecoder:
        def reset(self):
            return None

        def feed(self, _data):
            raise m.MLLPDecodeError("bad")

    sock = FakeSocket([b"anything", b""])

    def fake_create_connection(addr, timeout):
        create_calls["n"] += 1
        return sock

    def fake_sleep(_s):
        stop.flag = True

    monkeypatch.setattr(socket, "create_connection", fake_create_connection, raising=True)
    monkeypatch.setattr(time, "sleep", fake_sleep, raising=True)

    c = m.MLLPClient(("h", 1), reconnect_backoff_s=0.0, socket_timeout_s=0.01)
    c.decoder = BadDecoder()

    def handler(_hl7_bytes):
        return "AA"

    c.run(handler, stop_event=stop)
    assert create_calls["n"] == 1

