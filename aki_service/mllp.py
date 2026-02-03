from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

MLLP_START_OF_BLOCK = b"\x0b"
MLLP_END_OF_BLOCK = b"\x1c"
MLLP_CARRIAGE_RETURN = b"\x0d"
MLLP_END_SEQUENCE = MLLP_END_OF_BLOCK + MLLP_CARRIAGE_RETURN


class MLLPDecodeError(Exception):
    pass


@dataclass
class MLLPFrame:
    """A decoded MLLP frame containing the raw HL7 payload (no MLLP markers)."""
    hl7: bytes


class MLLPDecoder:
    """Incremental MLLP decoder for a TCP byte stream.

    Feed bytes via .feed(); it returns any complete HL7 messages decoded so far.
    """

    def __init__(self, *, max_frame_bytes: int = 2_000_000) -> None:
        self._buf = bytearray()
        self._max = int(max_frame_bytes)

    def feed(self, data: bytes) -> List[MLLPFrame]:
        if not data:
            return []
        self._buf.extend(data)
        if len(self._buf) > self._max:
            raise MLLPDecodeError(f"buffer exceeded max_frame_bytes={self._max}")

        out: List[MLLPFrame] = []

        while True:
            # Drop any garbage bytes before the first start marker.
            try:
                start = self._buf.index(MLLP_START_OF_BLOCK)
            except ValueError:
                # no start marker at all yet
                self._buf.clear()
                return out

            if start > 0:
                del self._buf[:start]

            # Now buffer begins with start marker
            try:
                end = self._buf.index(MLLP_END_OF_BLOCK, 1)
            except ValueError:
                # incomplete frame
                return out

            # Need CR after end-of-block
            if end + 1 >= len(self._buf):
                return out
            if self._buf[end:end+2] != MLLP_END_SEQUENCE:
                # Desync: drop the start marker and rescan
                del self._buf[0:1]
                continue

            payload = bytes(self._buf[1:end])  # exclude start marker and end marker
            # remove frame including end sequence
            del self._buf[:end+2]
            if payload:
                out.append(MLLPFrame(hl7=payload))

    def reset(self) -> None:
        self._buf.clear()


def build_hl7_ack(*, ack_code: str = "AA", timestamp_hl7: Optional[str] = None) -> bytes:
    """Build a minimal HL7 ACK message payload (without MLLP framing)."""
    if timestamp_hl7 is None:
        # HL7 timestamp: YYYYMMDDHHMMSS
        timestamp_hl7 = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    # Keep it minimal. Simulator requires MSH and MSA segments.
    # Note: segments are separated by carriage return in HL7v2.
    msg = (
        f"MSH|^~\\&|||||{timestamp_hl7}||ACK|||2.5\r"
        f"MSA|{ack_code}\r"
    )
    return msg.encode("ascii", errors="strict")


def frame_mllp(hl7_payload: bytes) -> bytes:
    return MLLP_START_OF_BLOCK + hl7_payload + MLLP_END_SEQUENCE


class MLLPClient:
    """Blocking TCP client that reads HL7 messages over MLLP and writes ACKs.

    This client is intentionally small and dependency-free (stdlib only).
    """

    def __init__(
        self,
        address: Tuple[str, int],
        *,
        recv_buf: int = 4096,
        socket_timeout_s: float = 10.0,
        reconnect_backoff_s: float = 1.0,
        max_frame_bytes: int = 2_000_000,
    ) -> None:
        self.address = address
        self.recv_buf = int(recv_buf)
        self.socket_timeout_s = float(socket_timeout_s)
        self.reconnect_backoff_s = float(reconnect_backoff_s)
        self.decoder = MLLPDecoder(max_frame_bytes=max_frame_bytes)

    def run(self, handler, *, stop_event: Optional[object] = None) -> None:
        """Connect and run until stopped; calls handler(hl7_bytes)->ack_code str."""

        def should_stop() -> bool:
            return bool(stop_event is not None and getattr(stop_event, "is_set", None) and stop_event.is_set())

        while not should_stop():
            try:
                with socket.create_connection(self.address, timeout=self.socket_timeout_s) as sock:
                    sock.settimeout(self.socket_timeout_s)
                    self.decoder.reset()
                    while not should_stop():
                        try:
                            data = sock.recv(self.recv_buf)
                        except socket.timeout:
                            # idle connection: keep waiting (do NOT reconnect)
                            continue

                        if not data:
                            # peer closed
                            break

                        frames = self.decoder.feed(data)
                        for frame in frames:
                            try:
                                ack_code = handler(frame.hl7)
                            except Exception:
                                # parser/handler blew up: NAK so simulator can resend.
                                ack_code = "AE"
                            ack = frame_mllp(build_hl7_ack(ack_code=ack_code))
                            sock.sendall(ack)
            except (OSError, MLLPDecodeError):
                if should_stop():
                    break
                time.sleep(self.reconnect_backoff_s)
                continue

            # clean close: reconnect after a short backoff (useful for simulator replay)
            if not should_stop():
                time.sleep(self.reconnect_backoff_s)
