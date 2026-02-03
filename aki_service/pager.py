from __future__ import annotations

import http.client
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PagerClient:
    host: str
    port: int
    timeout_s: float = 2.0

    def send_page(self, mrn: str, test_time_hl7: Optional[str]) -> Tuple[bool, str]:
        body = mrn if not test_time_hl7 else f"{mrn},{test_time_hl7}"
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
            conn.request("POST", "/page", body=body.encode("ascii", errors="ignore"), headers={"Content-Type": "text/plain"})
            resp = conn.getresponse()
            status = resp.status
            data = resp.read()  # consume
            conn.close()
            if 200 <= status < 300:
                return True, f"ok {status}"
            return False, f"http {status} {data[:200]!r}"
        except Exception as e:
            return False, f"error {e}"
