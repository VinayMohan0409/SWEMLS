# tests/unit/test_pager.py

import http.client

import pytest

from aki_service.pager import PagerClient


class FakeResponse:
    def __init__(self, status, data=b""):
        self.status = status
        self._data = data

    def read(self):
        return self._data


class FakeHTTPConnection:
    # Capture last instance for assertions
    last_instance = None

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.request_args = None
        self.closed = False
        self.response = FakeResponse(200, b"ok")
        FakeHTTPConnection.last_instance = self

    # Capture request payload
    def request(self, method, path, body=None, headers=None):
        self.request_args = dict(method=method, path=path, body=body, headers=headers)

    # Return configured response
    def getresponse(self):
        return self.response

    # Track connection close
    def close(self):
        self.closed = True


def patch_conn(monkeypatch, fake_cls=FakeHTTPConnection):
    monkeypatch.setattr(http.client, "HTTPConnection", fake_cls, raising=True)


def test_send_page_success_without_test_time(monkeypatch):
    # Happy path MRN only
    patch_conn(monkeypatch)

    pager = PagerClient(host="h", port=1)
    ok, info = pager.send_page("123", None)

    inst = FakeHTTPConnection.last_instance

    assert ok is True
    assert info == "ok 200"
    assert inst.host == "h"
    assert inst.port == 1
    assert inst.timeout == 2.0
    assert inst.request_args["method"] == "POST"
    assert inst.request_args["path"] == "/page"
    assert inst.request_args["body"] == b"123"
    assert inst.request_args["headers"]["Content-Type"] == "text/plain"
    assert inst.closed is True


def test_send_page_success_with_test_time(monkeypatch):
    # MRN with timestamp payload
    patch_conn(monkeypatch)

    pager = PagerClient(host="h", port=1)
    ok, info = pager.send_page("123", "20250101120000")

    inst = FakeHTTPConnection.last_instance

    assert ok is True
    assert info == "ok 200"
    assert inst.request_args["body"] == b"123,20250101120000"


def test_send_page_non_2xx_returns_http_error(monkeypatch):
    # Non success HTTP status
    class Conn(FakeHTTPConnection):
        def __init__(self, host, port, timeout):
            super().__init__(host, port, timeout)
            self.response = FakeResponse(500, b"boom")

    patch_conn(monkeypatch, Conn)

    pager = PagerClient(host="h", port=1)
    ok, info = pager.send_page("123", None)

    assert ok is False
    assert "http 500" in info
    assert "boom" in info


def test_send_page_ascii_ignore(monkeypatch):
    # Non ascii chars dropped
    patch_conn(monkeypatch)

    pager = PagerClient(host="h", port=1)
    ok, _ = pager.send_page("12é3", None)

    inst = FakeHTTPConnection.last_instance

    assert ok is True
    assert inst.request_args["body"] == b"123"


def test_send_page_connection_exception(monkeypatch):
    # Exception during connect or request
    class BoomConn:
        def __init__(self, *a, **k):
            raise RuntimeError("fail")

    patch_conn(monkeypatch, BoomConn)

    pager = PagerClient(host="h", port=1)
    ok, info = pager.send_page("123", None)

    assert ok is False
    assert "error" in info
    assert "fail" in info
