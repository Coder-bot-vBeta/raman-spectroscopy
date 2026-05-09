"""In-memory spectrum store keyed by UUID."""
import uuid

_store: dict = {}


def save(data: dict) -> str:
    sid = str(uuid.uuid4())
    _store[sid] = data
    return sid


def get(sid: str) -> dict | None:
    return _store.get(sid)


def delete(sid: str):
    _store.pop(sid, None)
