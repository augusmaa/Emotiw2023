import os

class Notifier:
    """Optional Pushover notifications via env vars (safe for GitHub)."""

    def __init__(self) -> None:
        self.enabled = bool(os.getenv("PUSHOVER_TOKEN")) and bool(os.getenv("PUSHOVER_USER"))
        self._client = None
        if self.enabled:
            from pushover_complete import PushoverAPI  # optional dependency
            self._client = PushoverAPI(os.environ["PUSHOVER_TOKEN"])
            self._user = os.environ["PUSHOVER_USER"]

    def send(self, message: str) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            self._client.send_message(self._user, message)
        except Exception:
            pass
