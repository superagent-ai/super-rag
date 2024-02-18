import logging
import time
import asyncio

from e2b import CodeInterpreter as CISandbox

logging.getLogger("e2b").setLevel(logging.ERROR)


class CodeInterpreterService:
    timeout = 3 * 60  # 3 minutes

    def _ensure_sandbox(
        self,
        session_id: str | None,
    ):
        if not session_id:
            # If we don't have a session_id we can create a new sandbox that will be deleted after timeout
            return CISandbox()

        # We can use the metadata to store the session_id and then reconnect to the sandbox if we find one with the same session_id
        sandboxes = CISandbox.list()
        for s in sandboxes:
            if not s.metadata:
                continue
            if s.metadata.get("session_id") == session_id:
                return CISandbox.reconnect(s.sandbox_id)

        # If we don't find a sandbox with the same session_id we can create a new one
        return CISandbox(metadata={"session_id": session_id})

    def __init__(
        self,
        session_id: str | None,
    ):
        self.session_id = session_id
        self.sandbox = self._ensure_sandbox(session_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # If the session_id is None the sandbox was a single use and we won't be reconnecting to it.
        if self.session_id:
            self.sandbox.keep_alive(self.timeout)

        # Closing sandbox here disconnects from the sandbox, but if the session_id was defined will still exists and we can reconnect to it until the timeout is reached.
        # After reconnecting the sandbox won't be deleted until we close it again. We can then still call keep_alive to keep it alive for a while.
        self.sandbox.close()

    async def run_python(self, code: str):
        # We can use a temporary file to run the code or execute the code directly while escaping special characters
        # If we need dependencies we can build a sandbox template where we install them or install them ad-hoc.

        epoch_time = time.time()
        codefile_path = f"/tmp/main-{epoch_time}.py"
        self.sandbox.filesystem.write(codefile_path, code)

        process = await asyncio.to_thread(
            self.sandbox.process.start_and_wait,
            f"python {codefile_path}",
        )

        return process
