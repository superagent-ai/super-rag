import logging
import time
import asyncio

from typing import List
from e2b import CodeInterpreter as CISandbox

logging.getLogger("e2b").setLevel(logging.ERROR)


class CodeInterpreterService:
    timeout = 3 * 60  # 3 minutes

    @staticmethod
    def _get_file_path(file_url: str):
        """Get the file path in the sandbox for a given file_url."""
        return "/code/" + str(hash(file_url))

    async def _upload_file(self, file_url: str):
        """Upload a file to the sandbox."""
        process = await asyncio.to_thread(
            self.sandbox.process.start_and_wait,
            f"wget -O {self._get_file_path(file_url)} {file_url}",
        )

        if process.exit_code != 0:
            raise Exception(
                f"Error downloading file {file_url} to sandbox {self.sandbox.id}"
            )

    def _ensure_sandbox(
        self,
        session_id: str | None,
    ):
        """
        Ensure we have a sandbox for the given session_id exists. If not, create a new one.
        If no session_id is given, create a new sandbox that will be deleted after exiting the object context.
        """

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
        file_urls: List[str],
    ):
        self.session_id = session_id
        self.file_urls = file_urls
        self._is_initialized = False

        self.sandbox = self._ensure_sandbox(session_id)

    async def __aenter__(self):
        if not self._is_initialized:
            self._is_initialized = True
            for file_url in self.file_urls:
                await self._upload_file(file_url)

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        # If the session_id is None the sandbox was a single use and we won't be reconnecting to it.
        if self.session_id:
            self.sandbox.keep_alive(self.timeout)

        # Closing sandbox here disconnects from the sandbox, but if the session_id was defined will still exists and we can reconnect to it until the timeout is reached.
        # After reconnecting the sandbox won't be deleted until we close it again. We can then still call keep_alive to keep it alive for a while.
        self.sandbox.close()

    def get_files_code(self):
        """
        Get the code to read the files in the sandbox.
        This can be used for instructing the LLM how to access the loaded files.
        """

        # TODO: Right now this only works for reading csv files, but we can add support for other file types.
        files_code = "\n".join(
            f'df{i} = read_csv("{self._get_file_path(url)}") # {url}'
            for i, url in enumerate(self.file_urls)
        )

        return f"""
import pandas as pd

{files_code}

"""

    async def run_python(self, code: str):
        # We can use a temporary file to run the code or execute the code directly while escaping special characters
        # If we need dependencies we can build a sandbox template where we install them or install them ad-hoc.

        files_code = self.get_files_code()

        templated_code = f"""
        {files_code}
        {code}
        """

        epoch_time = time.time()
        codefile_path = f"/tmp/main-{epoch_time}.py"
        self.sandbox.filesystem.write(codefile_path, templated_code)

        process = await asyncio.to_thread(
            self.sandbox.process.start_and_wait,
            f"python {codefile_path}",
        )

        return process
