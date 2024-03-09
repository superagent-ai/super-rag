import asyncio
import logging
import re
import textwrap
import time
from typing import List

import pandas as pd
from decouple import config
from e2b import Sandbox
from openai import AsyncOpenAI

from models.file import File, FileType

logging.getLogger("e2b").setLevel(logging.INFO)

client = AsyncOpenAI(
    api_key=config("OPENAI_API_KEY"),
)

SYSTEM_PROMPT = "You are a world-class python programmer that can complete any data analysis tasks by coding."


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
        If no session_id is given, create a new sandbox that will
        be deleted after exiting the object context.
        """
        if not session_id:
            return Sandbox("super-rag")

        sandboxes = Sandbox.list()
        for s in sandboxes:
            if not s.metadata:
                continue
            if s.metadata.get("session_id") == session_id:
                return Sandbox.reconnect(s.sandbox_id)

        return Sandbox(metadata={"session_id": session_id}, template="super-rag")

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
        try:
            if not self._is_initialized:
                self._is_initialized = True
                for file_url in self.file_urls:
                    await self._upload_file(file_url)
        except:
            self.self.sandbox.close()
            raise

        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback):
        try:
            if self.session_id:
                self.sandbox.keep_alive(self.timeout)
        finally:
            self.sandbox.close()

    def get_dataframe(self):
        """
        Get the code to read the files in the sandbox.
        This can be used for instructing the LLM how to access the loaded files.
        """
        dataframes = {}
        for file_url in self.file_urls:
            file = File(url=file_url)
            if file.type == FileType.csv:
                df = pd.read_csv(file_url)
            elif file.type == FileType.json:
                df = pd.read_json(file_url)
            elif file.type == FileType.xlsx:
                df = pd.read_excel(file_url)
            dataframes[file_url] = df
        return dataframes

    def generate_prompt(self, query: str) -> list:
        prompts = []
        dataframes = self.get_dataframe()
        for url, df in dataframes.items():
            prompts.append(
                textwrap.dedent(
                    f"""
                You are provided with a following pandas dataframe (`df`):
                {df.info()}

                Using the provided dataframe (`df`), update the following python code using pandas that returns the answer to question: \"{query}\"

                This is the initial python code to be updated:

                ```python
                import pandas as pd

                df = pd.read_csv("{url}") 
                1. Process: Manipulating data for analysis (grouping, filtering, aggregating, etc.)
                2. Analyze: Conducting the actual analysis
                3. Output: Returning the answer as a string
                ```
                """
                )
            )
        return prompts

    def extract_code(self, code: str) -> str:
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, code, re.DOTALL)
        if matches:
            return matches[0].strip()
        return ""

    async def generate_code(
        self,
        query: str,
    ) -> list:
        extracted_code = []
        generated_prompts = self.generate_prompt(query=query)
        for content in generated_prompts:
            completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                model="gpt-3.5-turbo-0125",
            )
            output = completion.choices[0].message.content
            extracted_code.append(self.extract_code(code=output))
        return extracted_code

    async def run_python(self, code: str):
        epoch_time = time.time()
        codefile_path = f"/tmp/main-{epoch_time}.py"
        self.sandbox.filesystem.write(codefile_path, code)
        process = await asyncio.to_thread(
            self.sandbox.process.start_and_wait,
            f"python {codefile_path}",
        )

        return process
