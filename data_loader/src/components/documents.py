import re
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from loguru import logger as log

REGEX_PATTERN = r"""
    (?:.*?/)?                    # Optional path prefix
    (?P<speech_name>[^｜]+)\s*   # Speech name up to ｜
    ｜\s*                        # Separator ｜
    (?P<speech_author>[^｜]+)\s* # Speech author up to ｜
    ｜\s*                        # Separator ｜
    (?P<speech_location>[^\[]+)  # Speech location up to [
    \[(?P<youtube_id>[^\]]+)\]   # YouTube ID in brackets
    """


def extract_metadata(filepath: str) -> str | None:
    """
    Extract metadata from a filepath using a specific naming pattern.

    Args:
        filepath (str or Path): Path to the file following the pattern:
            'speech_name｜speech_author｜speech_location[youtube_id]'

    Returns:
        dict or None: A dictionary containing extracted metadata with keys:
            - speech_name: Name of the speech
            - speech_author: Author of the speech
            - speech_location: Location of the speech
            - youtube_id: YouTube video ID
            Returns None if the filepath doesn't match the expected pattern
    """
    # Extract filename from path
    filepath = Path(filepath).name

    # Compile pattern with verbose flag (VERBOSE allows spaces and comments in regex)
    regex = re.compile(REGEX_PATTERN, re.VERBOSE)

    # Try to match the pattern
    match = regex.search(filepath)

    if match:
        return {
            "speech_name": match.group("speech_name").strip(),
            "speech_author": match.group("speech_author").strip(),
            "speech_location": match.group("speech_location").strip(),
            "youtube_id": match.group("youtube_id").strip(),
        }
    return None


def load_documents(input_dir: str) -> list[Document]:
    """
    Load documents from a directory and extract metadata from their filenames using LlamaIndex's SimpleDirectoryReader.

    Args:
        input_dir (str): Path to the directory containing documents to be indexed

    Returns:
        list[Document]: List of LlamaIndex Document objects with metadata extracted from filenames
    """
    reader = SimpleDirectoryReader(input_dir=input_dir, file_metadata=extract_metadata)
    docs = reader.load_data()
    log.info(f"Loaded {len(docs)} documents from {input_dir}")
    return docs
