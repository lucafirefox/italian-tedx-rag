import os

from loguru import logger as log


def check_openai_api_key() -> str:
    """
    Check if OPENAI_API_KEY environment variable is set.

    Returns:
        str: The API key if it exists

    Raises:
        OSError: If the API key is not set
    """
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        error_msg = "OPENAI_API_KEY environment variable not set"
        raise OSError(error_msg)

    log.info("OpenAI API key found")
    return OPENAI_API_KEY
