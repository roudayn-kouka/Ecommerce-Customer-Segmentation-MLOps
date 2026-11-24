"""
Download the raw Online Retail dataset (UCI) as a zip archive.
"""

from __future__ import annotations

import os

import requests

from src.utils import DATA_DIR, get_logger

logger = get_logger(__name__)

DEFAULT_FILE_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def ingest_data(
    file_url: str = DEFAULT_FILE_URL, timeout: int = 60, output_dir: str | None = None
) -> str:
    """
    Download the file at ``file_url`` into ``<output_dir>/data.zip``.

    :param file_url: URL of the zip archive (UCI Online Retail by default).
    :param timeout: HTTP timeout in seconds.
    :param output_dir: Target folder (defaults to the project ``data`` folder).
    :return: Path to the downloaded zip file.
    :raises requests.HTTPError: If the server does not answer with a 2xx status.
    """
    output_dir = output_dir or str(DATA_DIR)
    os.makedirs(output_dir, exist_ok=True)
    zipfile_path = os.path.join(output_dir, "data.zip")

    logger.info("Downloading data from %s", file_url)
    response = requests.get(file_url, timeout=timeout)
    response.raise_for_status()

    with open(zipfile_path, "wb") as file:
        file.write(response.content)
    logger.info("File downloaded successfully. Zip file available under %s", zipfile_path)
    return zipfile_path


if __name__ == "__main__":
    ingest_data()
