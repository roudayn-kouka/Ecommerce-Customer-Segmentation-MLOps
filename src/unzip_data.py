"""
Extract the downloaded zip archive and return the path of the Excel file.
"""

from __future__ import annotations

import os
import zipfile

from src.utils import DATA_DIR, get_logger

logger = get_logger(__name__)

ZIP_FILENAME = str(DATA_DIR / "data.zip")
EXTRACT_TO = str(DATA_DIR)
EXCEL_NAME = "Online Retail.xlsx"


def unzip_file(zip_filename: str = ZIP_FILENAME, extract_to: str = EXTRACT_TO) -> str:
    """
    Unzip ``zip_filename`` into ``extract_to``.

    :return: Path to the extracted ``Online Retail.xlsx`` file.
    :raises FileNotFoundError: If the archive does not exist.
    :raises zipfile.BadZipFile: If the archive is corrupted.
    """
    if not os.path.exists(zip_filename):
        raise FileNotFoundError(f"Zip file not found: {zip_filename}")
    try:
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        logger.error("Failed to unzip %s", zip_filename)
        raise
    logger.info("File %s successfully unzipped to %s", zip_filename, extract_to)
    return os.path.join(extract_to, EXCEL_NAME)


if __name__ == "__main__":
    unzip_file()
