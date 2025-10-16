import logging
import os
from logger_config import get_logger

import torch
from docling.document_converter import DocumentConverter

logger = get_logger(__name__)


def pdfs_to_markdown(source: str) -> str:
    """
    Converts pdf files to markdown using docling
    :param source: local file path or remote url
    :return: markdown string
    """

    converter = DocumentConverter()
    result = converter.convert(source)
    logger.info(f"Done Converting {os.path.basename(source)}")
    return result.document.save_as_markdown(os.path.basename(source).replace(".pdf", ".md"))


if "__main__" == __name__:
    pdfs_to_markdown(r"C:\Users\ahmed\PycharmProjects\AOUNet\data\pdfs\Faculty of Computer Studies\undergraduate programs\academic plans\Diploma Information Technology & Computing.pdf")
