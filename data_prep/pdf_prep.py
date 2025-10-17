import logging
import os
from pathlib import Path

from tqdm import tqdm

from logger_config import get_logger

import torch
from docling.document_converter import DocumentConverter

logger = get_logger(__name__)

pdf_parent_dir_path = Path(r"..\data\pdfs")
path_to_mds = Path(r"..\data\mds")

def pdfs_to_markdown(source: str) -> str:
    """
    Converts pdf files to markdown using docling
    :param source: local file path or remote url
    :return: markdown string
    """

    converter = DocumentConverter()
    result = converter.convert(source)
    logger.debug(f"Done Converting {os.path.basename(source)}")
    return result.document.save_as_markdown(path_to_mds / os.path.basename(source).replace(".pdf", ".md"))


def all_pdfs_markdown():

    pdfs_path = []

    if not pdf_parent_dir_path.exists():
        raise Exception("dir not found")
    if not pdf_parent_dir_path.is_dir():
        raise Exception("Not a dir")

    def iterate_child_dir(fromdir: Path):
        global original_doc_count

        logger.debug(f"within {os.path.basename(fromdir)}")
        children = fromdir.iterdir()
        for dir in children:
            if dir.is_dir():
                iterate_child_dir(dir)
            else:
                # document detected
                pdfs_path.append(dir)

    iterate_child_dir(pdf_parent_dir_path)


    for pdf_path in tqdm(pdfs_path, desc="Conveting PDFs to Markdown"):
        pdfs_to_markdown(pdf_path)

if "__main__" == __name__:
    # pdfs_to_markdown(r"C:\Users\ahmed\PycharmProjects\AOUNet\data\pdfs\Faculty of Computer Studies\undergraduate programs\academic plans\Diploma Information Technology & Computing.pdf")
    all_pdfs_markdown()