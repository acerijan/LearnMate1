import io
from typing import Union

import fitz  # PyMuPDF


class DocumentReader:
    """Minimal PDF text extractor using PyMuPDF.

    Accepts a path-like object or in-memory BytesIO for Streamlit uploads.
    """

    def __init__(self, file_obj: Union[str, io.BytesIO]):
        self.file_obj = file_obj

    def extract_text(self) -> str:
        # Open either from bytes or from file path
        if isinstance(self.file_obj, io.BytesIO):
            self.file_obj.seek(0)
            doc = fitz.open(stream=self.file_obj.read(), filetype="pdf")
        else:
            doc = fitz.open(self.file_obj)

        contents = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text:
                contents.append(text)
        return "\n".join(contents)


