# agent/pdf_report.py
from fpdf import FPDF
import base64
import io

class SimpleReport:
    def __init__(self, title="Agentes Autônomos - Relatório da Atividade Extra"):
        self.title = title
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def add_title(self, text):
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 10, text, ln=True, align="C")

    def add_paragraph(self, text):
        self.pdf.set_font("Arial", size=11)
        self.pdf.multi_cell(0, 6, text)
        self.pdf.ln(2)

    def add_image_from_buf(self, buf, w=160):
        # buf: BytesIO with PNG
        b64 = base64.b64encode(buf.getvalue()).decode()
        fname = "tmp_img.png"
        with open(fname, "wb") as f:
            f.write(base64.b64decode(b64))
        self.pdf.image(fname, w=w)
        self.pdf.ln(4)

    def output(self, path="Agentes Autônomos – Relatório da Atividade Extra.pdf"):
        self.pdf.output(path)
        return path
