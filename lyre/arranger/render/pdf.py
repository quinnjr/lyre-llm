import shutil
import subprocess


class PdfUnavailable(RuntimeError):
    pass


def write_pdf(musicxml_path, pdf_path):
    try:
        import verovio

        tk = verovio.toolkit()
        tk.loadFile(musicxml_path)
        tk.renderToPdf(pdf_path)
        return
    except Exception:
        pass
    musescore = shutil.which("musescore") or shutil.which("musescore3") or shutil.which("musescore4")
    if not musescore:
        raise PdfUnavailable(
            "PDF export needs the 'verovio' package or a MuseScore CLI "
            "(musescore/musescore3/musescore4) on PATH"
        )
    subprocess.run(
        [musescore, "-o", pdf_path, musicxml_path],
        check=True,
        capture_output=True,
    )