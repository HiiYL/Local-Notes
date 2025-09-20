from bs4 import BeautifulSoup
from markdownify import markdownify as md


def html_to_markdown(html: str) -> str:
    """Convert Apple Notes HTML body to Markdown.

    - Removes scripts/styles
    - Uses markdownify for HTML->MD
    - Normalizes excessive blank lines
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    body = soup.body or soup
    md_text = md(str(body), heading_style="ATX", bullets="-")

    # Normalize whitespace: collapse multiple blank lines
    lines = [line.rstrip() for line in md_text.splitlines()]
    out = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip()


# Backward-compatible alias
def html_to_text(html: str) -> str:
    return html_to_markdown(html)
