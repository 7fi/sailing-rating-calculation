"""Reading and writing the cached HTML under pages/, pagesTR/ and sailorPages/."""
import ast


def decodeBytesRepr(html):
    """Undo a page that was cached as str(response.content) rather than response.text.

    str() on a bytes object gives its repr, so the whole page gets wrapped in b'...',
    every non-ASCII byte becomes the literal text \\xe2\\x80\\x99, and apostrophes come
    back escaped as \\'. Anything read out of a page in that state is mangled - a JSON
    \\u2019 in the ts:data meta tag becomes a doubled \\\\u2019, which json.loads then
    decodes to the six literal characters \\u2019, so O'Gwen displays as O\\u2019Gwen.

    The encoding is reversible, so old caches are repaired on read instead of refetched.
    """
    if not html.startswith(("b'", 'b"')):
        return html
    try:
        return ast.literal_eval(html).decode("utf-8", errors="replace")
    except (ValueError, SyntaxError):
        # Not actually a bytes repr, or a truncated file - leave it as it is.
        return html


def readPage(path):
    with open(path, "r", encoding="utf-8") as f:
        return decodeBytesRepr(f.read())


def writePage(path, response):
    """Cache a response as decoded text. Never write str(response.content) here."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(response.text)
