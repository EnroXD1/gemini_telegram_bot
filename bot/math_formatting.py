from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from uuid import uuid4

from aiogram.types import InputRichMessage
from telegramify_markdown import convert, telegramify_rich

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})([^\n]*)\n?")
_INLINE_CODE = re.compile(r"(?<![\\`])(`+)(?!`)([\s\S]*?)(?<!`)\1(?!`)")
_MATH = re.compile(
    r"(?<!\\)\$\$(?!\$)(?P<display>[\s\S]*?)(?<!\\)\$\$(?!\$)"
    r"|(?<!\\)\\\[(?P<bracket>[\s\S]*?)\\\]"
    r"|(?<!\\)\\\((?P<paren>[\s\S]*?)\\\)"
    r"|(?<![\\$])\$(?![\s$])(?P<inline>[^\n$]*?\S)(?<!\\)\$(?![\d$])"
)
_RELATION = re.compile(r"[=<>≤≥∈∉]|\\(?:in|notin|leq?|geq?|neq)\b")
_TEXT_ROW = re.compile(r"[А-Яа-яЁё ]+")
_MEDIA_TAG = re.compile(r"<(?:img|video|audio|tg-document)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PreparedMath:
    markdown: str
    has_math: bool


@dataclass(frozen=True, slots=True)
class MathRichChunk:
    payload: InputRichMessage
    fallback_text: str


def _complete_relation(row: str) -> bool:
    relation = _RELATION.search(row)
    return bool(
        relation
        and row[: relation.start()].strip(" &")
        and row[relation.end() :].strip(" ,.;")
        and not row.rstrip().endswith(("+", "-", "*", "/", "=", "^", "_", "\\"))
    )


def _clean_math(body: str) -> str:
    # A single trailing slash is an LLM/Markdown artefact, not a LaTeX row break.
    body = re.sub(r"(?m)(?<!\\)\\[ \t]*$", "", body)
    rows = [line.strip() for line in body.splitlines() if line.strip()]
    # Repair the common pasted solution: complete equations separated by blank lines.
    # Do not split incomplete expressions, nested fractions or existing environments.
    if (
        len(rows) > 1
        and r"\begin" not in body
        and r"\\" not in body
        and sum(_complete_relation(row) for row in rows) >= 2
        and all(row.count("{") == row.count("}") for row in rows)
        and all(_complete_relation(row) or _TEXT_ROW.fullmatch(row) for row in rows)
    ):
        rows = [rf"\text{{{row}}}" if _TEXT_ROW.fullmatch(row) else row for row in rows]
        return "\\begin{aligned}\n" + " \\\\\n".join(rows) + "\n\\end{aligned}"
    return "\n".join(rows)


def prepare_math_markdown(source: str) -> PreparedMath:
    """Normalise math delimiters without rewriting ordinary code, URLs or prices."""
    protected: dict[str, str] = {}
    prefix = f"\ue000{uuid4().hex}:"
    math_fences = False

    def protect(value: str) -> str:
        token = f"{prefix}{len(protected)}\ue001"
        protected[token] = value
        return token

    pieces: list[str] = []
    offset = 0
    while opening := _FENCE.search(source, offset):
        pieces.append(source[offset : opening.start()])
        marker, language = opening.groups()
        closing = re.compile(
            rf"(?m)^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*(?:\n|$)"
        ).search(source, opening.end())
        end = closing.end() if closing else len(source)
        if language.strip().lower() == "math" and closing:
            body = _clean_math(source[opening.end() : closing.start()])
            pieces.append(f"\n\n$$\n{body}\n$$\n\n")
            math_fences = bool(body) or math_fences
        else:
            pieces.append(protect(source[opening.start() : end]))
        offset = end
    pieces.append(source[offset:])
    text = _INLINE_CODE.sub(lambda match: protect(match.group()), "".join(pieces))
    has_math = math_fences

    def normalise(match: re.Match[str]) -> str:
        nonlocal has_math
        # Never join formula delimiters across a protected code region.
        if prefix in match.group():
            return match.group()
        line_prefix = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
        if line_prefix.startswith(("    ", "\t")):
            return match.group()  # indented Markdown code
        body = next(value for value in match.groups() if value is not None)
        body = _clean_math(body)
        if not body:
            return match.group()
        has_math = True
        display = match.group("display") is not None or match.group("bracket") is not None
        return f"\n\n$$\n{body}\n$$\n\n" if display else f"${body}$"

    text = _MATH.sub(normalise, text)
    if has_math:
        text = re.sub(r"(?m)^ {0,3}\\[ \t]*(?=\n|$)", "", text)

        def prose_break(match: re.Match[str]) -> str:
            value = match.group(1)
            if not re.fullmatch(r"[A-Za-z]:", value) and (
                re.search(r"\s|[А-Яа-яЁё]", value) or value.endswith((":", ".", "!", "?", ";"))
            ):
                return value
            return match.group()

        text = re.sub(r"(?m)^([^\\\n]+)\\[ \t]*$", prose_break, text)
    for token, value in protected.items():
        text = text.replace(token, value)
    return PreparedMath(text, has_math)


class _RichTextFallback(HTMLParser):
    """Make each rejected rich chunk readable, without replaying already sent chunks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.math: list[str] | None = None
        self.block_math = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"tg-math", "tg-math-block"}:
            self.math = []
            self.block_math = tag == "tg-math-block"
            if self.block_math:
                self.parts.append("\n\n")
        elif tag in {"p", "pre", "blockquote", "li", "tr", "h1", "h2", "h3", "footer", "br"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"tg-math", "tg-math-block"} and self.math is not None:
            body = "".join(self.math)
            delimiter = "$$" if self.block_math else "$"
            try:
                plain = convert(f"{delimiter}{body}{delimiter}", latex_escape=False)[0]
            except Exception:
                plain = body  # do not lose unsupported expressions
            self.parts.append(plain)
            self.math = None
            if self.block_math:
                self.parts.append("\n\n")
        elif tag in {"p", "pre", "blockquote", "li", "tr", "h1", "h2", "h3", "footer"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        (self.math if self.math is not None else self.parts).append(data)


def render_math_rich_chunks(source: str) -> list[MathRichChunk]:
    prepared = prepare_math_markdown(source)
    if not prepared.has_math:
        return []
    try:
        payloads = [
            item.to_dict() for item in telegramify_rich(prepared.markdown, latex_escape=False)
        ]
        # Rich conversion must not introduce remote media downloads into AI replies.
        if any(_MEDIA_TAG.search(item.get("html", "")) for item in payloads):
            return []
        if not any("<tg-math" in item.get("html", "") for item in payloads):
            return []
        chunks = []
        for item in payloads:
            parser = _RichTextFallback()
            parser.feed(item["html"])
            parser.close()
            chunks.append(
                MathRichChunk(
                    InputRichMessage.model_validate(item), "".join(parser.parts).strip()
                )
            )
        return chunks
    except Exception as exc:
        logger.warning("Could not render rich math type=%s", type(exc).__name__)
        return []


def readable_math_text(source: str) -> str:
    """Inline results use ordinary text; normalise formula syntax there as well."""
    prepared = prepare_math_markdown(source)
    if not prepared.has_math:
        return source
    try:
        return convert(prepared.markdown, latex_escape=False)[0]
    except Exception:
        return source
