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
_PLAIN_MATH = re.compile(r"[\w\s+\-*/^=<>()\[\]{},.;!|\\]+", re.ASCII)
_MATH_WORDS = frozenset({
    "sin", "cos", "tan", "cot", "tg", "ctg", "ln", "log", "exp", "sqrt", "pi",
    "frac", "sum", "prod", "int", "lim", "in", "notin", "le", "leq", "ge", "geq",
    "ne", "neq", "cdot", "times", "div", "ldots", "cdots", "infty", "mathbb",
})
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ⁿⁱ", "0123456789+-()ni")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋₍₎", "0123456789+-()")
_MATH_SYMBOLS = {
    "−": "-", "⋅": r"\cdot ", "·": r"\cdot ", "×": r"\times ",
    "÷": r"\div ", "…": r"\ldots ", "π": r"\pi ", "∞": r"\infty ",
    "≤": r"\leq ", "≥": r"\geq ", "≠": r"\neq ", "∈": r"\in ",
    "∉": r"\notin ", "ℤ": r"\mathbb{Z}", "ℕ": r"\mathbb{N}",
    "½": r"\frac{1}{2}", "¼": r"\frac{1}{4}", "¾": r"\frac{3}{4}",
}
_ENV_NAMES = r"aligned|gathered|cases|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix"
_ENV_TOKEN = re.compile(r"\\(begin|end)\{(" + _ENV_NAMES + r")\}")
_MULTILINE_ENV = re.compile(
    r"(?<![\\$])\$(?!\$)(?P<body>\s*\\begin\{(?P<env>" + _ENV_NAMES
    + r")\}[\s\S]*?\\end\{(?P=env)\}\s*)\$(?!\$)"
)
_OPEN_ENV = re.compile(
    r"(?<![\\$])(?:(?:\$\$|\$|\\\[|\\\()[ \t\r\n]*)?\\begin\{(?:" + _ENV_NAMES + r")\}"
)
_EQUATION_TAG = re.compile(r"(?<!\\)\\tag\*?\s*(?:\{([A-Za-z0-9.:-]{1,40})\}|(\d+))")


@dataclass(frozen=True, slots=True)
class PreparedMath:
    markdown: str
    has_math: bool


@dataclass(frozen=True, slots=True)
class MathRichChunk:
    payload: InputRichMessage
    fallback_text: str


def has_unfinished_math(source: str) -> bool:
    """Join provider continuations verbatim only inside math, not code or prices."""
    parts: list[str] = []
    offset = 0
    while opening := _FENCE.search(source, offset):
        parts.append(source[offset:opening.start()])
        marker = opening[1]
        closing = re.compile(
            rf"(?m)^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*(?:\n|$)"
        ).search(source, opening.end())
        offset = closing.end() if closing else len(source)
    parts.append(source[offset:])
    text = _INLINE_CODE.sub("", "".join(parts))
    text = _MULTILINE_ENV.sub("", text)
    text = _MATH.sub("", text)
    if re.search(r"(?<!\\)(?:\$\$|\\\[|\\\()", text):
        return True
    tail = re.search(r"(?<![\\$])\$(?!\$)([^$]+)\Z", text)
    if tail and re.match(r"[A-Za-z(\\]", tail[1].lstrip()) and re.search(
        r"[=^_(){}\\]", tail[1]
    ):
        return True
    return bool(_OPEN_ENV.search(text) and not re.search(r"\\end\{", text))


def _close_typesetting_environments(body: str) -> str | None:
    """Close layout only; never add mathematical brackets, terms or values."""
    balance = 0
    for match in re.finditer(r"(?<!\\)[{}]", body):
        balance += 1 if match[0] == "{" else -1
        if balance < 0:
            return None
    if balance or re.search(r"(?<!\\)\\[A-Za-z]*$", body):
        return None
    stack: list[str] = []
    for token in _ENV_TOKEN.finditer(body):
        action, environment = token.groups()
        if action == "begin":
            stack.append(environment)
        elif not stack or stack.pop() != environment:
            return None
    if body.count(r"\left") != body.count(r"\right"):
        body = re.sub(r"\\(?:left|right)\b", "", body)
    return body + "".join(r"\end{" + environment + "}" for environment in reversed(stack))


def _complete_relation(row: str) -> bool:
    relation = _RELATION.search(row)
    return bool(
        relation
        and row[: relation.start()].strip(" &")
        and row[relation.end() :].strip(" ,.;")
        and not row.rstrip().endswith(("+", "-", "*", "/", "=", "^", "_", "\\"))
    )


def _closed_environment_spans(source: str) -> list[tuple[int, int]]:
    """Find balanced outer math layouts, including nested matrices/cases."""
    stack: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []
    for token in _ENV_TOKEN.finditer(source):
        action, environment = token.groups()
        if action == "begin":
            stack.append((token.start(), environment))
        elif stack and stack[-1][1] == environment:
            start, _ = stack.pop()
            if not stack:
                spans.append((start, token.end()))
        else:
            stack.clear()  # do not guess how a mismatched environment should close
    return spans


def _clean_math(body: str) -> str:
    # A single trailing slash is an LLM/Markdown artefact, not a LaTeX row break.
    body = re.sub(r"(?m)(?<!\\)\\[ \t]*$", "", body)
    aligned = re.fullmatch(r"\\begin\{aligned\}([\s\S]*)\\end\{aligned\}", body.strip())
    if aligned and r"\\" not in aligned[1] and r"\begin" not in aligned[1]:
        equations = [row.strip() for row in aligned[1].splitlines() if row.strip()]
        if len(equations) > 1 and all(
            _complete_relation(row) and row.count("{") == row.count("}")
            for row in equations
        ):
            return "\\begin{aligned}\n" + " \\\\\n".join(equations) + "\n\\end{aligned}"
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


def _plain_formula_latex(line: str) -> str | None:
    """Recognise standalone algebra, never infer division precedence or solve it."""
    if line.startswith(("    ", "\t")) or len(line) > 1500:
        return None
    body = line.strip()
    body = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ⁿⁱ]+", lambda m: "^{" + m[0].translate(_SUPERSCRIPTS) + "}", body
    )
    body = re.sub(
        r"[₀₁₂₃₄₅₆₇₈₉₊₋₍₎]+", lambda m: "_{" + m[0].translate(_SUBSCRIPTS) + "}", body
    )
    for symbol, latex in _MATH_SYMBOLS.items():
        body = body.replace(symbol, latex)
    if not _PLAIN_MATH.fullmatch(body):
        return None
    if any(word not in _MATH_WORDS for word in re.findall(r"[A-Za-z]{2,}", body)):
        return None
    if not re.search(r"[=<>^_]|\\(?:in|notin|leq|geq|neq|cdot|times|div)\b", body):
        return None
    if not re.search(r"[A-Za-z0-9]", body) or body.rstrip().endswith(
        ("+", "-", "=", "^", "_", "/", "\\")
    ):
        return None
    stack: list[str] = []
    for char in body:
        if char in "([{":
            stack.append(char)
        elif char in ")]}" and (
            not stack or stack.pop() != {")": "(", "]": "[", "}": "{"}[char]
        ):
            return None
    if stack:
        return None
    return re.sub(r"_\(([^()]+)\)", r"_{\1}", body)


def prepare_math_markdown(source: str, *, truncated: bool = False) -> PreparedMath:
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

    def protect_math(body: str, *, display: bool) -> str:
        nonlocal has_math
        numbers = [first or second for first, second in _EQUATION_TAG.findall(body)]
        body = _clean_math(_EQUATION_TAG.sub("", body))
        has_math = bool(body) or has_math
        rendered = protect(f"$$\n{body}\n$$") if display else protect(f"${body}$")
        suffix = " " + ", ".join(f"({number})" for number in numbers) if numbers else ""
        return f"\n\n{rendered}{suffix}\n\n" if display else rendered + suffix

    def normalise(match: re.Match[str]) -> str:
        nonlocal has_math
        # Never join formula delimiters across a protected code region.
        if prefix in match.group():
            return match.group()
        line_prefix = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
        if line_prefix.startswith(("    ", "\t")):
            return match.group()  # indented Markdown code
        body = next(value for value in match.groups() if value is not None)
        if truncated and _ENV_TOKEN.search(body):
            closed = _close_typesetting_environments(body)
            if closed is None:
                return match[0]  # recover completed rows below, outside the broken wrapper
            body = closed
        body = _clean_math(body)
        if not body:
            return match.group()
        display = match.group("display") is not None or match.group("bracket") is not None
        return protect_math(body, display=display)

    text = _MULTILINE_ENV.sub(
        lambda match: match[0] if prefix in match["body"]
        else protect_math(match["body"], display=True), text,
    )
    text = _MATH.sub(normalise, text)
    # A completed answer can contain a valid environment without any $ delimiters.
    # This is provider-independent and must not depend on finish_reason=length.
    pieces = []
    offset = 0
    for start, end in _closed_environment_spans(text):
        body = text[start:end]
        line_prefix = text[text.rfind("\n", 0, start) + 1:start]
        if (prefix in body or line_prefix.startswith(("    ", "\t"))
                or _close_typesetting_environments(body) is None):
            continue
        pieces.extend((text[offset:start], protect_math(body, display=True)))
        offset = end
    pieces.append(text[offset:])
    text = "".join(pieces)
    if truncated:
        offset = 0
        while opening := _OPEN_ENV.search(text, offset):
            start = text.find(r"\begin{", opening.start(), opening.end())
            remainder = text[start:]
            prose = re.search(
                r"(?m)^[ \t]*(?:⚠️|[А-Яа-яЁё]|[A-Za-z]{2,}(?:[ \t]+|[,:.!?])|"
                + re.escape(prefix) + ")", remainder
            )
            end = start + prose.start() if prose else len(text)
            body = text[start:end].rstrip()
            body = re.sub(r"(?:\$\$|\$|\\\]|\\\))\s*\Z", "", body).rstrip()
            # A broken last fraction must not invalidate all previous aligned rows.
            boundaries = [len(body), *reversed([
                match.end() for match in re.finditer(r"\\\\(?:\[[^\n\]]*\])?", body)
            ])]
            repaired = ""
            fragment = body
            for boundary in boundaries:
                closed = _close_typesetting_environments(body[:boundary].rstrip())
                if closed is not None:
                    repaired = protect_math(closed, display=True)
                    fragment = body[boundary:].strip()
                    break
            if fragment:
                fragment = _ENV_TOKEN.sub("", fragment)
                fragment = re.sub(r"(?m)^[ \t]*&", "", fragment).strip()
                # Preserve incomplete source explicitly rather than fake a denominator.
                fence = "`" * (
                    max((len(m[0]) for m in re.finditer(r"`+", fragment)), default=0) + 3
                )
                repaired += protect("\nНезавершённый фрагмент формулы:\n" + fence + "latex\n"
                                    + fragment + "\n" + fence + "\n")
            text = text[:opening.start()] + repaired + text[end:]
            offset = opening.start() + len(repaired)

        def unfinished(match: re.Match[str]) -> str:
            delimiter, body = match.groups()
            line_prefix = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
            if prefix in body or not body.strip() or line_prefix.startswith(("    ", "\t")):
                return match.group()
            if delimiter == "$" and not (
                re.match(r"[A-Za-z(\\]", body) and re.search(r"[=^_(){}\\]", body)
            ):
                return match.group()  # currency and ordinary unmatched dollar signs
            try:
                plain = convert(f"${body}$", latex_escape=False)[0]
            except Exception:
                plain = body
            # Keep the fragment readable, without inventing missing terms/brackets.
            return protect(plain)

        text = re.sub(
            r"(?<![\\$])(\$\$|\\\[|\\\(|\$)([^\n]*(?:\n(?!\s*\n)[^\n]+)*)",
            unfinished, text,
        )

    def standalone(match: re.Match[str]) -> str:
        nonlocal has_math
        body = _plain_formula_latex(match[0])
        if body is None:
            return match[0]
        has_math = True
        return protect(f"\n\n$$\n{body}\n$$\n\n")

    text = re.sub(r"(?m)^[^\n]+$", standalone, text)
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
