import re
from types import SimpleNamespace

import pytest
from aiogram.types import Message

from bot.config import ConfigError, Settings
from bot.markdown import render_markdown_chunks
from bot.math_formatting import (
    prepare_math_markdown,
    readable_math_text,
    render_math_rich_chunks,
)
from bot.media import MediaExtractor

BROKEN_SOLUTION = r"""Решаем:

sin(2x/3)=½
\
\
Общие решения синуса:\
\
$$2x/3=π/6+2π n\
\
 или \
\
2x/3=5π/6+2π n,\
\
n∈ Z.$$\
\
Отсюда:\
\
$$[x=π/4+3π n]\
\
 или \
\
[x=5π/4+3π n],  n∈ Z.$$\
\
В градусах это `x=45°+540° n` или `x=225°+540° n`."""


def test_original_broken_solution_becomes_two_native_math_blocks():
    chunks = render_math_rich_chunks(BROKEN_SOLUTION)
    assert len(chunks) == 1
    html = chunks[0].payload.html
    assert html.count("<tg-math-block>") == 2
    assert r"\begin{aligned}" in html and r"\text{или}" in html
    assert "[x=π/4+3π n]" in html and "[x=5π/4+3π n]" in html
    assert "$$" not in html and "синуса:\\" not in html and "Отсюда:\\" not in html
    assert "<code>x=45°+540° n</code>" in html
    plain = chunks[0].fallback_text
    assert "[x=π/4+3π n]" in plain and "[x=5π/4+3π n]" in plain
    assert "$$" not in plain and "\\" not in plain and "или" in plain


@pytest.mark.parametrize(
    "source,tag,latex",
    [
        (r"Ответ: $\frac{1}{2}$.", "tg-math", r"\frac{1}{2}"),
        (r"$$\sqrt{x^2+1}$$", "tg-math-block", r"\sqrt{x^2+1}"),
        (r"\(x^2+\alpha\beta\)", "tg-math", r"x^2+\alpha\beta"),
        (r"\[x=1\]", "tg-math-block", "x=1"),
        ("```math\nx_1=2\\pi\n```", "tg-math-block", r"x_1=2\pi"),
        (r"$$\sum_{n=1}^{\infty}\frac{1}{n^2}$$", "tg-math-block", r"\sum_{n=1}^{\infty}"),
        (r"$$\begin{pmatrix}a&b\\c&d\end{pmatrix}$$", "tg-math-block", r"\begin{pmatrix}"),
    ],
)
def test_common_delimiters_keep_real_latex_for_native_telegram(source, tag, latex):
    chunks = render_math_rich_chunks(source)
    assert chunks and f"<{tag}>" in chunks[0].payload.html
    assert latex in chunks[0].payload.html
    assert chunks[0].fallback_text


@pytest.mark.parametrize(
    "source",
    [
        r"Код: `\[\frac{a}{b}\]` и `$x^2$`.",
        '```python\ns = r"\\[x=1\\]"\nprice = "$5"\n```',
        "~~~~text\n$$\\frac{1}{2}$$\n~~~~",
        "````python\ns = '```'\ns = '$x$'\n````",
        "```python\nx = '$x$'",  # unclosed code fence protects the remaining source
        "Обычная цена $5 и $10, скидка 20%.",
        r"Путь C:\Users\User\file.py и https://example.com/a?x=1&y=2",
        "    print('$x^2$')",
        "Ответ без формул.",
    ],
)
def test_code_prices_paths_and_non_math_are_not_normalised(source):
    prepared = prepare_math_markdown(source)
    assert prepared.markdown == source and not prepared.has_math
    assert render_math_rich_chunks(source) == []
    assert readable_math_text(source) == source


def test_mixed_answer_keeps_fenced_and_inline_latex_source_untouched():
    source = r"""Формула: \[\frac{1}{2}\]

```python
path = r"C:\temp\"
regex = r"\[\frac{a}{b}\]"
price = "$5"
```
Исходник: `\(\sqrt{x}\)`.
"""
    text = "\n".join(chunk.text for chunk in render_markdown_chunks(source, 4000))
    assert "½" in text and 'regex = r"\\[\\frac{a}{b}\\]"' in text
    assert r"\(\sqrt{x}\)" in text and "C:\\temp\\" in text
    html = render_math_rich_chunks(source)[0].payload.html
    assert "<tg-math-block>" in html and r"\[\frac{a}{b}\]" in html


def test_cleaning_does_not_split_incomplete_or_already_aligned_expressions():
    for body in (
        "a +\nb",
        "a=\nb=1",
        "a=1+\nb=2",
        "\\frac{a\n+b}{c}",
        r"x=1\\y=2",
        r"\begin{aligned}a&=1\\b&=2\end{aligned}",
    ):
        prepared = prepare_math_markdown(f"$$\n{body}\n$$")
        assert prepared.markdown.count(r"\begin{aligned}") == body.count(r"\begin{aligned}")


def test_no_external_images_or_html_are_introduced_by_native_math():
    source = r"Формула $x^2$ и ![внешнее изображение](https://example.com/image.png)"
    assert render_math_rich_chunks(source) == []  # use legacy entities, not an uploaded image
    html = render_math_rich_chunks(r"$x^2$ <script>alert(1)</script>")[0].payload.html
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_readable_fallback_works_without_rich_message_support():
    source = r"Ответ: \(\frac{1}{2}\), \[x=\pi+2\pi n,\quad n\in\mathbb{Z}\]"
    text = readable_math_text(source)
    assert "½" in text and "π" in text and "ℤ" in text
    assert "\\frac" not in text and "$$" not in text
    chunks = render_markdown_chunks(BROKEN_SOLUTION, 80)
    assert len(chunks) > 1 and all(len(c.text.encode("utf-16-le")) // 2 <= 80 for c in chunks)
    assert not any("$$" in c.text or "\\" in c.text for c in chunks)


def test_long_native_answer_is_split_within_rich_api_byte_limit():
    source = ("Объяснение 🙂 " * 400 + r"$x^2+\pi$" + "\n\n") * 15
    chunks = render_math_rich_chunks(source)
    assert len(chunks) > 1
    assert all(len(chunk.payload.html.encode("utf-8")) <= 32768 for chunk in chunks)
    assert all(chunk.fallback_text for chunk in chunks)


def test_rendering_failure_is_safe_and_does_not_log_response_content(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise ValueError("PRIVATE-PROMPT-OR-KEY")

    monkeypatch.setattr("bot.math_formatting.telegramify_rich", broken)
    assert render_math_rich_chunks(r"$$\frac{1}{2}$$") == []
    assert "PRIVATE-PROMPT-OR-KEY" not in caplog.text
    monkeypatch.setattr("bot.math_formatting.convert", broken)
    assert readable_math_text(r"$x$ no change on failure") == r"$x$ no change on failure"


@pytest.mark.parametrize("mode", ["auto", "text", "bad"])
def test_math_render_mode_is_validated_at_startup(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:test-token")
    monkeypatch.setenv("AI_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MATH_RENDER_MODE", mode)
    if mode == "bad":
        with pytest.raises(ConfigError, match="MATH_RENDER_MODE"):
            Settings.from_env(tmp_path / "missing.env")
    else:
        assert Settings.from_env(tmp_path / "missing.env").math_render_mode == mode


def test_cleaned_solution_has_no_markdown_line_break_artefacts():
    prepared = prepare_math_markdown(BROKEN_SOLUTION)
    assert not re.search(r"(?m)^\\$|синуса:\\$|Отсюда:\\$", prepared.markdown)


async def test_reply_to_native_solution_keeps_latex_in_ai_context():
    original = Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 123456789, "is_bot": True, "first_name": "bot"},
            "rich_message": {
                "blocks": [
                    {"type": "mathematical_expression", "expression": r"x=\frac{\pi}{4}"}
                ]
            },
        }
    )
    current = Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": False, "first_name": "user"},
            "text": "Объясни этот шаг",
            "reply_to_message": original,
        }
    )
    bundle = await MediaExtractor(SimpleNamespace()).prepare(
        bot=None, messages=[current], user_text="Объясни этот шаг"
    )
    assert "Объясни этот шаг" in bundle.prompt and r"\frac" in bundle.prompt
    assert "mathematical_expression" in bundle.prompt and bundle.media == ()
