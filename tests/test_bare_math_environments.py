import re

import pytest

from bot.math_formatting import (
    prepare_math_markdown,
    readable_math_text,
    render_math_rich_chunks,
)

EUCLID_SOLUTION = r"""Задание 5. Найти обратный элемент с помощью расширенного алгоритма Евклида

Дано: $a=653, m=309$. Требуется найти $a^{-1}\pmod m$.

1. Прямой ход алгоритма Евклида

\begin{aligned}
653 &= 2 \cdot 309 + 35 \\
309 &= 8 \cdot 35 + 29 \\
35 &= 1 \cdot 29 + 6 \\
29 &= 4 \cdot 6 + 5 \\
6 &= 1 \cdot 5 + 1 \\
5 &= 5 \cdot 1 + 0
\end{aligned}

Последний ненулевой остаток равен 1, обратный элемент существует.

2. Обратный ход

\begin{aligned}
1 &= 6 - 1 \cdot 5 \\
1 &= 6 - 1 \cdot (29 - 4 \cdot 6) = 5 \cdot 6 - 1 \cdot 29 \\
1 &= 5 \cdot (35 - 1 \cdot 29) - 1 \cdot 29 = 5 \cdot 35 - 6 \cdot 29 \\
1 &= 5 \cdot 35 - 6 \cdot (309 - 8 \cdot 35) = 53 \cdot 35 - 6 \cdot 309 \\
1 &= 53 \cdot (653 - 2 \cdot 309) - 6 \cdot 309 = 53 \cdot 653 - 112 \cdot 309
\end{aligned}

3. Результат

$$53 \cdot 653 - 112 \cdot 309 = 1$$

По модулю 309:

$$53 \cdot 653 \equiv 1 \pmod{309}$$

Ответ:

$$a^{-1}=53 \pmod{309}$$
"""


def test_completed_euclid_solution_without_dollars_renders_all_five_blocks():
    prepared = prepare_math_markdown(EUCLID_SOLUTION, truncated=False)
    chunks = render_math_rich_chunks(prepared.markdown)
    html = "".join(chunk.payload.html for chunk in chunks)
    assert html.count("<tg-math-block>") == 5
    assert html.count(r"\begin{aligned}") == 2
    outside = re.sub(r"<tg-math(?:-block)?>.*?</tg-math(?:-block)?>", "", html, flags=re.S)
    assert r"\begin{aligned}" not in outside and r"\end{aligned}" not in outside
    assert "&amp;=" not in outside and r"\cdot" not in outside
    assert "53 \\cdot 653 - 112 \\cdot 309" in html
    assert "Последний ненулевой остаток" in outside and "Ответ:" in outside


@pytest.mark.parametrize("row_break", ["\\", "\\\\", ""])
def test_bare_aligned_repairs_only_unambiguous_physical_equation_row_breaks(row_break):
    source = "\\begin{aligned}\n653 &= 2 \\cdot 309 + 35 " + row_break
    source += "\n309 &= 8 \\cdot 35 + 29\n\\end{aligned}"
    chunks = render_math_rich_chunks(source)
    assert chunks and "<tg-math-block>" in chunks[0].payload.html
    assert r"35 \\" in chunks[0].payload.html
    assert "309 &amp;= 8" in chunks[0].payload.html


def test_nested_bare_math_environments_are_wrapped_once_as_one_block():
    source = r"\begin{aligned}A&=\begin{pmatrix}1&0\\0&1\end{pmatrix}\end{aligned}"
    chunks = render_math_rich_chunks(source)
    assert len(chunks) == 1 and chunks[0].payload.html.count("<tg-math-block>") == 1
    assert r"\end{pmatrix}\end{aligned}" in chunks[0].payload.html


@pytest.mark.parametrize("source", [
    r"`\begin{aligned}x&=1\end{aligned}`",
    "```latex\n" + EUCLID_SOLUTION + "\n```",
    "```tex\n" + EUCLID_SOLUTION + "\n```",
    "    " + r"\begin{aligned}x&=1\end{aligned}",
    r"\begin{aligned}x&=1\end{pmatrix}",
    r"\begin{aligned}x&=\frac{1}{\end{aligned}",
    r"\begin{document}x=1\end{document}",
])
def test_bare_environment_detection_keeps_code_unknown_or_broken_layout_untouched(source):
    prepared = prepare_math_markdown(source)
    assert prepared.markdown == source and not prepared.has_math


def test_math_does_not_absorb_a_code_region_between_environment_markers():
    source = "\\begin{aligned}\nx&=1\n```python\nprint(1)\n```\n\\end{aligned}"
    prepared = prepare_math_markdown(source)
    assert "```python\nprint(1)\n```" in prepared.markdown
    assert not prepared.has_math


def test_existing_math_wrapper_does_not_create_a_second_environment_wrapper():
    source = r"$$\begin{aligned}x&=1\\y&=2\end{aligned}$$"
    html = render_math_rich_chunks(source)[0].payload.html
    assert html.count("<tg-math-block>") == 1
    assert html.count(r"\begin{aligned}") == 1


def test_bare_aligned_plain_fallback_has_no_layout_commands():
    text = readable_math_text(r"\begin{aligned}x&=1\\y&=2\end{aligned}")
    assert "x" in text and "y" in text and "1" in text and "2" in text
    assert r"\begin" not in text and r"\end" not in text and "$$" not in text


def test_multiline_fraction_inside_bare_layout_does_not_gain_guessed_row_breaks():
    source = "\\begin{aligned}\nx&=\\frac{a\n+b}{c}\ny&=2\n\\end{aligned}"
    prepared = prepare_math_markdown(source)
    assert prepared.has_math and r"\\" not in prepared.markdown
    assert "\\frac{a\n+b}{c}" in prepared.markdown
