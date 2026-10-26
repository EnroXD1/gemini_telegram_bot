import re

import pytest

from bot.gemini import _merge_continuation
from bot.math_formatting import prepare_math_markdown, render_math_rich_chunks

SCREENSHOT_TAIL = r"""Подставим (1) в (2):

$$S_{k+1}=\frac{1}{30}k(k+1)(2k+1)(3k^2+3k-1)+(k+1)^4.\tag{3}$$

Нужно показать, что

$$6k^4+39k^3+91k^2+89k+30=(k+2)(2k+3)(3k^2+6k+2).\tag{6}$$

Проверим факторизацию:
$
\begin{aligned}

(k+2)(2k+3)(3k^2+6k+2)\\

&=(k+2)\bigl[6k^3+12k^2+4k+9k^2+18k+6\bigr]\\
&=(k+2)\bigl[6k^3+21k^2+20k+3\bigr]\\
&=6k^4+21k^3+20k^2+3k+12k^3+42k^2+40k+6

⚠️ Ответ всё ещё достиг лимита выбранной модели после автоматических продолжений.
"""


def outside_math(html):
    return re.sub(r"<tg-math(?:-block)?>.*?</tg-math(?:-block)?>", "", html, flags=re.S)


def test_exact_screenshot_open_aligned_is_one_native_block_without_raw_latex_leaks():
    prepared = prepare_math_markdown(SCREENSHOT_TAIL, truncated=True)
    html = "".join(chunk.payload.html for chunk in render_math_rich_chunks(prepared.markdown))
    assert html.count("<tg-math-block>") == 3
    assert r"\begin{aligned}" in html and r"\end{aligned}" in html
    assert "\\begin" not in outside_math(html)
    assert "&amp;=" not in outside_math(html)
    assert not re.search(r"(?m)^\$$", prepared.markdown)
    assert "⚠️" in html
    assert "6k^4+21k^3+20k^2+3k+12k^3+42k^2+40k+6" in html
    assert "3k^2+6k+2" in html  # typesetting cannot correct the model's algebra


@pytest.mark.parametrize("delimiter", ["$", "$$", r"\[", ""])
def test_open_aligned_across_blank_lines_is_repaired_with_all_rows(delimiter):
    source = delimiter + "\n" + r"\begin{aligned}" + "\nx&=1\\\\\n\n&=2"
    prepared = prepare_math_markdown(source, truncated=True)
    html = render_math_rich_chunks(prepared.markdown)[0].payload.html
    assert r"\begin{aligned}" in html and r"\end{aligned}" in html
    assert "x&amp;=1" in html and "&amp;=2" in html
    assert r"\begin" not in outside_math(html)


def test_incomplete_last_fraction_does_not_destroy_completed_rows_or_invent_denominator():
    source = "$$\n" + r"\begin{aligned}x&=1\\y&=\frac{1}{" + "\n\n⚠️ Лимит."
    prepared = prepare_math_markdown(source, truncated=True)
    html = "".join(chunk.payload.html for chunk in render_math_rich_chunks(prepared.markdown))
    assert "x&amp;=1" in html and r"\end{aligned}" in html
    assert "⚠️ Лимит." in html
    assert r"\begin" not in outside_math(html)
    assert r"\frac{1}{2}" not in html


def test_multiline_inline_aligned_with_closed_delimiters_is_promoted_to_display():
    source = "$\n" + r"\begin{aligned}x&=1\\y&=2\end{aligned}" + "\n$"
    html = render_math_rich_chunks(source)[0].payload.html
    assert "<tg-math-block>" in html
    assert r"\begin" not in outside_math(html)


@pytest.mark.parametrize("tag", [r"\tag{2}", r"\tag 2"])
def test_equation_tag_is_plain_number_outside_the_formula(tag):
    html = render_math_rich_chunks("$$x=1" + tag + "$$")[0].payload.html
    assert r"\tag" not in html and "(2)" in outside_math(html)
    assert re.search(r"<tg-math-block>\s*x=1\s*</tg-math-block>", html)


@pytest.mark.parametrize("first,second,expected", [
    (r"Ответ $x=\fra", "c{1}{2}$", r"Ответ $x=\frac{1}{2}$"),
    ("$$\n" + r"\begin{aligned}x&=1\\", r"y&=2\end{aligned}$$",
     "$$\n" + r"\begin{aligned}x&=1\\y&=2\end{aligned}$$"),
    (r"Ответ $x=\pi ", "n$", r"Ответ $x=\pi n$"),
    ("Первая часть.", "Вторая часть.", "Первая часть.\n\nВторая часть."),
    ("Цена $5.", "Цена $10.", "Цена $5.\n\nЦена $10."),
    ("Код `$x=1`.", "Далее.", "Код `$x=1`.\n\nДалее."),
])
def test_continuations_join_without_splitting_open_math_tokens(first, second, expected):
    assert _merge_continuation(first, second) == expected


def test_literal_latex_source_inside_code_is_not_repaired():
    source = "```latex\n" + SCREENSHOT_TAIL + "\n```"
    assert prepare_math_markdown(source, truncated=True).markdown == source


@pytest.mark.parametrize("source", [
    r"Ответ $x=1$", r"$$\begin{aligned}x=1\end{aligned}$$", "Цена $5 и $10.",
    "```python\ns = '$x='\n```", r"Путь C:\temp\file.py", r"Экранировано \$x=1",
])
def test_closed_math_code_paths_and_prices_keep_normal_paragraph_separator(source):
    assert _merge_continuation(source, "Далее.") == source + "\n\nДалее."


@pytest.mark.parametrize("prose", ["Далее пояснение.", "Therefore, the proof is incomplete."])
def test_recovering_open_environment_does_not_swallow_following_prose(prose):
    source = "$$\n" + r"\begin{aligned}x&=1\\y&=2" + "\n\n" + prose
    prepared = prepare_math_markdown(source, truncated=True)
    html = render_math_rich_chunks(prepared.markdown)[0].payload.html
    assert prose in outside_math(html)


def test_multiline_environment_never_absorbs_a_protected_code_region():
    source = "$\n" + r"\begin{aligned}" + "\n```python\nprint(1)\n```\n"
    source += r"\end{aligned}" + "\n$"
    prepared = prepare_math_markdown(source)
    assert "```python\nprint(1)\n```" in prepared.markdown
    assert not prepared.has_math


@pytest.mark.parametrize("tail", ["y&=2", r"y&=\frac{1}{"])
def test_closed_dollars_do_not_hide_an_unclosed_environment_in_a_partial_answer(tail):
    source = "$$\n" + r"\begin{aligned}x&=1\\" + tail + "\n$$\n\n⚠️ Лимит."
    prepared = prepare_math_markdown(source, truncated=True)
    html = "".join(chunk.payload.html for chunk in render_math_rich_chunks(prepared.markdown))
    assert r"\end{aligned}" in html and "x&amp;=1" in html
    assert r"\begin" not in outside_math(html)
    assert "⚠️ Лимит." in html
