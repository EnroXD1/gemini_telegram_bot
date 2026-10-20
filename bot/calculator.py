from __future__ import annotations

import ast
from decimal import Decimal, DecimalException, localcontext


class CalculationError(ValueError):
    pass


def calculate(expression: str) -> str:
    """Evaluate bounded arithmetic only; never execute Python or call user functions."""
    source = expression.strip().translate(
        str.maketrans({"×": "*", "÷": "/", "−": "-", ",": "."})
    )
    source = source.replace("^", "**")
    if not source or len(source) > 256:
        raise CalculationError("Укажите выражение длиной до 256 символов.")
    try:
        tree = ast.parse(source, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise CalculationError("Слишком сложное выражение.")
        with localcontext() as context:
            context.prec, context.Emax, context.Emin = 28, 100, -100
            result = _evaluate(tree.body, source)
        text = format(result, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    except (SyntaxError, DecimalException, OverflowError, RecursionError, ValueError) as exc:
        raise CalculationError(
            "Проверьте выражение: допустимы числа, скобки и + − * / // % **."
        ) from exc


def _bounded(value: Decimal) -> Decimal:
    if not value.is_finite() or (value and abs(value.adjusted()) > 100):
        raise CalculationError("Число вне допустимого диапазона.")
    return value


def _evaluate(node: ast.AST, source: str) -> Decimal:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        literal = ast.get_source_segment(source, node) or ""
        # Reject hex, imaginary numbers and non-decimal Python syntax.
        return _bounded(Decimal(literal.replace("_", "")))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, source)
        return value if isinstance(node.op, ast.UAdd) else -value
    if not isinstance(node, ast.BinOp):
        raise CalculationError("Допустима только арифметика.")
    left, right = _evaluate(node.left, source), _evaluate(node.right, source)
    if isinstance(node.op, ast.Add):
        value = left + right
    elif isinstance(node.op, ast.Sub):
        value = left - right
    elif isinstance(node.op, ast.Mult):
        value = left * right
    elif isinstance(node.op, ast.Div):
        value = left / right
    elif isinstance(node.op, ast.FloorDiv):
        value = (left / right).to_integral_value(rounding="ROUND_FLOOR")
    elif isinstance(node.op, ast.Mod):
        value = left - (left / right).to_integral_value(rounding="ROUND_FLOOR") * right
    elif isinstance(node.op, ast.Pow):
        if right != right.to_integral_value() or abs(right) > 100:
            raise CalculationError("Степень должна быть целой от -100 до 100.")
        value = left ** int(right)
    else:
        raise CalculationError("Недопустимая операция.")
    return _bounded(value)
