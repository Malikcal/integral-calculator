from http.server import BaseHTTPRequestHandler
import json, re
import sympy as sp
from sympy import (
    Symbol, integrate, latex, expand, Add, Mul, Pow,
    exp, sin, cos, tan, log, sqrt, pi, E, Rational,
    sec, csc, cot, asin, acos, atan, sinh, cosh, tanh,
    oo, simplify, factor, symbols, Number, Integer
)


# ── PREPROCESSING ─────────────────────────────────────────────────────────────

def preprocess(s):
    """Convert user-friendly notation to SymPy-parseable form."""
    s = s.strip()
    # e^x  or  e^(2x)  →  exp(...)
    s = re.sub(r'\be\^(\([^)]+\))', r'exp\1', s)
    s = re.sub(r'\be\^([a-zA-Z0-9_]+)', r'exp(\1)', s)
    # implicit multiplication: 2x → 2*x, 3sin → 3*sin
    s = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', s)
    s = re.sub(r'([a-zA-Z)])(\()', r'\1*\2', s)
    # ^ → **
    s = s.replace('^', '**')
    # ln → log
    s = re.sub(r'\bln\b', 'log', s)
    return s


SAFE = {
    'x': Symbol('x'), 'e': E, 'pi': pi, 'oo': oo,
    'sin': sin, 'cos': cos, 'tan': tan,
    'sec': sec, 'csc': csc, 'cot': cot,
    'asin': asin, 'acos': acos, 'atan': atan,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    'log': log, 'ln': log, 'exp': exp, 'sqrt': sqrt,
    'Rational': Rational, 'E': E,
}


def parse_expr(s):
    return sp.sympify(preprocess(s), locals=SAFE)


# ── STEP GENERATION ──────────────────────────────────────────────────────────

x = Symbol('x')


def _lx(expr):
    return latex(expr)


def steps_for_term(term):
    """Return list of LaTeX lines describing how to integrate a single term."""
    lines = []
    result = integrate(term, x)

    # ── constant ──
    if not term.has(x):
        c = _lx(term)
        lines.append(rf"\int {c} \, dx = {c} x")
        lines.append(rf"\text{{(constant rule: }} \int a \, dx = ax \text{{)}}")
        return lines, result

    # ── x^n  (power rule) ──
    if term == x:
        lines.append(r"\text{Power rule: } \int x^n \, dx = \dfrac{x^{n+1}}{n+1}")
        lines.append(rf"\int x \, dx = \dfrac{{x^2}}{{2}}")
        return lines, result

    if isinstance(term, Pow) and term.args[0] == x:
        n = term.args[1]
        if n == -1:
            lines.append(r"\int \dfrac{1}{x} \, dx = \ln|x|")
        else:
            lines.append(r"\text{Power rule: } \int x^n \, dx = \dfrac{x^{n+1}}{n+1}")
            lines.append(rf"\int {_lx(term)} \, dx = \dfrac{{x^{{{_lx(n+1)}}}}}{{{_lx(n+1)}}} = {_lx(result)}")
        return lines, result

    # ── c·x^n ──
    if isinstance(term, Mul):
        coeff, rest = term.as_coeff_Mul()
        # c * x^n
        if rest == x:
            lines.append(rf"\int {_lx(coeff)} x \, dx = {_lx(coeff)} \cdot \dfrac{{x^2}}{{2}} = {_lx(result)}")
            return lines, result
        if isinstance(rest, Pow) and rest.args[0] == x:
            n = rest.args[1]
            if n == -1:
                lines.append(rf"\int \dfrac{{{_lx(coeff)}}}{{x}} \, dx = {_lx(coeff)} \ln|x|")
            else:
                lines.append(rf"\int {_lx(coeff)} x^{{{_lx(n)}}} \, dx = {_lx(coeff)} \cdot \dfrac{{x^{{{_lx(n+1)}}}}}{{{_lx(n+1)}}} = {_lx(result)}")
            return lines, result
        # c * e^(ax)
        inner = None
        if isinstance(rest, exp):
            inner = rest.args[0]
        elif isinstance(rest, Pow) and rest.args[0] == E:
            inner = rest.args[1]
        if inner is not None:
            ic, ix = inner.as_coeff_Mul()
            if ix == x or ix == 1:
                a = ic
                if a == 1:
                    lines.append(rf"\int {_lx(coeff)} e^x \, dx = {_lx(coeff)} e^x")
                else:
                    lines.append(rf"\text{{Exponential rule: }} \int e^{{ax}} \, dx = \dfrac{{e^{{ax}}}}{{a}}")
                    lines.append(rf"\int {_lx(term)} \, dx = \dfrac{{{_lx(coeff)} e^{{{_lx(a)}x}}}}{{{_lx(a)}}} = {_lx(result)}")
                return lines, result
        # c * trig
        if isinstance(rest, (sin, cos, tan, sec, csc, cot)):
            rule = _trig_rule(rest)
            if rule:
                lines.append(rule)
            lines.append(rf"\int {_lx(term)} \, dx = {_lx(result)}")
            return lines, result

    # ── e^x, e^(ax) ──
    if isinstance(term, exp) or (isinstance(term, Pow) and term.args[0] == E):
        inner = term.args[0] if isinstance(term, exp) else term.args[1]
        ic, ix = inner.as_coeff_Mul()
        if ix == x or ix == 1:
            a = ic
            if a == 1:
                lines.append(r"\text{Exponential rule: } \int e^x \, dx = e^x")
            else:
                lines.append(r"\text{Exponential rule: } \int e^{ax} \, dx = \dfrac{e^{ax}}{a}")
                lines.append(rf"a = {_lx(a)}")
            lines.append(rf"\int {_lx(term)} \, dx = {_lx(result)}")
            return lines, result

    # ── trig ──
    if isinstance(term, (sin, cos, tan, sec, csc, cot)):
        rule = _trig_rule(term)
        if rule:
            lines.append(rule)
        lines.append(rf"\int {_lx(term)} \, dx = {_lx(result)}")
        return lines, result

    # ── log / ln ──
    if isinstance(term, log):
        arg = term.args[0]
        if arg == x:
            lines.append(r"\text{Integration by parts: } \int \ln x \, dx = x\ln x - x")
            lines.append(rf"\int \ln x \, dx = {_lx(result)}")
            return lines, result

    # ── fallback ──
    lines.append(rf"\int {_lx(term)} \, dx = {_lx(result)}")
    return lines, result


def _trig_rule(t):
    rules = {
        sin: r"\int \sin x \, dx = -\cos x",
        cos: r"\int \cos x \, dx = \sin x",
        tan: r"\int \tan x \, dx = -\ln|\cos x|",
        sec: r"\int \sec x \, dx = \ln|\sec x + \tan x|",
        csc: r"\int \csc x \, dx = -\ln|\csc x + \cot x|",
        cot: r"\int \cot x \, dx = \ln|\sin x|",
    }
    for k, v in rules.items():
        if isinstance(t, k):
            return v
    return None


def build_steps(expr, lower=None, upper=None):
    """Return list of LaTeX strings for the full worked solution."""
    lines = []
    expanded = expand(expr)
    result = integrate(expanded, x)

    # Opening line — write the integral
    if lower is not None and upper is not None:
        lines.append(
            rf"\int_{{{_lx(lower)}}}^{{{_lx(upper)}}} {_lx(expanded)} \, dx"
        )
    else:
        lines.append(rf"\int {_lx(expanded)} \, dx")

    # Sum rule applied?
    if isinstance(expanded, Add):
        terms = list(expanded.args)
        # Show the split
        split = " + ".join([rf"\int {_lx(t)} \, dx" for t in terms])
        lines.append(
            rf"= {split}"
        )
        lines.append(r"\text{(applying the sum/difference rule to each term)}")

        collected = []
        for term in terms:
            t_lines, t_result = steps_for_term(term)
            lines.extend(t_lines)
            collected.append(t_result)

        indef = result
    else:
        t_lines, _ = steps_for_term(expanded)
        lines.extend(t_lines)
        indef = result

    # Indefinite result
    if lower is None:
        lines.append(rf"= {_lx(indef)} + C")
        return lines

    # Definite — evaluate at bounds
    lines.append(r"\text{Now evaluate between the limits:}")
    lines.append(
        rf"\Bigl[{_lx(indef)}\Bigr]_{{{_lx(lower)}}}^{{{_lx(upper)}}}"
    )

    at_upper = simplify(indef.subs(x, upper))
    at_lower = simplify(indef.subs(x, lower))
    final    = simplify(at_upper - at_lower)

    lines.append(
        rf"= \left({_lx(at_upper)}\right) - \left({_lx(at_lower)}\right)"
    )
    lines.append(rf"= {_lx(final)}")
    return lines


# ── VERCEL HANDLER ────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length).decode())

            fn    = body.get("function", "").strip()
            itype = body.get("type", "indefinite")   # "indefinite" | "definite"
            lo    = body.get("lower", "").strip()
            hi    = body.get("upper", "").strip()

            if not fn:
                self._json({"success": False, "error": "No function provided."})
                return

            expr  = parse_expr(fn)
            lower = parse_expr(lo) if itype == "definite" and lo else None
            upper = parse_expr(hi) if itype == "definite" and hi else None

            steps = build_steps(expr, lower, upper)

            self._json({"success": True, "steps": steps})

        except Exception as e:
            self._json({"success": False, "error": str(e)})

    def _json(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._cors()
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass   # suppress default Vercel noise
