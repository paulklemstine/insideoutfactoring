# Code Style and Conventions

## Language
- Python 3.10+
- Type hints on all function signatures
- Google-style docstrings
- snake_case for functions/variables, PascalCase for classes

## Math Conventions
- Variable names follow paper notation: N for semiprime, p/q for factors, a/b/c for triple legs
- Use `sympy` or `gmpy2` for large integer arithmetic when needed
- Prefer exact integer arithmetic over floating point

## Testing
- pytest-based
- Test with small known semiprimes first, then scale up
- Always verify factors multiply back to N