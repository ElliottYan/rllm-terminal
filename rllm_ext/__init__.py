"""
Extension package for custom experiments built on top of rllm.

Design goal:
- Keep experimental / project-specific logic isolated from the core `rllm` package.
- Avoid modifying existing `rllm` code paths; only compose/extend via inheritance.
"""

