#!/usr/bin/env python3
"""Compatibility wrapper for the router-trace CLI."""
from lexa_hypermoe.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["trace", *__import__("sys").argv[1:]]))
