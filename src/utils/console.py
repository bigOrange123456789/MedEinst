"""Console-only language selection, shared by all inference worker threads.

Set the language once before starting a run. Keep model prompts, persisted data,
and external error/response payloads in their original language.
"""

import argparse
import sys


_language = "en"


def set_console_language(language: str) -> None:
    """Select the process-wide console language (English by default)."""
    if language not in ("en", "zh"):
        raise ValueError("console language must be 'en' or 'zh'")
    global _language
    _language = language


def tr(en: str, zh: str) -> str:
    """Choose display text without translating interpolated application data."""
    return zh if _language == "zh" else en


def configure_console_encoding() -> None:
    """Use UTF-8 for Windows CMD output, including redirected logs and emoji."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")


class ConsoleArgumentParser(argparse.ArgumentParser):
    """Localize argparse headings while retaining its original error details."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, add_help=False, **kwargs)
        self._positionals.title = tr("positional arguments", "位置参数")
        self._optionals.title = tr("options", "可选参数")
        self.add_argument("-h", "--help", action="help",
                          help=tr("show this help message and exit", "显示帮助信息并退出"))

    def format_usage(self):
        return super().format_usage().replace("usage:", tr("usage:", "用法:"), 1)

    def format_help(self):
        return super().format_help().replace("usage:", tr("usage:", "用法:"), 1)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: {tr('error', '参数错误（原始信息）')}: {message}\n")
