from __future__ import annotations

from dataclasses import dataclass

from .engine import InvestigationEngine
from .viewer import render_result_text


@dataclass(frozen=True)
class TuiConfig:
    default_mode: str = "investigation"
    top_k_per_pass: int = 5
    time_budget_sec: int = 120
    max_items: int = 8


def run_tui(
    engine: InvestigationEngine,
    *,
    default_mode: str = "investigation",
    top_k_per_pass: int = 5,
    time_budget_sec: int = 120,
) -> None:
    """Run a small terminal UI (requires optional dependency: textual)."""
    try:
        from textual.app import App, ComposeResult  # type: ignore[import-not-found]
        from textual.containers import Horizontal, Vertical, VerticalScroll  # type: ignore[import-not-found]
        from textual.widgets import Button, Footer, Header, Input, Static  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        from .bootstrap import auto_install_enabled, ensure_installed, requirements_path

        if auto_install_enabled():
            req = requirements_path("requirements-tui.txt")
            ensure_installed(
                requirements_files=[req] if req is not None else None,
                packages=("textual",),
                auto_install=True,
            )
            from textual.app import App, ComposeResult  # type: ignore[import-not-found]
            from textual.containers import Horizontal, Vertical, VerticalScroll  # type: ignore[import-not-found]
            from textual.widgets import Button, Footer, Header, Input, Static  # type: ignore[import-not-found]
        else:
            raise RuntimeError("TUI requires `textual`. Install: `pip install -r requirements-tui.txt`") from exc

    cfg = TuiConfig(default_mode=default_mode, top_k_per_pass=top_k_per_pass, time_budget_sec=time_budget_sec)

    class _SearchApp(App):
        BINDINGS = [
            ("ctrl+c", "quit", "Quit"),
            ("ctrl+d", "toggle_diagnostics", "Diagnostics"),
        ]

        CSS = """
        Screen { layout: vertical; }
        #controls { height: auto; padding: 1 2; }
        #status { height: auto; padding: 0 2; color: $text-muted; }
        #output_scroll { height: 1fr; padding: 1 2; }
        #output { width: 100%; }
        #mode_input { width: 24; }
        #query_input { width: 1fr; }
        """

        def __init__(self) -> None:
            super().__init__()
            self._engine = engine
            self._cfg = cfg
            self._show_diagnostics = False
            self._last_query: str | None = None

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header(show_clock=True)
            with Horizontal(id="controls"):
                yield Static("mode:")
                yield Input(value=self._cfg.default_mode, id="mode_input")
                yield Static("query:")
                yield Input(placeholder="Type a query and press Enter", id="query_input")
                yield Button("Search", id="search_btn", variant="primary")
            yield Static("", id="status")
            with VerticalScroll(id="output_scroll"):
                yield Static("", id="output")
            yield Footer()

        def on_mount(self) -> None:  # type: ignore[override]
            self.query_one("#status", Static).update(
                "Enter a web query and choose mode (investigation/report/fbi/collection/sniper/rumor/library). "
                "Ctrl+D toggles diagnostics."
            )

        def action_toggle_diagnostics(self) -> None:
            self._show_diagnostics = not self._show_diagnostics
            if self._last_query:
                self._rerender()

        def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
            if event.button.id == "search_btn":
                self._search()

        def on_input_submitted(self, event: Input.Submitted) -> None:  # type: ignore[override]
            if event.input.id == "query_input":
                self._search()

        def _search(self) -> None:
            query = self.query_one("#query_input", Input).value.strip()
            mode = self.query_one("#mode_input", Input).value.strip() or "investigation"
            if not query:
                return
            self._last_query = query
            self.query_one("#status", Static).update(f"Searching... mode={mode} top_k={self._cfg.top_k_per_pass}")
            try:
                result = self._engine.search(
                    query,
                    top_k_per_pass=self._cfg.top_k_per_pass,
                    time_budget_sec=self._cfg.time_budget_sec,
                    mode=mode,
                )
            except Exception as exc:  # pragma: no cover
                self.query_one("#output", Static).update(f"[ERROR] {type(exc).__name__}: {exc}")
                self.query_one("#status", Static).update("Search failed.")
                return

            session_id = result.diagnostics.get("knowledge_library_session_id")
            suffix = f" saved_session={session_id}" if session_id else ""
            self.query_one("#status", Static).update(f"Done. evidence={len(result.evidence)} sources={len(result.sources)}{suffix}")
            self._last_result = result
            self._rerender()

        def _rerender(self) -> None:
            result = getattr(self, "_last_result", None)
            if result is None or not self._last_query:
                return
            text = render_result_text(
                result,
                query=self._last_query,
                max_items=self._cfg.max_items,
                include_diagnostics=self._show_diagnostics,
            )
            self.query_one("#output", Static).update(text)

    _SearchApp().run()
