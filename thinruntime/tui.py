"""ThinTensor Terminal User Interface.

Uses the Textual library for a rich interactive TUI workspace.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        RichLog,
        Select,
        Static,
        TextArea,
        ContentSwitcher,
    )
    from textual.binding import Binding
    from textual import on

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

PROFILE_DESCRIPTIONS = {
    "auto": "Guarded quality on validated SmolLM3-3B; BF16 otherwise",
    "bf16": "Universal BF16 weights with exact-value BF16 KV",
    "quality": "Validated SmolLM3-3B body FP8 profile",
    "quality-guarded": "Validated body FP8 plus BF16-verified head",
    "experimental": "BF16 workbench for explicit experimental overrides",
}


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.2f} {unit}"


def _detect_gpu() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            return f"{name} ({_format_bytes(vram)})"
        return "No CUDA GPU detected"
    except Exception:
        return "PyTorch not available"


def _archive_model_for_tui(archive_path: str) -> dict[str, Any]:
    from .archive import ThinArchive

    archive = ThinArchive(archive_path)
    try:
        return dict(archive.manifest.get("model", {}))
    finally:
        archive.close()


def _get_path_completions(query: str) -> list[tuple[str, str]]:
    """Get dynamic list of matching cache models and local filesystem paths."""
    completions = []
    
    # 1. Search cached archives
    try:
        from .model_cache import list_cached_archives
        archives = list_cached_archives()
        for a in archives:
            if not query or query.lower() in a["name"].lower() or query.lower() in a["path"].lower():
                completions.append((f"[Cache] {a['name']}", a["path"]))
    except Exception:
        pass
        
    # 2. Search local filesystem directories and .thin archives
    if query and (query.startswith(".") or query.startswith("/") or "/" in query):
        try:
            path = Path(query)
            if query.endswith("/"):
                parent = path
                prefix = ""
            else:
                parent = path.parent
                prefix = path.name

            if parent.exists() and parent.is_dir():
                for item in sorted(parent.iterdir()):
                    if item.name.startswith(prefix) and not item.name.startswith("."):
                        if item.is_dir():
                            completions.append((f"[Dir] {item.name}/", str(item) + "/"))
                        elif item.name.endswith(".thin"):
                            completions.append((f"[File] {item.name}", str(item)))
        except Exception:
            pass
            
    return completions[:8]


# ---------------------------------------------------------------------------
# Custom Panes (Workspace Dashboard Views)
# ---------------------------------------------------------------------------

if HAS_TEXTUAL:

    class PathListItem(ListItem):
        """Custom ListItem containing the associated model file path."""

        def __init__(self, label_text: str, path: str, **kwargs):
            super().__init__(Label(label_text), **kwargs)
            self.model_path = path

    class HomePane(Vertical):
        """Dashboard home pane."""

        def compose(self) -> ComposeResult:
            yield Label("ThinTensor Dashboard", classes="pane-title")
            with Container(classes="card"):
                yield Label("Welcome to the unified ThinTensor Workspace TUI!", id="welcome-msg")
                yield Label(f"Detected GPU: {_detect_gpu()}", id="gpu-lbl")
            yield Label("Available Converted Archives:", classes="section-title")
            yield DataTable(id="home-archives-table")

        def on_mount(self) -> None:
            table = self.query_one("#home-archives-table", DataTable)
            table.add_columns("Name", "Size", "Path")
            table.cursor_type = "row"

            from .model_cache import list_cached_archives
            archives = list_cached_archives()
            for a in archives:
                table.add_row(a["name"], _format_bytes(a["size"]), a["path"])
            if not archives:
                table.add_row("(no cached archives found)", "", "")


    class RunPane(Vertical):
        """Integrated run and chat panel."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._running = False
            self._chat_mode = False
            self._chat_history = []
            self._messages = []
            self._weights = None
            self._kv_cache = None
            self._runtime = None
            self._token_index = 0
            self._loaded_profile = None
            self._reset_before_generation = True

        def compose(self) -> ComposeResult:
            yield Label("Model Run & Chat Workspace", classes="pane-title")
            with Horizontal(classes="split-layout"):
                # Configuration Column (Left)
                with ScrollableContainer(classes="form-column"):
                    yield Label("Search / Path to Model:")
                    yield Input(
                        placeholder="Search cache or type path (e.g. ./)...",
                        id="run-model-input",
                    )
                    yield ListView(id="run-model-search-list")

                    yield Label("Select Profile Preset:")
                    yield Select(
                        [(f"{k} — {v}", k) for k, v in PROFILE_DESCRIPTIONS.items()],
                        value="auto",
                        id="run-profile-select",
                    )

                    yield Label("Context Size Limit:")
                    yield Input(value="512", id="run-context-input")

                    yield Label("Max Gen Tokens:")
                    yield Input(value="200", id="run-tokens-input")

                    yield Label("Tokenizer (optional model ID or directory):")
                    yield Input(
                        placeholder="Auto-detect beside archive",
                        id="run-tokenizer-input",
                    )

                    yield Label("Temperature (greedy decode requires 0):")
                    yield Input(value="0", id="run-temp-input")

                    yield Label("Prompt / Message input:")
                    yield TextArea("Hello, what are you?", id="run-prompt-area")

                    with Horizontal():
                        yield Button("Run Prompt", variant="primary", id="btn-run-exec")
                        yield Button("Start Chat", variant="success", id="btn-run-chat")
                        yield Button("Stop", variant="error", id="btn-run-stop")

                    with Horizontal():
                        yield Button("Clear Context", variant="warning", id="btn-run-reset")
                        yield Button("Save Log", variant="default", id="btn-run-save")

                # Live Output Column (Right)
                with Vertical(classes="log-column"):
                    yield Label("Model Generation Stream:")
                    yield RichLog(id="run-output-log", highlight=True, markup=True)
                    yield Label("Stats: Idle", id="run-stats-lbl", classes="card")

        def on_mount(self) -> None:
            self.query_one("#run-model-search-list", ListView).styles.display = "none"

        @on(Input.Changed, "#run-model-input")
        def on_model_input_changed(self, event: Input.Changed) -> None:
            query = event.value.strip()
            completions = _get_path_completions(query)
            list_view = self.query_one("#run-model-search-list", ListView)
            list_view.clear()
            
            for display_name, path in completions:
                list_view.append(PathListItem(display_name, path))
                
            list_view.styles.display = "block" if completions else "none"

        @on(ListView.Selected, "#run-model-search-list")
        def on_list_selected(self, event: ListView.Selected) -> None:
            selected_path = getattr(event.item, "model_path", None)
            if selected_path:
                input_widget = self.query_one("#run-model-input", Input)
                input_widget.value = selected_path
                
                if selected_path.endswith("/"):
                    self.on_model_input_changed(Input.Changed(input_widget, selected_path))
                else:
                    self.query_one("#run-model-search-list", ListView).styles.display = "none"

        def on_button_pressed(self, event: Button.Pressed) -> None:
            button_id = event.button.id
            if button_id == "btn-run-exec":
                self.start_generation(chat_mode=False)
            elif button_id == "btn-run-chat":
                self.start_generation(chat_mode=True)
            elif button_id == "btn-run-stop":
                self.stop_generation()
            elif button_id == "btn-run-reset":
                self.reset_chat()
            elif button_id == "btn-run-save":
                self.save_chat()

        def start_generation(self, chat_mode: bool = False) -> None:
            if self._running:
                return

            model_path = self.query_one("#run-model-input", Input).value.strip()
            if not model_path:
                self.query_one("#run-output-log", RichLog).write("[red]Please search and select a model archive first.[/red]")
                return

            self._running = True
            self._reset_before_generation = (
                not chat_mode or (chat_mode and not self._chat_mode)
            )
            self._chat_mode = chat_mode
            self.run_worker(self._exec_generation(str(model_path)))

        def stop_generation(self) -> None:
            self._running = False
            self.query_one("#run-output-log", RichLog).write("\n[yellow]Generation stopped by user.[/yellow]")

        def reset_chat(self) -> None:
            self.stop_generation()
            self._chat_history.clear()
            self._messages.clear()
            self._token_index = 0
            self._chat_mode = False
            self._reset_before_generation = True
            if self._kv_cache:
                self._kv_cache.reset()
            self.query_one("#run-output-log", RichLog).clear()
            self.query_one("#run-output-log", RichLog).write("[cyan]Chat history and KV cache reset.[/cyan]")

        def save_chat(self) -> None:
            log = self.query_one("#run-output-log", RichLog)
            out_file = Path("transcript_tui.txt")
            try:
                out_file.write_text("\n".join(self._chat_history) + "\n", encoding="utf-8")
                log.write(f"\n[green]✓ Chat transcript saved to {out_file.absolute()}[/green]")
            except Exception as e:
                log.write(f"\n[red]✗ Failed to save transcript: {e}[/red]")

        async def _exec_generation(self, model_path: str) -> None:
            log = self.query_one("#run-output-log", RichLog)
            stats_lbl = self.query_one("#run-stats-lbl", Label)

            profile_name = self.query_one("#run-profile-select", Select).value
            context = int(self.query_one("#run-context-input", Input).value or "512")
            max_tokens = int(self.query_one("#run-tokens-input", Input).value or "200")
            temp = float(self.query_one("#run-temp-input", Input).value or "0")
            tokenizer_source = (
                self.query_one("#run-tokenizer-input", Input).value.strip()
                or None
            )
            prompt = self.query_one("#run-prompt-area", TextArea).text

            if not self._chat_mode:
                log.clear()
                log.write(f"[bold cyan]Running Model:[/bold cyan] {Path(model_path).name}")
                log.write(f"[bold cyan]Profile:[/bold cyan] {profile_name}")
                log.write(f"[bold cyan]Prompt:[/bold cyan] {prompt}\n")
            else:
                log.write(f"\n[bold green]User:[/bold green] {prompt}")
                self._chat_history.append(f"User: {prompt}")
                self._messages.append({"role": "user", "content": prompt})

            try:
                import torch
                from .gpu_runtime import ThinGpuWeights, ThinGpuQwenRuntime, PagedKVCache
                from .profile_presets import get_profile, profile_to_runtime_kwargs
                from .cli import (
                    _decode_tokens,
                    _is_eos,
                    _tokenize_chat_messages,
                    _tokenize_prompt,
                )

                if temp != 0:
                    raise ValueError(
                        "sampling is not implemented; set temperature to 0"
                    )

                archive_model = _archive_model_for_tui(model_path)
                profile = get_profile(str(profile_name), model=archive_model)
                runtime_kwargs = profile_to_runtime_kwargs(profile)

                # Lazy load model weights
                if (
                    not self._weights
                    or str(self._weights.archive_path) != model_path
                    or self._loaded_profile != profile["name"]
                ):
                    if self._weights is not None:
                        self._weights.close()
                    log.write("[yellow]Loading model weights into GPU...[/yellow]")
                    self._weights = ThinGpuWeights(model_path, device="cuda", dtype=torch.bfloat16)

                    model = self._weights.manifest.get("model", {})
                    layers = int(model.get("layers", 28))
                    heads = int(model.get("heads", 8))
                    kv_heads = int(model.get("kv_heads", heads))
                    head_dim = int(
                        model.get("head_dim")
                        or (int(model.get("hidden_size", 2048)) // heads)
                    )

                    self._kv_cache = PagedKVCache(
                        layers=layers,
                        kv_heads=kv_heads,
                        head_dim=head_dim,
                        device=torch.device("cuda"),
                        dtype=torch.bfloat16,
                    )
                    self._runtime = ThinGpuQwenRuntime(self._weights, kv_cache=self._kv_cache, **runtime_kwargs)
                    self._token_index = 0
                    self._loaded_profile = profile["name"]
                    log.write("[green]✓ Model loaded successfully![/green]\n")

                if self._reset_before_generation:
                    self._kv_cache.reset(reuse_pages=True)
                    self._token_index = 0
                    self._reset_before_generation = False

                # Tokenize
                if self._chat_mode:
                    token_ids = _tokenize_chat_messages(
                        self._messages,
                        model_path,
                        self._weights.manifest,
                        tokenizer_source=tokenizer_source,
                    )
                    self._kv_cache.reset(reuse_pages=True)
                    self._token_index = 0
                else:
                    token_ids = _tokenize_prompt(
                        prompt,
                        model_path,
                        self._weights.manifest,
                        tokenizer_source=tokenizer_source,
                    )
                if self._token_index + len(token_ids) + max_tokens > context:
                    if self._chat_mode:
                        self._messages.pop()
                        self._chat_history.pop()
                    raise ValueError(
                        "prompt and generation budget exceed the configured "
                        "context; reset or increase the context limit"
                    )

                # Feed prompt tokens
                for tid in token_ids:
                    hidden = self._runtime.forward_token(tid, token_index=self._token_index)
                    self._token_index += 1

                # Generate loop
                log.write("[bold cyan]Assistant:[/bold cyan]" if not self._chat_mode else "")
                response_ids: list[int] = []
                response = ""
                t_gen = time.perf_counter()
                generated = 0

                for step in range(max_tokens):
                    if not self._running:
                        break
                    await asyncio.sleep(0)

                    next_id_tensor = self._runtime.next_token_tensor(hidden)
                    torch.cuda.synchronize()
                    next_id = int(next_id_tensor.item())
                    generated += 1
                    response_ids.append(next_id)

                    next_response = _decode_tokens(
                        response_ids,
                        model_path,
                        self._weights.manifest,
                        tokenizer_source=tokenizer_source,
                    )
                    delta = (
                        next_response[len(response):]
                        if next_response.startswith(response)
                        else next_response
                    )
                    if delta:
                        log.write(delta)
                    response = next_response

                    if _is_eos(
                        next_id,
                        self._weights.manifest,
                        archive_path=model_path,
                        tokenizer_source=tokenizer_source,
                    ) or step + 1 == max_tokens:
                        break

                    hidden = self._runtime.forward_token(next_id, token_index=self._token_index)
                    self._token_index += 1

                    if generated % 10 == 0:
                        elapsed = time.perf_counter() - t_gen
                        tok_s = generated / elapsed if elapsed > 0 else 0
                        stats_lbl.update(
                            f"Speed: {tok_s:.1f} tok/s | Tokens: {generated} | "
                            f"Resident: {_format_bytes(self._weights.resident_weight_bytes)}"
                        )

                elapsed = time.perf_counter() - t_gen
                tok_s = generated / elapsed if elapsed > 0 else 0
                ms_tok = (elapsed / generated * 1000) if generated > 0 else 0

                self._chat_history.append(f"Assistant: {response}")
                if self._chat_mode:
                    self._messages.append(
                        {"role": "assistant", "content": response}
                    )
                stats_lbl.update(
                    f"✓ Done | Speed: {tok_s:.1f} tok/s ({ms_tok:.1f} ms/tok) | "
                    f"Peak GPU: {_format_bytes(torch.cuda.max_memory_allocated())}"
                )

            except Exception as e:
                log.write(f"\n[red]✗ Error during execution: {e}[/red]")
                stats_lbl.update(f"✗ Error: {e}")

            self._running = False


    class ConvertPane(Vertical):
        """Model conversion view."""

        def compose(self) -> ComposeResult:
            yield Label("HF to .thin Converter", classes="pane-title")
            with Horizontal(classes="split-layout"):
                with Vertical(classes="form-column"):
                    yield Label("Local HF Model Directory:")
                    yield Input(placeholder="./SmolLM3-3B", id="conv-hf-dir")

                    yield Label("Destination Archive Name:")
                    yield Input(placeholder="SmolLM3-3B.thin", id="conv-out-path")

                    yield Label("Override Architecture (Optional):")
                    yield Input(placeholder="qwen2", id="conv-arch")

                    yield Button("Run Conversion", variant="primary", id="btn-conv-run")

                with Vertical(classes="log-column"):
                    yield Label("Conversion Progress log:")
                    yield RichLog(id="conv-output-log", highlight=True, markup=True)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-conv-run":
                hf_dir = self.query_one("#conv-hf-dir", Input).value.strip()
                out_path = self.query_one("#conv-out-path", Input).value.strip()
                arch = self.query_one("#conv-arch", Input).value.strip() or None

                if not hf_dir or not out_path:
                    return

                log = self.query_one("#conv-output-log", RichLog)
                log.clear()
                log.write(f"Initiating conversion: {hf_dir} → {out_path}...")
                self.run_worker(self._do_convert(hf_dir, out_path, arch))

        async def _do_convert(self, hf_dir: str, out_path: str, arch: Optional[str]) -> None:
            log = self.query_one("#conv-output-log", RichLog)
            try:
                from .command_runner import convert_hf_model
                success = await asyncio.to_thread(
                    convert_hf_model,
                    hf_dir,
                    out_path,
                    arch=arch,
                )
                if success:
                    size = Path(out_path).stat().st_size
                    log.write(f"[green]✓ Conversion complete![/green]")
                    log.write(f"Archive: {out_path} ({_format_bytes(size)})")
                else:
                    log.write("[red]✗ Conversion task failed.[/red]")
            except Exception as e:
                log.write(f"[red]✗ Error: {e}[/red]")


    class PullPane(Vertical):
        """HuggingFace model pull panel."""

        def compose(self) -> ComposeResult:
            yield Label("Pull HuggingFace Repositories", classes="pane-title")
            with Horizontal(classes="split-layout"):
                with Vertical(classes="form-column"):
                    yield Label("HF Repository Name:")
                    yield Input(placeholder="HuggingFaceTB/SmolLM3-3B", id="pull-model-id")

                    yield Label("Target Revision:")
                    yield Input(value="main", id="pull-revision")

                    yield Button("Download Files", variant="primary", id="btn-pull-run")

                with Vertical(classes="log-column"):
                    yield Label("Download Stream:")
                    yield RichLog(id="pull-output-log", highlight=True, markup=True)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-pull-run":
                model_id = self.query_one("#pull-model-id", Input).value.strip()
                revision = self.query_one("#pull-revision", Input).value.strip() or "main"

                if not model_id:
                    return

                log = self.query_one("#pull-output-log", RichLog)
                log.clear()
                log.write(f"Starting HF download snapshots for: {model_id}...")
                self.run_worker(self._do_pull(model_id, revision))

        async def _do_pull(self, model_id: str, revision: str) -> None:
            log = self.query_one("#pull-output-log", RichLog)
            try:
                from .hf_pull import pull_model, check_huggingface_hub
                if not check_huggingface_hub():
                    log.write("[red]huggingface_hub is not installed.[/red]")
                    return
                result = await asyncio.to_thread(
                    pull_model,
                    model_id,
                    revision=revision,
                )
                log.write(f"[green]✓ Saved to: {result}[/green]")
            except Exception as e:
                log.write(f"[red]✗ Error: {e}[/red]")


    class BenchPane(Vertical):
        """Integrated benchmarking dashboard view."""

        def compose(self) -> ComposeResult:
            yield Label("Profile Benchmark Tests", classes="pane-title")
            with Horizontal(classes="split-layout"):
                with Vertical(classes="form-column"):
                    yield Label("Search / Path to Model:")
                    yield Input(
                        placeholder="Search cache or type path (e.g. ./)...",
                        id="bench-model-input",
                    )
                    yield ListView(id="bench-model-search-list")

                    yield Label("Target Profiles:")
                    yield Input(value="bf16,auto", id="bench-profiles")

                    yield Label("Steps:")
                    yield Input(value="200", id="bench-steps")

                    yield Button("Start Benchmark Test", variant="primary", id="btn-bench-run")

                with Vertical(classes="log-column"):
                    yield Label("Benchmark Metrics:")
                    yield RichLog(id="bench-output-log", highlight=True, markup=True)
                    yield DataTable(id="bench-results-table")

        def on_mount(self) -> None:
            self.query_one("#bench-model-search-list", ListView).styles.display = "none"

            table = self.query_one("#bench-results-table", DataTable)
            table.add_columns("Profile", "tok/s", "ms/tok", "Weights", "Peak GPU")

        @on(Input.Changed, "#bench-model-input")
        def on_model_input_changed(self, event: Input.Changed) -> None:
            query = event.value.strip()
            completions = _get_path_completions(query)
            list_view = self.query_one("#bench-model-search-list", ListView)
            list_view.clear()
            
            for display_name, path in completions:
                list_view.append(PathListItem(display_name, path))
                
            list_view.styles.display = "block" if completions else "none"

        @on(ListView.Selected, "#bench-model-search-list")
        def on_list_selected(self, event: ListView.Selected) -> None:
            selected_path = getattr(event.item, "model_path", None)
            if selected_path:
                input_widget = self.query_one("#bench-model-input", Input)
                input_widget.value = selected_path
                
                if selected_path.endswith("/"):
                    self.on_model_input_changed(Input.Changed(input_widget, selected_path))
                else:
                    self.query_one("#bench-model-search-list", ListView).styles.display = "none"

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-bench-run":
                model_path = self.query_one("#bench-model-input", Input).value.strip()
                profiles_str = self.query_one("#bench-profiles", Input).value.strip()
                steps = int(self.query_one("#bench-steps", Input).value or "200")

                if not model_path or not profiles_str:
                    return

                log = self.query_one("#bench-output-log", RichLog)
                log.clear()
                table = self.query_one("#bench-results-table", DataTable)
                table.clear()

                log.write(f"Starting benchmarks on: {Path(model_path).name}...")
                self.run_worker(self._do_bench(str(model_path), profiles_str, steps))

        async def _do_bench(self, model_path: str, profiles_str: str, steps: int) -> None:
            log = self.query_one("#bench-output-log", RichLog)
            table = self.query_one("#bench-results-table", DataTable)
            profiles = [p.strip() for p in profiles_str.split(",")]

            for p_name in profiles:
                log.write(f"Benchmarking profile: {p_name}...")
                try:
                    from .cli import _run_benchmark_profile
                    from .profile_presets import get_profile

                    profile = get_profile(
                        p_name,
                        model=_archive_model_for_tui(model_path),
                    )
                    result = await asyncio.to_thread(
                        _run_benchmark_profile,
                        archive_path=model_path,
                        profile=profile,
                        profile_name=profile["name"],
                        steps=steps,
                        warmup=10,
                        device="cuda",
                        dtype="bf16",
                        max_gpu_temp=87,
                        dry_run=False,
                        quiet=True,
                    )
                    if result is None or "error" in result:
                        raise RuntimeError(
                            (result or {}).get("error", "benchmark failed")
                        )

                    table.add_row(
                        profile["name"],
                        f"{result['tokens_per_s']:.1f}",
                        f"{result['ms_per_token']:.1f}",
                        _format_bytes(result["resident_weight_bytes"]),
                        _format_bytes(result["gpu_peak_allocated_bytes"]),
                    )
                    log.write(
                        f"[green]✓ Profile '{profile['name']}' completed: "
                        f"{result['tokens_per_s']:.1f} tok/s, causal KV[/green]"
                    )

                except Exception as e:
                    log.write(f"[red]✗ Profile '{p_name}' failed: {e}[/red]")


    class InspectPane(Vertical):
        """Archive verification and details pane."""

        def compose(self) -> ComposeResult:
            yield Label("Archive Inspector", classes="pane-title")
            with Horizontal(classes="split-layout"):
                with Vertical(classes="form-column"):
                    yield Label("Search / Path to Model:")
                    yield Input(
                        placeholder="Search cache or type path (e.g. ./)...",
                        id="inspect-model-input",
                    )
                    yield ListView(id="inspect-model-search-list")
                    yield Button("Inspect Structure", variant="primary", id="btn-inspect-run")

                with Vertical(classes="log-column"):
                    yield Label("Archive Specification:")
                    yield RichLog(id="inspect-output-log", highlight=True, markup=True)

        def on_mount(self) -> None:
            self.query_one("#inspect-model-search-list", ListView).styles.display = "none"

        @on(Input.Changed, "#inspect-model-input")
        def on_model_input_changed(self, event: Input.Changed) -> None:
            query = event.value.strip()
            completions = _get_path_completions(query)
            list_view = self.query_one("#inspect-model-search-list", ListView)
            list_view.clear()
            
            for display_name, path in completions:
                list_view.append(PathListItem(display_name, path))
                
            list_view.styles.display = "block" if completions else "none"

        @on(ListView.Selected, "#inspect-model-search-list")
        def on_list_selected(self, event: ListView.Selected) -> None:
            selected_path = getattr(event.item, "model_path", None)
            if selected_path:
                input_widget = self.query_one("#inspect-model-input", Input)
                input_widget.value = selected_path
                
                if selected_path.endswith("/"):
                    self.on_model_input_changed(Input.Changed(input_widget, selected_path))
                else:
                    self.query_one("#inspect-model-search-list", ListView).styles.display = "none"

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-inspect-run":
                model_path = self.query_one("#inspect-model-input", Input).value.strip()
                if not model_path:
                    return

                log = self.query_one("#inspect-output-log", RichLog)
                log.clear()
                self.run_worker(self._do_inspect(str(model_path)))

        async def _do_inspect(self, model_path: str) -> None:
            log = self.query_one("#inspect-output-log", RichLog)
            try:
                from .archive import ThinArchive
                from .command_runner import verify_archive

                archive = ThinArchive(model_path)
                manifest = archive.manifest
                model = manifest.get("model", {})
                pages = manifest.get("pages", [])

                log.write(f"[bold cyan]File:[/bold cyan] {model_path}")
                log.write(f"[bold cyan]Format Version:[/bold cyan] {manifest.get('version', 0)}")
                log.write(f"[bold cyan]Size:[/bold cyan] {_format_bytes(Path(model_path).stat().st_size)}")
                log.write("")
                log.write(f"[bold]Architecture:[/bold] {model.get('arch', '?')}")
                log.write(f"[bold]Layers:[/bold] {model.get('layers', '?')}")
                log.write(f"[bold]Hidden size:[/bold] {model.get('hidden_size', '?')}")
                log.write(f"[bold]Heads / KV heads:[/bold] {model.get('heads', '?')} / {model.get('kv_heads', '?')}")
                log.write(f"[bold]DType:[/bold] {model.get('dtype', '?')}")
                log.write(f"[bold]Pages:[/bold] {len(pages)}")

                mem = manifest.get("memory_plan", {})
                if mem:
                    log.write("")
                    log.write(f"[bold]Min VRAM requirement:[/bold] {_format_bytes(mem.get('min_vram_bytes', 0))}")
                    log.write(f"[bold]Recommended VRAM:[/bold] {_format_bytes(mem.get('recommended_vram_bytes', 0))}")

                log.write("")
                valid = verify_archive(model_path)
                if valid:
                    log.write("[bold green]Verification Status: PASS[/bold green]")
                else:
                    log.write("[bold red]Verification Status: FAIL[/bold red]")

                archive.close()
            except Exception as e:
                log.write(f"[red]✗ Error: {e}[/red]")


    class DoctorPane(Vertical):
        """Integrated system checker pane."""

        def compose(self) -> ComposeResult:
            yield Label("System Audits & Diagnostics", classes="pane-title")
            yield Button("Run Doctor Audit", variant="primary", id="btn-doctor-run")
            yield RichLog(id="doctor-output-log", highlight=True, markup=True)

        def on_mount(self) -> None:
            self.run_doctor()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-doctor-run":
                self.run_doctor()

        def run_doctor(self) -> None:
            log = self.query_one("#doctor-output-log", RichLog)
            log.clear()

            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            log.write(f"[green]✓[/green] Python version: {py_ver}")

            try:
                import torch
                log.write(f"[green]✓[/green] PyTorch version: {torch.__version__}")
                if torch.cuda.is_available():
                    log.write(f"[green]✓[/green] CUDA support: available")
                    log.write(f"[green]✓[/green] GPU core: {torch.cuda.get_device_name(0)}")
                    vram = torch.cuda.get_device_properties(0).total_memory
                    log.write(f"[green]✓[/green] VRAM size: {_format_bytes(vram)}")
                else:
                    log.write("[red]✗[/red] CUDA support: not available")
            except ImportError:
                log.write("[red]✗[/red] PyTorch: not installed")

            try:
                import triton  # noqa: F401
                log.write("[green]✓[/green] Triton kernels: installed")
            except ImportError:
                log.write("[red]✗[/red] Triton: not installed")

            from .command_runner import find_rust_binary
            rust_bin = find_rust_binary()
            if rust_bin:
                log.write(f"[green]✓[/green] Rust binary build: {rust_bin}")
            else:
                log.write("[red]✗[/red] Rust binary: not found")

            try:
                import huggingface_hub  # noqa: F401
                log.write("[green]✓[/green] HF Hub API: installed")
            except ImportError:
                log.write("[yellow]⚠[/yellow] HF Hub: not installed")

            from .model_cache import cache_root
            cache = cache_root()
            writable = os.access(str(cache.parent), os.W_OK)
            status = "[green]✓[/green]" if writable else "[red]✗[/red]"
            log.write(f"{status} Caching directories: {cache}")


    class ThinTensorApp(App):
        """Unified ThinTensor TUI App."""

        TITLE = "ThinTensor Dashboard"

        CSS = """
        Screen {
            background: #030712;
            color: #f3f4f6;
        }
        #app-grid {
            layout: grid;
            grid-size: 2;
            grid-columns: 28 1fr;
            height: 1fr;
        }
        Sidebar {
            background: #0f172a;
            border-right: tall #4f46e5;
            padding: 1 1;
            width: 28;
            height: 1fr;
        }
        #logo-sidebar {
            text-align: center;
            color: #818cf8;
            text-style: bold;
            margin: 1 0;
        }
        #nav-list {
            background: transparent;
        }
        #nav-list ListItem {
            padding: 1 2;
            color: #cbd5e1;
        }
        #nav-list ListItem:hover {
            background: #312e81;
            color: #f8fafc;
        }
        #nav-list ListItem.--highlight {
            background: #4f46e5;
            color: #ffffff;
            text-style: bold;
        }
        .pane {
            padding: 1 2;
            height: 1fr;
            layout: vertical;
        }
        .pane-title {
            text-style: bold;
            color: #818cf8;
            margin-bottom: 1;
        }
        .section-title {
            text-style: bold;
            margin: 1 0;
        }
        .card {
            background: #1e1b4b;
            border: tall #4f46e5;
            padding: 1 2;
            margin-bottom: 1;
        }
        .split-layout {
            layout: grid;
            grid-size: 2;
            grid-columns: 1fr 1fr;
            height: 1fr;
            grid-gutter: 1 2;
        }
        .form-column {
            layout: vertical;
            padding: 1;
            background: #0f172a;
            border: solid #1e293b;
            height: 1fr;
        }
        .log-column {
            layout: vertical;
            height: 1fr;
        }
        Input {
            border: tall #312e81;
            background: #020617;
            margin-bottom: 1;
        }
        Select {
            border: tall #312e81;
            background: #020617;
            margin-bottom: 1;
        }
        TextArea {
            border: tall #312e81;
            background: #020617;
            margin-bottom: 1;
            height: 6;
        }
        RichLog {
            background: #020617;
            border: solid #1e293b;
            height: 1fr;
        }
        DataTable {
            height: 1fr;
            background: #020617;
            border: solid #1e293b;
        }
        #run-model-search-list, #bench-model-search-list, #inspect-model-search-list {
            background: #020617;
            border: solid #312e81;
            max-height: 8;
            margin-bottom: 1;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit", show=True),
        ]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Container(id="app-grid"):
                with Vertical(id="nav-column"):
                    # Sidebar navigation
                    with Static(id="logo-sidebar"):
                        yield Label("ThinTensor")
                    with ListView(id="nav-list"):
                        yield ListItem(Label("Home"), id="nav-home")
                        yield ListItem(Label("Run & Chat"), id="nav-run")
                        yield ListItem(Label("Convert"), id="nav-convert")
                        yield ListItem(Label("Pull Model"), id="nav-pull")
                        yield ListItem(Label("Benchmark"), id="nav-bench")
                        yield ListItem(Label("Inspect"), id="nav-inspect")
                        yield ListItem(Label("Doctor"), id="nav-doctor")

                with ContentSwitcher(initial="pane-home", id="main-content"):
                    yield HomePane(id="pane-home", classes="pane")
                    yield RunPane(id="pane-run", classes="pane")
                    yield ConvertPane(id="pane-convert", classes="pane")
                    yield PullPane(id="pane-pull", classes="pane")
                    yield BenchPane(id="pane-bench", classes="pane")
                    yield InspectPane(id="pane-inspect", classes="pane")
                    yield DoctorPane(id="pane-doctor", classes="pane")
            yield Footer()

        @on(ListView.Selected, "#nav-list")
        def on_nav_selected(self, event: ListView.Selected) -> None:
            list_item_id = event.item.id
            switcher = self.query_one("#main-content", ContentSwitcher)

            mapping = {
                "nav-home": "pane-home",
                "nav-run": "pane-run",
                "nav-convert": "pane-convert",
                "nav-pull": "pane-pull",
                "nav-bench": "pane-bench",
                "nav-inspect": "pane-inspect",
                "nav-doctor": "pane-doctor",
            }
            target = mapping.get(list_item_id)
            if target:
                switcher.current = target


def run_tui() -> None:
    """Launch the ThinTensor TUI dashboard."""
    if not HAS_TEXTUAL:
        print("✗ TUI requires the 'textual' package.")
        print("  Install with: pip install textual")
        raise SystemExit(1)

    app = ThinTensorApp()
    app.run()


if __name__ == "__main__":
    run_tui()
