"""Z-Image Turbo Gradio UI -- text-to-image only.

Same shape as Kandinsky UI and Sana UI: no Jupyter, no Colab. Run
it, a page opens in your browser, type a prompt, hit Generate, a
picture appears underneath. Z-Image Turbo is Tongyi-MAI's (Alibaba)
6B distilled image model: 8 sampling steps, strong photorealistic
quality and bilingual (English/Chinese) text rendering, under the
Apache 2.0 license. The weights are NOT gated -- no token, no
license acceptance, no account step before the first download.

How to run (on a Windows PC with an NVIDIA GPU):

    install.bat      (once -- builds a Python 3.12 venv, installs
                      torch from cu130 + the packages)
    run.bat          (starts the app)

Python 3.12 is required because Diffusers 0.42 dropped Python 3.10.
The installer builds its own .venv from 3.12 and never touches any
other Python on the machine -- a Python 3.10 install used by other
apps is left completely alone.

The app file lives in the z-image-ui/ subfolder; run.bat runs it
from the folder above, so pictures land in images_out/ next to
run.bat -- one subfolder per batch name, reused across runs,
frames numbered continuously, and settings.txt appended once per
run, exactly like Kandinsky UI and Sana UI.

First Generate takes a while: the model downloads from Hugging
Face (about 15 GB on the default quantized path) and loads once.
Repeat generations reuse what is already loaded.

How it works (Diffusers 0.42.0, ZImagePipeline -- call shapes match
the Z-Image-Turbo model card's own Diffusers snippet):
  1. The pipeline loads from Tongyi-MAI/Z-Image-Turbo, except the
     transformer, which loads from a quantized GGUF file instead
     (see _load_pipe). Turbo is distilled: it always runs 9
     inference steps (8 DiT forwards) at guidance 0. There is no
     steps slider and no guidance slider in this UI -- both are
     locked, because Turbo has no use for them.
  2. The Qwen3-4B text encoder and the VAE ship with the model
     repo and load with the pipeline.

Fitting on a 12 GB card (RTX 4070) -- the lesson from the Klein
build, where a 12 GB BF16 transformer plus an 8 GB encoder did
NOT fit on a 12 GB card:
  - The DEFAULT precision is a quantized GGUF transformer
     (4-7 GB depending on which quant the repo offers), loaded
     through Diffusers (ZImageTransformer2DModel.from_single_file
     + GGUFQuantizationConfig, compute in bfloat16). The loader
     asks the GGUF repo what files it actually has and picks the
     best quant on offer (Q8_0 first, then Q6_K, Q5_K_M, Q4_K_M,
     Q3_K_M); the Q3_K_M file is the verified fallback because
     that exact file was confirmed in the repo. It never wires in
     a filename it has not seen.
  - CPU offload defaults ON, using the standard DiffusionPipeline
     enable_model_cpu_offload() (this exists on ZImagePipeline --
     unlike Klein's ModularPipeline, which had no such method).
     With offload on, the 8 GB text encoder stays on CPU except
     while encoding, and the quantized transformer moves to GPU
     for diffusion. Peak VRAM stays well under 12 GB.
  - Full BF16 is offered as a choice, but it loads the ~12 GB
     transformer plus the ~8 GB encoder: on a 12 GB card it only
     works with offload ON, and even then it is tight. It is here
     for bigger cards and for comparison, not as the default.
  - If the quantized load fails for any reason, the loader falls
     back to full BF16 WITH offload forced on, and says so in the
     status -- it never silently pretends the quant loaded, and
     never falls back to a full-GPU load that would OOM a 12 GB
     card.

Extras, in the same plain style as the other apps: size presets,
several pictures per prompt, seed shown in the status line, a Stop
button, a Random prompt button (word banks in code), a session-only
prompt history dropdown, Star batch (copies a batch into
images_out/favorites/), a prompt queue (one prompt per line),
single-level dynamic prompts {a|b|c}, and an Unload button to free
VRAM/RAM.

No animation, no upscaling, no editing, no ControlNet -- v1 is
text-to-image only, kept simple on purpose.
"""

import gc
import os
import random
import re
import shutil
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)  # folder holding run.bat
IMAGES_DIR = os.path.join(ROOT_DIR, "images_out")
FAVORITES_DIR = os.path.join(IMAGES_DIR, "favorites")

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
GGUF_REPO = "jayn7/Z-Image-Turbo-GGUF"
# Best quant first. The Q3_K_M entry is the verified anchor: that
# exact file was confirmed live in the GGUF repo (4.12 GB). The
# loader only picks a name the repo listing actually contains, so
# a missing better quant can never break the load -- it just
# falls through to the next name, ending at the verified file.
GGUF_PREFERENCE = [
    "z_image_turbo-Q8_0.gguf",
    "z_image_turbo-Q6_K.gguf",
    "z_image_turbo-Q5_K_M.gguf",
    "z_image_turbo-Q4_K_M.gguf",
    "z_image_turbo-Q3_K_M.gguf",
]
GGUF_FALLBACK = "z_image_turbo-Q3_K_M.gguf"

# Turbo is distilled: 9 steps == 8 DiT forwards, guidance always 0.
TURBO_STEPS = 9
TURBO_GUIDANCE = 0.0

DEFAULT_PROMPT = (
    "A lighthouse on a rocky cliff during a storm, waves crashing, "
    "dramatic sky, detailed"
)

_pipe = None
_pipe_key = None  # (precision, offload) of the loaded pipe
_pipe_note = ""


def _safe_batch_name(name):
    """Strip characters Windows forbids in folder names. Falls back
    to a timestamped name when the box is empty."""
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    if not name:
        name = datetime.now().strftime("zimage-%Y%m%d-%H%M%S")
    return name


def _batch_folder(base):
    """The one folder for a batch name: images_out/<base>, created
    if missing, reused if it already exists. Repeat runs never get
    a _2/_3 folder -- frames just keep numbering upward."""
    return os.path.join(IMAGES_DIR, base)


def _unique_folder(base, root=IMAGES_DIR):
    """<root>/<base>, or <base>_2, <base>_3... if it already exists.
    Used only by Star batch; generations reuse one folder per batch
    name (see _batch_folder) instead of numbered folders."""
    folder = os.path.join(root, base)
    if not os.path.exists(folder):
        return folder
    i = 2
    while True:
        candidate = "{}_{}".format(folder, i)
        if not os.path.exists(candidate):
            return candidate
        i += 1


_FRAME_RE = re.compile(r"^(\d{6})\.png$")


def _next_frame_number(folder):
    """First unused frame number in a folder: one past the highest
    existing 000001-style frame, or 1 for a fresh folder."""
    highest = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            match = _FRAME_RE.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _frame_range_text(start, count):
    """'frame 000006' for one image, 'frames 000006-000008' for
    several -- used in status lines and settings headers."""
    if count == 1:
        return "frame {:06d}".format(start)
    return "frames {:06d}-{:06d}".format(start, start + count - 1)


def _seeds_for(seed, count):
    """Resolve the seed box to one seed per image. -1 (or empty)
    draws a fresh random seed per image; a fixed seed S gives S,
    S+1, S+2... so a multi-image run with a fixed seed stays
    reproducible."""
    if seed is None:
        seed = -1
    seed = int(seed)
    if seed < 0:
        return [random.randint(0, 2**31 - 1) for _ in range(count)]
    return [seed + i for i in range(count)]


# Dynamic prompts: single level only, one random pick per group.
# "{a|b|c}" becomes a, b, or c. Groups do not nest; a group with no
# pipe is just its own text with the braces removed.
_TEMPLATE_RE = re.compile(r"\{([^{}]*)\}")


def _expand_templates(prompt):
    """Expand {a|b|c} groups, one random pick each. Plain prompts
    (no braces) come back unchanged."""
    return _TEMPLATE_RE.sub(
        lambda match: random.choice(match.group(1).split("|")), prompt
    )


def _append_settings(folder, header, entries):
    """Append one run's block to settings.txt: a header line (date,
    time, frame range) followed by the run's 'key: value' lines and
    a blank line. Creates the file on first use; earlier runs'
    blocks are never touched. A missing trailing newline is
    repaired so blocks never run together."""
    path = os.path.join(folder, "settings.txt")
    prefix = ""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(prefix + header + "\n")
        handle.write("\n".join(entries) + "\n\n")


def _run_header(start, count):
    """The settings.txt header for one run: date, time, and the
    frame-number range it produced."""
    return "Run {}, {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _frame_range_text(start, count))


def _seed_entry(seeds):
    """The settings.txt seed line: 'seed: X' for one image,
    'seeds: X, Y' for several."""
    if len(seeds) == 1:
        return "seed: {}".format(seeds[0])
    return "seeds: {}".format(", ".join(str(s) for s in seeds))


def _expanded_entries(expanded_list):
    """The settings.txt expanded-prompt line(s): one line for one
    image, a comma-joined list for several."""
    if len(expanded_list) == 1:
        return ["expanded_prompt: {}".format(expanded_list[0])]
    return ["expanded_prompts: {}".format(", ".join(expanded_list))]


def _round16(value):
    """Nearest multiple of 16, at least 16. Keeps the VAE's 8x
    downsampling exact even for odd-sized custom sizes."""
    return max(16, int(round(int(value) / 16.0)) * 16)


def _free_memory():
    """Drop cached GPU memory after unloading a model. Best effort:
    never raises, so an unload step cannot break the next load."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- freeing memory must not fail
        pass


def _unload_pipe():
    """Drop the loaded pipeline so the next Generate loads fresh
    (used when precision/offload change, and by Unload)."""
    global _pipe, _pipe_key, _pipe_note
    _pipe = None
    _pipe_key = None
    _pipe_note = ""
    _free_memory()


def _pick_gguf_file():
    """Best quantized transformer the GGUF repo actually offers,
    best first. The repo listing is the authority: a name is only
    picked if it is in the listing, so a renamed or missing file
    can never break the load. Any listing failure (offline, API
    change) falls back to the verified Q3_K_M file."""
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(GGUF_REPO, repo_type="model")
    except Exception:  # noqa: BLE001 -- listing is best-effort
        return GGUF_FALLBACK
    for name in GGUF_PREFERENCE:
        if name in files:
            return name
    return GGUF_FALLBACK


def _load_pipe(precision, offload):
    """Load (or reuse) the Z-Image Turbo pipeline. Returns
    (pipe, note). Precision is "gguf" (default, quantized
    transformer) or "bf16" (full weights). Offload True uses the
    standard enable_model_cpu_offload(); False loads everything
    onto the GPU (only for cards bigger than 12 GB).

    If the quantized load fails, this falls back to full BF16
    WITH offload forced on -- never to a full-GPU load -- and the
    note says so plainly. If even that fails, it raises a
    RuntimeError with a plain sentence for the status box."""
    global _pipe, _pipe_key, _pipe_note
    key = (precision, offload)
    if _pipe is not None and _pipe_key == key:
        return _pipe, _pipe_note
    _unload_pipe()

    import torch
    from diffusers import (
        GGUFQuantizationConfig,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )

    note = ""
    pipe = None
    if precision == "gguf":
        try:
            from huggingface_hub import hf_hub_download
            picked = _pick_gguf_file()
            local = hf_hub_download(GGUF_REPO, filename=picked)
            transformer = ZImageTransformer2DModel.from_single_file(
                local,
                quantization_config=GGUFQuantizationConfig(
                    compute_dtype=torch.bfloat16),
                dtype=torch.bfloat16)
            pipe = ZImagePipeline.from_pretrained(
                MODEL_ID, transformer=transformer,
                dtype=torch.bfloat16)
            note = "Quantized transformer: {}.".format(picked)
        except Exception as exc:  # noqa: BLE001 -- fall back, loudly
            note = ("Quantized load failed ({}); fell back to full "
                    "BF16 with CPU offload on.".format(exc))
            pipe = None
            offload = True
            key = (precision, True)
    if pipe is None:
        try:
            pipe = ZImagePipeline.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16)
        except Exception as exc:  # noqa: BLE001 -- report, don't crash
            raise RuntimeError(
                "Could not load Z-Image Turbo ({}). Check the "
                "internet connection and that Hugging Face is "
                "reachable, then try Generate again.".format(exc))
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    _pipe = pipe
    _pipe_key = key
    _pipe_note = note
    return pipe, note


def _history_add(history, prompt):
    """Add a prompt to the session history (most recent last, no
    duplicates, last 10 kept). Returns the new history list."""
    history = [item for item in (history or []) if item != prompt]
    history.append(prompt)
    return history[-10:]


def _history_update(history):
    """The history dropdown update: choices are the history, most
    recent first, value cleared so picking the same prompt twice
    still fires."""
    import gradio as gr
    return gr.update(choices=list(reversed(history or [])), value=None)


def _early(message, last_folder, history):
    """A single yield for an early exit (bad input, load failure):
    empty gallery, the message as status, history untouched."""
    yield [], message, _history_update(history), last_folder, history


def _preset_size(preset):
    """Size preset -> (width, height). Custom leaves the boxes
    alone (returns them unchanged)."""
    import gradio as gr
    sizes = {
        "Square 1024x1024": (1024, 1024),
        "Portrait 768x1024": (768, 1024),
        "Landscape 1024x768": (1024, 768),
        "Small square 768x768": (768, 768),
    }
    if preset in sizes:
        return sizes[preset]
    return gr.update(), gr.update()


def _noop():
    """Stop-button target: cancels the Generate event, does
    nothing itself."""
    return None


_SUBJECTS = [
    "a lighthouse", "a fox", "a mountain cabin", "a castle",
    "a harbor town", "a red panda", "a sailing ship",
    "a robot gardener", "a treehouse village", "a steam train",
]
_SETTINGS = [
    "a rocky cliff", "a misty forest", "desert dunes",
    "a snowy village", "a neon city street", "a quiet lake",
    "a sunflower field", "a space station", "a coral reef",
    "an old library",
]
_LIGHTS = [
    "golden hour light", "storm light", "moonlight",
    "neon dusk", "soft morning fog", "candlelight",
    "sunset clouds", "starlight",
]
_STYLES = [
    "photorealistic", "oil painting", "anime style",
    "watercolor", "cyberpunk", "storybook illustration",
]


def random_prompt():
    """One random prompt from the built-in word banks."""
    return "A {} in {}, {}, {}, detailed.".format(
        random.choice(_SUBJECTS),
        random.choice(_SETTINGS),
        random.choice(_LIGHTS),
        random.choice(_STYLES),
    )


def generate(prompt_text, queue_text, batch_name, width, height,
             seed, count, precision_choice, offload,
             last_folder, history):
    """Text-to-image generation. A generator function: it yields
    (gallery images, status, history dropdown update, last batch
    folder, history list) after every image, so the page updates
    live, and the Stop button can cancel it mid-run.

    Queue beats the single prompt: when the queue box has lines,
    each line is one job; otherwise the prompt box is the one job.
    Every job renders `count` images. Dynamics {a|b|c} are
    re-expanded for every single image, so no two images in a run
    share an expansion by accident. All jobs in one click share
    one batch folder, frames numbered continuously."""
    import torch

    count = int(count) if count else 1
    count = max(1, min(10, count))
    history = history or []

    lines = [line.strip() for line in (queue_text or "").splitlines()
             if line.strip()]
    if lines:
        templates = lines
    elif prompt_text and prompt_text.strip():
        templates = [prompt_text.strip()]
    else:
        yield from _early("Type a prompt first, or fill the queue.",
                          last_folder, history)
        return

    try:
        width = _round16(max(16, min(2048, int(width))))
        height = _round16(max(16, min(2048, int(height))))
    except (TypeError, ValueError):
        yield from _early("Width and height must be numbers.",
                          last_folder, history)
        return

    precision = "gguf" if str(precision_choice).startswith("Quantized") \
        else "bf16"

    base = _safe_batch_name(batch_name)
    folder = _batch_folder(base)
    os.makedirs(folder, exist_ok=True)

    try:
        pipe, note = _load_pipe(precision, bool(offload))
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return

    gallery = []
    status = ""
    for template in templates:
        start = _next_frame_number(folder)
        seeds = _seeds_for(seed, count)
        expanded_list = []
        for i in range(count):
            expanded = _expand_templates(template)
            expanded_list.append(expanded)
            try:
                image = pipe(
                    prompt=expanded,
                    height=height,
                    width=width,
                    num_inference_steps=TURBO_STEPS,
                    guidance_scale=TURBO_GUIDANCE,
                    generator=torch.Generator("cpu").manual_seed(
                        seeds[i]),
                ).images[0]
            except Exception as exc:  # noqa: BLE001 -- plain status
                yield (gallery,
                       "Generation failed on {}: {}".format(
                           _frame_range_text(start + i, 1), exc),
                       _history_update(history), folder, history)
                return
            frame = start + i
            image.save(os.path.join(folder, "{:06d}.png".format(frame)))
            gallery.append(image)
            status = (
                "Saved {} in {} ({}x{}, steps {}, guidance 0). "
                "Seeds: {}. {}".format(
                    _frame_range_text(start, i + 1), folder,
                    width, height, TURBO_STEPS,
                    ", ".join(str(s) for s in seeds[:i + 1]),
                    note))
            yield (gallery, status, _history_update(history),
                   folder, history)
            # Free cached memory between images. Without this, a
            # long queue creeps upward in RAM/VRAM until Windows
            # kills the process mid-batch (Oct 27, 2026: died 3
            # images into a 10-image queue, per-step time climbing
            # as it went). _free_memory is best-effort, never
            # raises, so cleanup cannot break a run.
            _free_memory()
        _append_settings(folder, _run_header(start, count), [
            "model: {}".format(MODEL_ID),
            "precision: {}".format(precision),
            "offload: {}".format(bool(offload)),
            "size: {}x{}".format(width, height),
            "steps: {}".format(TURBO_STEPS),
            "guidance: 0",
            "prompt: {}".format(template),
            _seed_entry(seeds),
        ] + _expanded_entries(expanded_list))
        history = _history_add(history, template)
        status = (
            "Done: {} in {} ({}x{}, steps {}, guidance 0). "
            "Seeds: {}. {}".format(
                _frame_range_text(start, count), folder,
                width, height, TURBO_STEPS,
                ", ".join(str(s) for s in seeds), note))
        yield (gallery, status, _history_update(history),
               folder, history)


def star_batch(last_folder):
    """Copy the most recently generated batch folder into
    images_out/favorites/ under a unique name (never overwrite).
    If nothing has run yet this session, say so in the status."""
    if not last_folder or not os.path.isdir(last_folder):
        return "No batch has run yet in this session."
    os.makedirs(FAVORITES_DIR, exist_ok=True)
    base = os.path.basename(os.path.normpath(last_folder))
    dest = _unique_folder(base, root=FAVORITES_DIR)
    shutil.copytree(last_folder, dest)
    return "Starred to {}".format(dest)


def unload_models():
    """Free the generator (VRAM and RAM)."""
    _unload_pipe()
    return "Models unloaded -- VRAM and RAM freed."


def build_ui():
    """The Gradio page: one tab, text-to-image only, kept simple
    on purpose. No steps slider and no guidance slider -- Turbo
    runs 9 steps at guidance 0, always."""
    import gradio as gr

    with gr.Blocks(title="Z-Image Turbo UI") as demo:
        gr.Markdown(
            "# Z-Image Turbo UI\n"
            "Type a prompt, hit Generate, pictures appear below. "
            "No account, no token, no license step -- the model is "
            "Apache 2.0 and ungated.\n"
            "Turbo always runs 9 steps at guidance 0, so there are "
            "no steps or guidance controls: that is the distilled "
            "model, not a missing feature.")
        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt", value=DEFAULT_PROMPT, lines=3)
            batch_box = gr.Textbox(
                label="Batch name (one folder per name)",
                value="zimage-pictures")
        with gr.Row():
            preset = gr.Dropdown(
                label="Size preset",
                choices=["Square 1024x1024", "Portrait 768x1024",
                         "Landscape 1024x768", "Small square 768x768",
                         "Custom"],
                value="Square 1024x1024")
            width_box = gr.Number(label="Width", value=1024,
                                  precision=0)
            height_box = gr.Number(label="Height", value=1024,
                                   precision=0)
        with gr.Row():
            seed_box = gr.Number(label="Seed (-1 = random)",
                                 value=-1, precision=0)
            count_box = gr.Slider(label="Images per prompt",
                                  minimum=1, maximum=10, value=1,
                                  step=1)
        with gr.Row():
            precision_box = gr.Radio(
                label="Precision",
                choices=["Quantized GGUF -- recommended for 12 GB",
                         "Full BF16 -- needs more VRAM"],
                value="Quantized GGUF -- recommended for 12 GB")
            offload_box = gr.Checkbox(
                label="CPU offload (keep ON for a 12 GB card)",
                value=True)
        queue_box = gr.Textbox(
            label="Prompt queue (one prompt per line -- when this "
                  "has lines, Generate runs the queue instead of "
                  "the prompt box; all lines share one batch folder)",
            lines=4)
        with gr.Row():
            random_btn = gr.Button("Random prompt")
            history_dd = gr.Dropdown(
                label="History (last 10 -- pick one to reload it)",
                choices=[])
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            stop_btn = gr.Button("Stop")
            star_btn = gr.Button("Star batch")
            unload_btn = gr.Button("Unload models")
        status_box = gr.Textbox(label="Status", lines=4)
        gallery = gr.Gallery(label="Pictures", columns=3)

        last_folder_state = gr.State("")
        history_state = gr.State([])

        preset.change(fn=_preset_size, inputs=preset,
                      outputs=[width_box, height_box])
        random_btn.click(fn=random_prompt, outputs=prompt_box)
        history_dd.change(fn=lambda choice: choice,
                          inputs=history_dd, outputs=prompt_box)
        generate_event = generate_btn.click(
            fn=generate,
            inputs=[prompt_box, queue_box, batch_box,
                    width_box, height_box, seed_box, count_box,
                    precision_box, offload_box,
                    last_folder_state, history_state],
            outputs=[gallery, status_box, history_dd,
                     last_folder_state, history_state])
        stop_btn.click(fn=_noop, cancels=[generate_event])
        star_btn.click(fn=star_batch, inputs=last_folder_state,
                       outputs=status_box)
        unload_btn.click(fn=unload_models, outputs=status_box)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", inbrowser=True)
