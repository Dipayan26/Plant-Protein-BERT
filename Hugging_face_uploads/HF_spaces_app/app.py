import os
import tempfile

import h5py
import numpy as np
import gradio as gr
import torch
from transformers import EsmForMaskedLM, EsmTokenizer

# ── constants ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, str | None] = {
    "PlantPLM-8M (7.5M params)": "dipayan26/PlantPLM-8M",
    "PlantPLM-35M (35M params) — coming soon": None,
    "PlantPLM-150M (150M params) — coming soon": None,
    "PlantPLM-650M (650M params) — coming soon": None,
}

MAX_LEN   = 256   # AA per sequence on CPU
MAX_SEQS  = 50    # sequences per job
MAX_FILES = 10    # uploaded files per job

EXAMPLE_SEQ = (
    "MSPQTETKASVGFKAGVKDYKLTYYTPEYETKDTDILAAFRVTPQPGVPPEEAGAAVAAESSTGT"
    "WTTPWTPTFGDDKIMASVGFKAGVKDYKLTYYTPEYETKDTDILAAFRVTPQPGVPPEEAGAAVA"
)  # RuBisCO large subunit fragment — Spinacia oleracea

EMB_PER_PROTEIN = "Per-protein  ·  mean pool  →  shape [hidden_dim]"
EMB_PER_AA      = "Per-amino-acid  ·  residue-level  →  shape [L × hidden_dim]"

FMT_PT = ".pt — PyTorch tensor"
FMT_H5 = ".h5 — HDF5"

device = "cuda" if torch.cuda.is_available() else "cpu"
ON_CPU = device == "cpu"

# ── load model ─────────────────────────────────────────────────────────────────

print("Loading PlantPLM-8M …")
tokenizer: EsmTokenizer    = EsmTokenizer.from_pretrained("dipayan26/PlantPLM-8M")
model:     EsmForMaskedLM  = EsmForMaskedLM.from_pretrained("dipayan26/PlantPLM-8M")
model.eval()
model.to(device)  # type: ignore[arg-type]
print(f"Model ready on {device}.")


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_fasta(text: str) -> list[tuple[str, str]]:
    seqs: list[tuple[str, str]] = []
    cur_id: str | None = None
    cur_seq: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None and cur_seq:
                seqs.append((cur_id, "".join(cur_seq)))
            header = line[1:].strip()
            cur_id = header.split()[0] if header else f"seq_{len(seqs) + 1}"
            cur_seq = []
        else:
            cur_seq.append(line.upper())
    if cur_id is not None and cur_seq:
        seqs.append((cur_id, "".join(cur_seq)))
    return seqs


def read_file(f) -> str:
    path = f if isinstance(f, str) else getattr(f, "name", str(f))
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def embed_sequence(seq: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        protein_emb : [hidden_dim]      — mean pool over residue tokens
        aa_emb      : [L, hidden_dim]   — per-residue, CLS and EOS excluded
    """
    inputs = tokenizer(
        seq, return_tensors="pt", truncation=True, max_length=MAX_LEN + 2
    ).to(device)

    with torch.no_grad():
        # last_hidden_state: [1, seq_len, hidden_dim]
        # seq_len = 1 (CLS) + L (AAs) + 1 (EOS)
        hidden = model.esm(**inputs).last_hidden_state

    hidden = hidden[0]  # [seq_len, hidden_dim]

    n_total = int(inputs["attention_mask"][0].sum().item())  # CLS + L + EOS
    L = n_total - 2

    aa_emb      = hidden[1 : L + 1].cpu().numpy()   # [L, hidden_dim]
    protein_emb = aa_emb.mean(axis=0)               # [hidden_dim]

    return protein_emb, aa_emb


# ── main generation function ───────────────────────────────────────────────────

def generate_embeddings(
    model_key:    str,
    input_method: str,
    single_seq:   str,
    fasta_text:   str,
    fasta_files,
    emb_type:     str,
    fmt:          str,
):
    if MODEL_REGISTRY.get(model_key) is None:
        return "❌ This model is not yet available. Please select PlantPLM-8M.", None

    # ── collect sequences ───────────────────────────────────────────────────────
    sequences: list[tuple[str, str]] = []

    if input_method == "Single sequence":
        seq = single_seq.strip().upper().replace(" ", "")
        if not seq:
            return "❌ Please enter a protein sequence.", None
        sequences = [("seq_1", seq)]

    elif input_method == "Paste FASTA":
        txt = (fasta_text or "").strip()
        if not txt:
            return "❌ Please paste at least one sequence.", None
        if not txt.startswith(">"):
            sequences = [("seq_1", txt.upper().replace(" ", "").replace("\n", ""))]
        else:
            sequences = parse_fasta(txt)
        if not sequences:
            return "❌ No valid sequences found in the pasted text.", None

    elif input_method == "Upload files":
        if not fasta_files:
            return "❌ Please upload at least one FASTA file.", None
        files = fasta_files if isinstance(fasta_files, list) else [fasta_files]
        files = [f for f in files if f is not None]
        if len(files) > MAX_FILES:
            return f"❌ Too many files ({len(files)}). Maximum: {MAX_FILES}.", None
        for f in files:
            content = read_file(f)
            parsed  = parse_fasta(content)
            if not parsed:
                raw = content.strip().upper().replace(" ", "")
                if raw:
                    name = os.path.basename(
                        f if isinstance(f, str) else f.name
                    ).rsplit(".", 1)[0]
                    parsed = [(name, raw)]
            sequences.extend(parsed)
        if not sequences:
            return "❌ No valid sequences found in uploaded files.", None

    # ── cap and truncate ────────────────────────────────────────────────────────
    capped_warn = ""
    if len(sequences) > MAX_SEQS:
        sequences   = sequences[:MAX_SEQS]
        capped_warn = f"⚠️ Only first {MAX_SEQS} sequences processed (per-job limit)."

    truncated: list[str] = []
    clean: list[tuple[str, str]] = []
    for sid, seq in sequences:
        if len(seq) > MAX_LEN:
            truncated.append(sid)
            seq = seq[:MAX_LEN]
        clean.append((sid, seq))

    # ── embed ───────────────────────────────────────────────────────────────────
    protein_embs: dict[str, np.ndarray] = {}
    aa_embs:      dict[str, np.ndarray] = {}

    for sid, seq in clean:
        p, a        = embed_sequence(seq)
        protein_embs[sid] = p
        aa_embs[sid]      = a

    # ── select data to export ───────────────────────────────────────────────────
    use_per_aa = EMB_PER_AA in emb_type
    data       = aa_embs if use_per_aa else protein_embs
    use_h5     = FMT_H5 in fmt

    ext    = ".h5" if use_h5 else ".pt"
    label  = "aa" if use_per_aa else "protein"
    tmp    = tempfile.NamedTemporaryFile(
        suffix=f"_{label}_emb{ext}", prefix="plantplm_", delete=False
    )
    tmp.close()

    if use_h5:
        with h5py.File(tmp.name, "w") as hf:
            for sid, emb in data.items():
                hf.create_dataset(sid, data=emb)
    else:
        torch.save(
            {sid: torch.from_numpy(emb) for sid, emb in data.items()},
            tmp.name,
        )

    # ── status log ──────────────────────────────────────────────────────────────
    D        = next(iter(protein_embs.values())).shape[0]
    type_str = f"per-amino-acid  [L × {D}]" if use_per_aa else f"per-protein  [{D}]"

    lines = [
        f"✅  {len(clean)} sequence(s) embedded on {device.upper()}.",
        f"    Embedding type : {type_str}",
        f"    File format    : {ext}",
        f"    Model          : {model_key}",
    ]
    if capped_warn:
        lines.append(capped_warn)
    if truncated:
        shown = ", ".join(truncated[:5]) + ("…" if len(truncated) > 5 else "")
        lines.append(f"⚠️  {len(truncated)} sequence(s) truncated to {MAX_LEN} AA: {shown}")

    return "\n".join(lines), tmp.name


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="PlantPLM — Embedding Generator", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style="background:#fff3cd;border-left:4px solid #e6a817;
                border-radius:6px;padding:10px 16px;margin-bottom:6px;
                font-size:0.9em;color:#111;font-weight:500">
      ⚠️ Running on CPU — sequences are capped at 256 amino acids.
      Expect ~3–6 s per sequence.
    </div>
    """)

    gr.Markdown("""
# PlantPLM — Protein Embedding Generator

Generate protein embeddings from plant-adapted ESM-2 models trained on **19.9 million Viridiplantae sequences**.
Part of the [Plant-Protein-BERT collection](https://huggingface.co/collections/dipayan26/plant-protein-bert).
""")

    # ── model ───────────────────────────────────────────────────────────────────
    model_dropdown = gr.Dropdown(
        choices=list(MODEL_REGISTRY.keys()),
        value="PlantPLM-8M (7.5M params)",
        label="Model",
    )
    model_warn = gr.Markdown(visible=False)

    gr.Markdown("---\n### Input")

    # ── input method ────────────────────────────────────────────────────────────
    input_radio = gr.Radio(
        choices=["Single sequence", "Paste FASTA", "Upload files"],
        value="Single sequence",
        label="Input method",
    )
    single_box = gr.Textbox(
        label="Protein sequence (uppercase amino acids)",
        placeholder="MSPQTETKASVGFKAGVKDYKLTYYTPEYETK…",
        lines=3,
        visible=True,
    )
    example_btn = gr.Button(
        "Load example  (RuBisCO large subunit · Spinacia oleracea)",
        size="sm", variant="secondary", visible=True,
    )
    fasta_box = gr.Textbox(
        label=f"FASTA text — up to {MAX_SEQS} sequences",
        placeholder=">protein_A\nMSPQTETKASVGFK…\n\n>protein_B\nMALSSRTLS…",
        lines=9, visible=False,
    )
    file_box = gr.File(
        label=f"Upload FASTA files  (max {MAX_FILES} · .fasta / .fa / .faa / .txt)",
        file_count="multiple",
        file_types=[".fasta", ".fa", ".faa", ".txt"],
        visible=False,
    )

    # ── embedding options ────────────────────────────────────────────────────────
    gr.Markdown("---\n### Embedding options")

    with gr.Row():
        emb_radio = gr.Radio(
            choices=[EMB_PER_PROTEIN, EMB_PER_AA],
            value=EMB_PER_PROTEIN,
            label="Embedding type",
        )
        fmt_radio = gr.Radio(
            choices=[FMT_PT, FMT_H5],
            value=FMT_PT,
            label="Download format",
        )

    emb_info = gr.Markdown(
        "> **Per-protein (mean pool)** — averages all residue embeddings into one "
        "fixed-size vector of shape `[hidden_dim]` (320 for 8M).  \n"
        "> Use for sequence-level tasks: classification, clustering, similarity search."
    )

    fmt_info = gr.Markdown(
        "> **`.pt` (PyTorch)** — load with `torch.load(path)`, returns a `dict[str, Tensor]`.  \n"
        "> **`.h5` (HDF5)** — load with `h5py.File(path)`, each key is a sequence ID."
    )

    # ── generate ─────────────────────────────────────────────────────────────────
    gr.Markdown("---")
    generate_btn = gr.Button("Generate embeddings", variant="primary", size="lg")
    status_box   = gr.Textbox(label="Status / log", lines=7, interactive=False)

    gr.Markdown("### Download")
    file_dl = gr.File(label="Embedding file", visible=False)

    gr.Markdown(
        "---\n"
        "**Model:** [`dipayan26/PlantPLM-8M`](https://huggingface.co/dipayan26/PlantPLM-8M) · "
        "**Base:** `facebook/esm2_t6_8M_UR50D` · "
        "**Training data:** 19.9M Viridiplantae proteins · "
        "**Code:** [GitHub](https://github.com/Dipayan26/Plant-Protein-BERT)"
    )

    # ── event handlers ────────────────────────────────────────────────────────────

    def toggle_inputs(method: str):
        is_single = method == "Single sequence"
        return (
            gr.update(visible=is_single),
            gr.update(visible=is_single),
            gr.update(visible=(method == "Paste FASTA")),
            gr.update(visible=(method == "Upload files")),
        )

    input_radio.change(
        toggle_inputs, inputs=input_radio,
        outputs=[single_box, example_btn, fasta_box, file_box],
    )

    example_btn.click(lambda: EXAMPLE_SEQ, outputs=single_box)

    def on_model_change(key: str):
        if MODEL_REGISTRY.get(key) is None:
            return gr.update(
                value="> ⚠️ Not yet available. Only **PlantPLM-8M** is currently active.",
                visible=True,
            )
        return gr.update(visible=False)

    model_dropdown.change(on_model_change, inputs=model_dropdown, outputs=model_warn)

    def on_emb_change(emb_type: str):
        if EMB_PER_AA in emb_type:
            desc = (
                "> **Per-amino-acid** — one vector per residue, shape `[L × hidden_dim]`.  \n"
                "> CLS and EOS tokens are excluded; `L` = actual sequence length.  \n"
                "> Use for residue-level tasks: binding site prediction, contact maps, annotation."
            )
        else:
            desc = (
                "> **Per-protein (mean pool)** — averages all residue embeddings into one "
                "fixed-size vector of shape `[hidden_dim]` (320 for 8M).  \n"
                "> Use for sequence-level tasks: classification, clustering, similarity search."
            )
        return gr.update(value=desc)

    emb_radio.change(on_emb_change, inputs=emb_radio, outputs=emb_info)

    def run(model_key, method, single, fasta_text, files, emb_type, fmt):
        status, path = generate_embeddings(
            model_key, method, single, fasta_text, files, emb_type, fmt
        )
        return status, gr.update(value=path, visible=path is not None)

    generate_btn.click(
        run,
        inputs=[model_dropdown, input_radio, single_box, fasta_box, file_box, emb_radio, fmt_radio],
        outputs=[status_box, file_dl],
    )

demo.launch()
