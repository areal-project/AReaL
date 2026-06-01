import argparse
import os
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from flask import Flask, jsonify, render_template_string, request
from transformers import AutoTokenizer

from prefix_tree_utils_v1 import PrefixTreeDataCollatorForCompletionOnlyLM


DEFAULT_SPECIAL_TOKENS = [
    "<Think>",
    "</Think>",
    "<Parallel>",
    "</Parallel>",
    "<Outlines>",
    "</Outlines>",
    "<Outline>",
    "</Outline>",
    "<Trial>",
    "</Trial>",
    "<Subtask>",
    "</Subtask>",
    "<Thread>",
    "</Thread>",
    "<Conclusion>",
    "</Conclusion>",
]


def _get_templates(template_name: str) -> Tuple[str, str, str]:
    """
    Return (instruction_template, response_template, pad_token) for the given template.
    """
    template_name = template_name.lower()
    if template_name == "qwen":
        return "<|im_start|>user", "<|im_start|>assistant\n", "<|fim_pad|>"
    if template_name == "llama":
        return "<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|reserved_special_token_5|>"
    if template_name == "ds":
        return "<｜User｜>", "<｜Assistant｜>", "<|fim_pad|>"
    raise ValueError(f"Unsupported template '{template_name}'. Options: qwen, llama, ds.")


def build_tokenizer(model_name: str, template_name: str) -> AutoTokenizer:
    """
    Load tokenizer locally (no weights) and ensure parallel reasoning special tokens exist.
    """
    instruction_template, response_template, pad_token = _get_templates(template_name)
    _ = instruction_template, response_template  # Unused, but keeps the relationship explicit.

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )

    added = tokenizer.add_special_tokens({"additional_special_tokens": DEFAULT_SPECIAL_TOKENS})
    if added:
        # Make sure tokenizer knows about newly added pad token if it was missing.
        pass

    if tokenizer.pad_token is None:
        # Prefer template-specific pad token; fall back to eos if needed.
        if pad_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [pad_token]})
        tokenizer.pad_token = pad_token

    return tokenizer


def build_collator(tokenizer: AutoTokenizer, template_name: str, max_length: int):
    instruction_template, response_template, _ = _get_templates(template_name)
    return PrefixTreeDataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        max_length=max_length,
        tokenizer=tokenizer,
        mlm=False,
    )


@lru_cache(maxsize=1)
def _load_dataset(dataset_path: str, text_field: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    if text_field not in df.columns:
        raise ValueError(f"Text field '{text_field}' not found in dataset columns: {list(df.columns)}")
    return df


def prepare_sample(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    collator: PrefixTreeDataCollatorForCompletionOnlyLM,
    index: int,
    text_field: str,
    max_tokens: Optional[int],
) -> Dict:
    row = df.iloc[index]
    raw_text = row[text_field]

    encoded = tokenizer(
        raw_text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    examples = [{"input_ids": encoded["input_ids"][0].tolist()}]
    batch = collator.torch_call(examples)

    input_ids = batch["input_ids"][0]
    positions = batch["position_ids"][0]
    # attention_mask is (batch, heads=1, L, L); convert to allow/deny mask
    attention = (batch["attention_mask"][0, 0] == 0).to(torch.int)

    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]
        positions = positions[:max_tokens]
        attention = attention[:max_tokens, :max_tokens]

    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    meta = {
        "index": int(index),
        "uuid": row.get("uuid"),
        "num_tokens": len(tokens),
    }

    return {
        "meta": meta,
        "tokens": tokens,
        "token_ids": input_ids.tolist(),
        "position_ids": positions.tolist(),
        "attention": attention.tolist(),
    }


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Parallel Reasoning Flattened Prefix-Tree Visualizer</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --accent: #1890ff;
      --text: #2c3e50;
      --muted: #6b7280;
      --danger: #f59e0b;
      --grid-on: #3b82f6;
      --grid-off: #e5e7eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
      color: var(--text);
      font-family: "Space Grotesk", "Fira Sans", "Helvetica Neue", Arial, sans-serif;
      line-height: 1.6;
    }
    header {
      padding: 20px 24px 10px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
    }
    h1 {
      margin: 0;
      font-weight: 700;
      letter-spacing: 0.3px;
    }
    .subtitle { color: var(--muted); margin-top: 6px; }
    main { padding: 16px 24px 28px; }
    .panel {
      background: var(--panel);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
    label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 4px; }
    input[type="number"], input[type="text"] {
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #d1d5db;
      background: #ffffff;
      color: var(--text);
      min-width: 120px;
      font-family: inherit;
    }
    input[type="checkbox"] {
      cursor: pointer;
      width: 16px;
      height: 16px;
      margin-right: 6px;
      vertical-align: middle;
    }
    button {
      background: linear-gradient(135deg, #1890ff, #0ea5e9);
      color: #ffffff;
      border: none;
      padding: 10px 16px;
      border-radius: 10px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.1s ease, box-shadow 0.1s ease;
      box-shadow: 0 2px 8px rgba(24,144,255,0.2);
    }
    button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(24,144,255,0.3); }
    .info { color: var(--muted); font-size: 14px; }
    .tokens {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 12px;
      background: #f9fafb;
      border-radius: 10px;
      min-height: 60px;
      border: 1px solid #e5e7eb;
    }
    .token {
      padding: 6px 8px;
      border-radius: 8px;
      background: #ffffff;
      cursor: pointer;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      color: var(--text);
      border: 1px solid #e5e7eb;
      transition: background 0.1s ease, border 0.1s ease;
      white-space: pre;
    }
    .token:hover { background: #e0f2fe; border-color: #0ea5e9; }
    .token.active { background: #dbeafe; border-color: var(--accent); font-weight: bold; }
    .token.tag {
      background: #fef3c7;
      border-color: #f59e0b;
      color: #d97706;
      font-weight: 600;
    }
    .token.tag:hover { background: #fde68a; }
    .ellipsis {
      padding: 6px 12px;
      border-radius: 8px;
      background: #dbeafe;
      cursor: pointer;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      color: #1e40af;
      border: 1px dashed #60a5fa;
      transition: all 0.1s ease;
      font-weight: 600;
    }
    .ellipsis:hover {
      background: #bfdbfe;
      border-color: #3b82f6;
      transform: scale(1.05);
    }
    .span-boundary {
      padding: 4px 8px;
      border-radius: 6px;
      background: #f3f4f6;
      color: var(--muted);
      font-size: 11px;
      font-family: "Space Grotesk", sans-serif;
      border: 1px solid #e5e7eb;
      user-select: none;
    }
    .grid-container {
      background: #f9fafb;
      border-radius: 10px;
      border: 1px solid #e5e7eb;
      overflow-x: auto;
      position: relative;
    }
    .grid-wrapper {
      display: flex;
      padding: 10px;
    }
    .position-labels {
      display: flex;
      flex-direction: column;
      margin-right: 12px;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      color: var(--text);
      font-weight: 500;
      min-width: 40px;
    }
    .position-label {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      padding: 0 8px 0 4px;
      cursor: pointer;
      transition: all 0.1s ease;
      background: #ffffff;
      margin-bottom: 1px;
      border-radius: 3px;
    }
    .position-label:hover {
      color: var(--accent);
      background: #e0f2fe;
      transform: translateX(-2px);
    }
    .position-label.active {
      color: #ffffff;
      background: var(--danger);
      font-weight: bold;
    }
    .grid {
      display: grid;
      gap: 0;
    }
    .cell {
      border-radius: 1px;
      cursor: pointer;
      transition: transform 0.05s ease;
    }
    .cell:hover { transform: scale(1.2); }
    .cell.on { background: var(--grid-on); }
    .cell.off { background: var(--grid-off); }
    .cell.highlighted-row { box-shadow: 0 0 0 2px var(--danger); }
    .zoom-controls {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
    }
    .zoom-btn {
      background: #ffffff;
      color: var(--text);
      border: 1px solid #d1d5db;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.1s ease;
    }
    .zoom-btn:hover { background: #f3f4f6; }
    .row-label {
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    .chip {
      background: #ffffff;
      border-radius: 10px;
      padding: 10px 12px;
      border: 1px solid #e5e7eb;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Parallel Reasoning Flattened Prefix-Tree Visualizer</h1>
    <div class="subtitle">
      This tool visualizes the attention mask and position IDs for samples using a flattened prefix-tree structure in parallel reasoning tasks.<br/>
      Dataset: {{ dataset_label }} | Text field: {{ text_field }} | Model: {{ model_name }}
    </div>
  </header>
  <main>
    <div class="panel">
      <form class="controls" onsubmit="loadSample(event)">
        <div>
          <label for="idx">Sample index</label>
          <input id="idx" name="idx" type="number" value="0" min="0" />
        </div>
        <div>
          <label for="limit">Max tokens (optional)</label>
          <input id="limit" name="limit" type="number" placeholder="e.g. 512" />
        </div>
        <button type="submit">Load sample</button>
        <div class="info" id="status"></div>
      </form>
      <div class="zoom-controls">
        <label>Cell size:</label>
        <button type="button" class="zoom-btn" onclick="adjustZoom(-1)">−</button>
        <span id="zoom-level" style="min-width: 60px; text-align: center;">6px</span>
        <button type="button" class="zoom-btn" onclick="adjustZoom(1)">+</button>
        <button type="button" class="zoom-btn" onclick="resetZoom()">Reset</button>
        <span style="margin-left: 16px; color: var(--muted);">|</span>
        <label style="margin-left: 8px;">
          <input type="checkbox" id="abbreviate" onchange="toggleAbbreviate()" checked />
          Abbreviated mode
        </label>
        <label style="margin-left: 8px;">Show:</label>
        <input id="context-size" type="number" value="3" min="1" max="20" style="width: 60px;" onchange="renderAll()" />
        <span style="font-size: 12px; color: var(--muted);">tokens per span edge</span>
      </div>
    </div>

    <div class="panel">
      <div class="meta" id="meta"></div>
    </div>

    <div class="panel">
      <div class="row-label">Tokens (click to inspect row)</div>
      <div id="tokens" class="tokens"></div>
    </div>

    <div class="panel">
      <div class="row-label" id="mask-label">Attention matrix (rows: tokens querying, columns: tokens being attended to)</div>
      <div class="grid-container">
        <div class="grid-wrapper">
          <div id="position-labels" class="position-labels"></div>
          <div id="grid" class="grid"></div>
        </div>
      </div>
    </div>
  </main>

  <script>
    let current = null;
    let selectedRow = 0;
    let cellSize = 6;
    let abbreviated = true;
    let expandedSpans = new Set();
    let visibleIndices = [];

    // Clean token display (handle Ġ for space, etc.)
    function cleanToken(tok) {
      return tok.replace(/^Ġ/, ' ').replace(/^Ċ/, '\\n').replace(/^Ï/, ' ');
    }

    // Check if token is an XML tag
    function isTag(tok) {
      const cleaned = cleanToken(tok).trim();
      return cleaned.startsWith('<') && cleaned.endsWith('>');
    }

    // Parse tokens into spans based on XML tags
    function parseSpans(tokens) {
      const spans = [];
      let currentSpan = { start: 0, end: 0, type: 'text' };

      for (let i = 0; i < tokens.length; i++) {
        const tok = tokens[i];
        if (isTag(tok)) {
          // End current span
          if (currentSpan.start !== i) {
            currentSpan.end = i;
            spans.push(currentSpan);
          }
          // Add tag as its own span
          spans.push({ start: i, end: i + 1, type: 'tag', tag: cleanToken(tok).trim() });
          currentSpan = { start: i + 1, end: i + 1, type: 'text' };
        }
      }
      // Add final span
      if (currentSpan.start < tokens.length) {
        currentSpan.end = tokens.length;
        spans.push(currentSpan);
      }
      return spans.filter(s => s.start < s.end);
    }

    // Calculate visible token indices in abbreviated mode
    function calculateVisibleIndices() {
      if (!current || !abbreviated) {
        visibleIndices = current ? Array.from({length: current.tokens.length}, (_, i) => i) : [];
        return;
      }

      const contextSize = parseInt(document.getElementById("context-size").value) || 3;
      const spans = parseSpans(current.tokens);
      const visible = [];

      spans.forEach((span, spanIdx) => {
        if (span.type === 'tag') {
          visible.push(span.start);
        } else if (expandedSpans.has(spanIdx)) {
          // Show all tokens in expanded span
          for (let i = span.start; i < span.end; i++) {
            visible.push(i);
          }
        } else {
          const len = span.end - span.start;
          if (len <= contextSize * 2) {
            // Show all if span is small
            for (let i = span.start; i < span.end; i++) {
              visible.push(i);
            }
          } else {
            // Show first and last contextSize tokens
            for (let i = span.start; i < span.start + contextSize; i++) {
              visible.push(i);
            }
            for (let i = span.end - contextSize; i < span.end; i++) {
              visible.push(i);
            }
          }
        }
      });

      visibleIndices = visible;
    }

    async function loadSample(evt) {
      if (evt) evt.preventDefault();
      const idx = document.getElementById("idx").value || 0;
      const limit = document.getElementById("limit").value;
      const status = document.getElementById("status");
      status.textContent = "Loading…";
      try {
        const url = `/api/sample?index=${idx}` + (limit ? `&limit=${limit}` : "");
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(await res.text());
        }
        current = await res.json();
        selectedRow = 0;
        expandedSpans.clear();
        renderAll();
        status.textContent = "";
      } catch (err) {
        status.textContent = err.message || "Error";
      }
    }

    function toggleAbbreviate() {
      abbreviated = document.getElementById("abbreviate").checked;
      expandedSpans.clear();
      renderAll();
    }

    function renderAll() {
      if (!current) return;
      calculateVisibleIndices();
      renderMeta();
      renderTokens();
      renderGrid();
    }

    function renderMeta() {
      if (!current) return;
      const m = current.meta || {};
      const meta = document.getElementById("meta");
      meta.innerHTML = `
        <div class="chip"><strong>Index</strong><br>${m.index}</div>
        <div class="chip"><strong>UUID</strong><br>${m.uuid ?? "N/A"}</div>
        <div class="chip"><strong>Tokens</strong><br>${current.tokens.length}</div>
      `;
    }

    function renderTokens() {
      if (!current) return;
      const container = document.getElementById("tokens");
      container.innerHTML = "";

      if (!abbreviated) {
        // Full mode: show all tokens
        current.tokens.forEach((tok, i) => {
          const span = document.createElement("span");
          const cleaned = cleanToken(tok);
          span.className = "token" + (i === selectedRow ? " active" : "") + (isTag(tok) ? " tag" : "");
          span.textContent = cleaned;
          span.title = `idx=${i} | token_id=${current.token_ids[i]} | pos=${current.position_ids[i]} | raw="${tok}"`;
          span.onclick = () => {
            selectedRow = i;
            renderAll();
          };
          container.appendChild(span);
        });
        return;
      }

      // Abbreviated mode
      const contextSize = parseInt(document.getElementById("context-size").value) || 3;
      const spans = parseSpans(current.tokens);

      spans.forEach((span, spanIdx) => {
        if (span.type === 'tag') {
          // Render tag token
          const tok = current.tokens[span.start];
          const elem = document.createElement("span");
          elem.className = "token tag" + (span.start === selectedRow ? " active" : "");
          elem.textContent = cleanToken(tok);
          elem.title = `idx=${span.start} | token_id=${current.token_ids[span.start]} | pos=${current.position_ids[span.start]}`;
          elem.onclick = () => {
            selectedRow = span.start;
            renderAll();
          };
          container.appendChild(elem);
        } else {
          const len = span.end - span.start;
          const isExpanded = expandedSpans.has(spanIdx);

          if (len <= contextSize * 2 || isExpanded) {
            // Show all tokens
            for (let i = span.start; i < span.end; i++) {
              const tok = current.tokens[i];
              const elem = document.createElement("span");
              elem.className = "token" + (i === selectedRow ? " active" : "");
              elem.textContent = cleanToken(tok);
              elem.title = `idx=${i} | token_id=${current.token_ids[i]} | pos=${current.position_ids[i]}`;
              elem.onclick = () => {
                selectedRow = i;
                renderAll();
              };
              container.appendChild(elem);
            }
            if (isExpanded && len > contextSize * 2) {
              // Add collapse button
              const collapse = document.createElement("span");
              collapse.className = "ellipsis";
              collapse.textContent = "[collapse]";
              collapse.title = `Collapse span (${len} tokens)`;
              collapse.onclick = () => {
                expandedSpans.delete(spanIdx);
                renderAll();
              };
              container.appendChild(collapse);
            }
          } else {
            // Show first contextSize tokens
            for (let i = span.start; i < span.start + contextSize; i++) {
              const tok = current.tokens[i];
              const elem = document.createElement("span");
              elem.className = "token" + (i === selectedRow ? " active" : "");
              elem.textContent = cleanToken(tok);
              elem.title = `idx=${i} | token_id=${current.token_ids[i]} | pos=${current.position_ids[i]}`;
              elem.onclick = () => {
                selectedRow = i;
                renderAll();
              };
              container.appendChild(elem);
            }

            // Add ellipsis
            const ellipsis = document.createElement("span");
            ellipsis.className = "ellipsis";
            const hiddenCount = len - contextSize * 2;
            ellipsis.textContent = `... (${hiddenCount} tokens)`;
            ellipsis.title = `Click to expand ${hiddenCount} hidden tokens`;
            ellipsis.onclick = () => {
              expandedSpans.add(spanIdx);
              renderAll();
            };
            container.appendChild(ellipsis);

            // Show last contextSize tokens
            for (let i = span.end - contextSize; i < span.end; i++) {
              const tok = current.tokens[i];
              const elem = document.createElement("span");
              elem.className = "token" + (i === selectedRow ? " active" : "");
              elem.textContent = cleanToken(tok);
              elem.title = `idx=${i} | token_id=${current.token_ids[i]} | pos=${current.position_ids[i]}`;
              elem.onclick = () => {
                selectedRow = i;
                renderAll();
              };
              container.appendChild(elem);
            }
          }
        }
      });
    }

    function adjustZoom(delta) {
      cellSize = Math.max(2, Math.min(20, cellSize + delta));
      document.getElementById("zoom-level").textContent = cellSize + "px";
      renderGrid();
    }

    function resetZoom() {
      cellSize = 6;
      document.getElementById("zoom-level").textContent = cellSize + "px";
      renderGrid();
    }

    function renderGrid() {
      if (!current) return;
      const grid = document.getElementById("grid");
      const posLabels = document.getElementById("position-labels");
      const totalRows = current.attention.length;
      const totalCols = current.attention[0]?.length || 1;

      // In abbreviated mode, only show visible rows/cols
      const displayRows = abbreviated ? visibleIndices : Array.from({length: totalRows}, (_, i) => i);
      const displayCols = abbreviated ? visibleIndices : Array.from({length: totalCols}, (_, i) => i);

      grid.style.gridTemplateColumns = `repeat(${displayCols.length}, ${cellSize}px)`;
      grid.innerHTML = "";
      posLabels.innerHTML = "";

      displayRows.forEach(rowIdx => {
        // Add position label
        const posLabel = document.createElement("div");
        posLabel.className = "position-label" + (rowIdx === selectedRow ? " active" : "");
        posLabel.textContent = current.position_ids[rowIdx];
        posLabel.style.height = cellSize + "px";
        posLabel.onclick = () => {
          selectedRow = rowIdx;
          renderAll();
        };
        posLabels.appendChild(posLabel);

        // Add attention cells
        const row = current.attention[rowIdx] || [];
        displayCols.forEach(colIdx => {
          const val = row[colIdx];
          const cell = document.createElement("div");
          const isHighlighted = rowIdx === selectedRow;
          cell.className = "cell " + (val ? "on" : "off") + (isHighlighted ? " highlighted-row" : "");
          cell.style.width = cellSize + "px";
          cell.style.height = cellSize + "px";
          cell.title = `row ${rowIdx} (pos=${current.position_ids[rowIdx]}) -> col ${colIdx} (pos=${current.position_ids[colIdx]}) | allow=${!!val}`;
          cell.onclick = () => {
            selectedRow = rowIdx;
            renderAll();
          };
          grid.appendChild(cell);
        });
      });

      const lbl = document.getElementById("mask-label");
      const modeText = abbreviated ? `Abbreviated (${displayRows.length}/${totalRows} rows)` : "Full";
      lbl.textContent = `Attention matrix [${modeText}] - ${totalRows}×${totalCols} - Cell size: ${cellSize}px - Click to select row`;
    }

    loadSample();
  </script>
</body>
</html>
"""


def create_app(args) -> Flask:
    tokenizer = build_tokenizer(args.model_name, args.template_name)
    collator = build_collator(tokenizer, args.template_name, args.max_length)
    df = _load_dataset(args.dataset_path, args.text_field)

    app = Flask(__name__)

    @app.route("/api/sample")
    def get_sample():
        try:
            idx = int(request.args.get("index", 0))
        except ValueError:
            return ("Invalid index", 400)
        if idx < 0 or idx >= len(df):
            return (f"Index out of range. Dataset has {len(df)} rows.", 400)

        limit_raw = request.args.get("limit")
        max_tokens = int(limit_raw) if limit_raw else None

        sample = prepare_sample(
            df=df,
            tokenizer=tokenizer,
            collator=collator,
            index=idx,
            text_field=args.text_field,
            max_tokens=max_tokens,
        )
        return jsonify(sample)

    @app.route("/")
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            dataset_label=os.path.basename(args.dataset_path),
            text_field=args.text_field,
            model_name=args.model_name,
        )

    @app.route("/health")
    def health():
        return {"status": "ok"}

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention mask and position ids for parallel reasoning samples.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to a parquet file containing samples (e.g., data_generation/dataset/.../train.parquet).",
    )
    parser.add_argument(
        "--text-field",
        default="qwen_text",
        help="Field/column containing the serialized text to tokenize (default: qwen_text).",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-8B-131072",
        help="Tokenizer to use (default: Qwen/Qwen3-8B-131072).",
    )
    parser.add_argument(
        "--template-name",
        default="qwen",
        choices=["qwen", "llama", "ds"],
        help="Template type used for training (default: qwen).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=40960,
        help="Truncate to this many tokens before visualizing (keeps attention grid reasonable).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for the Flask server.")
    parser.add_argument("--port", type=int, default=8008, help="Port for the Flask server.")
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
