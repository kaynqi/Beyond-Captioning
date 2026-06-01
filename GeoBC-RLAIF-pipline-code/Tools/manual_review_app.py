# -*- coding: utf-8 -*-
"""
manual_review_app.py

1) 左侧双时相图片始终保持在网页视口内，不再随右侧长答案整体滚动
2) 保存人工选择后，立即在右侧顶部显示：
   - 当前样本是否与 AI 最终 best 一致
   - AI 选择的模型、槽位和答案
   - Overall agreement rate、分模型一致率

"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for


# =========================
# 在这里填写固定路径
# =========================
CONFIG = {
    "blind_review_jsonl": r"check/check-change-train/blind_review.jsonl",
    "answer_key_csv": r"check/check-change-train/answer_key.csv",
    "image_dir_t1": r"check/check-change-train/pre_sampled_images",
    "image_dir_t2": r"check/check-change-train/post_sampled_images",
    "output_dir": r"check/check-change-train/after-check-result",
    "host": "127.0.0.1",
    "port": 5000,
    "normalize_model_name": True,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Manual Blind Review (Dual Time Phase + Live Agreement Rate)</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --card: #ffffff;
      --line: #e5e7eb;
      --muted: #6b7280;
      --text: #111827;
      --ok: #047857;
      --warn: #b45309;
      --bad: #b91c1c;
      --chip: #eef2ff;
      --chip-text: #3730a3;
    }

    * { box-sizing: border-box; }
    html, body {
      height: 100%;
      margin: 0;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    .page {
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .header-wrap {
      flex: 0 0 auto;
      padding: 14px 16px 10px 16px;
    }

    .shell {
      flex: 1 1 auto;
      min-height: 0;
      padding: 0 16px 16px 16px;
    }

    .header-card, .pane, .summary-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .header-card {
      padding: 16px 20px;
    }

    .header-flex {
      display: flex;
      gap: 20px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }

    .main-grid {
      display: grid;
      grid-template-columns: minmax(560px, 0.95fr) minmax(760px, 1.15fr);
      gap: 16px;
      height: 100%;
      min-height: 0;
    }

    .pane {
      height: 100%;
      min-height: 0;
      overflow-y: auto;
      padding: 16px;
    }

    .meta {
      color: #4b5563;
      line-height: 1.8;
      font-size: 14px;
    }

    .progress {
      font-weight: 700;
      font-size: 18px;
      margin-bottom: 8px;
    }

    h1, h2, h3, h4 {
      margin-top: 0;
    }

    .image-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    .image-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fafafa;
    }

    .image-title {
      font-weight: 700;
      margin-bottom: 10px;
    }

    .img-box {
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 280px;
      max-height: 68vh;
      background: white;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }

    .preview {
      width: 100%;
      height: auto;
      max-height: 66vh;
      object-fit: contain;
      background: white;
    }

    .path-tip {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
      word-break: break-all;
    }

    .section {
      margin-bottom: 16px;
    }

    .option {
      border: 1px solid #d1d5db;
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 14px;
      background: #fcfcfd;
    }

    .option:hover {
      border-color: #9ca3af;
      background: #ffffff;
    }

    .option-title {
      font-weight: 700;
      margin-bottom: 8px;
    }

    .option-meta {
      color: var(--muted);
      font-size: 12px;
      margin-left: 8px;
    }

    .answer-box {
      white-space: pre-wrap;
      line-height: 1.65;
      font-size: 14px;
      color: #1f2937;
      background: #f9fafb;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-top: 8px;
    }

    .small {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }

    input[type="text"], textarea {
      width: 100%;
      border: 1px solid #d1d5db;
      border-radius: 12px;
      padding: 12px;
      font-size: 14px;
      margin-bottom: 12px;
      background: white;
    }

    textarea {
      min-height: 110px;
      resize: vertical;
    }

    .btn-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 10px;
    }

    button, .btn-link {
      border: none;
      background: #111827;
      color: white;
      border-radius: 12px;
      padding: 12px 18px;
      cursor: pointer;
      text-decoration: none;
      font-size: 14px;
      display: inline-block;
    }

    .btn-secondary {
      background: white;
      color: #111827;
      border: 1px solid #d1d5db;
    }

    .warn {
      color: var(--warn);
      font-size: 13px;
      margin-top: 8px;
    }

    .ok {
      color: var(--ok);
      font-size: 13px;
      margin-top: 8px;
    }

    .bad {
      color: var(--bad);
      font-size: 13px;
      margin-top: 8px;
    }

    .chip {
      display: inline-block;
      background: var(--chip);
      color: var(--chip-text);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      margin-right: 8px;
      margin-top: 8px;
    }

    .timer-panel {
      min-width: 230px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fafafa;
    }

    .timer-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }

    .timer-value {
      font-size: 28px;
      font-weight: 800;
      letter-spacing: 0.04em;
      font-variant-numeric: tabular-nums;
    }

    .timer-tip {
      color: var(--warn);
      font-size: 12px;
      line-height: 1.5;
      margin-top: 6px;
    }

    .timer-ok {
      color: var(--ok);
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
    }

    .stat-box {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: #fafafa;
    }

    .stat-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }

    .stat-value {
      font-size: 22px;
      font-weight: 800;
    }

    .reveal-box {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      background: #fbfbff;
    }

    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
    }

    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }

    th, td {
      border-bottom: 1px solid #f0f0f0;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }

    th {
      background: #fafafa;
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 1;
    }

    .summary-card {
      padding: 24px;
      max-width: 1200px;
      margin: 24px auto;
    }

    .dis-item {
      padding: 12px 0;
      border-bottom: 1px solid #f0f0f0;
    }

    @media (max-width: 1280px) {
      html, body {
        overflow: auto;
      }
      .page {
        height: auto;
      }
      .shell {
        padding-bottom: 16px;
      }
      .main-grid {
        grid-template-columns: 1fr;
        height: auto;
      }
      .pane {
        height: auto;
        overflow: visible;
      }
      .image-grid {
        grid-template-columns: 1fr;
      }
      .stats-grid {
        grid-template-columns: 1fr 1fr;
      }
      .img-box {
        max-height: none;
      }
      .preview {
        max-height: 60vh;
      }
    }
  </style>
</head>
<body>
<div class="page">

  {% if mode == "review" %}
    <div class="header-wrap">
      <div class="header-card">
        <div class="header-flex">
          <div>
            <div class="progress">Progress: {{ reviewed_count }} / {{ total_count }}</div>
            <div class="meta">
              Current sample: <strong>{{ idx + 1 }}</strong> / {{ total_count }}<br>
              task_id: <strong>{{ item.task_id }}</strong><br>
              blind_review image filename: {{ item.image_filename }}
            </div>

            {% if item.missing_models %}
              <div class="warn">Missing models: {{ item.missing_models | join(", ") }}</div>
            {% endif %}
            {% if not item.t1_found %}
              <div class="warn">2024 time-phase image not found (matched by the first 6 characters of task_id)</div>
            {% endif %}
            {% if not item.t2_found %}
              <div class="warn">2025 time-phase image not found (matched by the first 6 characters of task_id)</div>
            {% endif %}
            {% if not item.has_answer_key %}
              <div class="warn">answer_key reference not found; this item cannot be compared for AI agreement</div>
            {% endif %}
            {% if saved %}
              <div class="ok">The current manual selection has been saved. The AI final selection and agreement statistics are shown below.</div>
            {% endif %}
          </div>

          <div>
            <div class="timer-panel" aria-label="Current-question dwell timer">
              <div class="timer-label">Current-question timer</div>
              <div id="question-timer" class="timer-value">00:00</div>
              <div id="timer-tip" class="timer-tip">Select the most appropriate answer. Please spend no less than 120 seconds to read the question carefully.</div>
            </div>
            <span class="chip">Overall agreement rate: {{ stats.agreement_rate_percent }}</span>
            <span class="chip">Compared: {{ stats.comparable_count }}</span>
            <span class="chip">Agreed: {{ stats.agreement_count }}</span>
            <span class="chip">Disagreed: {{ stats.disagreement_count }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="shell">
      <div class="main-grid">
        <div class="pane">
          <h3>Before/After Time-Phase Images</h3>

          <div class="image-grid">
            <div class="image-card">
              <div class="image-title">T1 / PRE_RGB</div>
              <div class="img-box">
                {% if item.t1_found %}
                  <img class="preview" src="{{ url_for('serve_dual_image', idx=idx, which='t1') }}" alt="t1 image">
                {% else %}
                  <div class="warn">Matching image not found</div>
                {% endif %}
              </div>
              <div class="path-tip">{{ item.t1_filename or '' }}</div>
            </div>

            <div class="image-card">
              <div class="image-title">T2 / POST_RGB</div>
              <div class="img-box">
                {% if item.t2_found %}
                  <img class="preview" src="{{ url_for('serve_dual_image', idx=idx, which='t2') }}" alt="t2 image">
                {% else %}
                  <div class="warn">Matching image not found</div>
                {% endif %}
              </div>
              <div class="path-tip">{{ item.t2_filename or '' }}</div>
            </div>
          </div>

          <div class="section" style="margin-top:16px;">
            <h4>Current Global Statistics</h4>
            <div class="stats-grid">
              <div class="stat-box">
                <div class="stat-label">Completed selections</div>
                <div class="stat-value">{{ stats.reviewed_count }}</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">Compared samples</div>
                <div class="stat-value">{{ stats.comparable_count }}</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">Agreed samples</div>
                <div class="stat-value">{{ stats.agreement_count }}</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">Agreement rate</div>
                <div class="stat-value">{{ stats.agreement_rate_percent }}</div>
              </div>
            </div>
          </div>

          <div class="section" style="margin-top:16px;">
            <h4>Statistics by AI Final Selected Model</h4>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>AI selected count</th>
                    <th>Manual agreement count</th>
                    <th>Agreement rate</th>
                  </tr>
                </thead>
                <tbody>
                  {% for row in stats.per_model_rows %}
                    <tr>
                      <td>{{ row.model_name }}</td>
                      <td>{{ row.total }}</td>
                      <td>{{ row.agree }}</td>
                      <td>{{ row.rate_percent }}</td>
                    </tr>
                  {% endfor %}
                  {% if not stats.per_model_rows %}
                    <tr><td colspan="4">No statistics available yet</td></tr>
                  {% endif %}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div class="pane">
          {% if show_reveal %}
            <div class="section">
              <div class="reveal-box">
                <h3>Current Sample: Manual Selection vs AI Final Selection</h3>

                {% if existing.agreed == 1 %}
                  <div class="ok"><strong>Result for this item: agrees with AI final best</strong></div>
                {% elif existing.agreed == 0 %}
                  <div class="bad"><strong>Result for this item: does not agree with AI final best</strong></div>
                {% else %}
                  <div class="warn"><strong>Result for this item: reference information is missing, so comparison is unavailable</strong></div>
                {% endif %}

                <div class="meta" style="margin-top:10px;">
                  Manual selection: 
                  <strong>{{ existing.selected_slot or '' }}</strong>
                  {% if existing.selected_model %} / {{ existing.selected_model }}{% endif %}
                  <br>
                  AI final selection: 
                  <strong>{{ existing.auto_best_slot or 'Unknown slot' }}</strong>
                  {% if existing.auto_best_model %} / {{ existing.auto_best_model }}{% endif %}
                </div>

                <div style="margin-top:14px;">
                  <div class="small">Manual selected answer</div>
                  <div class="answer-box">{{ existing.selected_answer or 'None' }}</div>
                </div>

                <div style="margin-top:14px;">
                  <div class="small">AI final selected answer</div>
                  <div class="answer-box">{{ existing.auto_best_answer or 'None' }}</div>
                </div>
              </div>
            </div>
          {% endif %}

          <div class="section">
            <h3>Select the answer you consider best</h3>
            <form id="review-form" method="post">
              {% for opt in item.blind_options %}
                <div class="option">
                  <label>
                    <input
                      type="radio"
                      name="selected_slot"
                      value="{{ opt.slot }}"
                      {% if existing and existing.selected_slot == opt.slot %}checked{% endif %}
                    >
                    <span class="option-title">Candidate {{ opt.slot }}</span>
                    <div class="answer-box">{{ opt.answer }}</div>
                  </label>
                </div>
              {% endfor %}

              <div class="small">Reviewer</div>
              <input
                type="text"
                name="reviewer"
                value="{{ existing.reviewer if existing else '' }}"
                placeholder="Enter name or ID"
              >

              <div class="small">Notes</div>
              <textarea
                name="note"
                placeholder="Optional: write down your reason for selection or any questions"
              >{{ existing.note if existing else '' }}</textarea>

              <div class="btn-row">
                <button type="submit">Save selection and view AI result</button>
                <a class="btn-link btn-secondary" href="{{ url_for('review', idx=(idx-1 if idx > 0 else 0)) }}">Previous</a>
                <a class="btn-link btn-secondary" href="{{ url_for('review', idx=(idx+1 if idx < total_count-1 else total_count-1)) }}">Next</a>
                <a class="btn-link btn-secondary" href="{{ url_for('summary') }}">View summary page</a>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>

  {% elif mode == "summary" %}
    <div class="summary-card">
      <h2>Manual Selection and AI Agreement Statistics</h2>

      <div class="stats-grid" style="margin-bottom:18px;">
        <div class="stat-box">
          <div class="stat-label">Total samples</div>
          <div class="stat-value">{{ stats.total_count }}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Completed manual selections</div>
          <div class="stat-value">{{ stats.reviewed_count }}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Compared samples</div>
          <div class="stat-value">{{ stats.comparable_count }}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Overall agreement rate</div>
          <div class="stat-value">{{ stats.agreement_rate_percent }}</div>
        </div>
      </div>

      <div class="btn-row">
        <a class="btn-link" href="{{ url_for('review', idx=stats.next_unreviewed_idx) }}">Continue reviewing</a>
        <a class="btn-link btn-secondary" href="{{ url_for('download_csv') }}">Download manual_selection.csv</a>
        <a class="btn-link btn-secondary" href="{{ url_for('download_json') }}">Download manual_selection.json</a>
      </div>

      <hr style="margin: 24px 0; border: none; border-top: 1px solid #e5e7eb;">

      <h3>Statistics by AI Final Selected Model</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>AI selected count</th>
              <th>Manual agreement count</th>
              <th>Disagreement count</th>
              <th>Agreement rate</th>
            </tr>
          </thead>
          <tbody>
            {% for row in stats.per_model_rows %}
              <tr>
                <td>{{ row.model_name }}</td>
                <td>{{ row.total }}</td>
                <td>{{ row.agree }}</td>
                <td>{{ row.disagree }}</td>
                <td>{{ row.rate_percent }}</td>
              </tr>
            {% endfor %}
            {% if not stats.per_model_rows %}
              <tr><td colspan="5">No statistics available yet</td></tr>
            {% endif %}
          </tbody>
        </table>
      </div>

      <hr style="margin: 24px 0; border: none; border-top: 1px solid #e5e7eb;">

      <h3>Saved Records (latest first, top 20)</h3>
      {% for row in stats.latest_rows %}
        <div class="dis-item">
          <div><strong>{{ row.task_id }}</strong></div>
          <div class="meta">
            Manual: {{ row.selected_slot }}{% if row.selected_model %} / {{ row.selected_model }}{% endif %}<br>
            AI: {{ row.auto_best_slot or 'Unknown slot' }}{% if row.auto_best_model %} / {{ row.auto_best_model }}{% endif %}<br>
            Agreed: {{ 'Yes' if row.agreed == 1 else ('No' if row.agreed == 0 else 'Unable to determine') }}<br>
            Reviewer: {{ row.reviewer or 'Not provided' }}
          </div>
        </div>
      {% endfor %}
    </div>
  {% endif %}

</div>
<script>
(function () {
  const timerEl = document.getElementById("question-timer");
  const tipEl = document.getElementById("timer-tip");
  const formEl = document.getElementById("review-form");
  if (!timerEl) return;

  const startedAt = Date.now();
  const warningThresholdMs = 2 * 60 * 1000;

  function formatElapsed(ms) {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    return `${minutes}:${seconds}`;
  }

  function updateTimer() {
    const elapsedMs = Date.now() - startedAt;
    timerEl.textContent = formatElapsed(elapsedMs);
    if (elapsedMs >= warningThresholdMs) {
      timerEl.classList.add("timer-ok");
      if (tipEl) {
        tipEl.textContent = "At least 2 minutes have elapsed. This timer is for reference only and will not be saved.";
      }
    }
  }

  updateTimer();
  window.setInterval(updateTimer, 1000);

  if (formEl) {
    formEl.addEventListener("submit", function (event) {
      const elapsedMs = Date.now() - startedAt;
      if (elapsedMs < warningThresholdMs) {
        const elapsedText = formatElapsed(elapsedMs);
        const ok = window.confirm(
          `Current-question dwell time is only ${elapsedText}, less than 2 minutes.\n\nDo you still want to save this selection?`
        );
        if (!ok) {
          event.preventDefault();
        }
      }
    });
  }
})();
</script>
</body>
</html>
"""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_model_name(name: str) -> str:
    s = normalize_text(name).lower()
    for ch in [" ", "_", "-"]:
        s = s.replace(ch, "")
    return s


def models_equal(a: str, b: str, do_normalize: bool) -> bool:
    if do_normalize:
        return normalize_model_name(a) == normalize_model_name(b)
    return normalize_text(a) == normalize_text(b)


def validate_config(cfg: Dict[str, Any]) -> None:
    required_keys = [
        "blind_review_jsonl",
        "answer_key_csv",
        "image_dir_t1",
        "image_dir_t2",
        "output_dir",
        "host",
        "port",
        "normalize_model_name",
    ]
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"CONFIG 缺少字段: {key}")

    blind_review_jsonl = Path(cfg["blind_review_jsonl"])
    answer_key_csv = Path(cfg["answer_key_csv"])
    image_dir_t1 = Path(cfg["image_dir_t1"])
    image_dir_t2 = Path(cfg["image_dir_t2"])

    if not blind_review_jsonl.exists():
        raise FileNotFoundError(f"blind_review.jsonl 不存在: {blind_review_jsonl}")
    if not answer_key_csv.exists():
        raise FileNotFoundError(f"answer_key.csv 不存在: {answer_key_csv}")
    if not image_dir_t1.exists():
        raise FileNotFoundError(f"T1 图片目录不存在: {image_dir_t1}")
    if not image_dir_t2.exists():
        raise FileNotFoundError(f"T2 图片目录不存在: {image_dir_t2}")

    port = int(cfg["port"])
    if port <= 0 or port > 65535:
        raise ValueError("port 必须在 1-65535 之间")


def first6_from_name(name: str) -> str:
    return Path(name).stem[:6]


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in sorted(image_dir.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            task_id = first6_from_name(p.name)
            if task_id and task_id not in index:
                index[task_id] = p
    return index


def load_blind_review_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def build_answer_key_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    index = {}
    for row in rows:
        task_id = normalize_text(row.get("task_id"))
        if task_id:
            index[task_id] = row
    return index


def load_manual_selection_json(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_manual_selection_json(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    cleaned: Dict[str, Dict[str, Any]] = {}
    for key, row in data.items():
        cleaned_row = dict(row)
        cleaned_row.pop("reviewed_at", None)
        cleaned[key] = cleaned_row

    with path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)


def export_manual_selection_csv(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    rows = sorted(data.values(), key=lambda x: x.get("task_id", ""))
    fieldnames = [
        "task_id",
        "image_filename",
        "t1_filename",
        "t2_filename",
        "selected_slot",
        "selected_model",
        "selected_answer",
        "auto_best_slot",
        "auto_best_model",
        "auto_best_answer",
        "agreed",
        "reviewer",
        "note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_answer_by_slot(blind_options: List[Dict[str, Any]], slot: str) -> str:
    for opt in blind_options:
        if normalize_text(opt.get("slot")).upper() == normalize_text(slot).upper():
            return normalize_text(opt.get("answer"))
    return ""


def find_slot_by_model(option_model_map: Dict[str, str], model_name: str, do_normalize: bool) -> str:
    for slot, mapped_model in option_model_map.items():
        if models_equal(mapped_model, model_name, do_normalize):
            return slot
    return ""


def attach_answer_key(
    items: List[Dict[str, Any]],
    answer_key_index: Dict[str, Dict[str, str]],
    do_normalize: bool,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []

    for item in items:
        task_id = normalize_text(item.get("task_id"))[:6]
        row = answer_key_index.get(task_id, {})
        option_model_map: Dict[str, str] = {}

        for key, value in row.items():
            if key.startswith("option_"):
                slot = key.replace("option_", "").strip().upper()
                if slot:
                    option_model_map[slot] = normalize_text(value)

        auto_best_model = normalize_text(row.get("auto_best_model"))
        auto_best_slot = find_slot_by_model(option_model_map, auto_best_model, do_normalize)
        auto_best_answer = find_answer_by_slot(item.get("blind_options", []), auto_best_slot)

        new_item = dict(item)
        new_item["has_answer_key"] = bool(row)
        new_item["option_model_map"] = option_model_map
        new_item["auto_best_model"] = auto_best_model
        new_item["auto_best_slot"] = auto_best_slot
        new_item["auto_best_answer"] = auto_best_answer
        enriched.append(new_item)

    return enriched


def attach_dual_time_images(
    items: List[Dict[str, Any]],
    index_t1: Dict[str, Path],
    index_t2: Dict[str, Path],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in items:
        task_id = normalize_text(item.get("task_id"))[:6]
        t1_path = index_t1.get(task_id)
        t2_path = index_t2.get(task_id)

        new_item = dict(item)
        new_item["t1_path"] = str(t1_path.resolve()) if t1_path else ""
        new_item["t2_path"] = str(t2_path.resolve()) if t2_path else ""
        new_item["t1_found"] = bool(t1_path)
        new_item["t2_found"] = bool(t2_path)
        new_item["t1_filename"] = t1_path.name if t1_path else ""
        new_item["t2_filename"] = t2_path.name if t2_path else ""
        enriched.append(new_item)
    return enriched


def compute_stats(items: List[Dict[str, Any]], selections: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total_count = len(items)
    reviewed_count = 0
    comparable_count = 0
    agreement_count = 0
    disagreement_count = 0
    next_unreviewed_idx = 0
    found_next = False

    per_model: Dict[str, Dict[str, Any]] = {}

    for idx, item in enumerate(items):
        task_id = normalize_text(item.get("task_id"))
        row = selections.get(task_id)

        if row and normalize_text(row.get("selected_slot")):
            reviewed_count += 1
        elif not found_next:
            next_unreviewed_idx = idx
            found_next = True

        if not row:
            continue

        agreed_val = row.get("agreed")
        auto_best_model = normalize_text(row.get("auto_best_model"))

        if agreed_val in [0, 1] and auto_best_model:
            comparable_count += 1

            if int(agreed_val) == 1:
                agreement_count += 1
            else:
                disagreement_count += 1

            if auto_best_model not in per_model:
                per_model[auto_best_model] = {
                    "model_name": auto_best_model,
                    "total": 0,
                    "agree": 0,
                    "disagree": 0,
                    "rate_percent": "0.00%",
                }

            per_model[auto_best_model]["total"] += 1
            if int(agreed_val) == 1:
                per_model[auto_best_model]["agree"] += 1
            else:
                per_model[auto_best_model]["disagree"] += 1

    if reviewed_count == total_count and total_count > 0:
        next_unreviewed_idx = total_count - 1

    reviewed_rate = f"{(reviewed_count / total_count * 100):.2f}%" if total_count else "0.00%"
    agreement_rate = (agreement_count / comparable_count) if comparable_count else 0.0
    agreement_rate_percent = f"{agreement_rate * 100:.2f}%"

    per_model_rows = list(per_model.values())
    for row in per_model_rows:
        total = row["total"]
        agree = row["agree"]
        row["rate_percent"] = f"{(agree / total * 100):.2f}%" if total else "0.00%"

    per_model_rows.sort(key=lambda x: (-x["total"], x["model_name"]))

    latest_rows = list(selections.values())[-20:][::-1]

    return {
        "total_count": total_count,
        "reviewed_count": reviewed_count,
        "reviewed_rate": reviewed_rate,
        "next_unreviewed_idx": next_unreviewed_idx,
        "comparable_count": comparable_count,
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "agreement_rate": agreement_rate,
        "agreement_rate_percent": agreement_rate_percent,
        "per_model_rows": per_model_rows,
        "latest_rows": latest_rows,
    }


def create_app(
    blind_review_jsonl: Path,
    answer_key_csv: Path,
    image_dir_t1: Path,
    image_dir_t2: Path,
    output_dir: Path,
    do_normalize: bool,
) -> Flask:
    app = Flask(__name__)

    raw_items = load_blind_review_jsonl(blind_review_jsonl)
    answer_key_index = build_answer_key_index(read_csv_rows(answer_key_csv))
    raw_items = attach_answer_key(raw_items, answer_key_index, do_normalize)

    index_t1 = build_image_index(image_dir_t1)
    index_t2 = build_image_index(image_dir_t2)
    items = attach_dual_time_images(raw_items, index_t1, index_t2)

    selection_json_path = output_dir / "manual_selection.json"
    selection_csv_path = output_dir / "manual_selection.csv"

    ensure_dir(output_dir)

    def get_selections() -> Dict[str, Dict[str, Any]]:
        return load_manual_selection_json(selection_json_path)

    def save_selections(data: Dict[str, Dict[str, Any]]) -> None:
        save_manual_selection_json(selection_json_path, data)
        export_manual_selection_csv(selection_csv_path, data)

    @app.route("/")
    def home():
        selections = get_selections()
        stats = compute_stats(items, selections)
        return redirect(url_for("review", idx=stats["next_unreviewed_idx"]))

    @app.route("/review/<int:idx>", methods=["GET", "POST"])
    def review(idx: int):
        if idx < 0 or idx >= len(items):
            abort(404)

        item = items[idx]
        task_id = normalize_text(item.get("task_id"))

        if request.method == "POST":
            selected_slot = normalize_text(request.form.get("selected_slot")).upper()
            reviewer = normalize_text(request.form.get("reviewer"))
            note = normalize_text(request.form.get("note"))

            if selected_slot:
                selected_answer = find_answer_by_slot(item.get("blind_options", []), selected_slot)
                selected_model = normalize_text(item.get("option_model_map", {}).get(selected_slot))
                auto_best_model = normalize_text(item.get("auto_best_model"))
                auto_best_slot = normalize_text(item.get("auto_best_slot"))
                auto_best_answer = normalize_text(item.get("auto_best_answer"))

                agreed: Optional[int] = None
                if selected_model and auto_best_model:
                    agreed = 1 if models_equal(selected_model, auto_best_model, do_normalize) else 0

                selections = get_selections()
                if task_id in selections:
                    selections.pop(task_id)
                selections[task_id] = {
                    "task_id": task_id,
                    "image_filename": normalize_text(item.get("image_filename")),
                    "t1_filename": normalize_text(item.get("t1_filename")),
                    "t2_filename": normalize_text(item.get("t2_filename")),
                    "selected_slot": selected_slot,
                    "selected_model": selected_model,
                    "selected_answer": selected_answer,
                    "auto_best_slot": auto_best_slot,
                    "auto_best_model": auto_best_model,
                    "auto_best_answer": auto_best_answer,
                    "agreed": agreed,
                    "reviewer": reviewer,
                    "note": note,
                }
                save_selections(selections)
                return redirect(url_for("review", idx=idx, saved=1))

        selections = get_selections()
        stats = compute_stats(items, selections)
        existing = selections.get(task_id)
        show_reveal = bool(existing and normalize_text(existing.get("selected_slot")))
        saved = request.args.get("saved") == "1"

        return render_template_string(
            HTML_TEMPLATE,
            mode="review",
            idx=idx,
            item=item,
            total_count=len(items),
            reviewed_count=stats["reviewed_count"],
            existing=existing,
            stats=stats,
            show_reveal=show_reveal,
            saved=saved,
        )

    @app.route("/dual-image/<int:idx>/<string:which>")
    def serve_dual_image(idx: int, which: str):
        if idx < 0 or idx >= len(items):
            abort(404)

        item = items[idx]
        path_str = item.get("t1_path", "") if which == "t1" else item.get("t2_path", "") if which == "t2" else ""
        if not path_str:
            abort(404, description="Image not found")

        image_path = Path(path_str)
        if not image_path.exists():
            abort(404, description="Image not found")

        return send_file(image_path)

    @app.route("/summary")
    def summary():
        selections = get_selections()
        stats = compute_stats(items, selections)
        return render_template_string(
            HTML_TEMPLATE,
            mode="summary",
            stats=stats,
        )

    @app.route("/download/manual_selection.csv")
    def download_csv():
        selections = get_selections()
        export_manual_selection_csv(selection_csv_path, selections)
        return send_file(selection_csv_path, as_attachment=True)

    @app.route("/download/manual_selection.json")
    def download_json():
        selections = get_selections()
        save_manual_selection_json(selection_json_path, selections)
        return send_file(selection_json_path, as_attachment=True)

    return app


def main():
    validate_config(CONFIG)

    blind_review_jsonl = Path(CONFIG["blind_review_jsonl"])
    answer_key_csv = Path(CONFIG["answer_key_csv"])
    image_dir_t1 = Path(CONFIG["image_dir_t1"])
    image_dir_t2 = Path(CONFIG["image_dir_t2"])
    output_dir = Path(CONFIG["output_dir"])
    host = str(CONFIG["host"])
    port = int(CONFIG["port"])
    do_normalize = bool(CONFIG["normalize_model_name"])

    ensure_dir(output_dir)

    print(f"[INFO] blind_review.jsonl: {blind_review_jsonl}")
    print(f"[INFO] answer_key.csv: {answer_key_csv}")
    print(f"[INFO] T1 dir: {image_dir_t1}")
    print(f"[INFO] T2 dir: {image_dir_t2}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] server: http://{host}:{port}")

    app = create_app(
        blind_review_jsonl=blind_review_jsonl,
        answer_key_csv=answer_key_csv,
        image_dir_t1=image_dir_t1,
        image_dir_t2=image_dir_t2,
        output_dir=output_dir,
        do_normalize=do_normalize,
    )
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()