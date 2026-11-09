"""Colab GPU runtime: nvidia-smi sampler + llama.cpp /metrics reader (stdlib only).

Metric names confirmed in the llama.cpp server README (tools/server/README.md, master,
section /metrics, enabled with --metrics):
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
  llamacpp:prompt_tokens_total, llamacpp:prompt_seconds_total, llamacpp:prompt_tokens_seconds,
  llamacpp:tokens_predicted_total, llamacpp:tokens_predicted_seconds_total,
  llamacpp:predicted_tokens_seconds, llamacpp:requests_processing, llamacpp:requests_deferred,
  llamacpp:n_tokens_max, llamacpp:n_decode_total, llamacpp:n_busy_slots_per_decode.

NOT confirmed (absent from the README table; older builds exposed it):
  llamacpp:kv_cache_usage_ratio. It is read if present, otherwise the value is None.

The *_tokens_seconds gauges are averages since server start. llama_metrics() therefore
prefers a recent rate computed from counter deltas between two calls, and falls back to
the gauge on the first call or when no new tokens were produced.
"""
import collections
import subprocess
import threading
import time
import urllib.request

_QUERY = ("utilization.gpu,utilization.memory,memory.used,memory.total,"
          "power.draw,power.limit,clocks.sm,temperature.gpu")
_FIELDS = ("gpu_util_pct", "mem_bw_util_pct", "mem_used_mib", "mem_total_mib",
           "power_w", "power_limit_w", "sm_clock_mhz", "temp_c")

_lock = threading.Lock()
_samples = collections.deque(maxlen=120)  # (timestamp, parsed GPU 0 record)
_state = {"thread": None, "gpu_count": 0}


def _num(s):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None  # "[N/A]", "[Not Supported]", ""


def parse_smi_line(line: str) -> dict:
    """Parse one csv,noheader,nounits line; N/A fields become None; {} if malformed."""
    parts = [p.strip() for p in (line or "").split(",")]
    if len(parts) < len(_FIELDS):
        return {}
    return {k: _num(v) for k, v in zip(_FIELDS, parts)}


def _reader(interval_s):
    ms = max(100, int(interval_s * 1000))
    cmd = ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits",
           "-lms", str(ms)]
    while True:
        proc = None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, bufsize=1)
            last_tick, idx = 0.0, 0
            for line in proc.stdout:
                rec = parse_smi_line(line)
                if not rec:
                    continue
                now = time.time()
                # -lms prints all GPUs at once per tick; a time gap marks a new tick (GPU 0 first).
                if now - last_tick > interval_s * 0.3:
                    if idx:
                        with _lock:
                            _state["gpu_count"] = idx
                    idx, last_tick = 0, now
                if idx == 0:
                    with _lock:
                        _samples.append((now, rec))
                        _state["gpu_count"] = max(_state["gpu_count"], 1)
                idx += 1
        except FileNotFoundError:
            return  # no nvidia-smi on this host: snapshot() reports unavailable
        except Exception:
            pass
        finally:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
        time.sleep(2)  # process died: restart


def start_sampler(interval_s: float = 1.0):
    """Start the background sampler; a no-op while it is already running."""
    with _lock:
        t = _state["thread"]
        if t is not None and t.is_alive():
            return
        t = threading.Thread(target=_reader, args=(interval_s,), name="gpu-sampler", daemon=True)
        _state["thread"] = t
    t.start()


def snapshot() -> dict:
    """Latest GPU 0 stats, 60 s average and history. Never raises."""
    try:
        with _lock:
            samples = list(_samples)
            gpu_count = _state["gpu_count"]
        if not samples:
            return {"available": False}
        ts, cur = samples[-1]
        if time.time() - ts > 15:  # sampler stalled
            return {"available": False}
        recent = [s["gpu_util_pct"] for t, s in samples
                  if t >= ts - 60 and s["gpu_util_pct"] is not None]
        used, total = cur["mem_used_mib"], cur["mem_total_mib"]
        return {
            "available": True,
            "gpu_count": gpu_count,
            "gpu_util_pct": cur["gpu_util_pct"],
            "gpu_util_avg_60s": round(sum(recent) / len(recent), 1) if recent else None,
            "mem_bw_util_pct": cur["mem_bw_util_pct"],
            "vram_used_gb": round(used / 1024, 2) if used is not None else None,
            "vram_total_gb": round(total / 1024, 2) if total is not None else None,
            "power_w": cur["power_w"],
            "power_limit_w": cur["power_limit_w"],
            "sm_clock_mhz": cur["sm_clock_mhz"],
            "temp_c": cur["temp_c"],
            "history": [s["gpu_util_pct"] for t, s in samples if t >= ts - 60],
        }
    except Exception:
        return {"available": False}


def parse_prometheus(text: str) -> dict:
    """Parse Prometheus text exposition into {series: float}; labelled series keep '{labels}'."""
    out = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line and "}" in line:
            end = line.index("}") + 1
            name, rest = line[:end].replace(" ", ""), line[end:].split()
        else:
            tokens = line.split()
            name, rest = tokens[0], tokens[1:]
        if not rest:
            continue
        try:
            out[name] = float(rest[0])
        except ValueError:
            continue
    return out


_prev = {"url": None, "tok": None, "sec": None, "ptok": None, "psec": None}


def _rate(n, s, pn, ps):
    if None in (n, s, pn, ps):
        return None
    dn, ds = n - pn, s - ps
    return dn / ds if dn > 0 and ds > 0 else None


def llama_metrics(base_url: str, timeout: float = 3) -> dict:
    """Fetch {base_url}/metrics from llama-server; return key figures, or {} on any error."""
    try:
        url = base_url.rstrip("/") + "/metrics"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            m = parse_prometheus(r.read().decode("utf-8", "replace"))
        if not m:
            return {}
        g = m.get
        tok, sec = g("llamacpp:tokens_predicted_total"), g("llamacpp:tokens_predicted_seconds_total")
        ptok, psec = g("llamacpp:prompt_tokens_total"), g("llamacpp:prompt_seconds_total")
        decode = prompt = None
        with _lock:
            if _prev["url"] == url:
                decode = _rate(tok, sec, _prev["tok"], _prev["sec"])
                prompt = _rate(ptok, psec, _prev["ptok"], _prev["psec"])
            _prev.update(url=url, tok=tok, sec=sec, ptok=ptok, psec=psec)
        if decode is None:
            decode = g("llamacpp:predicted_tokens_seconds")
        if prompt is None:
            prompt = g("llamacpp:prompt_tokens_seconds")
        return {
            "prompt_tps": prompt,
            "decode_tps": decode,
            "requests_processing": g("llamacpp:requests_processing"),
            "requests_deferred": g("llamacpp:requests_deferred"),
            "kv_cache_usage_ratio": g("llamacpp:kv_cache_usage_ratio"),  # name unconfirmed
            "n_decode_total": g("llamacpp:n_decode_total"),
        }
    except Exception:
        return {}
