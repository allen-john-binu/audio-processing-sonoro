#!/usr/bin/env python3
"""
Robot DOA Visualizer — local server
Usage:
    python server.py                        # looks for CSVs in ./data/
    python server.py --csv-dir /path/to/csvs
    python server.py --port 8765
"""

import argparse
import json
import mimetypes
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

CSV_DIR = Path("mergeData")   # default folder; override with --csv-dir


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy request logs; show only errors
        if args[1] not in ("200", "304"):
            super().log_message(format, *args)

    def do_GET(self):
        parsed = urlparse(self.path)

        # ── /api/files  →  list CSV filenames ───────────────────────────────
        if parsed.path == "/api/files":
            try:
                files = sorted(
                    f.name
                    for f in CSV_DIR.iterdir()
                    if f.suffix.lower() == ".csv"
                )
            except FileNotFoundError:
                files = []
            body = json.dumps(files).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        # ── /api/csv?name=filename.csv  →  raw CSV content ──────────────────
        if parsed.path == "/api/csv":
            qs = parse_qs(parsed.query)
            name_list = qs.get("name", [])
            if not name_list:
                self._send_error(400, "Missing ?name= parameter")
                return
            # Sanitise: strip any path components
            filename = Path(name_list[0]).name
            csv_path = CSV_DIR / filename
            if not csv_path.exists():
                self._send_error(404, f"File not found: {filename}")
                return
            body = csv_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        # ── /api/video?name=filename.csv  →  matching video (same stem) ─────
        if parsed.path == "/api/video":
            qs = parse_qs(parsed.query)
            name_list = qs.get("name", [])
            if not name_list:
                self._send_error(400, "Missing ?name= parameter")
                return
            stem = Path(name_list[0]).stem
            video_path = self._find_video(stem)
            if video_path is None:
                self._send_error(
                    404,
                    f"No video found for '{stem}' "
                    f"(looked for {stem}.mp4/.mov/.webm/.m4v in {CSV_DIR})",
                )
                return
            self._serve_video(video_path)
            return

        # ── everything else  →  serve static files (index.html etc.) ────────
        super().do_GET()

    def _find_video(self, stem):
        """Case-insensitive lookup of a video file matching `stem`, trying a
        few common extensions. Videos live in a `video/` subfolder inside
        CSV_DIR (falls back to CSV_DIR itself if that subfolder is absent)."""
        exts = (".mp4", ".mov", ".webm", ".m4v")
        search_dirs = [CSV_DIR / "video", CSV_DIR]
        for d in search_dirs:
            try:
                for f in d.iterdir():
                    if f.is_file() and f.stem.lower() == stem.lower() and f.suffix.lower() in exts:
                        return f
            except FileNotFoundError:
                continue
        return None

    def _serve_video(self, path):
        """Serve a video file with HTTP Range support so the <video> element
        can seek/scrub instead of only playing sequentially."""
        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        range_header = self.headers.get("Range")

        try:
            if range_header:
                units, _, rng = range_header.partition("=")
                start_str, _, end_str = rng.partition("-")
                try:
                    start = int(start_str) if start_str else 0
                    end = int(end_str) if end_str else file_size - 1
                except ValueError:
                    start, end = 0, file_size - 1
                end = min(end, file_size - 1)
                if start > end or start >= file_size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return

                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    chunk_size = 256 * 1024
                    while remaining > 0:
                        chunk = f.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(path, "rb") as f:
                    chunk_size = 256 * 1024
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Client (video player) closed/aborted the connection — normal
            # when seeking or navigating away; nothing to do.
            pass

    def _send_error(self, code, message):
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global CSV_DIR

    parser = argparse.ArgumentParser(description="Robot DOA Visualizer server")
    parser.add_argument("--csv-dir", default="mergedData",
                        help="Directory containing CSV files (default: ./data)")
    parser.add_argument("--port", type=int, default=8083,
                        help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    CSV_DIR = Path(args.csv_dir).resolve()
    if not CSV_DIR.exists():
        print(f"[warn] CSV directory '{CSV_DIR}' does not exist — creating it.")
        CSV_DIR.mkdir(parents=True)

    # Serve static files from the directory where this script lives
    os.chdir(Path(__file__).parent)

    addr = ("", args.port)
    httpd = HTTPServer(addr, Handler)
    print(f"✓ Serving on  http://localhost:{args.port}")
    print(f"✓ CSV folder  {CSV_DIR}")
    print(f"  Drop your .csv files into that folder, then refresh the browser.")
    print(f"  Press Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)


if __name__ == "__main__":
    main()