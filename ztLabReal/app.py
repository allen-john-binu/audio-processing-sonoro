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

        # ── everything else  →  serve static files (index.html etc.) ────────
        super().do_GET()

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