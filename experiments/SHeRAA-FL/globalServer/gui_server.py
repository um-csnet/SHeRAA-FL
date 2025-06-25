import json, subprocess, threading, queue, os, tkinter as tk
from datetime import datetime
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText

# ─── Configuration ─────────────────────────────────────────
CONFIG_PATH = "config.json"
CLIENT_LIST_PATH = "predefined_client_list.json"
SCRIPT_TO_RUN = "global_server_attestator.py"
THEME = "darkly"
FONT_HEADER = ("Segoe UI", 11)
FONT_SECTION = ("Segoe UI", 11, "bold")
FONT_ITALIC = ("Segoe UI", 11, "italic")
FONT_MONO = ("Consolas", 11)
# ──────────────────────────────────────────────────────────


class ServerGUI:
    def __init__(self, root: ttk.Window):
        self.root = root
        self.root.title("SHeRAA-FL Global Server Program")
        self.output_q = queue.Queue()
        self.proc = None
        self.spinner = None

        # Load configuration
        self.cfg = self._load_json(CONFIG_PATH)
        self.clients = self._load_json(CLIENT_LIST_PATH)

        self._build_header()
        self._build_console()
        self._build_footer()

        self.root.after(100, self._poll_output)

        # Auto-size window to fit all content
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

    def _load_json(self, path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {}

    def _build_header(self):
        hdr = ttk.Frame(self.root, padding=12)
        hdr.pack(fill=tk.X)

        # ─ Global Server Info ─
        server_frame = ttk.LabelFrame(hdr, text="Global Server Information", padding=10)
        server_frame.pack(fill=tk.X, pady=(0, 10))

        fields = [
            ("Server IP", self.cfg.get("server_ip")),
            ("Server Host", self.cfg.get("server_host")),
            ("Port", self.cfg.get("server_port")),
            ("Batch Size", self.cfg.get("training_model_batch_size")),
            ("Epochs", self.cfg.get("training_model_epochs")),
            ("TPM Index", self.cfg.get("tpm_index")),
            ("FL Model", self.cfg.get("fl_training_model")),
        ]

        for label, value in fields:
            ttk.Label(server_frame, text=f">{label}: {value}", font=FONT_HEADER).pack(anchor=tk.W)

        # ─ Client Info by Domain ─
        domain_frame = ttk.LabelFrame(hdr, text="Client Information by Domain", padding=10)
        domain_frame.pack(fill=tk.X, pady=(0, 10))

        domains = {}
        for client_id, data in self.clients.items():
            domain = data.get("client_domain", "Unknown")
            domains.setdefault(domain, []).append((client_id, data))

        for domain in sorted(domains.keys()):
            ttk.Label(domain_frame, text=f"Domain {domain}", font=FONT_SECTION).pack(anchor=tk.W, pady=(6, 2))
            for cid, info in domains[domain]:
                text = f"    >{cid} - IP: {info['client_ip']}, Port: {info['client_source_port']}, Cert: {info['client_cert_path']}"
                ttk.Label(domain_frame, text=text, font=FONT_HEADER).pack(anchor=tk.W)

        # ─ Dataset Info ─
        dataset_frame = ttk.LabelFrame(hdr, text="Dataset Information", padding=10)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))

        dataset_type = "ISCXVPN2016 Dataset"
        for val in self.cfg.values():
            if isinstance(val, str) and "nbiot" in val.lower():
                dataset_type = "NBIOT Dataset"
                break

        ttk.Label(dataset_frame, text=f"Detected Dataset: {dataset_type}",
                  font=FONT_ITALIC, foreground="#28b463").pack(anchor=tk.W)

        # ─ Run Button ─
        self.run_btn = ttk.Button(hdr, text="Run Global Server Attestator",
                                  bootstyle=SUCCESS, width=40,
                                  command=self._run_script)
        self.run_btn.pack(pady=(10, 0))

    def _build_console(self):
        self.console = ScrolledText(self.root, height=18, font=FONT_MONO, autohide=True)
        self.console.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

    def _build_footer(self):
        year = datetime.now().year
        footer = ttk.Frame(self.root, padding=(12, 4))
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(
            footer,
            text=f"Copyright Azizi & CSNET {year}. All Rights Reserved\nSHeRAA-FL v0.10",
            anchor=tk.W,
            justify=tk.LEFT,
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _run_script(self):
        if self.proc:
            return

        self.run_btn.config(state=tk.DISABLED)
        self._log(f"\n Starting {SCRIPT_TO_RUN}\n")

        def worker():
            self.proc = subprocess.Popen(
                ["python3", "-u", SCRIPT_TO_RUN],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in self.proc.stdout:
                self.output_q.put(line)
            self.proc.wait()
            self.output_q.put(f"\n Finished with code {self.proc.returncode}\n")
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.proc = None

        threading.Thread(target=worker, daemon=True).start()

    def _poll_output(self):
        while not self.output_q.empty():
            line = self.output_q.get()
            self.console.insert(tk.END, line)
            self.console.see(tk.END)
        self.root.after(100, self._poll_output)

    def _log(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)


# ─── Main ────────────────────────────────────────
if __name__ == "__main__":
    app = ServerGUI(ttk.Window(themename=THEME))
    app.root.mainloop()
