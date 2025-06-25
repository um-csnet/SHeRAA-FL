#!/usr/bin/env python3
"""
SHeRAA-FL Client Launcher with config display, live output, and footer info.
"""

import sys, subprocess, threading, queue, os, pathlib, itertools, json, tkinter as tk
from datetime import datetime

import ttkbootstrap as ttk
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.constants import *

# ───────────────────────── CONFIG ─────────────────────────
CONFIG_FILE = "config.json"
THEME = "darkly"
POLL_MS = 50
FONT_CONSOLE = ("Consolas", 11)
'''
SCRIPTS = {
    "script_a.py": "Run Remote Attestation Process",
    "script_b.py": "Run Domain-Level Verification",
    "script_c.py": "Run Hierarchical Training",
}
'''
VERSION = "SHeRAA-FL v0.10"
# ──────────────────────────────────────────────────────────


class RunnerGUI:
    def __init__(self, root: ttk.Window):
        self.root = root
        self.cfg = self._load_config(CONFIG_FILE)
        
        # Map JSON script keys to display labels (ordered)
        script_keys = [
            ("attestation_program_path", "1.Run Remote Attestation Process"),
            ("domain_verification_program_path", "2.Run Domain-Level Verification"),
            ("FL_program_path", "3.Run Hierarchical Training"),
        ]

        self.SCRIPTS = {}
        for key, label in script_keys:
            script_path = self.cfg.get(key)
            if script_path:
                self.SCRIPTS[script_path] = label
        
        self.client_id = self.cfg.get("client_id", "<unknown>")
        self.client_domain = self.cfg.get("client_domain", "<unknown>")
        self.client_ip = self.cfg.get("client_ip", "<unknown>")
        self.client_port = self.cfg.get("client_source_port", "<unknown>")

        self.root.title(f"SHeRAA-FL Client Program – {self.client_id}")
        #self.root.geometry("960x600")

        self.output_q = queue.Queue()
        self.procs: dict[str, subprocess.Popen] = {}
        #self.spinner_cycle = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        self.spinner_cycle = itertools.cycle("............................")
        self.spinner_job = None
        
        # Track dynamic labels for instruction statuses
        # Track dynamic labels for instruction statuses
        self.instruction_status_labels = {
            "A": None,
            "B": None,
            "C": None,
        }

        self._build_header()
        self._build_toolbar()
        self._build_console()
        self._build_statusbar()
        self.root.after(POLL_MS, self._drain_output)
        # Let Tk compute the size, then lock it
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())


    def _load_config(self, path: str) -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as exc:
            Messagebox.show_error(f"Could not load {path}:\n{exc}")
            return {}

    def _build_header(self):
        hdr = ttk.Frame(self.root, padding=(12, 10))
        hdr.pack(fill=tk.X)

        # ────────────── Global Server Info Block ──────────────
        server_frame = ttk.LabelFrame(hdr, text="Global Server Information", padding=10)
        server_frame.pack(fill=tk.X, pady=(0, 10))

        server_host = self.cfg.get("server_host", "<unknown>")
        server_port = self.cfg.get("server_port", "<unknown>")
        server_cert = self.cfg.get("server_cert_path", "<unknown>")

        ttk.Label(server_frame, text=f">Host       : {server_host}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(server_frame, text=f">Port       : {server_port}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(server_frame, text=f">Cert Path  : {server_cert}", font=("Segoe UI", 11)).pack(anchor=tk.W)

        # ────────────── Client Info Block ──────────────
        client_frame = ttk.LabelFrame(hdr, text="Client Information", padding=10)
        client_frame.pack(fill=tk.X, pady=(0, 10))

        cid = self.cfg.get("client_id", "<unknown>")
        dom = self.cfg.get("client_domain", "<unknown>")
        cip = self.cfg.get("client_ip", "<unknown>")
        cport = self.cfg.get("client_source_port", "<unknown>")
        batch = self.cfg.get("training_model_batch_size", "<unknown>")
        epochs = self.cfg.get("training_model_epochs", "<unknown>")
        tpm = self.cfg.get("tpm_index", "<unknown>")
        model = self.cfg.get("fl_training_model", "<unknown>")

        ttk.Label(client_frame, text=f">ID             : {cid}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(client_frame, text=f">Domain         : {dom}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(client_frame, text=f">IP             : {cip}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(client_frame, text=f">Port           : {cport}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(client_frame, text=f">Batch Size     : {batch}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(client_frame, text=f">Epochs         : {epochs}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(client_frame, text=f">TPM Index      : {tpm}", font=("Segoe UI", 11)).pack(anchor=tk.W)
        ttk.Label(client_frame, text=f">FL Model       : {model}", font=("Segoe UI", 11)).pack(anchor=tk.W)

        # ────────────── Dataset Info Block ──────────────
        dataset_frame = ttk.LabelFrame(hdr, text="Dataset Information", padding=10)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))

        dataset_label = "ISCXVPN2016 Dataset"
        for val in self.cfg.values():
            if isinstance(val, str) and "nbiot" in val.lower():
                dataset_label = "NBIOT Dataset"
                break

        ttk.Label(dataset_frame, text=f"Detected Dataset: {dataset_label}",
                  font=("Segoe UI", 11, "italic"), foreground="#28b463").pack(anchor=tk.W)

        # ────────────── Instruction Block ──────────────
        inst_frame = ttk.LabelFrame(hdr, text="Running Instructions", padding=10)
        inst_frame.pack(fill=tk.X, pady=(0, 5))

        # Instruction 1
        row1 = ttk.Frame(inst_frame)
        row1.pack(fill=tk.X, anchor=tk.W)
        ttk.Label(row1, text="1. Perform remote attestation process with the Global Server",
                  font=("Segoe UI", 11)).pack(side=tk.LEFT)
        lbl1 = ttk.Label(row1, text="Not Yet", foreground="red", font=("Segoe UI", 11, "bold"))
        lbl1.pack(side=tk.LEFT, padx=(10, 0))
        self.instruction_status_labels["A"] = lbl1

        # Instruction 2
        row2 = ttk.Frame(inst_frame)
        row2.pack(fill=tk.X, anchor=tk.W)
        ttk.Label(row2, text="2. Perform domain-level verification with the Local Aggregator",
                  font=("Segoe UI", 11)).pack(side=tk.LEFT)
        lbl2 = ttk.Label(row2, text="Not Yet", foreground="red", font=("Segoe UI", 11, "bold"))
        lbl2.pack(side=tk.LEFT, padx=(10, 0))
        self.instruction_status_labels["B"] = lbl2

        # Instruction 3
        row3 = ttk.Frame(inst_frame)
        row3.pack(fill=tk.X, anchor=tk.W)
        ttk.Label(row3, text="3. Perform Hierarchical Training",
                  font=("Segoe UI", 11)).pack(side=tk.LEFT)
        lbl3 = ttk.Label(row3, text="Not Yet", foreground="red", font=("Segoe UI", 11, "bold"))
        lbl3.pack(side=tk.LEFT, padx=(10, 0))
        self.instruction_status_labels["C"] = lbl3

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(6, 4, 6, 0))
        bar.pack(fill=tk.X)

        self.buttons = {}
        for script, caption in self.SCRIPTS.items():
            btn = ttk.Button(bar, text=caption, bootstyle=PRIMARY, width=30,
                             command=lambda s=script: self._start_script(s))
            btn.pack(side=tk.LEFT, padx=(0, 6))
            self.buttons[script] = btn

        ttk.Button(bar, text="Settings", bootstyle=OUTLINE,
                   command=self._show_prefs).pack(side=tk.LEFT)

    def _build_console(self):
        self.console = ScrolledText(self.root, font=FONT_CONSOLE, autohide=True,
                                    highlightthickness=0, wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _build_statusbar(self):
        sb = ttk.Frame(self.root, padding=6)
        sb.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_lbl = ttk.Label(sb, text="Ready", anchor=tk.W)
        self.status_lbl.pack(side=tk.TOP, fill=tk.X)

        # Footer note
        year = datetime.now().year
        footer = f"Copyright Azizi & CSNET {year}. All Rights Reserved\n{VERSION}"
        footer_lbl = ttk.Label(sb, text=footer, anchor=tk.W, font=("Segoe UI", 12), justify=tk.LEFT)
        footer_lbl.pack(side=tk.BOTTOM, fill=tk.X)

        self.spin_lbl = ttk.Label(sb, text="")
        self.spin_lbl.place(relx=1.0, rely=0.0, x=-12, y=2, anchor="ne")

    def _start_script(self, script_file: str):
        if script_file in self.procs:
            Messagebox.info(f"{script_file} is already running.")
            return

        full = pathlib.Path(__file__).with_name(script_file)
        if not full.exists():
            Messagebox.show_error(f"Cannot find {script_file}")
            return

        self._log(f"\n=== {self._ts()} : starting {script_file} ===\n")
        self.status_lbl.config(text=f"Running {script_file}")
        self.buttons[script_file].config(state=tk.DISABLED)
        self._spin_start()

        def reader():
            proc = subprocess.Popen(
                [sys.executable, "-u", str(full)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.procs[script_file] = proc
            for line in proc.stdout:
                self.output_q.put(line)
            proc.wait()
            self.output_q.put(f"[{script_file} exited {proc.returncode}]\n")
            self.root.after(0, lambda: self._on_proc_end(script_file))

        threading.Thread(target=reader, daemon=True).start()

    def _on_proc_end(self, script_file: str):
        self.buttons[script_file].config(state=tk.NORMAL)
        self.procs.pop(script_file, None)

        # Update dynamic instruction label
        try:
            script_index = list(self.SCRIPTS.keys()).index(script_file)
            label_key = ["A", "B", "C"][script_index]
            if label_key in self.instruction_status_labels:
                self.instruction_status_labels[label_key].config(text="Done", foreground="green")
        except ValueError:
            pass

        if not self.procs:
            self.status_lbl.config(text="Ready")
            self._spin_stop()

    def _drain_output(self):
        while not self.output_q.empty():
            self.console.insert(tk.END, self.output_q.get())
            self.console.see(tk.END)
        self.root.after(POLL_MS, self._drain_output)

    def _spin_once(self):
        self.spin_lbl.config(text=next(self.spinner_cycle))
        self.spinner_job = self.root.after(90, self._spin_once)

    def _spin_start(self):
        if self.spinner_job is None:
            self._spin_once()

    def _spin_stop(self):
        if self.spinner_job:
            self.root.after_cancel(self.spinner_job)
            self.spinner_job = None
            self.spin_lbl.config(text="")

    def _show_prefs(self):
        Messagebox.ok("Preference dialog stub – add your settings here!")

    @staticmethod
    def _ts():
        return datetime.now().strftime("%H:%M:%S")

    def _log(self, msg: str):
        self.console.insert(tk.END, msg)
        self.console.see(tk.END)


# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    app = RunnerGUI(ttk.Window(themename=THEME))
    app.root.mainloop()