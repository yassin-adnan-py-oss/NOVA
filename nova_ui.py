# NOVA - GUI (Tkinter) - English UI  [v2 - hardened]
# Full graphical interface for the NOVA vision assistant.
# Requirements:
#   pip install opencv-python ultralytics pyttsx3 pillow torch torchvision timm ollama
# Optional modules (YOLO, OCR, Agent, MiDaS) are wrapped in try/except and
# report their status clearly - nothing crashes silently.

import os
import sys
import time
import traceback
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import cv2

# ------------------------------------------------------------------
# Crash diagnostics: log ANY unhandled error (main thread + threads)
# ------------------------------------------------------------------
def _log_crash(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open("nova_crash.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{msg}\n")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _log_crash

def _thread_excepthook(args):
    msg = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    try:
        with open("nova_crash.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== THREAD {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{msg}\n")
    except Exception:
        pass
    print(f"[thread error] {msg}", file=sys.stderr)

threading.excepthook = _thread_excepthook

# ------------------------------------------------------------------
# NOVA modules (each imported independently)
# ------------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

try:
    from nova_color_organizer import ColorOrganizer
    COLOR_OK = True
except ImportError:
    COLOR_OK = False

try:
    from nova_memory import Memory
    MEMORY_OK = True
except ImportError:
    MEMORY_OK = False

try:
    from nova_agent_llm import Agent
    AGENT_OK = True
except ImportError:
    AGENT_OK = False

try:
    from nova_ocr import OCRReader
    OCR_OK = True
except ImportError:
    OCR_OK = False

# Check the ollama package separately so we can show a precise status message
try:
    import ollama  # noqa: F401
    OLLAMA_PKG_OK = True
except ImportError:
    OLLAMA_PKG_OK = False

# Pillow is required to display video frames in Tkinter
try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False


class NovaApp:
    """Main NOVA application window."""

    def __init__(self, root):
        self.root = root
        self.root.title("NOVA - AI Vision Assistant")
        self.root.configure(bg="#1e1e2e")
        self.root.geometry("1250x720")

        # ---------- Internal state ----------
        self.cap = None
        self.cap_lock = threading.Lock()   # guards self.cap (read/release race)
        self.running = False
        self.video_source = None           # 0 = webcam, or a video file path
        self.voice_enabled = tk.BooleanVar(value=True)
        self.depth_enabled = tk.BooleanVar(value=False)
        self._pil_warned = False

        self.model = None
        self.color_organizer = ColorOrganizer() if COLOR_OK else None
        self.memory = Memory() if MEMORY_OK else None

        # Agent availability: module import AND ollama package must both work
        self.agent = None
        if AGENT_OK and self.memory and OLLAMA_PKG_OK:
            try:
                self.agent = Agent(self.memory, self.get_current_objects)
            except Exception as e:
                self.log(f"Agent init failed: {e}")

        self.ocr_reader = None             # loaded lazily, it's heavy
        self.depth_model = None
        self.depth_transform = None

        self.previous_sizes = {}
        self.current_objects = []
        self.latest_frame = None        # snapshot for OCR (never read cap from another thread)
        self.last_spoken_text = ""
        self.last_spoken_time = 0

        self._build_ui()

        if not PIL_OK:
            self.log("WARNING: pillow is not installed - video display disabled. Run: pip install pillow")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # TTS - engine is created INSIDE the speaking thread.
    # Creating a pyttsx3 engine in the main thread and calling it from
    # another thread is a known cause of hard crashes on Windows.
    # ------------------------------------------------------------------
    def speak_async(self, text):
        if not self.voice_enabled.get():
            return
        def run():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                self.log(f"TTS error: {e}")
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # Building the UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # ---------- Top bar ----------
        top = tk.Frame(self.root, bg="#11111b", height=55)
        top.pack(side="top", fill="x")

        tk.Label(top, text="NOVA", font=("Segoe UI", 18, "bold"),
                 bg="#11111b", fg="#89b4fa").pack(side="left", padx=15, pady=6)

        ttk.Button(top, text="Start Camera", command=self.start_camera).pack(side="left", padx=5, pady=10)
        ttk.Button(top, text="Open Video...", command=self.open_video).pack(side="left", padx=5)
        ttk.Button(top, text="Stop", command=self.stop_stream).pack(side="left", padx=5)
        ttk.Checkbutton(top, text="Voice", variable=self.voice_enabled).pack(side="left", padx=10)
        ttk.Checkbutton(top, text="Depth Map (MiDaS)", variable=self.depth_enabled,
                        command=self.on_depth_toggle).pack(side="left", padx=5)

        # ---------- Main body ----------
        body = tk.Frame(self.root, bg="#1e1e2e")
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg="#1e1e2e")
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="Stopped - press 'Start Camera' or open a video")
        tk.Label(left, textvariable=self.status_var, bg="#1e1e2e",
                 fg="#a6e3a1", font=("Segoe UI", 10)).pack(pady=4)

        right = ttk.Notebook(body)
        right.pack(side="right", fill="both", padx=8, pady=8)

        # --- Tab: Objects ---
        tab_objects = tk.Frame(right, bg="#313244")
        right.add(tab_objects, text=" Objects ")
        cols = ("Object", "Color", "Direction", "Status")
        self.objects_tree = ttk.Treeview(tab_objects, columns=cols, show="headings", height=14)
        for c in cols:
            self.objects_tree.heading(c, text=c)
            self.objects_tree.column(c, width=95, anchor="center")
        self.objects_tree.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Tab: Memory ---
        tab_memory = tk.Frame(right, bg="#313244")
        right.add(tab_memory, text=" Memory ")
        mem_top = tk.Frame(tab_memory, bg="#313244")
        mem_top.pack(fill="x", padx=6, pady=6)
        self.mem_entry = ttk.Entry(mem_top, width=28)
        self.mem_entry.pack(side="left", padx=4)
        self.mem_entry.bind("<Return>", lambda e: self.search_memory())
        ttk.Button(mem_top, text="Ask NOVA", command=self.search_memory).pack(side="left")
        ttk.Button(mem_top, text="Clear Memory", command=self.clear_memory).pack(side="left", padx=4)
        self.mem_result = tk.StringVar(value='Type a question like: "where are my keys?"')
        tk.Label(tab_memory, textvariable=self.mem_result, bg="#313244", fg="#f9e2af",
                 font=("Segoe UI", 10), wraplength=380, justify="left").pack(padx=6, anchor="w")
        tk.Label(tab_memory, text="Saved records:", bg="#313244", fg="#cdd6f4").pack(anchor="w", padx=6)
        self.memory_list = tk.Listbox(tab_memory, bg="#45475a", fg="#ffffff",
                                      font=("Consolas", 10))
        self.memory_list.pack(fill="both", expand=True, padx=6, pady=6)
        self.refresh_memory_list()

        # --- Tab: Chat ---
        tab_chat = tk.Frame(right, bg="#313244")
        right.add(tab_chat, text=" Chat ")
        self.agent_status_var = tk.StringVar()
        tk.Label(tab_chat, textvariable=self.agent_status_var, bg="#313244",
                 fg="#a6e3a1", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=6, pady=(4, 0))
        self._update_agent_status()
        self.chat_box = scrolledtext.ScrolledText(tab_chat, state="disabled",
                                                  bg="#313244", fg="#ffffff",
                                                  font=("Segoe UI", 10))
        self.chat_box.pack(fill="both", expand=True, padx=6, pady=6)
        chat_bottom = tk.Frame(tab_chat, bg="#313244")
        chat_bottom.pack(fill="x", padx=6, pady=4)
        self.chat_entry = ttk.Entry(chat_bottom, width=30)
        self.chat_entry.pack(side="left", padx=4, fill="x", expand=True)
        self.chat_entry.bind("<Return>", lambda e: self.send_chat())
        ttk.Button(chat_bottom, text="Send", command=self.send_chat).pack(side="left")

        # --- Tab: OCR ---
        tab_ocr = tk.Frame(right, bg="#313244")
        right.add(tab_ocr, text=" OCR ")
        ttk.Button(tab_ocr, text="Capture Frame & Read Text", command=self.run_ocr).pack(pady=10)
        self.ocr_text = scrolledtext.ScrolledText(tab_ocr, bg="#313244", fg="#ffffff",
                                                  font=("Consolas", 10))
        self.ocr_text.pack(fill="both", expand=True, padx=6, pady=6)

        # --- Tab: Log ---
        tab_log = tk.Frame(right, bg="#313244")
        right.add(tab_log, text=" Log ")
        self.log_box = scrolledtext.ScrolledText(tab_log, state="disabled",
                                                 bg="#313244", fg="#a6e3a1",
                                                 font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True, padx=6, pady=6)

    def _update_agent_status(self):
        if self.agent:
            from nova_agent_llm import MODEL_NAME
            self.agent_status_var.set(f"Agent: ONLINE (model: {MODEL_NAME})")
        elif not AGENT_OK:
            self.agent_status_var.set("Agent: OFFLINE - nova_agent_llm.py not found")
        elif not OLLAMA_PKG_OK:
            self.agent_status_var.set("Agent: OFFLINE - run: pip install ollama")
        elif not self.memory:
            self.agent_status_var.set("Agent: OFFLINE - nova_memory.py not found")
        else:
            self.agent_status_var.set("Agent: OFFLINE - unknown error (see Log tab)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def log(self, msg):
        def _append():
            try:
                self.log_box.configure(state="normal")
                self.log_box.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
            except Exception:
                pass
        try:
            self.root.after(0, _append)
        except Exception:
            pass

    def get_current_objects(self):
        return list(self.current_objects)

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------
    def _load_yolo(self):
        if not YOLO_OK:
            self.log("Ultralytics is not installed! Run: pip install ultralytics")
            return False
        if self.model is None:
            try:
                self.log("Loading YOLO...")
                self.model = YOLO("yolov8s.pt")
                self.log("YOLO ready.")
            except Exception as e:
                self.log(f"YOLO failed to load: {e}")
                return False
        return True

    def start_camera(self):
        self.video_source = 0
        self._start_stream()

    def open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.video_source = path
            self._start_stream()

    def _start_stream(self):
        self.stop_stream()

        cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW) \
            if self.video_source == 0 else cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the video source.")
            return

        with self.cap_lock:
            self.cap = cap
        self.running = True
        self.status_var.set("Running")
        threading.Thread(target=self._video_loop, daemon=True).start()
        self.log(f"Stream started: {self.video_source}")

    def stop_stream(self):
        self.running = False
        with self.cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        self.status_var.set("Stopped")

    def on_close(self):
        self.stop_stream()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Video loop (separate thread)
    # ------------------------------------------------------------------
    def _video_loop(self):
        try:
            if not self._load_yolo():
                self.running = False
                return

            while self.running:
                with self.cap_lock:
                    cap = self.cap
                if cap is None:
                    break

                success, frame = cap.read()
                if not success:
                    if isinstance(self.video_source, str):
                        # Loop by reopening the file. Seeking with CAP_PROP_POS_FRAMES
                        # can trigger "Assertion fctx->async_lock failed" in libavcodec.
                        with self.cap_lock:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            if not self.running:
                                break
                            new_cap = cv2.VideoCapture(self.video_source)
                            self.cap = new_cap if new_cap.isOpened() else None
                        continue
                    break

                frame_width = frame.shape[1]
                annotated = frame.copy()
                self.latest_frame = frame.copy()   # safe snapshot for OCR
                self.current_objects = []

                try:
                    results = self.model.track(frame, persist=True)
                    annotated = results[0].plot()
                    boxes = results[0].boxes
                    rows = []

                    if boxes is not None and boxes.id is not None:
                        for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):
                            x1, y1, x2, y2 = box.tolist()
                            track_id = int(track_id.item())
                            area = (x2 - x1) * (y2 - y1)
                            center_x = (x1 + x2) / 2
                            label = self.model.names[int(cls)]

                            direction = ("left" if center_x < frame_width / 3
                                         else "right" if center_x > frame_width * 2 / 3
                                         else "center")

                            old = self.previous_sizes.get(track_id, area)
                            diff = area - old
                            if diff > old * 0.03:
                                status = "approaching"
                            elif diff < -old * 0.03:
                                status = "moving away"
                            else:
                                status = "stable"
                            self.previous_sizes[track_id] = area

                            color_name = ""
                            if self.color_organizer:
                                try:
                                    colors = self.color_organizer.get_dominant_colors(
                                        frame, bbox=(x1, y1, x2, y2), top_n=2)
                                    color_name = colors[0]["color"] if colors else ""
                                except Exception:
                                    pass

                            if self.memory:
                                try:
                                    self.memory.remember(label, direction, color_name)
                                except Exception as e:
                                    self.log(f"Memory write error: {e}")

                            obj = {"label": label, "color": color_name, "direction": direction}
                            self.current_objects.append(obj)
                            rows.append((label, color_name, direction, status))

                            cv2.putText(annotated, f"{color_name} {label} ({direction}) - {status}",
                                        (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                            spoken = f"{color_name} {label}".strip()
                            sentence = f"{spoken} on your {direction}, {status}"
                            now = time.time()
                            if sentence != self.last_spoken_text and now - self.last_spoken_time > 2.5:
                                self.speak_async(sentence)
                                self.last_spoken_text = sentence
                                self.last_spoken_time = now

                    self.root.after(0, lambda r=rows: self._update_objects_table(r))

                except Exception as e:
                    self.log(f"Tracking error: {e}")

                # Depth map overlay
                if self.depth_enabled.get() and self.depth_model is not None:
                    try:
                        import torch
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inp = self.depth_transform(img_rgb).to(self._depth_device)
                        with torch.no_grad():
                            pred = self.depth_model(inp)
                            pred = torch.nn.functional.interpolate(
                                pred.unsqueeze(1), size=img_rgb.shape[:2],
                                mode="bicubic", align_corners=False).squeeze()
                        dnorm = cv2.normalize(pred.cpu().numpy(), None, 0, 255,
                                              cv2.NORM_MINMAX, cv2.CV_8U)
                        depth_colored = cv2.applyColorMap(dnorm, cv2.COLORMAP_INFERNO)
                        annotated = cv2.addWeighted(annotated, 0.6, depth_colored, 0.4, 0)
                    except Exception as e:
                        self.log(f"Depth error: {e}")

                self._show_frame(annotated)

        except Exception:
            # Never let the loop thread die silently - full trace goes to log + file
            err = traceback.format_exc()
            self.log(f"Video loop crashed:\n{err}")
            with open("nova_crash.log", "a", encoding="utf-8") as f:
                f.write(f"\n===== VIDEO LOOP {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{err}\n")
        finally:
            self.running = False
            try:
                self.root.after(0, lambda: self.status_var.set("Stopped"))
            except Exception:
                pass

    def _update_objects_table(self, rows):
        try:
            self.objects_tree.delete(*self.objects_tree.get_children())
            for r in rows:
                self.objects_tree.insert("", "end", values=r)
        except Exception:
            pass

    def _show_frame(self, frame_bgr):
        if not PIL_OK:
            if not self._pil_warned:
                self._pil_warned = True
                self.log("Install pillow to see the video: pip install pillow")
            return
        try:
            h, w = frame_bgr.shape[:2]
            canvas_w = max(self.video_label.winfo_width(), 320)
            canvas_h = max(self.video_label.winfo_height(), 240)
            scale = min(canvas_w / w, canvas_h / h)
            disp = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(disp)
            imgtk = ImageTk.PhotoImage(image=img)
            self.root.after(0, lambda im=imgtk: self._set_image(im))
        except Exception as e:
            self.log(f"Display error: {e}")

    def _set_image(self, imgtk):
        try:
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def search_memory(self):
        query = self.mem_entry.get().strip()
        if not query:
            return
        if not self.memory:
            self.mem_result.set("Memory module is not available.")
            return
        try:
            answer = self.memory.recall(query)
        except Exception as e:
            answer = f"Error: {e}"
        self.mem_result.set(answer)
        self.log(f"Memory query: {query} -> {answer}")

    def refresh_memory_list(self):
        try:
            self.memory_list.delete(0, "end")
            if not self.memory:
                return
            for label, info in self.memory.records.items():
                self.memory_list.insert(
                    "end",
                    f"{label}  |  {info.get('color','')}  |  {info.get('direction','')}  |  {info.get('readable_time','')}")
        except Exception:
            pass

    def clear_memory(self):
        if self.memory and messagebox.askyesno("Confirm", "Clear all memory records?"):
            try:
                self.memory.forget_all()
                self.refresh_memory_list()
                self.log("Memory cleared.")
            except Exception as e:
                self.log(f"Memory clear error: {e}")

    # ------------------------------------------------------------------
    # Chat (Agent)
    # ------------------------------------------------------------------
    def _chat_append(self, who, text):
        try:
            self.chat_box.configure(state="normal")
            self.chat_box.insert("end", f"{who}: {text}\n")
            self.chat_box.see("end")
            self.chat_box.configure(state="disabled")
        except Exception:
            pass

    def send_chat(self):
        query = self.chat_entry.get().strip()
        if not query:
            return
        self.chat_entry.delete(0, "end")
        self._chat_append("You", query)
        self.log(f"Chat: {query}")

        if not self.agent:
            self._chat_append(
                "NOVA",
                "Agent is offline. Check the status line above - usually you need:\n"
                "  1) pip install ollama\n"
                "  2) ollama running + model pulled:  ollama run llama3.2:3b")
            return

        def work():
            try:
                reply = self.agent.handle(query)
            except Exception as e:
                reply = (f"Error: {e}\n"
                         f"Make sure Ollama is running and the model is pulled "
                         f"(ollama run llama3.2:3b).")
            self.root.after(0, lambda r=reply: self._chat_append("NOVA", r))
            self.root.after(0, self.refresh_memory_list)
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def run_ocr(self):
        if self.cap is None or not self.cap.isOpened():
            messagebox.showinfo("OCR", "Start the camera first.")
            return
        if not OCR_OK:
            messagebox.showinfo("OCR", "nova_ocr module is not available. Install easyocr.")
            return

        if self.latest_frame is None:
            messagebox.showinfo("OCR", "No frame available yet - wait for the video to start.")
            return
        frame = self.latest_frame.copy()   # NEVER call cap.read() outside the video thread

        def work():
            try:
                if self.ocr_reader is None:
                    self.log("Loading OCR reader (first run may take a while)...")
                    self.ocr_reader = OCRReader()
                tmp = "_nova_ocr_tmp.png"
                cv2.imwrite(tmp, frame)
                text, info = self.ocr_reader.read_and_analyze(tmp)
                if os.path.exists(tmp):
                    os.remove(tmp)
                out = f"Detected text:\n{text}\n\nExtracted info:\n"
                out += "\n".join(f"{k}: {v}" for k, v in info.items()) if info else "Nothing found."
                def _fill():
                    self.ocr_text.delete("1.0", "end")
                    self.ocr_text.insert("end", out)
                self.root.after(0, _fill)
                self.log("OCR done.")
            except Exception as e:
                self.log(f"OCR error: {e}")
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------
    def on_depth_toggle(self):
        if not self.depth_enabled.get():
            return
        if self.depth_model is not None:
            return
        def work():
            try:
                import torch
                self.log("Loading MiDaS...")
                self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                self._depth_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                self.depth_model.to(self._depth_device).eval()
                t = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.depth_transform = t.small_transform
                self.log("MiDaS ready.")
            except Exception as e:
                self.log(f"Failed to load MiDaS: {e}")
                self.depth_enabled.set(False)
        threading.Thread(target=work, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = NovaApp(root)
    root.mainloop()
