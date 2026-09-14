"""Local desktop monitor using the previously verified v3.1 inference engine."""
import argparse
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

import cv2
from PIL import Image, ImageTk
from intent_engine_v3_1_windows import IntentEngine, EngineConfig, CameraFactory

# Locate the project independently of the optional external video collection.
BASE = next((p for p in Path(__file__).resolve().parents
             if (p / 'intentpredictionattempt4-main').is_dir()),
            Path(__file__).resolve().parent)
_DATA_RELATIVE = Path('raw videos for mecha lab-20260904T185555Z-1-001') / 'raw videos for mecha lab'
_DATA_CANDIDATES = [BASE / _DATA_RELATIVE, BASE.parent / _DATA_RELATIVE]
DATA = (Path(os.environ['INTENT_DATA_DIR']).expanduser()
        if os.environ.get('INTENT_DATA_DIR')
        else next((p for p in _DATA_CANDIDATES if p.is_dir()), _DATA_CANDIDATES[1]))
WEIGHTS = (Path(os.environ['INTENT_WEIGHTS_PATH']).expanduser()
           if os.environ.get('INTENT_WEIGHTS_PATH')
           else DATA / 'cnnlstm-Epoch-196-Loss-0.01737015192823795.pth')
LABELS = ['INTERACTION', 'PASSTHRU', 'WAIT']
BG, PANEL, FG = '#0d0f14', '#171c27', '#eef2fa'
COLORS = ['#ff625b', '#f5cc52', '#51d89a']


class PreviewCamera:
    def __init__(self, camera):
        self.camera = camera
        self.backend_name = camera.backend_name
        self.frame = None
        self.last_frame_time = 0.0

    def read(self):
        ok, frame = self.camera.read()
        if ok and frame is not None:
            self.frame = frame
            self.last_frame_time = time.monotonic()
        return ok, frame

    def release(self):
        self.camera.release()


class PreviewEngine(IntentEngine):
    def _camera_loop(self):
        self.preview = PreviewCamera(self._camera)
        self._camera = self.preview
        super()._camera_loop()


def config(weights, video='', camera=0):
    return EngineConfig(resume_path=str(weights), class_labels=LABELS,
                        n_classes=3, sample_size=150, sample_duration=8,
                        smoothing_window=3, confidence_threshold=0.75,
                        camera_warmup_frames=0, max_infer_hz=10,
                        video_path=str(video), camera_index=camera, loop_video=True)


class Monitor:
    def __init__(self, root):
        self.root = root
        self.engine = None
        self.busy = False
        self.closing = False
        self.messages = queue.Queue()
        self.last_state = None
        root.title('Intent Monitor — Local Model')
        root.geometry('1180x760')
        root.minsize(1000, 680)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self.close)
        tk.Label(root, text='INTENT MONITOR', bg=BG, fg=FG,
                 font=('Segoe UI', 24, 'bold')).pack(anchor='w', padx=24, pady=(18, 4))
        tk.Label(root, text='Local video / webcam • Epoch 196 • ResNet-101 + LSTM',
                 bg=BG, fg='#a6b2c8').pack(anchor='w', padx=24)
        controls = tk.Frame(root, bg=BG)
        controls.pack(fill='x', padx=24, pady=14)
        self.video = tk.StringVar(value=str(DATA / 'INTERACTION' / 'int1.mov'))
        self.weights = tk.StringVar(value=str(WEIGHTS))
        self.mode = tk.StringVar(value='Video')
        self.index = tk.StringVar(value='0')
        self.inputs = []
        for row, (label, variable, callback) in enumerate([
                ('Weights', self.weights, self.choose_weights),
                ('Video', self.video, self.choose_video)]):
            tk.Label(controls, text=label, bg=BG, fg=FG, width=8, anchor='w').grid(row=row, column=0)
            entry = ttk.Entry(controls, textvariable=variable)
            entry.grid(row=row, column=1, sticky='ew', pady=3)
            button = ttk.Button(controls, text='Browse…', command=callback)
            button.grid(row=row, column=2, padx=(8, 0))
            self.inputs.extend([entry, button])
        controls.columnconfigure(1, weight=1)
        modes = tk.Frame(controls, bg=BG)
        modes.grid(row=2, column=1, sticky='w', pady=(8, 0))
        for name in ['Video', 'Webcam']:
            radio = ttk.Radiobutton(modes, text=name, variable=self.mode, value=name)
            radio.pack(side='left', padx=(0, 12))
            self.inputs.append(radio)
        tk.Label(modes, text='Camera index:', bg=BG, fg=FG).pack(side='left')
        index = ttk.Entry(modes, textvariable=self.index, width=4)
        index.pack(side='left', padx=8)
        self.inputs.append(index)
        self.start_button = ttk.Button(modes, text='Start', command=self.start)
        self.start_button.pack(side='left', padx=8)
        self.stop_button = ttk.Button(modes, text='Stop', command=self.stop, state='disabled')
        self.stop_button.pack(side='left')
        body = tk.Frame(root, bg=BG)
        body.pack(fill='both', expand=True, padx=24)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self.preview_label = tk.Label(body, text='Choose a source and press Start', bg='#080b10', fg='#a6b2c8')
        self.preview_label.grid(row=0, column=0, sticky='nsew', padx=(0, 16))
        side = tk.Frame(body, bg=PANEL, padx=18, pady=16)
        side.grid(row=0, column=1, sticky='nsew')
        self.banner = tk.Label(side, text='IDLE', font=('Segoe UI', 25, 'bold'), bg=PANEL, fg=FG)
        self.banner.pack(fill='x', pady=(0, 8))
        self.detail = tk.Label(side, text='Waiting for input', bg=PANEL, fg=FG)
        self.detail.pack(fill='x', pady=(0, 16))
        self.bars = []
        for label, color in zip(LABELS, COLORS):
            title = tk.Label(side, text=label + '   —', bg=PANEL, fg=color, anchor='w', font=('Segoe UI', 12, 'bold'))
            title.pack(fill='x', pady=(8, 4))
            bar = tk.Canvas(side, bg='#293143', height=17, highlightthickness=0)
            bar.pack(fill='x')
            fill = bar.create_rectangle(0, 0, 0, 17, fill=color, outline='')
            self.bars.append((title, bar, fill))
        tk.Label(side, text='EVENT LOG', bg=PANEL, fg='#a6b2c8').pack(anchor='w', pady=(24, 6))
        self.events = tk.Text(side, height=6, bg=PANEL, fg=FG, relief='flat', state='disabled', font=('Consolas', 10))
        self.events.pack(fill='both', expand=True)
        self.status = tk.Label(root, text='Ready • Model status display only; no robot commands are sent.', bg=BG, fg='#a6b2c8', anchor='w')
        self.status.pack(fill='x', padx=24, pady=14)
        self.root.after(50, self.update)

    def choose_weights(self):
        value = filedialog.askopenfilename(title='Select epoch-196 checkpoint', filetypes=[('Model weights', '*.pth')])
        if value:
            self.weights.set(value)

    def choose_video(self):
        value = filedialog.askopenfilename(title='Select a video', initialdir=str(DATA if DATA.is_dir() else BASE), filetypes=[('Videos', '*.mov *.mp4 *.avi *.mkv'), ('All files', '*.*')])
        if value:
            self.video.set(value)
            self.mode.set('Video')

    def set_controls(self, running):
        for item in self.inputs:
            item.configure(state='disabled' if running else 'normal')
        self.start_button.configure(state='disabled' if running else 'normal')
        self.stop_button.configure(state='normal' if running and not self.busy else 'disabled')

    def start(self):
        if self.busy or self.engine:
            return
        try:
            weights = Path(self.weights.get())
            video = self.video.get() if self.mode.get() == 'Video' else ''
            if not weights.is_file():
                raise ValueError('Select the epoch-196 weights using Browse.')
            if self.mode.get() == 'Video' and not Path(video).is_file():
                raise ValueError('Select an existing video using Browse.')
            cfg = config(weights, video, int(self.index.get()))
        except ValueError as exc:
            self.status.configure(text=str(exc))
            return
        self.busy = True
        self.last_state = None
        self.set_controls(True)
        self.banner.configure(text='LOADING', fg=FG)
        self.status.configure(text='Loading checkpoint and opening source…')
        def worker():
            engine = None
            try:
                engine = PreviewEngine(cfg)
                engine.start()
                self.messages.put(('started', engine))
            except Exception as exc:
                if engine:
                    engine.stop()
                self.messages.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        if self.busy:
            return
        if not self.engine:
            return
        self.busy = True
        engine, self.engine = self.engine, None
        self.set_controls(True)
        self.banner.configure(text='STOPPING', fg=FG)
        def worker():
            engine.stop()
            self.messages.put(('stopped', None))
        threading.Thread(target=worker, daemon=True).start()

    def close(self):
        self.closing = True
        self.stop()

    def update(self):
        while not self.messages.empty():
            kind, value = self.messages.get_nowait()
            self.busy = False
            if kind == 'started':
                self.engine = value
                self.set_controls(True)
            else:
                self.set_controls(False)
                self.banner.configure(text='ERROR' if kind == 'error' else 'IDLE', fg=FG)
                self.detail.configure(text='No active predictions')
                self.status.configure(text=value if kind == 'error' else 'Stopped. Choose another source or press Start.')
        if self.closing:
            if self.engine and not self.busy:
                self.stop()
            if not self.busy and self.engine is None:
                self.root.destroy()
                return
        if self.engine:
            state = self.engine.get_state()
            preview = getattr(self.engine, 'preview', None)
            if preview and preview.frame is not None:
                rgb = cv2.cvtColor(preview.frame, cv2.COLOR_BGR2RGB)
                picture = Image.fromarray(rgb)
                picture.thumbnail((max(1, self.preview_label.winfo_width()), max(1, self.preview_label.winfo_height())))
                self.photo = ImageTk.PhotoImage(picture)
                self.preview_label.configure(image=self.photo, text='')
            stale = not preview or time.monotonic() - preview.last_frame_time > 2
            ready = bool(state['probs'].sum())
            flag = 'NO VIDEO' if stale else state['flag'] if ready else 'WARMING UP'
            color = {'STOP': COLORS[0], 'CAUTION': COLORS[1], 'CLEAR': COLORS[2]}.get(flag, FG)
            self.banner.configure(text=flag, fg=color)
            self.detail.configure(text=f"{state['label']} • {state['confidence']:.1%}" if ready and not stale else 'Waiting for frames / predictions')
            for label, probability, (title, bar, fill) in zip(LABELS, state['probs'], self.bars):
                title.configure(text=f'{label}   {probability:.1%}' if ready and not stale else label + '   —')
                bar.coords(fill, 0, 0, bar.winfo_width() * float(probability) if ready and not stale else 0, 17)
            if flag != self.last_state:
                self.last_state = flag
                self.events.configure(state='normal')
                self.events.insert('end', time.strftime('%H:%M:%S') + '  ' + flag + '\n')
                if int(self.events.index('end-1c').split('.')[0]) > 200:
                    self.events.delete('1.0', '2.0')
                self.events.see('end')
                self.events.configure(state='disabled')
            self.status.configure(text=f"Device: {self.engine.device} • Predictions/sec: {state['infer_fps']:.1f} • {state['camera']} • Display only")
        self.root.after(50, self.update)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    root = tk.Tk()
    app = Monitor(root)
    if args.smoke_test:
        root.after(300, app.start)
        root.after(18000, app.close)
    root.mainloop()
