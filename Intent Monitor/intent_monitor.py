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
from camera_devices import discover_cameras, display_names, identity, NamedCamera, RESOLUTIONS
from clip_recorder import ClipRecorder
from recording_session import CameraPreview, RecordingSession, validate_identifiers

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


def make_inference_engine(cfg, recorder):
    # Recording and its preview never import torch or allocate model memory.
    from intent_engine_v3_1_windows import IntentEngine

    class PreviewEngine(IntentEngine):
        record_only = False

        def _camera_loop(self):
            self.preview = CameraPreview(self._camera, recorder)
            self._camera = self.preview
            super()._camera_loop()

    return PreviewEngine(cfg)


def config(weights, video='', camera=0, sample_duration=16):
    from intent_engine_v3_1_windows import EngineConfig
    if sample_duration not in (8, 16):
        raise ValueError('Choose 16 frames, or legacy 8 for comparison.')
    return EngineConfig(resume_path=str(weights), class_labels=LABELS,
                        n_classes=3, sample_size=150, sample_duration=sample_duration,
                        smoothing_window=3, confidence_threshold=0.75,
                        camera_warmup_frames=0, max_infer_hz=10,
                        video_path=str(video), camera_index=camera, loop_video=True)


class Monitor:
    def __init__(self, root, scan_cameras=True, recordings_folder=None):
        self.root = root
        self.engine = None
        self.busy = False
        self.closing = False
        self.messages = queue.Queue()
        self.last_state = None
        self.recorder = ClipRecorder(recordings_folder or BASE / 'Camera Recordings')
        self.scan_cameras = scan_cameras
        self.stop_failed = False
        self.recording_error = ''
        self.cameras = []
        self.scanning = False
        root.title('Intent Monitor — Local Model')
        root.geometry('1180x740')
        root.minsize(1040, 680)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self.close)
        tk.Label(root, text='INTENT MONITOR', bg=BG, fg=FG,
                 font=('Segoe UI', 24, 'bold')).pack(anchor='w', padx=24, pady=(18, 4))
        tk.Label(root, text='Record original camera footage • Optional local intent predictions',
                 bg=BG, fg='#a6b2c8').pack(anchor='w', padx=24)
        controls = tk.Frame(root, bg=BG)
        controls.pack(fill='x', padx=24, pady=14)
        self.video = tk.StringVar(value=str(DATA / 'INTERACTION' / 'int1.mov'))
        self.weights = tk.StringVar(value=str(WEIGHTS))
        self.mode = tk.StringVar(value='Record only')
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
        for name in ['Record only', 'Video', 'Webcam']:
            radio = ttk.Radiobutton(modes, text=name, variable=self.mode, value=name,
                                    command=lambda: self.set_controls(False))
            radio.pack(side='left', padx=(0, 12))
            self.inputs.append(radio)
        tk.Label(modes, text='Camera:', bg=BG, fg=FG).pack(side='left')
        self.camera_dropdown = ttk.Combobox(modes, state='disabled', width=24)
        self.camera_dropdown.set('Scanning for cameras…')
        self.camera_dropdown.pack(side='left', padx=8)
        self.refresh_button = ttk.Button(modes, text='Refresh', command=self.refresh_cameras)
        self.refresh_button.pack(side='left')
        self.start_button = ttk.Button(modes, text='Start', command=self.start)
        self.start_button.pack(side='left', padx=8)
        self.stop_button = ttk.Button(modes, text='Stop', command=self.stop, state='disabled')
        self.stop_button.pack(side='left')
        capture = tk.Frame(controls, bg=BG)
        capture.grid(row=3, column=1, sticky='w', pady=(8, 0))
        tk.Label(capture, text='Camera resolution:', bg=BG, fg=FG).pack(side='left')
        self.resolution = ttk.Combobox(capture, values=list(RESOLUTIONS), state='readonly', width=16)
        self.resolution.set('1920 x 1080')
        self.resolution.pack(side='left', padx=8)
        tk.Label(capture, text='Requested 30 FPS', bg=BG, fg='#a6b2c8').pack(side='left', padx=8)
        tk.Label(capture, text='Inference frames:', bg=BG, fg=FG).pack(side='left')
        self.sample_frames = ttk.Combobox(capture, values=['16', '8'], state='readonly', width=5)
        self.sample_frames.set('16')
        self.sample_frames.pack(side='left', padx=8)
        tk.Label(capture, text='8 = legacy comparison', bg=BG, fg='#a6b2c8').pack(side='left')
        metadata = tk.Frame(controls, bg=BG)
        metadata.grid(row=4, column=1, sticky='ew', pady=(8, 0))
        self.collection = {}
        self.metadata_inputs = []
        settings = self.recorder.sessions.settings()
        for column, (key, label, width) in enumerate([
                ('session_id', 'Session ID', 20), ('participant_id', 'Participant ID', 14),
                ('camera_setup_id', 'Camera setup ID', 16), ('tool', 'Tool', 16)]):
            cell = tk.Frame(metadata, bg=BG)
            cell.grid(row=0, column=column, sticky='w', padx=(0, 12))
            tk.Label(cell, text=label, bg=BG, fg=FG, anchor='w').pack(anchor='w')
            variable = tk.StringVar(value=settings.get(key, ''))
            entry = ttk.Entry(cell, textvariable=variable, width=width)
            entry.pack()
            self.collection[key] = variable
            self.metadata_inputs.append(entry)
        recordings = tk.Frame(controls, bg=BG)
        recordings.grid(row=5, column=1, sticky='w', pady=(10, 0))
        tk.Label(recordings, text='Record example:', bg=BG, fg=FG).pack(side='left')
        self.record_scenario = ttk.Combobox(recordings, values=LABELS, state='readonly', width=14)
        self.record_scenario.set('WAIT')
        self.record_scenario.pack(side='left', padx=8)
        self.record_button = ttk.Button(recordings, text='Record 10 sec', command=self.toggle_recording, state='disabled')
        self.record_button.pack(side='left')
        self.take_label = tk.Label(recordings, text='', bg=BG, fg='#a6b2c8')
        self.take_label.pack(side='left', padx=12)
        self.record_status = tk.Label(controls, text='Records the camera image, not the desktop. Scenario labels need review before training.', bg=BG, fg='#a6b2c8', anchor='w')
        self.record_status.grid(row=6, column=1, sticky='w', pady=(5, 0))
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
        self.set_controls(False)
        if scan_cameras:
            self.root.after(100, self.refresh_cameras)

    def toggle_recording(self):
        active, _ = self.recorder.status()
        if active:
            self.recorder.stop()
        elif self.engine and not self.busy and self.mode.get() in ('Webcam', 'Record only'):
            preview = getattr(self.engine, 'preview', None)
            if preview is None or time.perf_counter()-preview.last_frame_time > 2:
                self.record_status.configure(text='Wait for fresh camera frames before recording.')
                return
            try:
                self.recording_error = ''
                metadata = validate_identifiers({key: var.get() for key, var in self.collection.items()})
                self.recorder.start(self.record_scenario.get(), self.engine.camera_backend,
                                    metadata=metadata, camera_settings=preview.capture_metadata)
            except Exception as exc:
                self.recording_error = str(exc)
                self.record_status.configure(text=str(exc))

    def refresh_cameras(self):
        if self.busy or self.engine or self.scanning:
            return
        selected = self.camera_dropdown.current()
        previous = identity(self.cameras[selected]) if 0 <= selected < len(self.cameras) else None
        self.scanning = True
        self.set_controls(False)
        def worker():
            try:
                self.messages.put(('cameras', (discover_cameras(), previous, '')))
            except Exception as exc:
                self.messages.put(('cameras', ([], previous, str(exc))))
        threading.Thread(target=worker, daemon=True).start()

    def choose_weights(self):
        value = filedialog.askopenfilename(title='Select epoch-196 checkpoint', filetypes=[('Model weights', '*.pth')])
        if value:
            self.weights.set(value)

    def choose_video(self):
        value = filedialog.askopenfilename(title='Select a video', initialdir=str(DATA if DATA.is_dir() else BASE), filetypes=[('Videos', '*.mov *.mp4 *.avi *.mkv'), ('All files', '*.*')])
        if value:
            self.video.set(value)
            self.mode.set('Video')
            self.set_controls(False)

    def set_controls(self, running):
        for item in self.inputs:
            item.configure(state='disabled' if running else 'normal')
        if self.mode.get() == 'Record only':
            for item in self.inputs[:4]:
                item.configure(state='disabled')
        webcam = self.mode.get() in ('Webcam', 'Record only')
        self.start_button.configure(state='disabled' if self.stop_failed or running or (webcam and (self.scanning or not self.cameras)) else 'normal')
        self.refresh_button.configure(state='disabled' if running or self.scanning else 'normal')
        self.stop_button.configure(state='normal' if running and not self.busy else 'disabled')
        self.resolution.configure(state='readonly' if not running and webcam else 'disabled')
        self.sample_frames.configure(state='readonly' if not running and self.mode.get() != 'Record only' else 'disabled')
        self.camera_dropdown.configure(
            state='readonly' if not running and webcam and self.cameras and not self.scanning else 'disabled')

    def start(self):
        if self.busy or self.engine:
            return
        try:
            weights = Path(self.weights.get())
            video = self.video.get() if self.mode.get() == 'Video' else ''
            record_only = self.mode.get() == 'Record only'
            if not record_only and not weights.is_file():
                raise ValueError('Select the epoch-196 weights using Browse.')
            if self.mode.get() == 'Video' and not Path(video).is_file():
                raise ValueError('Select an existing video using Browse.')
            selected_camera = None
            if self.mode.get() in ('Webcam', 'Record only'):
                selected = self.camera_dropdown.current()
                if self.scanning or not 0 <= selected < len(self.cameras):
                    raise ValueError('No camera selected. Connect a camera and click Refresh.')
                selected_camera = self.cameras[selected]
            width, height = RESOLUTIONS[self.resolution.get()]
            frames = int(self.sample_frames.get())
        except ValueError as exc:
            self.status.configure(text=str(exc))
            return
        self.busy = True
        self.last_state = None
        self.set_controls(True)
        self.banner.configure(text='LOADING', fg=FG)
        self.status.configure(text='Opening camera…' if record_only else 'Loading checkpoint and opening source…')
        def worker():
            engine = None
            try:
                if record_only:
                    camera = NamedCamera(selected_camera, width, height, 30)
                    engine = RecordingSession(camera, self.recorder)
                    engine.start()
                else:
                    cfg = config(weights, video, sample_duration=frames)
                    engine = make_inference_engine(cfg, self.recorder)
                    camera = NamedCamera(selected_camera, width, height, 30) if selected_camera is not None else None
                    engine.start(camera=camera)
                self.messages.put(('started', engine))
            except Exception as exc:
                if engine:
                    try:
                        engine.stop()
                    except Exception:
                        pass
                self.messages.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        if self.busy:
            return
        if not self.engine:
            return
        self.busy = True
        self.recorder.stop()
        engine, self.engine = self.engine, None
        self.set_controls(True)
        self.banner.configure(text='STOPPING', fg=FG)
        def worker():
            try:
                engine.stop()
                self.messages.put(('stopped', None))
            except Exception as exc:
                self.messages.put(('stop_error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def close(self):
        self.closing = True
        self.stop()

    def update(self):
        while not self.messages.empty():
            kind, value = self.messages.get_nowait()
            if kind == 'cameras':
                self.cameras, previous, error = value
                self.scanning = False
                self.camera_dropdown.configure(values=display_names(self.cameras))
                if self.cameras:
                    selected = next((i for i, d in enumerate(self.cameras) if identity(d) == previous), 0)
                    self.camera_dropdown.current(selected)
                else:
                    self.camera_dropdown.set('Camera discovery failed' if error else 'No cameras detected')
                if not self.engine and not self.busy:
                    self.status.configure(text=('Camera discovery failed: ' + error) if error else f'{len(self.cameras)} camera(s) detected. Choose Record only or Webcam to use one.')
                self.set_controls(self.busy or self.engine is not None)
                continue
            self.busy = False
            if kind == 'started':
                self.engine = value
                self.set_controls(True)
            else:
                if kind == 'stop_error':
                    self.stop_failed = True
                self.set_controls(False)
                self.banner.configure(text='ERROR' if kind in ('error', 'stop_error') else 'IDLE', fg=FG)
                self.detail.configure(text='No active predictions')
                self.status.configure(text=value if kind in ('error', 'stop_error') else 'Stopped. Choose another source or press Start.')
        if self.closing:
            if self.engine and not self.busy:
                self.stop()
            if not self.busy and self.engine is None:
                self.root.destroy()
                return
        recording, recording_message = self.recorder.status()
        can_record = bool(self.engine) and not self.busy and self.mode.get() in ('Webcam', 'Record only')
        self.record_button.configure(text='Stop recording' if recording else 'Record 10 sec', state='normal' if can_record else 'disabled')
        self.record_scenario.configure(state='disabled' if recording else 'readonly')
        for entry in self.metadata_inputs:
            entry.configure(state='disabled' if recording else 'normal')
        session_id = self.collection['session_id'].get().strip()
        self.take_label.configure(text=f'Next take: {self.recorder.sessions.next_take(session_id):04d}')
        if self.recorder.active or hasattr(self.recorder, 'path'):
            self.record_status.configure(text=self.recording_error or recording_message)
        if self.engine:
            preview = getattr(self.engine, 'preview', None)
            if preview and preview.frame is not None:
                rgb = cv2.cvtColor(preview.frame, cv2.COLOR_BGR2RGB)
                picture = Image.fromarray(rgb)
                picture.thumbnail((max(1, self.preview_label.winfo_width()), max(1, self.preview_label.winfo_height())))
                self.photo = ImageTk.PhotoImage(picture)
                self.preview_label.configure(image=self.photo, text='')
            stale = not preview or time.perf_counter() - preview.last_frame_time > 2
            if getattr(self.engine, 'record_only', False):
                failed = self.engine.error
                self.banner.configure(text='CAMERA ERROR' if failed else 'RECORDING' if recording else 'RECORD ONLY', fg=FG)
                self.detail.configure(text=failed or ('Waiting for camera frames' if stale else 'Camera preview • No model loaded'))
                for label, (title, bar, fill) in zip(LABELS, self.bars):
                    title.configure(text=label + '   —')
                    bar.coords(fill, 0, 0, 0, 17)
                info = preview.capture_metadata
                delivered = info.get('delivered_frame_size', ['?', '?'])
                driver_fps = info.get('driver_reported_fps')
                rate = f'{driver_fps:g}' if driver_fps else 'unknown'
                self.status.configure(text=f'{self.engine.camera_backend} • Delivered {delivered[0]} × {delivered[1]} • Requested 30 FPS / driver reports {rate} • Check saved capture timing')
                if failed:
                    self.record_button.configure(state='disabled')
                self.root.after(50, self.update)
                return
            state = self.engine.get_state()
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
            self.status.configure(text=f"Device: {self.engine.device} • Predictions/sec: {state['infer_fps']:.1f} • {state['camera']} • Display only" + (' • Camera setup not validated' if self.mode.get() == 'Webcam' else ''))
        self.root.after(50, self.update)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Run the legacy video inference smoke test.')
    parser.add_argument('--ui-smoke-test', action='store_true', help='Check UI controls without opening a camera or model.')
    args = parser.parse_args()
    root = tk.Tk()
    app = Monitor(root, scan_cameras=not args.ui_smoke_test)
    if args.ui_smoke_test:
        import sys
        assert 'torch' not in sys.modules
        for mode in ['Record only', 'Video', 'Webcam']:
            app.mode.set(mode)
            app.set_controls(False)
        app.mode.set('Record only')
        app.set_controls(False)
        root.after(400, app.close)
    elif args.smoke_test and not args.record_only:
        app.mode.set('Video')
        app.set_controls(False)
        root.after(300, app.start)
        root.after(18000, app.close)
    elif args.smoke_test:
        root.after(400, app.close)
    root.mainloop()
