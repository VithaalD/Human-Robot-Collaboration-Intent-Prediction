"""Desktop launcher for the local fusion viewer; uses the existing GPU environment."""
from pathlib import Path
import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT=Path(__file__).resolve().parent
PYTHON=Path(sys.executable).with_name('python.exe')
RUNTIME=ROOT/'.runtime/launcher'


def viewer_command(mode, value=None, stop_file=None):
    command=[str(PYTHON),str(ROOT/'tools/run_fusion_viewer.py')]
    if mode=='demo':command+=['--demo']
    elif mode=='video':command+=['--video',str(value)]
    elif mode=='camera':
        index=int(value)
        if index<0:raise ValueError('Camera index must be zero or greater')
        command+=['--camera',str(index)]
    else:raise ValueError('Unknown source')
    if stop_file is not None:command+=['--stop-file',str(stop_file)]
    return command


class Launcher:
    def __init__(self, root):
        self.root=root; self.process=None; self.log=None; self.closing=False; self.stopping=False
        root.title('Fusion Viewer'); root.geometry('560x440'); root.resizable(False,False)
        root.configure(bg='#101923')
        style=ttk.Style(root); style.theme_use('clam')
        style.configure('TButton',font=('Segoe UI',11),padding=12)
        panel=tk.Frame(root,bg='#101923'); panel.pack(fill='both',expand=True,padx=28,pady=24)
        tk.Label(panel,text='Fusion Viewer',font=('Segoe UI',25,'bold'),fg='white',bg='#101923').pack(anchor='w')
        tk.Label(panel,text='Object detection + intent prediction',font=('Segoe UI',11),fg='#a9bdcf',bg='#101923').pack(anchor='w',pady=(2,20))
        self.buttons=[]
        self.button(panel,'Play lab demo',lambda:self.start('demo'))
        self.button(panel,'Choose a video…',self.choose_video)
        row=tk.Frame(panel,bg='#101923'); row.pack(fill='x',pady=6)
        self.camera_index=tk.StringVar(value='0')
        self.button(row,'Start live camera',lambda:self.start('camera',self.camera_index.get()),pack=False).pack(side='left',fill='x',expand=True)
        tk.Label(row,text='Index',bg='#101923',fg='#a9bdcf').pack(side='left',padx=(12,4))
        ttk.Spinbox(row,from_=0,to=20,textvariable=self.camera_index,width=4).pack(side='left')
        self.stop=ttk.Button(panel,text='Stop viewer',command=self.stop_viewer,state='disabled'); self.stop.pack(fill='x',pady=6)
        self.status=tk.StringVar(value='Ready. Choose a video or camera.')
        tk.Label(panel,textvariable=self.status,wraplength=500,justify='left',font=('Segoe UI',10),fg='#83d9bb',bg='#101923').pack(anchor='w',pady=(14,6))
        tk.Label(panel,text='Camera 0 may be your laptop webcam. Select the RealSense RGB index in the lab.\nExperimental model • display only • Q/Esc closes the viewer.',wraplength=500,justify='left',font=('Segoe UI',9),fg='#a9bdcf',bg='#101923').pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.after(250,self.poll)

    def button(self,parent,text,command,pack=True):
        button=ttk.Button(parent,text=text,command=command); self.buttons.append(button)
        if pack:button.pack(fill='x',pady=6)
        return button

    def choose_video(self):
        path=filedialog.askopenfilename(title='Choose a video',filetypes=[('Videos','*.avi *.mp4 *.mov *.mkv'),('All files','*.*')])
        if path:self.start('video',path)

    def start(self,mode,value=None):
        if self.process is not None:return
        try:
            RUNTIME.mkdir(parents=True,exist_ok=True)
            token=str(time.time_ns()); self.stop_file=RUNTIME/(token+'.stop')
            command=viewer_command(mode,value,self.stop_file)
            self.log_path=RUNTIME/(token+'.log'); self.log=self.log_path.open('w',encoding='utf-8')
            env=os.environ.copy(); env['YOLO_CONFIG_DIR']=str(ROOT/'.runtime/ultralytics'); env['PYTHONDONTWRITEBYTECODE']='1'
            self.process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=self.log,stderr=subprocess.STDOUT,
                                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.stopping=False
            for b in self.buttons:b.configure(state='disabled')
            self.stop.configure(state='normal'); self.status.set('Starting viewer… model loading can take 20–30 seconds.')
        except Exception as exc:
            if self.log:self.log.close(); self.log=None
            self.process=None; messagebox.showerror('Unable to start viewer',str(exc))

    def stop_viewer(self):
        if self.process is not None and not self.stopping:
            self.stopping=True; self.stop_file.touch(); self.status.set('Stopping viewer…')
            self.stop.configure(state='disabled')
            process=self.process
            self.root.after(5000,lambda:self.force_stop(process))

    def force_stop(self,process):
        if self.process is process and process.poll() is None:process.terminate()

    def poll(self):
        if self.process is not None:
            code=self.process.poll()
            if code is not None:
                self.process=None
                if self.log:self.log.close(); self.log=None
                self.stop.configure(state='disabled')
                for b in self.buttons:b.configure(state='normal')
                self.status.set('Viewer stopped. Choose another source.' if self.stopping or code==0 else 'Viewer could not run. See the error message.')
                if code!=0 and not self.stopping:
                    tail='\n'.join(self.log_path.read_text(errors='replace').splitlines()[-14:])
                    messagebox.showerror('Viewer error',tail)
                if self.closing:self.root.destroy(); return
        self.root.after(250,self.poll)

    def close(self):
        if self.process is None:self.root.destroy()
        else:self.closing=True; self.stop_viewer()


if __name__=='__main__':
    root=tk.Tk(); app=Launcher(root)
    if '--smoke-test' in sys.argv:root.after(700,root.destroy)
    root.mainloop()
