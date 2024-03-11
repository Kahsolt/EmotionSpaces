#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/11

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
from functools import partial
from argparse import ArgumentParser
from traceback import print_exc, format_exc

import yaml
from PIL import Image
from PIL.ImageTk import PhotoImage
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from train import *

device = 'cpu'

WINDOW_TITLE = 'EmotionSpaces'
WINDOW_SIZE  = (1000, 800)
IMG_RESIZE = (384, 384)


class App:

  def __init__(self, model):
    self.model: MultiTaskNet = model

    # {'head': [var]} for discreted heads
    self.var_heads: Dict[str, List[tk.DoubleVar]] = {}
    # {'head': [lbl]} for discreted heads
    self.lbl_heads: Dict[str, List[ttk.Label]] = {}

    self.setup_gui()
    self.setup_inits()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_inits(self):
    for vars in self.var_heads.values():
      v_init = 1 / len(vars)
      for var in vars:
        var.set(v_init)

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    W, H = wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    w, h = WINDOW_SIZE
    wnd.geometry(f'{w}x{h}+{(W-w)//2}+{(H-h)//2}')
    #wnd.resizable(False, False)
    wnd.title(WINDOW_TITLE)
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: open file
    frm1 = ttk.Label(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      self.var_fp = tk.StringVar(wnd)
      frm11 = ttk.Entry(frm1, textvariable=self.var_fp)
      frm11.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)

      btn = tk.Button(frm1, text='Open..', command=self.open_)
      btn.pack(side=tk.RIGHT)

    # bottom: display & controls
    frm2 = ttk.Frame(wnd)
    frm2.pack(expand=tk.YES, fill=tk.BOTH)
    if True:
      # left: img + VA
      frm21 = ttk.Frame(frm2)
      frm21.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
      if True:
        # img
        frm211 = ttk.LabelFrame(frm21, text='Image')
        frm211.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
        if True:
          pv = ttk.Label(frm211, anchor=tk.CENTER)
          pv.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
          self.pv: ttk.Label = pv

        # VA
        frm212 = ttk.LabelFrame(frm21, text=HeadType.VA.value)
        frm212.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
        if True:
          head = 'VA'
          self.var_heads[head] = []
          for ihead in range(HEAD_DIMS[head]):
            self.var_heads[head].append(tk.DoubleVar(wnd))
          class_names = HEAD_CLASS_NAMES[head]

          # https://blog.csdn.net/qq_44864262/article/details/107738440
          fig = plt.figure(figsize=(4, 4))
          fig.tight_layout()
          ax = fig.gca()
          ax.annotate('', xy=(10.1, 0), xytext=(-10.1, 0), arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3'})
          ax.annotate('', xy=(0, 10.1), xytext=(0, -10.1), arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3'})
          ax.text(7.0, 0.5, class_names[0])
          ax.text(0.5, 9.0, class_names[1])
          ax.set_xlim(-10.1, 10.1)
          ax.set_ylim(-10.1, 10.1)
          cvs = FigureCanvasTkAgg(fig, frm212)
          cvstk = cvs.get_tk_widget()
          cvstk.bind('<Button-1>', lambda _: tkmsg.showinfo('are you ok?'))
          cvstk.pack(expand=tk.YES, fill=tk.BOTH)
          if not 'toolbar':
            toolbar = NavigationToolbar2Tk(cvs, frm212, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
          self.fig, self.ax, self.cvs = fig, ax, cvs

      # right: discreted spaces
      frm22 = ttk.Frame(frm2)
      frm22.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)
      if True:
        for head in [e.value for e in [HeadType.Polar, HeadType.Ekman, HeadType.EkmanN, HeadType.Mikels]]:
          refresh_fn = lambda _: partial(self.refresh, head)()
          frm22x = ttk.LabelFrame(frm22, text=head)
          frm22x.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
          if True:
            self.var_heads[head] = []
            self.lbl_heads[head] = []
            for ihead in range(HEAD_DIMS[head]):
              frm22xy = ttk.Frame(frm22x)
              frm22xy.pack(side=tk.LEFT, expand=tk.YES, fill=tk.Y)
              if True:
                var = tk.DoubleVar(wnd)
                self.var_heads[head].append(var)
                sc = tk.Scale(frm22xy, command=refresh_fn, variable=var, orient=tk.VERTICAL, from_=1.0, to=0.0, resolution=0.001)
                sc.pack(side=tk.TOP, expand=tk.YES, fill=tk.Y)
                lbl = ttk.Label(frm22xy, text=HEAD_CLASS_NAMES[head][ihead], foreground='blue')
                lbl.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.Y)
                self.lbl_heads[head].append(lbl)

  def open_(self):
    fp = tkfdlg.askopenfilename()
    if not fp: return
    if not Path(fp).is_file():
      tkmsg.showerror('Error', f'path {fp} is not a file!')
      return
    self.var_fp.set(fp)

    img = Image.open(fp).convert('RGB')
    self.img = img
    img = PhotoImage(img.resize(IMG_RESIZE))
    self.pv.configure(image=img)
    self.pv.img = img
    self.refresh()

  @torch.inference_mode()
  def refresh(self, head:str=None):
    def get_head_vars(head:str) -> ndarray:
      return np.asarray([var.get() for var in self.var_heads[head]], dtype=np.float32)
    def set_head_vars(head:str, vals:ndarray):
      # probdist norm
      if is_clf(head) and vals.sum() > 0:
        vals /= vals.sum()
      # update widgets
      vars = self.var_heads[head]
      for var, val in zip(vars, vals):
        var.set(val.item())
      # highlight the hotest label
      if head in self.lbl_heads:
        lbls = self.lbl_heads[head]
        idx = np.argmax(vals)
        for i, lbl in enumerate(lbls):
          lbl['foreground'] = 'red' if i == idx else 'blue'

    if head is None:    # predict on image
      img = transform_test(self.img).unsqueeze(dim=0).to(device)
      for head in self.model.heads:
        ev = self.model.infer(img, head)[0].cpu().numpy()
        set_head_vars(head, ev)
    else:               # space tranx
      # probdist renorm
      set_head_vars(head, get_head_vars(head))
      # inv to Xspace
      ev = torch.from_numpy(get_head_vars(head)).float().unsqueeze(dim=0).to(device)
      xv = self.model.ev_to_xv(ev, head)
      # map to Espaces
      for to_head in self.model.heads:
        if to_head == head: continue
        ev = self.model.xv_to_ev(xv, to_head)[0].cpu().numpy()
        set_head_vars(to_head, ev)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--load', type=Path, help='load pretrained weights')
  args = parser.parse_args()

  ''' Model & Ckpt '''
  if not args.load:
    print('>> You are NOT loading any pretrained weights, predictions should be nonsense!!')
    print('>> This should only happen in bare testing case')
    model = MultiTaskNet(pretrain=True)
  else:
    fp = Path(args.load).parent.parent / 'hparams.yaml'
    with open(fp, 'r', encoding='utf-8') as fh:
      hp = yaml.safe_load(fh)
    model = MultiTaskNet(hp['model'], hp['head'], hp['d_x'], pretrain=False)
    model = LitModel.load_from_checkpoint(args.load, model=model).model.to(device).eval()

  App(model)
