#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/11

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfdlg
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
WINDOW_SIZE  = (912, 860)
IMG_RESIZE = (384, 384)

PLOT_VA_VLIM = 2
TEXT_VA_OFFSET = 0.1
TEXT_V_OFFSET = 0.7
TEXT_A_OFFSET = 0.2
TEXT_MARK_OFFSET = 0.15
EPS = 1e-3


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
      # left: img + VA plot
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
          ax: Axes = fig.gca()
          arrowprops = {'arrowstyle': '->', 'connectionstyle': 'arc3'}
          ax.annotate('', xy=(PLOT_VA_VLIM+0.1, 0), xytext=(-PLOT_VA_VLIM-0.1, 0), arrowprops=arrowprops)
          ax.annotate('', xy=(0, PLOT_VA_VLIM+0.1), xytext=(0, -PLOT_VA_VLIM-0.1), arrowprops=arrowprops)
          ax.text(PLOT_VA_VLIM-TEXT_V_OFFSET, TEXT_VA_OFFSET, class_names[0])
          ax.text(TEXT_VA_OFFSET, PLOT_VA_VLIM-TEXT_A_OFFSET, class_names[1])
          ax.set_xlim(-PLOT_VA_VLIM-0.1, PLOT_VA_VLIM+0.1)
          ax.set_ylim(-PLOT_VA_VLIM-0.1, PLOT_VA_VLIM+0.1)
          self.mark = ax.text(0, 0, '★', c='r', fontsize='xx-large')
          cvs = FigureCanvasTkAgg(fig, frm212)
          cvstk = cvs.get_tk_widget()
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
        # VA and Polar share the same row
        frm22z = ttk.Frame(frm22)
        frm22z.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
        if True:
          for head in [e.value for e in [HeadType.VA, HeadType.Polar]]:
            self.setup_gui_control_group(head, frm22z, tk.LEFT)
        # one per row for others
        for head in [e.value for e in [HeadType.Ekman, HeadType.EkmanN, HeadType.Mikels]]:
          self.setup_gui_control_group(head, frm22, tk.TOP)

  def setup_gui_control_group(self, head:str, master:tk.Widget, side:str=tk.TOP):
    if head == 'VA':
      vstart, vend, vstep = PLOT_VA_VLIM, -PLOT_VA_VLIM, 0.1
    else:
      vstart, vend, vstep = 1.0, 0.0, 0.001

    frm = ttk.LabelFrame(master, text=head)
    frm.pack(side=side, expand=tk.YES, fill=tk.BOTH)
    if True:
      # NOTE: here head must be provided as default value for early value binding :(
      refresh_fn = lambda value, head=head: self.refresh(head)
      class_names = HEAD_CLASS_NAMES[head]
      self.var_heads[head] = []
      self.lbl_heads[head] = []
      for ihead in range(HEAD_DIMS[head]):
        sfrm = ttk.Frame(frm)
        sfrm.pack(side=tk.LEFT, expand=tk.YES, fill=tk.Y)
        if True:
          var = tk.DoubleVar(self.wnd)
          sc = tk.Scale(sfrm, command=refresh_fn, variable=var, orient=tk.VERTICAL, from_=vstart, to=vend, resolution=vstep)
          sc.pack(side=tk.TOP, expand=tk.YES, fill=tk.Y)
          lbl = ttk.Label(sfrm, text=class_names[ihead], foreground='blue')
          lbl.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.Y)
          self.var_heads[head].append(var)
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

  def refresh(self, head:str=None):
    def get_head_vars(head:str) -> ndarray:
      return np.asarray([max(var.get(), EPS) for var in self.var_heads[head]], dtype=np.float32)
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
      if head != 'VA': set_head_vars(head, get_head_vars(head))
      # inv to Xspace
      ev = torch.from_numpy(get_head_vars(head)).float().unsqueeze(dim=0).to(device)
      xv = self.model.ev_to_xv(ev, head)
      # map to Espaces
      for to_head in self.model.heads:
        if to_head == head: continue
        ev = self.model.xv_to_ev(xv, to_head)[0].cpu().numpy()
        set_head_vars(to_head, ev)

    # update the mark on VA-plot
    var_V, var_A = self.var_heads['VA']
    self.mark.set_x(var_V.get() - TEXT_MARK_OFFSET)
    self.mark.set_y(var_A.get() - TEXT_MARK_OFFSET)
    self.cvs.draw()


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
      hp = yaml.unsafe_load(fh)
    model = MultiTaskNet(hp['model'], hp['head'], hp['d_x'], pretrain=False)
    model = LitModel.load_from_checkpoint(args.load, model=model).model.to(device).eval()

  App(model)
