# 그림 설정. 보고서 그림과 같은 색, 같은 크기
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm
import matplotlib.ticker as mticker

B1, B2, B3 = '#2a78d6', '#eb6834', '#1baf7a'; INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#898781'; GRID = '#e1e0d9'; BASE = '#c3c2b7'
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

# 한글 폰트
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  # findfont 경고가 너무 많이 떠서
_avail = {f.name for f in fm.fontManager.ttflist}
KFONT = next((f for f in ['Malgun Gothic', 'AppleGothic', 'Noto Sans CJK KR', 'NanumGothic'] if f in _avail), 'DejaVu Sans')
print('그림 글꼴:', KFONT)
plt.rcParams.update({'font.family': KFONT,
                     'font.size': 8.3, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 9.2,
                     'axes.titleweight': 'normal', 'axes.titlelocation': 'left', 'axes.unicode_minus': False,
                     'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.facecolor': 'white', 'legend.frameon': False, 'legend.fontsize': 7.8})

def clean(ax, ygrid=True):
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if ygrid: ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

def dot(ax, x, y, col, **kw):
    ax.plot(x, y, color=col, linewidth=1.3, marker='o', markersize=4.6, markeredgecolor='white', markeredgewidth=1.0, **kw)

WD = 6.7  # inch
