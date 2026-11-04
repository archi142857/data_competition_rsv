# 유행 시기 지표 비교 (보고서 3.2절): 그림 3
from rsv_core import *
from plot_style import *

c, W, yw, info = load(DATA_DIR)
P = list(W.columns); idx = {p: i for i, p in enumerate(P)}; isl = info.island.to_dict()

res = []
for s in (1, 14, 27, 40):
    t0 = yw.index[(yw.year == 2022) & (yw.week == s)][0]
    arr = W.loc[t0:t0 + 51].to_numpy()
    pk = arr.argmax(0)
    com = (arr * np.arange(52)[:, None]).sum(0) / arr.sum(0)
    out = {'start': s, 'n_peak_last_week': int((pk == 51).sum())}
    for I in ('Honshu', 'Kyushu', 'Shikoku'):  # 홋카이도, 오키나와는 현이 하나뿐
        ps = [p for p in P if isl[p] == I]
        out[f'{I}_peak'] = np.median([abs(pk[idx[x]] - pk[idx[y]]) for x, y in combinations(ps, 2)])
        out[f'{I}_com'] = round(float(np.median([abs(com[idx[x]] - com[idx[y]]) for x, y in combinations(ps, 2)])), 2)
    res.append(out)
SB = pd.DataFrame(res); display(SB)

fig, axes = plt.subplots(1, 2, figsize=(WD, 2.45))
for ax, key, ttl in [(axes[0], 'peak', '최고 주 기준 차이'), (axes[1], 'com', '무게중심(평균 시기) 기준 차이')]:
    for I, col, lab in [('Honshu', B1, '혼슈'), ('Kyushu', B2, '규슈'), ('Shikoku', B3, '시코쿠')]:
        dot(ax, SB.start, SB[f'{I}_{key}'], col, label=lab)
    clean(ax); ax.set_xticks(SB.start); ax.set_xlabel('분석 창 시작 주 (2022년)'); ax.set_title(ttl, fontsize=8.4, pad=4); ax.set_ylim(0, 42)
axes[0].set_ylabel('같은 섬 현 쌍 차이의 중앙값 (주)'); axes[1].legend(loc='upper right')
# 글자 위치는 눈으로 맞춤
axes[0].annotate('13개 현의 최고 주가\n창의 마지막 주에 위치', (26.4, 38), xytext=(13.5, 37), fontsize=7, color=INK2, ha='right', va='center',
                 arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.7))
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig3_boundary.png'); plt.close(fig)
