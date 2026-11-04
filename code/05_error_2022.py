# 2022년 원래 단위 오차 진단 (보고서 5.3절): 그림 5
from rsv_core import *
from plot_style import *

c, W, yw, info = load(DATA_DIR)
E = evaluate_years(build_panel(W, yw, info, h=4, k=3), W, yw)

d = E[E.eval_year == 2022].copy()
a, b = 'pred_M2 인근 3현 추가', 'pred_M1 자기 현 이력'
d['dSE'] = (d[a] - d.y_tgt)**2 - (d[b] - d.y_tgt)**2  # +면 M2가 더 틀림
bp = d.groupby('pref').dSE.sum().sort_values(ascending=False)
print('제곱 오차 증가분 중 상위 3개 현의 비중: %.2f' % (bp.head(3).sum() / d.dSE.sum()), bp.head(3).round(0).to_dict())
display(d.sort_values('dSE', ascending=False).head(8)[['pref', 'o_week', 't_week', 'y_now', 'y_tgt', b, a]].round(1))

peak = {p: int(yw.loc[W[p].loc[yw.index[yw.year == 2022]].idxmax(), 'week']) for p in ['Aichi', 'Osaka', 'Hyogo']}
print('2022년 정점 주:', peak)
# 본문 오사카 예시 (29주에 33주 예측)
display(d[(d.pref == 'Osaka') & (d.o_week == 29)][['o_week', 't_week', 'y_now', 'y_tgt', b, a]].round(0))

fig, axes = plt.subplots(1, 2, figsize=(WD, 2.45))
for ax, p, nm in [(axes[0], 'Osaka', '오사카'), (axes[1], 'Aichi', '아이치')]:
    g = d[d.pref == p].sort_values('t_week'); obs = W[p].loc[yw.index[yw.year == 2022]]
    ax.plot(np.arange(1, len(obs) + 1), obs.values, color=INK2, linewidth=1.2, label='실제 보고 수')
    ax.plot(g.t_week, g[b], color=B1, linewidth=1.3, label='M1 자기 현 이력'); ax.plot(g.t_week, g[a], color=B2, linewidth=1.3, label='M2 인근 3현 추가')
    ax.axvspan(30.5, 35.5, color='#f0efec', zorder=0)  # 31~35주
    clean(ax); ax.set_xlim(1, 52); ax.set_xlabel('목표 주 (2022년)'); ax.set_title(nm, fontsize=8.4, pad=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, q: f'{int(v):,}'))
axes[0].set_ylabel('주간 보고 수'); axes[0].legend(loc='upper left', fontsize=7.2)
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig5_2022_examples.png'); plt.close(fig)
