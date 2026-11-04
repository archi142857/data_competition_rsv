# 현 사이 동조성 (보고서 3장, 3.1절): 그림 1, 그림 2
from rsv_core import *
from plot_style import *

c, W, yw, info = load(DATA_DIR)
D = distance_matrix(info)

order = info.lat.sort_values(ascending=False).index
M = W[order].T
R = M.div(M.quantile(0.99, axis=1).replace(0, 1), axis=0).clip(upper=1.0)
yr0 = yw.reset_index().groupby('year').time.min()

fig, ax = plt.subplots(figsize=(WD, 5.2))
im = ax.imshow(R.to_numpy(), aspect='auto', cmap=LinearSegmentedColormap.from_list('seq', ['#ffffff'] + SEQ), vmin=0, vmax=1,
               interpolation='nearest', extent=[0.5, M.shape[1] + 0.5, len(order) - 0.5, -0.5])
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5.4)
ax.set_xticks(yr0.values); ax.set_xticklabels(yr0.index.astype(str))
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
for t in yr0.values: ax.axvline(t - 0.5, color='white', linewidth=0.6)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=7)
cb.set_label('현별 상대 강도 (각 현의 99백분위수 = 1)', color=INK2, fontsize=7.3)
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig1_heatmap.png'); plt.close(fig)

L = np.log1p(W); P = list(W.columns); idx = {p: i for i, p in enumerate(P)}; isl = info.island.to_dict()
# 2016(전환기), 2020(공백)은 뺌
periods = {'2013–2015 (겨울 유행기)': (2013, 2015), '2017–2019 (가을 유행기)': (2017, 2019), '2021–2023 (코로나 이후)': (2021, 2023)}
rows = []
for nm, (a, b) in periods.items():
    t = yw.index[(yw.year >= a) & (yw.year <= b)]
    Cm = np.corrcoef(L.loc[t].to_numpy().T)
    for p, q in combinations(P, 2):
        rows.append({'period': nm, 'r': Cm[idx[p], idx[q]], 'dist': D[idx[p], idx[q]], 'same': isl[p] == isl[q]})
S = pd.DataFrame(rows)
bins = ['<150', '150–300', '300–500', '500–800', '>800']
S['dbin'] = pd.cut(S.dist, [0, 150, 300, 500, 800, 3000], labels=bins)
display(S.groupby(['period', 'same', 'dbin'], observed=True).r.mean().unstack('dbin').round(2))  # same=True: 같은 섬
print('기간별 전체 평균 상관:', S.groupby('period', sort=False).r.mean().round(3).to_dict())

fig, axes = plt.subplots(1, 3, figsize=(WD, 2.45), sharey=True)
for ax, (per, g) in zip(axes, S.groupby('period', sort=False)):
    tb = g.groupby(['same', 'dbin'], observed=True).r.mean().unstack('dbin')[bins]
    x = np.arange(len(bins))
    dot(ax, x, tb.loc[True].values, B1, label='같은 섬'); dot(ax, x, tb.loc[False].values, B2, label='다른 섬')
    clean(ax); ax.set_xticks(x); ax.set_xticklabels(bins, fontsize=6.6); ax.set_title(per, fontsize=8.2, pad=4); ax.set_ylim(0.2, 0.9)
axes[0].set_ylabel('현 쌍 상관계수 평균'); axes[1].set_xlabel('현 대표 좌표 사이 거리 (km)'); axes[0].legend(loc='lower left')
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig2_sync.png'); plt.close(fig)
