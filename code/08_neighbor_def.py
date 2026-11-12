# 인근 현 정의를 바꿔 보기 (최종보고서 6.2절): 표 8, 그림 8
from rsv_core import *
from plot_style import *

c, W, yw, info = load(DATA_DIR)
PAIRS = [(M2, M1), (M4, M3)]

def run(nb_w=None, k=3):
    return evaluate_years(build_panel(W, yw, info, k=k, nb_w=nb_w), W, yw)

def summary(E, label):
    ef = effect_table(E, PAIRS)
    r = {'인근 정의': label, 'M2': log_mae(E, M2), 'M4': log_mae(E, M4)}
    for a, b in PAIRS:
        x = ef[ef['비교'] == f'{a} - {b}'].set_index('연도')
        for Y in (2021, 2022, 2023, '전체'):
            r[f'{a[:2]}-{b[:2]} {Y}'] = x.loc[Y, '차이']
        r[f'{a[:2]}-{b[:2]} 하한'] = x.loc['전체', '하한']; r[f'{a[:2]}-{b[:2]} 상한'] = x.loc['전체', '상한']
    return r

# 1) 현의 수, 같은 섬, 거리 가중
D = distance_matrix(info); Dn = D.copy(); np.fill_diagonal(Dn, np.inf)
defs = [(f'가까운 {k}현' + (' (기본)' if k == 3 else ''), dict(k=k)) for k in range(1, 6)]
defs += [('같은 섬 안에서 가까운 3현', dict(nb_w=nb_weights(info, 3, 'island'))),
         ('가까운 3현, 거리 역수 가중', dict(nb_w=nb_weights(info, 3, 'invdist'))),
         ('가까운 5현, 거리 역수 가중', dict(nb_w=nb_weights(info, 5, 'invdist')))]
E3 = run()
T8 = pd.DataFrame([summary(run(**kw), nm) for nm, kw in defs]); save(T8, 'neighbor_defs.csv'); display(T8.round(3))
wi = nb_weights(info, 3, 'island'); w3 = nb_weights(info, 3)
print('같은 섬 제한으로 인근이 바뀐 현: %d곳' % int(((wi > 0) != (w3 > 0)).any(axis=1).sum()))
island_changed = int(((wi > 0) != (w3 > 0)).any(axis=1).sum())

# 2) 섬별 효과 (기본 정의). 시코쿠는 4현, 홋카이도와 오키나와는 1현이라 구간은 혼슈, 규슈만
print(info.island.value_counts().to_dict())
rows = []
for I, g in info.groupby('island'):
    d = E3[E3.pref.isin(g.index)]
    r = {'섬': I, '현 수': len(g), 'M1': log_mae(d, M1), 'M2': log_mae(d, M2), 'M3': log_mae(d, M3), 'M4': log_mae(d, M4)}
    for a, b in PAIRS:
        pt, lo, hi = cluster_boot_diff(d, 'pred_' + a, 'pred_' + b)
        r[f'{a[:2]}-{b[:2]}'] = pt
        if len(g) >= 7: r[f'{a[:2]}-{b[:2]} 하한'] = lo; r[f'{a[:2]}-{b[:2]} 상한'] = hi
    rows.append(r)
T9 = pd.DataFrame(rows).sort_values('현 수', ascending=False); save(T9, 'island_effects.csv'); display(T9.round(3))

# 3) 대조 실험: 가까운 3현 대신 더 먼 현, 아무 현 3곳을 넣었을 때
rows = []
for s in range(0, 45, 3):
    wt = nb_weights(info, 3, 'rank', start=s)
    E = run(nb_w=wt)
    dist = float(np.median(Dn[wt.to_numpy() > 0]))
    blk = block_boot_diff(E, 'pred_' + M4, 'pred_' + M3)  # 주 묶음으로 다시 뽑았을 때도 구간이 0을 벗어나는지
    rows.append({'거리 순위': f'{s + 1}~{s + 3}', '시작 순위': s + 1, '거리 중앙값(km)': dist, **{k: v for k, v in summary(E, '').items() if k != '인근 정의'},
                 'M4-M3 주 묶음 하한': blk[1], 'M4-M3 주 묶음 상한': blk[2]})
RK = pd.DataFrame(rows); save(RK, 'neighbor_rank.csv')
display(RK[['거리 순위', '거리 중앙값(km)', 'M2-M1 전체', 'M4-M3 전체', 'M4-M3 하한', 'M4-M3 상한', 'M4-M3 주 묶음 하한', 'M4-M3 주 묶음 상한']].round(3))
fx = {}
for s in (1, 4):
    r = RK.set_index('시작 순위').loc[s]; fx[f'blk{s}_lo'] = r['M4-M3 주 묶음 하한']; fx[f'blk{s}_hi'] = r['M4-M3 주 묶음 상한']

# 인구가 비슷한 현이라서 닮은 것은 아닌지: 거리와 상관없이 인구가 가장 비슷한 3현
wp = nb_weights(info, 3, 'pop'); Ep = run(nb_w=wp)
ef = effect_table(Ep, PAIRS); ef = ef[ef['연도'] == '전체'].set_index('비교')
print('인구가 비슷한 3현 (거리 중앙값 %.0f km):' % np.median(Dn[wp.to_numpy() > 0])); display(ef.round(3))
for a, b in PAIRS:
    k = f'{a[:2]}{b[:2]}'; r = ef.loc[f'{a} - {b}']
    fx[f'pop_{k}'] = r['차이']; fx[f'pop_{k}_lo'] = r['하한']; fx[f'pop_{k}_hi'] = r['상한']
fx['pop_dist'] = np.median(Dn[wp.to_numpy() > 0])

NR = 200  # 무작위 반복 수. 한 번에 1초쯤 걸림
rnd = []
for seed in range(NR):
    E = run(nb_w=nb_weights(info, 3, 'random', seed=seed))
    rnd.append({'seed': seed, 'M2-M1': log_mae(E, M2) - log_mae(E, M1), 'M4-M3': log_mae(E, M4) - log_mae(E, M3),
                **{f'M2-M1 {Y}': log_mae(E[E.eval_year == Y], M2) - log_mae(E[E.eval_year == Y], M1) for Y in (2021, 2022, 2023)},
                **{f'M4-M3 {Y}': log_mae(E[E.eval_year == Y], M4) - log_mae(E[E.eval_year == Y], M3) for Y in (2021, 2022, 2023)}})
RD = pd.DataFrame(rnd); save(RD, 'neighbor_random.csv')
facts('neighbor', {'island_changed': island_changed, **fx, 'rand_neg_M4M3': int((RD['M4-M3'] < 0).sum())})
base = {'M2-M1': log_mae(E3, M2) - log_mae(E3, M1), 'M4-M3': log_mae(E3, M4) - log_mae(E3, M3)}
for k in ('M2-M1', 'M4-M3'):
    q = RD[k].quantile([0.025, 0.5, 0.975])
    print('%s | 가까운 3현 %.3f | 무작위 3현 중앙값 %.3f [%.3f, %.3f] | 무작위가 가까운 3현보다 나은 횟수 %d/%d' % (
        k, base[k], q[0.5], q[0.025], q[0.975], int((RD[k] <= base[k]).sum()), NR))
for Y in (2021, 2022, 2023):
    d = E3[E3.eval_year == Y]
    print(Y, '| M4-M3 가까운 3현 %.3f, 무작위 중앙값 %.3f [%.3f, %.3f]' % (log_mae(d, M4) - log_mae(d, M3), *RD[f'M4-M3 {Y}'].quantile([0.5, 0.025, 0.975])))

fig, axes = plt.subplots(1, 2, figsize=(WD, 2.6), sharey=True)
for ax, k, ttl in ((axes[0], 'M2-M1', '자기 현 모델에 더했을 때 (M2 − M1)'), (axes[1], 'M4-M3', '전국 모델에 더했을 때 (M4 − M3)')):
    lo, hi = RD[k].quantile([0.025, 0.975])
    ax.axhspan(lo, hi, color='#f0efec', zorder=0)
    ax.axhline(RD[k].median(), color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    dot(ax, RK['거리 중앙값(km)'], RK[k + ' 전체'], B1 if k == 'M2-M1' else B3)
    ax.axhline(0, color=INK2, linewidth=0.7); clean(ax); ax.set_title(ttl, fontsize=8.2, pad=4); ax.set_xlabel('입력에 쓴 3현까지 거리의 중앙값 (km)')
axes[0].set_ylabel('로그 MAE 차이 (음수 = 개선)')
axes[1].text(RK['거리 중앙값(km)'].max(), RD['M4-M3'].quantile(0.975) + 0.004, '무작위 3현 (95% 범위)', fontsize=7, color=INK2, ha='right', va='bottom')
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig8_distance.png'); plt.close(fig)
