# 예측 평가 (보고서 4장, 5.1~5.2절): 표 2, 표 3, 신뢰구간, 그림 4
from rsv_core import *
from plot_style import *

c, W, yw, info = load(DATA_DIR)

Pn = build_panel(W, yw, info, h=4, k=3)
E = evaluate_years(Pn, W, yw)
print('연도별 평가 행 수:', E.groupby('eval_year').size().to_dict())

names = ['M0 현재값 유지'] + list(MODELS)
rows = []
for Y in (2021, 2022, 2023):
    d = E[E.eval_year == Y]
    rows += [{'연도': str(Y), '모델': nm, **metrics(d, 'pred_' + nm).to_dict()} for nm in names]
rows += [{'연도': '전체', '모델': nm, **metrics(E, 'pred_' + nm).to_dict()} for nm in names]
T1 = pd.DataFrame(rows); T1.to_csv(OUT_DIR / 'metrics_by_year.csv', index=False, encoding='utf-8-sig')
display(T1.round({'MAE': 2, 'RMSE': 2, 'logMAE': 3, 'scaledMAE': 3}))

YRS = ['2021', '2022', '2023', '전체']
T2 = T1.pivot(index='모델', columns='연도', values='logMAE').loc[names, YRS].round(3)
display(T2)
T3 = T1.pivot(index='모델', columns='연도', values=['MAE', 'RMSE']).swaplevel(axis=1)
T3 = T3[[(y, k) for y in YRS for k in ('MAE', 'RMSE')]].loc[names].round(1)
display(T3)
print('2022년 규모 보정 MAE: M1 %.3f, M2 %.3f' % tuple(T1[(T1['연도'] == '2022') & T1['모델'].isin(['M1 자기 현 이력', 'M2 인근 3현 추가'])].scaledMAE))

# 앞의 셋이 그림 4
pairs = [('M2 인근 3현 추가', 'M1 자기 현 이력'), ('M3 전국 추가', 'M1 자기 현 이력'), ('M4 인근과 전국 추가', 'M3 전국 추가'),
         ('M2 인근 3현 추가', 'M3 전국 추가'), ('M2 인근 3현 추가', 'M0 현재값 유지'), ('M1 자기 현 이력', 'M0 현재값 유지')]
cis = []
for Y in (2021, 2022, 2023, '전체'):
    d = E if Y == '전체' else E[E.eval_year == Y]
    for a, b in pairs:
        pt, lo, hi = cluster_boot_diff(d, 'pred_' + a, 'pred_' + b)
        cis.append({'연도': Y, '비교': f'{a} - {b}', '차이': pt, '하한': lo, '상한': hi})
CI = pd.DataFrame(cis)
display(CI[CI['비교'] == 'M2 인근 3현 추가 - M1 자기 현 이력'].round(3))

fig, ax = plt.subplots(figsize=(WD, 2.95))
cats = [2021, 2022, 2023, '전체']
labels = ['인근 3현 추가 (M2 − M1)', '전국 정보 추가 (M3 − M1)', '전국 모델에 인근 3현 추가 (M4 − M3)']
for j, ((a, b), lab, col) in enumerate(zip(pairs[:3], labels, [B1, B2, B3])):
    sub = CI[CI['비교'] == f'{a} - {b}']
    xs = np.arange(len(cats)) + (j - 1) * 0.2  # 겹치지 않게
    ax.vlines(xs, sub['하한'], sub['상한'], color=col, linewidth=1.4)
    ax.plot(xs, sub['차이'], 'o', color=col, markersize=5, markeredgecolor='white', markeredgewidth=1.0, label=lab, linestyle='none')
ax.axhline(0, color=INK2, linewidth=0.8); clean(ax); ax.set_xticks(range(len(cats))); ax.set_xticklabels([str(x) for x in cats])
ax.set_ylabel('로그 MAE 차이 (음수 = 개선)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.2, handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(); fig.subplots_adjust(bottom=0.24); fig.savefig(FIG_DIR / 'fig4_effects.png'); plt.close(fig)
CI.to_csv(OUT_DIR / 'effects_ci.csv', index=False, encoding='utf-8-sig')

show = ['M4 인근과 전국 추가 - M3 전국 추가', 'M3 전국 추가 - M1 자기 현 이력', 'M2 인근 3현 추가 - M3 전국 추가',
        'M1 자기 현 이력 - M0 현재값 유지', 'M2 인근 3현 추가 - M0 현재값 유지']
display(CI[CI['비교'].isin(show)].sort_values(['비교', '연도'], key=lambda s: s.astype(str)).round(3))
