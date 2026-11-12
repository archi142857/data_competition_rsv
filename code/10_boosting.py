# 그래디언트 부스팅으로 같은 평가 반복 (최종보고서 6.4절): 표 10
from rsv_core import *
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

c, W, yw, info = load(DATA_DIR)
P = build_panel(W, yw, info)
PAIRS = [(M2, M1), (M3, M1), (M4, M3), (M2, M3)]

def hgb(lr=0.05, it=300, leaf=31):
    # 학습률과 반복 수는 주제 정할 때 시험 삼아 쓰던 값 그대로. 이번 결과를 보고 바꾸지 않음. 조기 종료를 켜면 검증용 행을 무작위로 떼어 내서 결과가 흔들림
    return lambda: HistGradientBoostingRegressor(learning_rate=lr, max_iter=it, max_leaf_nodes=leaf, early_stopping=False, random_state=0)

ER = evaluate_years(P, W, yw)
EG = evaluate_years(P, W, yw, learner=hgb())
names = [M0, M1, M2, M3, M4]
rows = []
for Y in (2021, 2022, 2023, '전체'):
    d = EG if Y == '전체' else EG[EG.eval_year == Y]
    rows += [{'연도': str(Y), '모델': nm, **metrics(d, 'pred_' + nm).to_dict()} for nm in names]
TG = pd.DataFrame(rows); save(TG, 'boosting_metrics.csv')
display(TG.pivot(index='모델', columns='연도', values='logMAE').loc[names, ['2021', '2022', '2023', '전체']].round(3))
display(TG.pivot(index='모델', columns='연도', values='RMSE').loc[names, ['2021', '2022', '2023', '전체']].round(1))
CG = effect_table(EG, PAIRS); save(CG, 'boosting_effects.csv'); display(CG.round(3))

# 같은 입력에서 릿지와 부스팅 중 어느 쪽 오차가 작은지
EG2 = EG.rename(columns={'pred_' + m: 'pred_부스팅 ' + m for m in (M1, M2, M3, M4)})
for m in (M1, M2, M3, M4): EG2['pred_릿지 ' + m] = ER['pred_' + m].to_numpy()
BR = effect_table(EG2, [('부스팅 ' + m, '릿지 ' + m) for m in (M1, M2, M3, M4)]); save(BR, 'boosting_vs_ridge.csv')
display(BR.query("연도 == '전체'").round(3))

# 2022년 세 현의 정점 직후 예측 (6.1절과 같은 15행)
ratio = lambda d, m: float(np.exp((np.log1p(d['pred_' + m]) - np.log1p(d.y_tgt)).mean()))
pick = lambda E: E[(E.eval_year == 2022) & E.pref.isin(['Osaka', 'Aichi', 'Hyogo']) & E.o_week.between(27, 31)]
print('세 현 15행 예측/실제 | 릿지 M1 %.2f, M2 %.2f | 부스팅 M1 %.2f, M2 %.2f' % (ratio(pick(ER), M1), ratio(pick(ER), M2), ratio(pick(EG), M1), ratio(pick(EG), M2)))
display(pick(EG)[['pref', 'o_week', 'y_now', 'y_tgt', 'pred_' + M1, 'pred_' + M2]].round(0))
rm22 = lambda E, m: np.sqrt(((E[E.eval_year == 2022]['pred_' + m] - E[E.eval_year == 2022].y_tgt)**2).mean())
fb = {'ratio_ridge_M1': ratio(pick(ER), M1), 'ratio_ridge_M2': ratio(pick(ER), M2), 'ratio_hgb_M1': ratio(pick(EG), M1), 'ratio_hgb_M2': ratio(pick(EG), M2),
      'rmse22_ridge_M1': rm22(ER, M1), 'rmse22_ridge_M2': rm22(ER, M2), 'rmse22_hgb_M1': rm22(EG, M1), 'rmse22_hgb_M2': rm22(EG, M2)}

# 유행 단계별 (5.4절과 같은 기준)
EG['phase'] = np.where(EG.y_tgt > EG.y_now * 1.2, '상승', np.where(EG.y_tgt < EG.y_now / 1.2, '하강', '정체'))
ph = EG.groupby(['phase', 'eval_year']).apply(lambda x: pd.Series({'행 수': len(x), 'M1': log_mae(x, M1), 'M2': log_mae(x, M2)}), include_groups=False)
display(ph.round(3).unstack('eval_year').swaplevel(axis=1).sort_index(axis=1).loc[['상승', '하강', '정체']]); save(ph.reset_index(), 'boosting_phase.csv')

# 정점 순서에 따른 차이도 부스팅에서 다시 (6.1절과 같은 계산)
LG = peak_lead(EG, W, yw, info)
rows = []
for k in ('인근이 1주 넘게 먼저', '차이 1주 이내', '자기 현이 1주 넘게 먼저'):
    d = LG[LG['구분'] == k]; pt, lo, hi = boot_mean(d['절대 오차 차이'], d.pref)
    rows.append({'구분': k, '현×연도': len(d), 'M1 편향': d['M1 편향'].mean(), 'M2 편향': d['M2 편향'].mean(), '절대 오차 차이': pt, '하한': lo, '상한': hi})
TL = pd.DataFrame(rows); display(TL.round(3)); save(TL, 'boosting_lead_groups.csv')
fb['lead_rho_abs'] = spearmanr(LG['인근 정점 − 자기 정점'], LG['절대 오차 차이'])[0]; fb['lead_bias_M1'] = LG['M1 편향'].mean(); fb['lead_bias_M2'] = LG['M2 편향'].mean()
print('정점 순서와 절대 오차 차이의 순위 상관 %.2f | 정점 전후 편향 M1 %.2f, M2 %.2f' % (fb['lead_rho_abs'], fb['lead_bias_M1'], fb['lead_bias_M2']))
facts('boosting', fb)

# 설정을 바꿔도 결론이 같은지
rows = []
for lr, it, leaf in ((0.05, 300, 31), (0.05, 100, 31), (0.1, 300, 31), (0.05, 300, 15), (0.05, 300, 63)):
    E = EG if (lr, it, leaf) == (0.05, 300, 31) else evaluate_years(P, W, yw, learner=hgb(lr, it, leaf))
    rows.append({'학습률': lr, '반복': it, '잎 수': leaf, **{m[:2]: log_mae(E, m) for m in (M1, M2, M3, M4)},
                 'M2-M1': log_mae(E, M2) - log_mae(E, M1), 'M4-M3': log_mae(E, M4) - log_mae(E, M3),
                 **{f'M4-M3 {Y}': log_mae(E[E.eval_year == Y], M4) - log_mae(E[E.eval_year == Y], M3) for Y in (2021, 2022, 2023)}})
HS = pd.DataFrame(rows); save(HS, 'boosting_settings.csv'); display(HS.round(3))

# 부스팅은 입력을 구간으로 나눠 쓰기 때문에 끝자리가 조금만 달라도 경계가 바뀌고 결과가 흔들림 (인근 평균을 더하는 순서만 바꿔도 셋째 자리가 달라졌음)
# 인근 입력에 1e-12 크기의 잡음을 더해 10번 다시 돌려서 그 폭을 봄. M1, M3는 인근 입력이 없어서 그대로
def phase_gap(E, g, Y):
    d = E[(E.phase == g) & (E.eval_year == Y)]
    return log_mae(d, M2) - log_mae(d, M1)

rows = []
for seed in range(10):
    Pn = P.copy(); Pn[NB] = Pn[NB] + np.random.default_rng(seed).normal(0, 1e-12, (len(Pn), len(NB)))
    En = evaluate_years(Pn, W, yw, models={M2: MODELS[M2], M4: MODELS[M4]}, learner=hgb())
    for m in (M1, M3): En['pred_' + m] = EG['pred_' + m].to_numpy()
    En['phase'] = EG.phase.to_numpy()
    r = {'seed': seed, 'M2': log_mae(En, M2), 'M4': log_mae(En, M4)}
    r['M2-M1'], _, r['M2-M1 상한'] = cluster_boot_diff(En, 'pred_' + M2, 'pred_' + M1)
    r['M4-M3'], _, r['M4-M3 상한'] = cluster_boot_diff(En, 'pred_' + M4, 'pred_' + M3)
    r['M4-M3 2022'], r['M4-M3 2022 하한'], r['M4-M3 2022 상한'] = cluster_boot_diff(En[En.eval_year == 2022], 'pred_' + M4, 'pred_' + M3)
    r['2022 RMSE M2'] = rm22(En, M2); r['세 현 예측/실제 M2'] = ratio(pick(En), M2)
    for Y in (2021, 2022, 2023): r[f'하강 {Y}'] = phase_gap(En, '하강', Y)
    r['상승 최대'] = max(phase_gap(En, '상승', Y) for Y in (2021, 2022, 2023))
    Ln = peak_lead(En, W, yw, info); r['정점 전후 M2 편향'] = Ln['M2 편향'].mean()
    for k, nm in (('인근이 1주 넘게 먼저', '인근 먼저'), ('자기 현이 1주 넘게 먼저', '자기 먼저')):
        d = Ln[Ln['구분'] == k]; r[nm], r[nm + ' 하한'], r[nm + ' 상한'] = boot_mean(d['절대 오차 차이'], d.pref)
    rows.append(r)
NZ = pd.DataFrame(rows); save(NZ, 'boosting_noise.csv'); display(NZ.drop(columns='seed').agg(['min', 'max']).T.round(3))
