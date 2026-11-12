# 유행 시기 오차 (최종보고서 6.5절): 표 11
# 4주 앞 예측값을 목표 주 순서로 이어 붙인 곡선과 실제 곡선에서 시기를 각각 재고 그 차이를 봄
from rsv_core import *

c, W, yw, info = load(DATA_DIR)
E = evaluate_years(build_panel(W, yw, info), W, yw)
names = [M0, M1, M2, M3, M4]

def onset(w, v, thr):
    # 기준 이상인 주가 2주 이어지는 첫 주. 한 주만 튀는 값에 걸리지 않게
    ok = v >= thr
    hit = np.where(ok[:-1] & ok[1:])[0]
    return w[hit[0]] if len(hit) else np.nan

rows = []
for (Y, p), d in E.groupby(['eval_year', 'pref']):
    d = d.sort_values('t_week'); w = d.t_week.to_numpy(); y = d.y_tgt.to_numpy()
    thr = 0.25 * y.max()  # 목표 주 5~52주 안에서 실제 최댓값의 25%. 예측 곡선에도 같은 기준을 씀 (사후 평가용)
    r = {'연도': Y, 'pref': p, '실제 무게중심': (w * y).sum() / y.sum(), '실제 시작 주': onset(w, y, thr), '실제 최고 주': w[y.argmax()]}
    for nm in names:
        v = d['pred_' + nm].to_numpy()
        r['무게중심 ' + nm] = (w * v).sum() / v.sum() - r['실제 무게중심']
        r['시작 주 ' + nm] = onset(w, v, thr) - r['실제 시작 주']
        r['최고 주 ' + nm] = w[v.argmax()] - r['실제 최고 주']
    rows.append(r)
TM = pd.DataFrame(rows)
print('목표 주 범위:', int(E.t_week.min()), '~', int(E.t_week.max()), '| 현×연도', len(TM))
cut = TM['실제 시작 주'] == E.t_week.min()
print('실제 시작 주가 첫 목표 주(5주)에 걸린 경우:', int(cut.sum()), '| 예측 곡선이 기준에 못 미친 경우:', {nm[:2]: int(TM['시작 주 ' + nm].isna().sum()) for nm in names})
# 연초에 이미 기준을 넘어 있던 현은 시작 시기를 잴 수 없어서 시작 주 비교에서 뺌
TM.loc[cut, [c_ for c_ in TM.columns if c_.startswith('시작 주 ')]] = np.nan
save(TM, 'timing_by_pref.csv')

def boot(a, b, B=2000, seed=0):
    # 현마다 세 해가 있으므로 현 단위로 뽑음
    g = pd.DataFrame({'p': TM.pref, 'd': TM[a].abs() - TM[b].abs()}).dropna().groupby('p').d.agg(['sum', 'count'])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(g), size=(B, len(g)))
    s = g['sum'].to_numpy()[idx].sum(1) / g['count'].to_numpy()[idx].sum(1)
    return g['sum'].sum() / g['count'].sum(), np.quantile(s, 0.025), np.quantile(s, 0.975)

rows = []
for kind in ('무게중심', '시작 주', '최고 주'):
    for nm in names:
        col = f'{kind} {nm}'
        r = {'지표': kind, '모델': nm, **{str(Y): TM.loc[TM['연도'] == Y, col].abs().mean() for Y in (2021, 2022, 2023)},
             '전체': TM[col].abs().mean(), '평균 부호 오차': TM[col].mean(), '현×연도': int(TM[col].notna().sum())}
        rows.append(r)
T12 = pd.DataFrame(rows); save(T12, 'timing_summary.csv'); display(T12.round(2))
rows = []
for kind in ('무게중심', '시작 주', '최고 주'):
    for a, b in ((M2, M1), (M4, M3), (M2, M0)):
        pt, lo, hi = boot(f'{kind} {a}', f'{kind} {b}')
        print('%s 절대 오차 차이 %s − %s: %.2f주 [%.2f, %.2f]' % (kind, a[:2], b[:2], pt, lo, hi))
        rows.append({'지표': kind, '비교': f'{a[:2]}-{b[:2]}', '차이': pt, '하한': lo, '상한': hi})
save(pd.DataFrame(rows), 'timing_diff.csv')

# 시작 주 기준(최댓값의 25%)은 임의로 정한 값이라 10%, 50%로도 확인. 연초에 기준을 넘어 있던 조합을 빼지 않았을 때도 같이 봄
def onset_err(q, drop_cut=True):
    rows = []
    for (Y, p), d in E.groupby(['eval_year', 'pref']):
        d = d.sort_values('t_week'); w = d.t_week.to_numpy(); y = d.y_tgt.to_numpy(); thr = q * y.max(); a = onset(w, y, thr)
        if drop_cut and a == w[0]: continue
        rows.append({'연도': Y, 'pref': p, **{nm: onset(w, d['pred_' + nm].to_numpy(), thr) - a for nm in names}})
    return pd.DataFrame(rows)

rows = []
for q, drop in ((0.10, True), (0.25, True), (0.50, True), (0.25, False)):
    T = onset_err(q, drop); v = (T[M2].abs() - T[M1].abs()).dropna()
    g = pd.DataFrame({'p': T.pref[v.index], 'd': v}).groupby('p').d.agg(['sum', 'count'])
    idx = np.random.default_rng(0).integers(0, len(g), size=(2000, len(g)))
    bs = g['sum'].to_numpy()[idx].sum(1) / g['count'].to_numpy()[idx].sum(1)
    rows.append({'기준': f'{q:.0%}', '연초 조합': '제외' if drop else '포함', '현×연도': len(v), **{nm[:2]: T.loc[v.index, nm].abs().mean() for nm in (M0, M1, M2)},
                 'M2-M1': v.mean(), '하한': np.quantile(bs, 0.025), '상한': np.quantile(bs, 0.975)})
OS = pd.DataFrame(rows); save(OS, 'timing_onset_sens.csv'); display(OS.round(2))

# 현 사이에 유행 시기가 얼마나 흩어졌는지 (실제 무게중심의 사분위 범위)
iqr = TM.groupby('연도')['실제 무게중심'].agg(lambda v: v.quantile(0.75) - v.quantile(0.25))
print('현별 실제 무게중심의 사분위 범위(주):', iqr.round(1).to_dict())
facts('timing', {'n': len(TM), 'onset_cut': int(cut.sum()), **{f'iqr_{Y}': iqr[Y] for Y in (2021, 2022, 2023)}})
