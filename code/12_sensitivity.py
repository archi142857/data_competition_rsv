# 특이 관측, 기준 모델, 설정을 바꿨을 때 결론이 유지되는지 (최종보고서 6.6~6.7절): 표 12, 표 13, 표 14
from rsv_core import *

c, W, yw, info = load(DATA_DIR)
P = build_panel(W, yw, info)
PAIRS = [(M2, M1), (M4, M3)]

def line(label, E, n_train=None):
    ef = effect_table(E, PAIRS).set_index(['비교', '연도'])
    r = {'조건': label, '평가 행': len(E), **{m[:2]: log_mae(E, m) for m in (M1, M2, M3, M4)}}
    for a, b in PAIRS:
        k = f'{a} - {b}'; s = f'{a[:2]}-{b[:2]}'
        r[s] = ef.loc[(k, '전체'), '차이']; r[s + ' 하한'] = ef.loc[(k, '전체'), '하한']; r[s + ' 상한'] = ef.loc[(k, '전체'), '상한']
        for Y in (2021, 2022, 2023): r[f'{s} {Y}'] = ef.loc[(k, Y), '차이']
    return r

E0 = evaluate_years(P, W, yw)
rows = [line('기본 설계', E0)]

# 오키나와: 인근 3현이 모두 657 km 넘게 떨어져 있고 코로나 전에도 여름에 유행함. 자료에서 통째로 빼고 다시 계산
keep = [p for p in W.columns if p != 'Okinawa']
rows.append(line('오키나와 제외', evaluate_years(build_panel(W[keep], yw, info.loc[keep]), W[keep], yw)))

# 코로나 유행기: 그 구간의 행을 학습에서 뺌 (예측 시점이나 목표 시점이 구간에 들면 제외)
# 구간은 자료로 먼저 잡음. 전국 합계가 2013~2019년 최솟값보다 낮았던 2020년 주의 처음부터 끝까지
nat = W.sum(axis=1); floor = nat[yw.index[yw.year <= 2019]].min()
low = yw[(yw.year == 2020) & (nat < floor)].week
print('전국 합계가 2013~2019년 최솟값(%d건)보다 낮았던 2020년 주: %d~%d주 중 %d주' % (floor, low.min(), low.max(), len(low)))
out_of = lambda ts: (lambda tr: ~(tr.time.isin(ts) | tr.t_time.isin(ts)))
cov0 = set(yw.index[(yw.year == 2020) & yw.week.between(low.min(), low.max())])
rows.append(line(f'2020년 {low.min()}~{low.max()}주의 행을 학습에서 제외', evaluate_years(P, W, yw, keep=out_of(cov0))))
# 더 넓게: 전국 휴교 요청 직후인 2020년 10주부터 그해 말까지
cov = set(yw.index[(yw.year == 2020) & (yw.week >= 10)])
no_cov = out_of(cov)
n_all = len(train_rows(P, yw, 2023)); n_cut = int(no_cov(train_rows(P, yw, 2023)).sum())
print('2023년 평가용 학습 행: %d → %d' % (n_all, n_cut))
rows.append(line('2020년 10주 이후 2020년 행을 학습에서 제외', evaluate_years(P, W, yw, keep=no_cov)))
# 유행이 다시 커지기 전인 2021년 26주까지 넓혀도 같은지
cov2 = cov | set(yw.index[(yw.year == 2021) & (yw.week <= 26)])
rows.append(line('2020년 10주부터 2021년 26주까지의 행을 학습에서 제외', evaluate_years(P, W, yw, keep=out_of(cov2))))

# 보고 수가 아주 작은 행: 0건과 2건의 로그 차이가 1,000건과 3,000건의 차이와 같게 잡힘. 예측 시점 보고 수 5건 미만을 평가에서 뺌
rows.append(line('예측 시점 보고 수 5건 미만 행을 평가에서 제외', E0[E0.y_now >= 5]))
rows.append(line('오사카, 아이치, 효고를 평가에서 제외', E0[~E0.pref.isin(['Osaka', 'Aichi', 'Hyogo'])]))

# 기준 모델 M1이 약해서 인근 정보의 이득이 크게 잡힌 것은 아닌지. M1은 세 해를 합쳐도 현재값 유지와 비슷했고, 코로나 전에 배운 계절 입력이 2021년 이후에는 오히려 해가 됨
# 계절 입력(sin, cos)을 네 모델에서 모두 빼고 같은 평가를 다시 함
OWN_NS = [c_ for c_ in OWN if c_ not in ('ws', 'wc')]
NS = {M1: OWN_NS, M2: OWN_NS + NB, M3: OWN_NS + NAT, M4: OWN_NS + NB + NAT}
ENS = evaluate_years(P, W, yw, models=NS)
rows.append(line('계절 입력을 네 모델에서 제외', ENS))
save(effect_table(ENS, PAIRS + [(M1, M0)]), 'noseason_effects.csv')
assert (ENS.time.to_numpy() == E0.time.to_numpy()).all() and (ENS.pref.to_numpy() == E0.pref.to_numpy()).all()
X = E0[['pref', 'y_tgt', 'eval_year']].copy(); X['a'] = E0['pred_' + M2].to_numpy(); X['b'] = ENS['pred_' + M1].to_numpy()
ns_fix = cluster_boot_diff(X, 'a', 'b')
print('고정 설계의 M2 − 계절 입력을 뺀 M1: %.3f [%.3f, %.3f]' % ns_fix)
nb22 = ENS[(ENS.eval_year == 2022) & (ENS.y_now >= 5)]; ns22 = cluster_boot_diff(nb22, 'pred_' + M2, 'pred_' + M1)
print('계절 입력 제외, 2022년 M2-M1 | 5건 이상 %d행: %.3f [%.3f, %.3f]' % (len(nb22), *ns22))
# 계절 입력을 뺀 모델에서도 전국 합계를 넘는 이득이 가까운 현에서만 나오는지 (무작위 3현 50번)
ns_rand = []
for seed in range(50):
    Er = evaluate_years(build_panel(W, yw, info, nb_w=nb_weights(info, 3, 'random', seed=seed)), W, yw, models=NS)
    ns_rand.append(log_mae(Er, M4) - log_mae(Er, M3))
print('계절 입력 제외, 무작위 3현 50번의 M4-M3: 중앙값 %.3f [%.3f, %.3f]' % tuple(np.quantile(ns_rand, [0.5, 0.025, 0.975])))

for a in (0.1, 10, 100):
    rows.append(line(f'릿지 α = {a:g}', evaluate_years(P, W, yw, alpha=a)))
T13 = pd.DataFrame(rows); save(T13, 'sensitivity.csv'); display(T13.round(3))

# 2022년의 M2-M1은 작은 보고 수 행을 빼면 어떻게 되는지 따로 확인
big = E0[(E0.y_now >= 5) & (E0.eval_year == 2022)]; sm = E0[(E0.y_now < 5) & (E0.eval_year == 2022)]
s22 = cluster_boot_diff(big, 'pred_' + M2, 'pred_' + M1)
print('2022년 M2-M1 | 5건 이상 %d행: %.3f [%.3f, %.3f] | 5건 미만 %d행: %.3f' % (len(big), *s22, len(sm), log_mae(sm, M2) - log_mae(sm, M1)))

# 주 묶음(8주) 부트스트랩. 현 단위 구간과 나란히 저장
rows = []
for a, b in PAIRS:
    for Y in (2021, 2022, 2023, '전체'):
        d = E0 if Y == '전체' else E0[E0.eval_year == Y]
        c1 = cluster_boot_diff(d, 'pred_' + a, 'pred_' + b); c2 = block_boot_diff(d, 'pred_' + a, 'pred_' + b)
        rows.append({'비교': f'{a[:2]}-{b[:2]}', '연도': Y, '차이': c1[0], '현 하한': c1[1], '현 상한': c1[2], '주 묶음 하한': c2[1], '주 묶음 상한': c2[2]})
BB = pd.DataFrame(rows); save(BB, 'block_boot.csv'); display(BB.round(3))

d22 = E0[(E0.eval_year == 2022) & ~E0.pref.isin(['Osaka', 'Aichi', 'Hyogo'])]
rm = lambda d, m: np.sqrt(((d['pred_' + m] - d.y_tgt)**2).mean())
print('2022년 원래 단위 RMSE, 세 현 제외: M1 %.1f, M2 %.1f | MAE M1 %.1f, M2 %.1f' % (rm(d22, M1), rm(d22, M2), (d22['pred_' + M1] - d22.y_tgt).abs().mean(), (d22['pred_' + M2] - d22.y_tgt).abs().mean()))
facts('sensitivity', {'floor': floor, 'low_first': low.min(), 'low_last': low.max(), 'low_n': len(low), 'train_all': n_all, 'train_cut': n_cut,
                      'ns_fix_d': ns_fix[0], 'ns_fix_lo': ns_fix[1], 'ns_fix_hi': ns_fix[2], 'ns_rand_med': np.median(ns_rand), 'ns_rand_n': len(ns_rand),
                      'ns_big22_d': ns22[0], 'ns_big22_lo': ns22[1], 'ns_big22_hi': ns22[2],
                      'ns_rand_better': int((np.array(ns_rand) <= log_mae(ENS, M4) - log_mae(ENS, M3)).sum()),
                      'rmse22_no3_M1': rm(d22, M1), 'rmse22_no3_M2': rm(d22, M2), 'n_small': int((E0.y_now < 5).sum()),
                      'big22_n': len(big), 'big22_d': s22[0], 'big22_lo': s22[1], 'big22_hi': s22[2], 'small22_n': len(sm), 'small22_d': log_mae(sm, M2) - log_mae(sm, M1)})

# 예측 기간. 4주는 중간보고서에서 정한 값이고 나머지는 비교용
rows = []
for h in (1, 2, 3, 4, 6, 8):
    E = evaluate_years(build_panel(W, yw, info, h=h), W, yw)
    rows.append({**line(f'{h}주 앞', E), 'M0': log_mae(E, M0), 'h': h})
T14 = pd.DataFrame(rows); save(T14, 'horizon.csv'); display(T14[['조건', '평가 행', 'M0', 'M1', 'M2', 'M3', 'M4', 'M2-M1', 'M2-M1 하한', 'M2-M1 상한', 'M4-M3', 'M4-M3 하한', 'M4-M3 상한']].round(3))

# 유행 단계를 나누는 기준 (중간보고서는 20%)
rows = []
for th in (0.15, 0.20, 0.25):
    ph = np.where(E0.y_tgt > E0.y_now * (1 + th), '상승', np.where(E0.y_tgt < E0.y_now / (1 + th), '하강', '정체'))
    for g in ('상승', '하강', '정체'):
        d = E0[ph == g]
        rows.append({'기준': f'{th:.0%}', '단계': g, '행 수': len(d), 'M1': log_mae(d, M1), 'M2': log_mae(d, M2), 'M2-M1': log_mae(d, M2) - log_mae(d, M1)})
PH = pd.DataFrame(rows); save(PH, 'phase_threshold.csv'); display(PH.round(3))
