# 현들을 연결한 SIRS 모의실험 (최종보고서 부록 B): 표 B1
# 실제 자료에 맞춘 모형이 아님. 현 사이 연결 방식만 바꿔 가며 만든 가상 자료에 같은 예측 평가를 돌려 본 것
from rsv_core import *
from scipy.stats import spearmanr

c, W, yw, info = load(DATA_DIR)
D = distance_matrix(info); N = info['pop'].to_numpy().astype(float); n = len(N)
mu = W.loc[yw.index[yw.year <= 2019]].mean().to_numpy()
YR, WK = yw.year.to_numpy(), yw.week.to_numpy()

def coupling(kind, ell=100.0):
    # local: 가까울수록, 인구가 많을수록 강한 연결. global: 거리와 무관하게 인구에만 비례
    A = np.tile(N, (n, 1))
    if kind == 'local': A = A * np.exp(-D / ell)
    np.fill_diagonal(A, 0)
    return A / A.sum(axis=1, keepdims=True)

def simulate(kind, m, npi=0.5, seed=0, R0=3.0, amp=0.25, wane=1 / 52, imp=1e-8, burn=520):
    # 주 단위 SIRS. 감염 기간 1주, 면역은 평균 52주 뒤 소실. 자료에 맞춘 값이 아니고 임의로 잡음. 전파율 50% 감소는 대회 FAQ가 예로 든 값
    rng = np.random.default_rng(seed); Cw = coupling(kind)
    t0 = burn + int(np.where((YR == 2020) & (WK == 10))[0][0])  # 코로나 구간 시작
    S = N * 0.4; I = N * 1e-4; R = N - S - I
    out = np.zeros((burn + len(W), n))
    for t in range(len(out)):
        beta = R0 * (1 + amp * np.cos(2 * np.pi * ((t - burn) % 52.18 - 44) / 52.18))
        if t0 <= t < t0 + 52: beta *= 1 - npi  # 1년 동안 전파율을 낮춤
        prev = I / N
        lam = beta * ((1 - m) * prev + m * (Cw @ prev)) + imp  # imp: 나라 밖에서 들어오는 감염. 없으면 연결 없는 경우에 유행이 안 돌아옴
        new = rng.binomial(S.astype(np.int64), 1 - np.exp(-lam)).astype(float)
        back = rng.binomial(R.astype(np.int64), wane).astype(float)
        S, R, I = S - new + back, R + I - back, new
        out[t] = new
    X = out[burn:]
    rho = mu / X[YR <= 2019].mean(axis=0)  # 코로나 전 평균 보고 수가 실제 자료와 같아지도록 보고율을 맞춤
    return pd.DataFrame(rng.poisson(X * rho), index=W.index, columns=W.columns)

def score(Ws, nrand=5):
    E = evaluate_years(build_panel(Ws, yw, info), Ws, yw)
    g = lambda E, a, b: log_mae(E, a) - log_mae(E, b)
    r = {'M0': log_mae(E, M0), 'M1': log_mae(E, M1), 'M3-M1': g(E, M3, M1), 'M2-M1 인근': g(E, M2, M1), 'M4-M3 인근': g(E, M4, M3),
         '0인 값 비율': float((Ws.to_numpy()[YR >= 2021] == 0).mean())}
    rr = []
    for s in range(nrand):
        Er = evaluate_years(build_panel(Ws, yw, info, nb_w=nb_weights(info, 3, 'random', seed=s)), Ws, yw)
        rr.append((g(Er, M2, M1), g(Er, M4, M3)))
    r['M2-M1 무작위'], r['M4-M3 무작위'] = np.mean(rr, axis=0)
    pk = lambda Y: WK[YR == Y][Ws.to_numpy()[YR == Y].argmax(axis=0)]
    r['정점 주 IQR 2019'] = np.subtract(*np.percentile(pk(2019), [75, 25])); r['정점 주 IQR 2021'] = np.subtract(*np.percentile(pk(2021), [75, 25]))
    return r

# 연결 비중 0.5%와 외부 유입 1e-8은 몇 번 돌려 본 뒤에 잡은 값. 비중이 5%면 모든 현이 한꺼번에 움직여서 비교할 것이 없었고, 유입이 1e-9면 연결 없는 조건에서 유행이 안 돌아왔음
CASES = [('연결 없음', 'local', 0.0, 0.5, 1e-8), ('거리와 무관한 연결', 'global', 0.005, 0.5, 1e-8), ('거리에 따라 약해지는 연결', 'local', 0.005, 0.5, 1e-8),
         ('거리에 따라 약해지는 연결, 전파율 30% 감소', 'local', 0.005, 0.3, 1e-8), ('거리에 따라 약해지는 연결, 전파율 70% 감소', 'local', 0.005, 0.7, 1e-8),
         ('거리에 따라 약해지는 연결, 연결 비중 2%', 'local', 0.02, 0.5, 1e-8), ('거리와 무관한 연결, 연결 비중 2%', 'global', 0.02, 0.5, 1e-8),
         ('연결 없음, 외부 유입 10배', 'local', 0.0, 0.5, 1e-7), ('거리와 무관한 연결, 외부 유입 10배', 'global', 0.005, 0.5, 1e-7), ('거리에 따라 약해지는 연결, 외부 유입 10배', 'local', 0.005, 0.5, 1e-7)]
NREP = 20
rows = []
for label, kind, m, npi, imp in CASES:
    for seed in range(NREP):
        rows.append({'조건': label, 'seed': seed, **score(simulate(kind, m, npi, seed, imp=imp))})
    print(label, '끝', flush=True)
SM = pd.DataFrame(rows); save(SM, 'sim_runs.csv')
SM['M4-M3 차이'] = SM['M4-M3 인근'] - SM['M4-M3 무작위']; SM['M2-M1 차이'] = SM['M2-M1 인근'] - SM['M2-M1 무작위']
cols = ['M0', 'M1', 'M3-M1', 'M2-M1 인근', 'M2-M1 무작위', 'M2-M1 차이', 'M4-M3 인근', 'M4-M3 무작위', 'M4-M3 차이', '0인 값 비율', '정점 주 IQR 2019', '정점 주 IQR 2021']
T = SM.groupby('조건', sort=False)[cols].mean()
T['인근이 나은 횟수'] = SM.assign(w=SM['M4-M3 차이'] < 0).groupby('조건', sort=False).w.sum()
T['인근이 오차를 줄인 횟수'] = SM.assign(w=SM['M4-M3 인근'] < 0).groupby('조건', sort=False).w.sum()
save(T.reset_index(), 'sim_summary.csv'); display(T.round(3))

# 연결이 없어도 인근 3현이 무작위 3현보다 덜 나빴던 이유 확인: 인구가 많은 현일수록 유행이 먼저 돌아오고, 인구가 비슷한 현끼리 모여 있음
nb3 = nearest_k(info); lp = np.log(info['pop'])
rows = []
for seed in range(NREP):
    X = simulate('local', 0.0, 0.5, seed).to_numpy(); thr = 0.1 * X[YR <= 2019].max(axis=0)
    post = np.where(YR >= 2021)[0]
    first = pd.Series([next((t for t in post if X[t, j] >= thr[j]), np.nan) for j in range(n)], index=W.columns)
    f = first.fillna(len(W) + 52)  # 2023년 말까지 유행이 안 돌아온 현은 맨 뒤 순위로
    rows.append({'seed': seed, '복귀 못 한 현': int(first.isna().sum()), '인구와 복귀 시점': spearmanr(lp, f)[0],
                 '자기 현과 인근의 복귀 시점': spearmanr(f, pd.Series({p: np.mean([f[q] for q in nb3[p]]) for p in W.columns}))[0]})
NC = pd.DataFrame(rows); save(NC, 'sim_none_check.csv'); display(NC.drop(columns='seed').mean().round(2))
rho_pop = spearmanr(lp, pd.Series({p: np.mean([lp[q] for q in nb3[p]]) for p in W.columns}))[0]
print('자기 현 인구와 인근 3현 평균 인구의 순위 상관: %.2f' % rho_pop)
facts('sim', {'nrep': NREP, 'rho_pop_nb': rho_pop, **NC.drop(columns='seed').mean().to_dict()})
