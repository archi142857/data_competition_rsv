# 공통 함수. 최종 노트북의 공통 함수 셀과 같은 코드
import os
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
# 자료는 저장소에 안 올림. 저장소 맨 위에 RSV_transmission_in_Japan 폴더를 두면 됨
CANDIDATES = [ROOT / 'RSV_transmission_in_Japan', ROOT.parent / '일본 RSV' / 'RSV_transmission_in_Japan']
DATA_DIR = next((p for p in CANDIDATES if (p / 'cases.csv').exists()), None)
assert DATA_DIR is not None, 'cases.csv 못 찾음 (README 참고)'
FIG_DIR = ROOT / 'figures'; FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / 'results'; OUT_DIR.mkdir(exist_ok=True)

def display(x):
    # 노트북 밖에서 돌릴 때 표 찍는 용
    print(x.to_string() if hasattr(x, 'to_string') else x, end='\n\n')

def save(df, name, index=False):
    df.to_csv(OUT_DIR / name, index=index, encoding='utf-8-sig')  # 엑셀에서 한글 안 깨지게

def facts(name, d):
    # 표로 만들기 애매한 낱개 숫자. 보고서 본문에 쓰는 값은 여기서 읽어 감
    save(pd.Series(d, name='value').rename_axis('key').reset_index(), f'facts_{name}.csv')

# 기상 파일 쪽 표기가 다름 (Naoasaki는 오타인 듯)
FIX = {'Hyōgo': 'Hyogo', 'Naoasaki': 'Nagasaki'}

def load(d):
    c = pd.read_csv(os.path.join(d, 'cases.csv'), index_col=0)
    pop = pd.read_csv(os.path.join(d, 'population.csv'), thousands=',')
    pop['prefecture'] = pop.prefecture.str.replace(r'-(ken|fu|to)$', '', regex=True)  # Aichi-ken -> Aichi
    W = c.pivot(index='time', columns='prefecture', values='cases').sort_index()
    yw = c.drop_duplicates('time').set_index('time')[['year', 'week']].sort_index()
    info = c.groupby('prefecture')[['lat', 'long']].first().join(pop.set_index('prefecture')[['pop', 'island']])
    info = info.loc[W.columns]
    return c, W, yw, info

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def distance_matrix(info):
    la, lo = info.lat.to_numpy(), info.long.to_numpy()
    return haversine(la[:, None], lo[:, None], la[None, :], lo[None, :])

def nearest_k(info, k=3):
    D = distance_matrix(info).copy(); np.fill_diagonal(D, np.inf)  # 자기 자신 제외
    idx = np.argsort(D, axis=1)[:, :k]
    names = info.index.to_numpy()
    return {names[i]: list(names[idx[i]]) for i in range(len(names))}

def nb_weights(info, k=3, kind='near', start=0, seed=None):
    # 행이 대상 현, 열이 평균에 들어가는 현. rank는 가까운 순으로 start번째부터 k곳 (먼 현 대조용)
    D = distance_matrix(info).copy(); np.fill_diagonal(D, np.inf)
    n = len(info); isl = info.island.to_numpy(); rng = np.random.default_rng(seed)
    Wt = np.zeros((n, n))
    for i in range(n):
        d = D[i].copy()
        if kind == 'island' and (isl == isl[i]).sum() > 1:  # 홋카이도, 오키나와는 같은 섬에 다른 현이 없어서 그대로 둠
            d[isl != isl[i]] = np.inf
        if kind == 'random':
            pick = rng.choice(np.delete(np.arange(n), i), size=k, replace=False)
        elif kind == 'pop':  # 거리와 상관없이 인구가 가장 비슷한 k곳
            gap = np.abs(np.log(info['pop'].to_numpy()) - np.log(info['pop'].to_numpy()[i])); gap[i] = np.inf
            pick = np.argsort(gap)[:k]
        else:
            pick = np.argsort(d)[start:start + k]
            pick = pick[np.isfinite(d[pick])]
        w = 1 / D[i, pick] if kind == 'invdist' else np.ones(len(pick))
        Wt[i, pick] = w / w.sum()
    return pd.DataFrame(Wt, index=info.index, columns=info.index)

def build_panel(W, yw, info, h=4, k=3, nb_lag=0, nb_w=None, nat_lag=0):
    # nb_lag=1이면 인근 값을 한 주 늦게 씀 (공개 지연 점검용). nat_lag는 전국 합계에 같은 처리
    # nb_w를 주면 가까운 k곳 평균 대신 그 가중치로 인근 평균을 만듦
    L = np.log1p(W)  # 0이 많아서 log1p
    NAT = np.log1p(W.sum(axis=1)).shift(nat_lag)
    nb = nearest_k(info, k)
    if nb_w is not None:
        NBM = pd.DataFrame(L.to_numpy() @ nb_w.loc[W.columns, W.columns].to_numpy().T, index=L.index, columns=L.columns)
    T = W.index.to_numpy()
    rows = []
    for p in W.columns:
        s = L[p]
        nbm = (L[nb[p]].mean(axis=1) if nb_w is None else NBM[p]).shift(nb_lag)
        df = pd.DataFrame({
            'time': T, 'pref': p, 'y_now': W[p].to_numpy(), 'y_tgt': W[p].shift(-h).to_numpy(),
            'l0': s, 'l1': s.shift(1), 'l2': s.shift(2), 'l3': s.shift(3), 'd4': s - s.shift(4),
            'nb0': nbm, 'nb_d4': nbm - nbm.shift(4), 'nb_gap': nbm - s,
            'nat0': NAT, 'nat_d4': NAT - NAT.shift(4),
            # 아래는 최종보고서 6.1절에서 시험한 입력. M0~M4에는 안 들어감
            'nb_d1': nbm - nbm.shift(1), 'nb_d2': nbm - nbm.shift(2), 'nat_d1': NAT - NAT.shift(1), 'nat_d2': NAT - NAT.shift(2),
            'pk': s - s.rolling(8, min_periods=1).max(), 'nb_pk': nbm - nbm.rolling(8, min_periods=1).max(),
            'cum': np.log1p(W[p].rolling(26, min_periods=1).mean()) - np.log1p(W[p].expanding().mean()), 'typ': s.expanding().mean(),
        }).reset_index(drop=True)  # 이거 없으면 아래 merge에서 time이 인덱스랑 겹쳐서 에러남
        rows.append(df)
    P = pd.concat(rows, ignore_index=True)
    P = P.merge(yw.rename(columns={'year': 'o_year', 'week': 'o_week'}), left_on='time', right_index=True)
    P['t_time'] = P.time + h
    P = P.merge(yw.rename(columns={'year': 't_year', 'week': 't_week'}), left_on='t_time', right_index=True, how='left')
    # 2015, 2020년은 53주라서 52 대신 365.25/7
    P['ws'] = np.sin(2 * np.pi * P.t_week / 52.18); P['wc'] = np.cos(2 * np.pi * P.t_week / 52.18)
    P['tgt_log'] = np.log1p(P.y_tgt)
    return P

OWN = ['l0', 'l1', 'l2', 'l3', 'd4', 'ws', 'wc']
NB = ['nb0', 'nb_d4', 'nb_gap']
NAT = ['nat0', 'nat_d4']
MODELS = {'M1 자기 현 이력': OWN, 'M2 인근 3현 추가': OWN + NB, 'M3 전국 추가': OWN + NAT, 'M4 인근과 전국 추가': OWN + NB + NAT}
M0, M1, M2, M3, M4 = ['M0 현재값 유지'] + list(MODELS)

def ridge(alpha=1.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def train_rows(P, yw, Y, need=()):
    # 목표 시점 기준으로 자름. 예측 시점으로 자르면 작년 마지막 4주 행의 목표값이 올해 값이 됨
    y0 = yw.index[yw.year == Y].min()
    return P[(P.t_time < y0)].dropna(subset=OWN + NB + NAT + list(need) + ['tgt_log'])

def evaluate_years(P, W, yw, years=(2021, 2022, 2023), alpha=1.0, models=None, learner=None, keep=None):
    # models, learner, keep을 안 주면 중간보고서에서 고정한 설계 그대로
    # keep: 학습 행을 골라내는 함수 (특이 관측을 빼고 다시 학습할 때)
    models = MODELS if models is None else models
    learner = (lambda: ridge(alpha)) if learner is None else learner
    need = sorted({c for cols in models.values() for c in cols} - set(OWN + NB + NAT))
    out = []
    for Y in years:
        y0 = yw.index[yw.year == Y].min()
        te = P[(P.o_year == Y) & (P.t_year == Y)].copy()
        tr = train_rows(P, yw, Y, need)
        if keep is not None: tr = tr[keep(tr)]
        te = te.dropna(subset=OWN + NB + NAT + need + ['tgt_log'])
        scale = W.loc[W.index < y0].mean().clip(lower=1.0)  # 규모 보정 MAE 분모
        te['scale'] = te.pref.map(scale)
        te['pred_' + M0] = te.y_now
        for name, cols in models.items():
            m = learner().fit(tr[cols], tr.tgt_log)
            te['pred_' + name] = np.clip(np.expm1(m.predict(te[cols])), 0, None)
        te['eval_year'] = Y
        out.append(te)
    return pd.concat(out, ignore_index=True)

def metrics(df, pcol):
    e = df[pcol] - df.y_tgt
    return pd.Series({'MAE': e.abs().mean(), 'RMSE': np.sqrt((e**2).mean()),
                      'logMAE': (np.log1p(df[pcol]) - np.log1p(df.y_tgt)).abs().mean(),
                      'scaledMAE': (e.abs() / df.scale).mean(), 'n': len(df)})

def log_mae(df, name):
    return (np.log1p(df['pred_' + name]) - np.log1p(df.y_tgt)).abs().mean()

def cluster_boot_diff(df, pa, pb, B=2000, seed=0):
    # 같은 현 행끼리는 독립이 아니라서 현 단위로 뽑음. 음수면 pa가 나음
    rng = np.random.default_rng(seed)
    ea = (np.log1p(df[pa]) - np.log1p(df.y_tgt)).abs(); eb = (np.log1p(df[pb]) - np.log1p(df.y_tgt)).abs()
    g = pd.DataFrame({'p': df.pref.to_numpy(), 'a': ea.to_numpy(), 'b': eb.to_numpy()}).groupby('p').agg(['sum', 'count'])
    sa, sb, n = g[('a', 'sum')].to_numpy(), g[('b', 'sum')].to_numpy(), g[('a', 'count')].to_numpy()
    k = len(sa); idx = rng.integers(0, k, size=(B, k))
    d = sa[idx].sum(1) / n[idx].sum(1) - sb[idx].sum(1) / n[idx].sum(1)
    point = sa.sum() / n.sum() - sb.sum() / n.sum()
    return point, np.quantile(d, 0.025), np.quantile(d, 0.975)

def effect_table(E, pairs, years=(2021, 2022, 2023)):
    rows = []
    for Y in list(years) + ['전체']:
        d = E if Y == '전체' else E[E.eval_year == Y]
        for a, b in pairs:
            pt, lo, hi = cluster_boot_diff(d, 'pred_' + a, 'pred_' + b)
            rows.append({'연도': Y, '비교': f'{a} - {b}', '차이': pt, '하한': lo, '상한': hi})
    return pd.DataFrame(rows)

def block_boot_diff(df, pa, pb, block=8, B=2000, seed=0):
    # 현 단위 부트스트랩은 같은 주에 여러 현이 함께 틀리는 것을 못 잡음. 예측 시점을 8주씩 묶어서 묶음 단위로도 뽑아 봄
    rng = np.random.default_rng(seed)
    ea = (np.log1p(df[pa]) - np.log1p(df.y_tgt)).abs(); eb = (np.log1p(df[pb]) - np.log1p(df.y_tgt)).abs()
    key = df.eval_year.astype(int) * 100 + (df.o_week.astype(int) - 1) // block
    g = pd.DataFrame({'k': key.to_numpy(), 'd': (ea - eb).to_numpy()}).groupby('k').d.agg(['sum', 'count'])
    s_, n_ = g['sum'].to_numpy(), g['count'].to_numpy()
    idx = rng.integers(0, len(g), size=(B, len(g)))
    d = s_[idx].sum(1) / n_[idx].sum(1)
    return s_.sum() / n_.sum(), np.quantile(d, 0.025), np.quantile(d, 0.975)

def peak_lead(E, W, yw, info, a=M2, b=M1):
    # 현과 연도마다 정점 1주 전~2주 후에 낸 예측(목표는 정점 3~6주 뒤)의 로그 편향과, 인근 3현 정점이 자기 현보다 얼마나 늦은지
    nb = nearest_k(info)
    ea = np.log1p(E['pred_' + a]) - E.tgt_log; eb = np.log1p(E['pred_' + b]) - E.tgt_log
    rows = []
    for Y in sorted(E.eval_year.unique()):
        t = yw.index[yw.year == Y]; wk = yw.loc[t, 'week'].to_numpy()
        pk = {p: int(yw.loc[W.loc[t, p].idxmax(), 'week']) for p in W.columns}
        com = {p: (W.loc[t, p].to_numpy() * wk).sum() / W.loc[t, p].sum() for p in W.columns}
        for p in W.columns:
            m = ((E.eval_year == Y) & (E.pref == p) & E.o_week.between(pk[p] - 1, pk[p] + 2)).to_numpy()
            if m.sum() < 3: continue  # 정점이 연말이면 목표 주가 다음 해로 넘어가서 행이 없음
            rows.append({'연도': Y, 'pref': p, '정점 주': pk[p], '인근 정점 − 자기 정점': np.mean([pk[q] for q in nb[p]]) - pk[p],
                         '인근 무게중심 − 자기 무게중심': np.mean([com[q] for q in nb[p]]) - com[p],
                         'M1 편향': eb[m].mean(), 'M2 편향': ea[m].mean(), '절대 오차 차이': (ea[m].abs() - eb[m].abs()).mean()})
    LD = pd.DataFrame(rows); LD['편향 차이'] = LD['M2 편향'] - LD['M1 편향']
    x = LD['인근 정점 − 자기 정점']
    LD['구분'] = np.where(x < -1, '인근이 1주 넘게 먼저', np.where(x > 1, '자기 현이 1주 넘게 먼저', '차이 1주 이내'))
    return LD

def boot_mean(v, g, B=2000, seed=0):
    # 값 v의 평균을 묶음 g(현) 단위로 다시 뽑아 95% 구간을 구함
    t = pd.DataFrame({'g': np.asarray(g), 'v': np.asarray(v)}).groupby('g').v.agg(['sum', 'count'])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(t), size=(B, len(t)))
    d = t['sum'].to_numpy()[idx].sum(1) / t['count'].to_numpy()[idx].sum(1)
    return t['sum'].sum() / t['count'].sum(), np.quantile(d, 0.025), np.quantile(d, 0.975)
