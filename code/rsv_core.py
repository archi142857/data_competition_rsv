# 공통 함수. 노트북의 공통 함수 셀과 같은 코드
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

def build_panel(W, yw, info, h=4, k=3, nb_lag=0):
    # nb_lag=1이면 인근 값을 한 주 늦게 씀 (공개 지연 점검용)
    L = np.log1p(W)  # 0이 많아서 log1p
    NAT = np.log1p(W.sum(axis=1))
    nb = nearest_k(info, k)
    T = W.index.to_numpy()
    rows = []
    for p in W.columns:
        s = L[p]
        nbm = L[nb[p]].mean(axis=1).shift(nb_lag)
        df = pd.DataFrame({
            'time': T, 'pref': p, 'y_now': W[p].to_numpy(), 'y_tgt': W[p].shift(-h).to_numpy(),
            'l0': s, 'l1': s.shift(1), 'l2': s.shift(2), 'l3': s.shift(3), 'd4': s - s.shift(4),
            'nb0': nbm, 'nb_d4': nbm - nbm.shift(4), 'nb_gap': nbm - s,
            'nat0': NAT, 'nat_d4': NAT - NAT.shift(4),
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
# TODO 최종보고서: 인근 k 바꿔 보기(1~5), 같은 섬 안에서만 고르기
MODELS = {'M1 자기 현 이력': OWN, 'M2 인근 3현 추가': OWN + NB, 'M3 전국 추가': OWN + NAT, 'M4 인근과 전국 추가': OWN + NB + NAT}

def evaluate_years(P, W, yw, years=(2021, 2022, 2023), alpha=1.0):
    out = []
    for Y in years:
        y0 = yw.index[yw.year == Y].min()
        te = P[(P.o_year == Y) & (P.t_year == Y)].copy()
        # 목표 시점 기준으로 자름. 예측 시점으로 자르면 작년 마지막 4주 행의 목표값이 올해 값이 됨
        tr = P[(P.t_time < y0)].dropna(subset=OWN + NB + NAT + ['tgt_log'])
        te = te.dropna(subset=OWN + NB + NAT + ['tgt_log'])
        scale = W.loc[W.index < y0].mean().clip(lower=1.0)  # 규모 보정 MAE 분모
        te['scale'] = te.pref.map(scale)
        te['pred_M0 현재값 유지'] = te.y_now
        for name, cols in MODELS.items():
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(tr[cols], tr.tgt_log)
            te['pred_' + name] = np.clip(np.expm1(m.predict(te[cols])), 0, None)
        te['eval_year'] = Y
        out.append(te)
    return pd.concat(out, ignore_index=True)

def metrics(df, pcol):
    e = df[pcol] - df.y_tgt
    return pd.Series({'MAE': e.abs().mean(), 'RMSE': np.sqrt((e**2).mean()),
                      'logMAE': (np.log1p(df[pcol]) - np.log1p(df.y_tgt)).abs().mean(),
                      'scaledMAE': (e.abs() / df.scale).mean(), 'n': len(df)})

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
