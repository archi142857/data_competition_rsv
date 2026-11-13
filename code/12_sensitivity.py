# 특이 관측, 기준 모델, 설정을 바꿨을 때 결론이 유지되는지 (최종보고서 6.6~6.7절): 표 12, 표 13, 표 14, 표 15
from rsv_core import *
from sklearn.ensemble import HistGradientBoostingRegressor

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

# 묶음 길이를 4주, 13주, 26주로 바꿔도 세 해 전체의 구간이 0 아래인지
rows = []
for a, b in PAIRS:
    for blk in (4, 8, 13, 26):
        pt, lo, hi = block_boot_diff(E0, 'pred_' + a, 'pred_' + b, block=blk)
        rows.append({'비교': f'{a[:2]}-{b[:2]}', '묶음(주)': blk, '차이': pt, '하한': lo, '상한': hi})
BL = pd.DataFrame(rows); save(BL, 'block_length.csv'); display(BL.round(3))

# 예측 시점의 보고 수 구간별 M2 − M1. 2022년의 개선이 작은 보고 수 쪽에 몰렸는지, 세 해 전체도 그런지
e21 = (np.log1p(E0['pred_' + M2]) - np.log1p(E0.y_tgt)).abs() - (np.log1p(E0['pred_' + M1]) - np.log1p(E0.y_tgt)).abs()
BINS = {'0건': E0.y_now == 0, '1~4건': E0.y_now.between(1, 4), '5~19건': E0.y_now.between(5, 19), '20건 이상': E0.y_now >= 20}
CB = pd.DataFrame([{'연도': lab, '구간': k, '행 수': int((m & ym).sum()), 'M2-M1': e21[m & ym].mean()}
                   for lab, ym in (('2022', E0.eval_year == 2022), ('전체', E0.eval_year > 0)) for k, m in BINS.items()])
save(CB, 'count_bins.csv'); display(CB.round(3))

def group_gap(d, e, mask, key=None, B=2000, seed=0):
    # 두 집단의 차이의 구간. 같은 현(key를 주면 그 묶음)을 두 집단에서 함께 뽑음
    t = pd.DataFrame({'p': d.pref.to_numpy() if key is None else key, 'a': np.where(mask, e, np.nan), 'b': np.where(~mask, e, np.nan)}).groupby('p').agg(['sum', 'count'])
    idx = np.random.default_rng(seed).integers(0, len(t), size=(B, len(t))); full = np.arange(len(t))
    m = lambda k, i: t[(k, 'sum')].to_numpy()[i].sum(-1) / t[(k, 'count')].to_numpy()[i].sum(-1)
    dd = m('a', idx) - m('b', idx)
    return m('a', full) - m('b', full), np.quantile(dd, 0.025), np.quantile(dd, 0.975)

y22 = (E0.eval_year == 2022).to_numpy()
gap22 = group_gap(E0[y22], e21[y22].to_numpy(), (E0[y22].y_now < 5).to_numpy())
gap22b = group_gap(E0[y22], e21[y22].to_numpy(), (E0[y22].y_now < 5).to_numpy(), key=((E0[y22].o_week.astype(int) - 1) // 8).to_numpy())  # 8주 묶음은 2022년에 6개뿐
print('2022년 M2-M1, 5건 미만과 5건 이상의 차이: %.3f [%.3f, %.3f] | 주 묶음 [%.3f, %.3f]' % (*gap22, *gap22b[1:]))

# 현별로 보면 몇 개 현에서 나아졌는지
rows = []
for a, b in PAIRS:
    e = (np.log1p(E0['pred_' + a]) - np.log1p(E0.y_tgt)).abs() - (np.log1p(E0['pred_' + b]) - np.log1p(E0.y_tgt)).abs()
    for Y in (2021, 2022, 2023, '전체'):
        m = (E0.eval_year > 0) if Y == '전체' else (E0.eval_year == Y)
        bp = e[m].groupby(E0.pref[m].to_numpy()).mean()
        rows.append({'비교': f'{a[:2]}-{b[:2]}', '연도': Y, '나아진 현': int((bp < 0).sum()), '현 수': len(bp), '최소': bp.min(), '중앙값': bp.median(), '최대': bp.max()})
PS = pd.DataFrame(rows); save(PS, 'pref_share.csv'); display(PS.round(3))

# 기준을 더 강하게 잡아도 이득이 남는지. 자기 현 이력을 늘리고, 전국 입력에는 인근 입력과 같은 형태(자기 현을 뺀 46개 현의 로그 평균, 그 4주 변화, 자기 현과의 차이)를 더함
# 가장 강한 기준은 이력 12주에 1, 2, 8주 변화와 이동평균, 전국 쪽에는 지연값과 1, 2주 변화까지 넣음. 입력은 모두 예측 시점까지의 값으로만 만듦
L = np.log1p(W); natL = np.log1p(W.sum(axis=1)); ext = []
for p in W.columns:
    s = L[p]; om = (L.sum(axis=1) - s) / (len(W.columns) - 1)
    d = {'time': W.index.to_numpy(), 'pref': p, **{f'l{k}': s.shift(k).to_numpy() for k in range(4, 13)}}
    d.update({'d1': (s - s.shift(1)).to_numpy(), 'd2': (s - s.shift(2)).to_numpy(), 'd8': (s - s.shift(8)).to_numpy(),
              'm4': s.rolling(4).mean().to_numpy(), 'm8': s.rolling(8).mean().to_numpy(), 'm13': s.rolling(13).mean().to_numpy(),
              'om0': om.to_numpy(), 'om_d4': (om - om.shift(4)).to_numpy(), 'om_gap': (om - s).to_numpy(), 'om_d1': (om - om.shift(1)).to_numpy(), 'om_d2': (om - om.shift(2)).to_numpy(),
              'nat1': natL.shift(1).to_numpy(), 'nat2': natL.shift(2).to_numpy(), 'nat3': natL.shift(3).to_numpy()})
    ext.append(pd.DataFrame(d))
EXT = pd.concat(ext, ignore_index=True)
H8 = ['l4', 'l5', 'l6', 'l7', 'pk', 'cum']; H12 = H8 + ['l8', 'l9', 'l10', 'l11', 'l12', 'd1', 'd2', 'd8', 'm4', 'm8', 'm13']
OM = ['om0', 'om_d4', 'om_gap']; NATX = OM + ['nat1', 'nat2', 'nat3', 'nat_d1', 'nat_d2', 'om_d1', 'om_d2']
STRONG = {'이력 8주, 전국 입력 확장': (OWN + H8, NAT + OM), '이력 8주, 전국 입력 확장, 계절 입력 제외': (OWN_NS + H8, NAT + OM),
          '이력 12주, 전국 입력 더 확장, 계절 입력 제외': (OWN_NS + H12, NAT + NATX)}
strong_models = lambda own, natc: {M1: own, M2: own + NB, M3: own + natc, M4: own + natc + NB}

def hgb(lr=0.05, it=300, leaf=31):
    # 6.4절과 같은 설정
    return lambda: HistGradientBoostingRegressor(learning_rate=lr, max_iter=it, max_leaf_nodes=leaf, early_stopping=False, random_state=0)

PX = P.merge(EXT, on=['time', 'pref'])
YV = tuple(range(2015, 2024))  # 6.8절과 같은 아홉 해
rows, yrows = [], []
for name, (own, natc) in STRONG.items():
    for lab, kw in (('릿지', {'years': YV}), ('부스팅', {'learner': hgb()})):  # 릿지는 빨라서 아홉 해를 모두 평가하고 연도별 값도 봄
        Ea = evaluate_years(PX, W, yw, models=strong_models(own, natc), **kw)
        E = Ea[Ea.eval_year >= 2021]
        groups = [(str(Y), Ea.eval_year == Y) for Y in sorted(Ea.eval_year.unique())]
        if lab == '릿지': groups += [('2015~2019', Ea.eval_year <= 2019), ('2020 제외 여덟 해', Ea.eval_year != 2020)]
        for glab, gm in groups:
            d = Ea[gm]; q = {'기준': name, '모델': lab, '연도': glab, '평가 행': len(d), 'M1': log_mae(d, M1), 'M2': log_mae(d, M2)}
            for a, b in PAIRS:
                k = f'{a[:2]}-{b[:2]}'; q[k], q[k + ' 하한'], q[k + ' 상한'] = cluster_boot_diff(d, 'pred_' + a, 'pred_' + b)
            yrows.append(q)
        r = {'기준': name, '모델': lab, '평가 행': len(E), '입력 수(M4)': len(own + natc + NB), **{m[:2]: log_mae(E, m) for m in (M1, M2, M3, M4)}}
        for a, b in PAIRS:
            k = f'{a[:2]}-{b[:2]}'
            r[k], r[k + ' 하한'], r[k + ' 상한'] = cluster_boot_diff(E, 'pred_' + a, 'pred_' + b)
            _, r[k + ' 주 묶음 하한'], r[k + ' 주 묶음 상한'] = block_boot_diff(E, 'pred_' + a, 'pred_' + b)
        rows.append(r)
SB = pd.DataFrame(rows); save(SB, 'strong_baseline.csv'); display(SB.round(3))
SY = pd.DataFrame(yrows); save(SY, 'strong_years.csv'); display(SY.pivot(index=['기준', '모델'], columns='연도', values='M2-M1').round(3))
# 강한 기준에서도 가까운 현이어야 하는지: 무작위 3현 30번 (릿지)
rows = []
for seed in range(30):
    Pr = build_panel(W, yw, info, nb_w=nb_weights(info, 3, 'random', seed=seed)).merge(EXT, on=['time', 'pref'])
    for name, (own, natc) in STRONG.items():
        Er = evaluate_years(Pr, W, yw, models=strong_models(own, natc))
        rows.append({'seed': seed, '기준': name, 'M2-M1': log_mae(Er, M2) - log_mae(Er, M1), 'M4-M3': log_mae(Er, M4) - log_mae(Er, M3)})
SR_ = pd.DataFrame(rows); save(SR_, 'strong_random.csv'); display(SR_.groupby('기준', sort=False)[['M2-M1', 'M4-M3']].agg(['median', 'min', 'max']).round(3))

# 설정을 평가 연도 이전의 자료만으로 골라도 결론이 같은지. 인근 현의 수, 릿지 α, 계절 입력을 넣을지를
# 모델마다 그 전 여러 해(2015년부터)를 같은 방식으로 평가한 로그 MAE가 가장 낮은 조합으로 고름
KS = (1, 2, 3, 5, 8, 12, 16, 24); ALPHAS = (0.01, 0.1, 1, 10, 100)
PART = {M1: [], M2: NB, M3: NAT, M4: NB + NAT}
pred, base = {}, None
for k in KS:
    Pk = build_panel(W, yw, info, k=k)
    for al in ALPHAS:
        Ek = evaluate_years(Pk, W, yw, years=YV, alpha=al, models={f'{m}|{s}': own + PART[m] for m in PART for s, own in (('넣음', OWN), ('뺌', OWN_NS))})
        if base is None: base = Ek[['time', 'pref', 'y_tgt', 'eval_year', 'o_week']].copy()
        assert (Ek.time.to_numpy() == base.time.to_numpy()).all() and (Ek.pref.to_numpy() == base.pref.to_numpy()).all()
        for m in PART:
            if k != 3 and m in (M1, M3): continue  # 인근 입력이 없는 모델은 k와 무관
            for s in ('넣음', '뺌'): pred[(m, k, al, s)] = Ek[f'pred_{m}|{s}'].to_numpy()
tl = np.log1p(base.y_tgt.to_numpy()); yr = base.eval_year.to_numpy()
err = {key: np.abs(np.log1p(v) - tl) for key, v in pred.items()}
assert all(np.allclose(pred[(m, 3, 1, '넣음')][yr >= 2021], E0['pred_' + m].to_numpy()) for m in PART)  # 고정 설계가 후보에 그대로 들어 있는지
def pick(val):
    # val(Y): 평가 연도 Y의 설정을 고를 때 검증에 쓰는 행
    EN = base.copy(); rows = []
    for Y in (2021, 2022, 2023):
        va, te = val(Y), yr == Y
        for m in PART:
            best = min((key for key in err if key[0] == m), key=lambda key: err[key][va].mean())
            EN.loc[te, 'pred_' + m] = pred[best][te]
            rows.append({'평가 연도': Y, '모델': m[:2], '인근 현 수': best[1] if m in (M2, M4) else np.nan, 'α': best[2], '계절 입력': best[3],
                         '후보 수': sum(key[0] == m for key in err), '검증 로그 MAE': err[best][va].mean(), '평가 로그 MAE': err[best][te].mean(), '고정 설계 로그 MAE': err[(m, 3, 1, '넣음')][te].mean()})
    return pd.DataFrame(rows), EN[yr >= 2021]

# 보고가 급감한 2020년이 검증에 들어가면 선택이 그해에 끌려갈 수 있어서, 2020년을 뺀 검증으로도 고름
VAL = {'2015년부터 전 해까지': lambda Y: yr < Y, '2020년 제외': lambda Y: (yr < Y) & (yr != 2020)}
sel, rows = [], []
for vname, val in VAL.items():
    S1, EN = pick(val); S1.insert(0, '검증', vname); sel.append(S1)
    for a, b in PAIRS:
        for Y in (2021, 2022, 2023, '전체'):
            d = EN if Y == '전체' else EN[EN.eval_year == Y]
            c1 = cluster_boot_diff(d, 'pred_' + a, 'pred_' + b); c2 = block_boot_diff(d, 'pred_' + a, 'pred_' + b)
            rows.append({'검증': vname, '비교': f'{a[:2]}-{b[:2]}', '연도': Y, '차이': c1[0], '현 하한': c1[1], '현 상한': c1[2], '주 묶음 하한': c2[1], '주 묶음 상한': c2[2],
                         **{m[:2]: log_mae(d, m) for m in (a, b)}})
SEL = pd.concat(sel, ignore_index=True); save(SEL, 'nested_select.csv'); display(SEL.round(3))
NE = pd.DataFrame(rows); save(NE, 'nested_effects.csv'); display(NE.round(3))
# 인근 현 수에 따른 검증 오차 (계절 입력 제외, α는 가장 나은 값, 검증에 2020년 포함)
VK = pd.DataFrame([{'평가 연도': Y, '모델': m[:2], '인근 현 수': k, '검증 로그 MAE': min(err[(m, k, al, '뺌')][yr < Y].mean() for al in ALPHAS)}
                   for Y in (2021, 2022, 2023) for m in (M2, M4) for k in KS])
save(VK, 'nested_k.csv'); display(VK.pivot(index=['평가 연도', '모델'], columns='인근 현 수', values='검증 로그 MAE').round(4))

d22 = E0[(E0.eval_year == 2022) & ~E0.pref.isin(['Osaka', 'Aichi', 'Hyogo'])]
rm = lambda d, m: np.sqrt(((d['pred_' + m] - d.y_tgt)**2).mean())
print('2022년 원래 단위 RMSE, 세 현 제외: M1 %.1f, M2 %.1f | MAE M1 %.1f, M2 %.1f' % (rm(d22, M1), rm(d22, M2), (d22['pred_' + M1] - d22.y_tgt).abs().mean(), (d22['pred_' + M2] - d22.y_tgt).abs().mean()))
facts('sensitivity', {'floor': floor, 'low_first': low.min(), 'low_last': low.max(), 'low_n': len(low), 'train_all': n_all, 'train_cut': n_cut,
                      'ns_fix_d': ns_fix[0], 'ns_fix_lo': ns_fix[1], 'ns_fix_hi': ns_fix[2], 'ns_rand_med': np.median(ns_rand), 'ns_rand_n': len(ns_rand),
                      'ns_big22_d': ns22[0], 'ns_big22_lo': ns22[1], 'ns_big22_hi': ns22[2],
                      'ns_rand_better': int((np.array(ns_rand) <= log_mae(ENS, M4) - log_mae(ENS, M3)).sum()),
                      'rmse22_no3_M1': rm(d22, M1), 'rmse22_no3_M2': rm(d22, M2), 'n_small': int((E0.y_now < 5).sum()),
                      'gap22_d': gap22[0], 'gap22_lo': gap22[1], 'gap22_hi': gap22[2], 'gap22_blo': gap22b[1], 'gap22_bhi': gap22b[2],
                      'nv_first': YV[0], 'nv_k_n': len(KS), 'nv_k_min': min(KS), 'nv_k_max': max(KS), 'nv_al_n': len(ALPHAS), 'nv_al_min': min(ALPHAS), 'nv_al_max': max(ALPHAS),
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
