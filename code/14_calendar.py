# 평가 연도를 2015~2023년으로 넓혀 계절 입력과 인근 입력의 값어치를 해마다 비교 (최종보고서 6.8절): 표 16, 그림 9
from rsv_core import *
from plot_style import *
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

c, W, yw, info = load(DATA_DIR)
P = build_panel(W, yw, info)
# 2014년은 학습에 쓸 자료가 2013년 한 해뿐이라 2015년부터 평가함. 학습 구간을 자르는 규칙은 고정 설계와 같음
YEARS = tuple(range(2015, 2024))
EPI = [Y for Y in YEARS if Y != 2020]  # 보고가 급감한 2020년은 여러 해를 묶을 때 뺌
PRE, POST = list(range(2015, 2020)), [2021, 2022, 2023]
# 이름 끝의 x는 계절 입력(sin, cos)을 뺀 모델. M1x는 자기 현의 최근 값과 변화만 씀
SP = {**{m[:2]: cols for m, cols in MODELS.items()}, **{m[:2] + 'x': [c_ for c_ in cols if c_ not in ('ws', 'wc')] for m, cols in MODELS.items()}}
short = lambda m: 'M0' if m == M0 else m

def hgb(lr=0.05, it=300, leaf=31):
    # 6.4절과 같은 설정
    return lambda: HistGradientBoostingRegressor(learning_rate=lr, max_iter=it, max_leaf_nodes=leaf, early_stopping=False, random_state=0)

EV = {'릿지': evaluate_years(P, W, yw, years=YEARS, models=SP), '부스팅': evaluate_years(P, W, yw, years=YEARS, models=SP, learner=hgb())}
print('연도별 평가 행 수:', EV['릿지'].groupby('eval_year').size().to_dict())  # 2015, 2020년은 53주라서 한 주 많음
# 2021~2023년의 M1~M4는 고정 설계의 예측과 같아야 함
E0 = evaluate_years(P, W, yw); fx = EV['릿지'][EV['릿지'].eval_year >= 2021]
assert all(np.allclose(E0['pred_' + m], fx['pred_' + m[:2]]) for m in MODELS)
LM = pd.DataFrame([{'모델': k, '연도': Y, '평가 행': int((E.eval_year == Y).sum()), **{short(m): log_mae(E[E.eval_year == Y], m) for m in [M0] + list(SP)}}
                   for k, E in EV.items() for Y in YEARS])
save(LM, 'calendar_logmae.csv'); display(LM.round(3))

# 차이와 신뢰구간. 음수면 앞의 모델이 나음. 계절 입력과 인근 3현의 값어치는 둘 다 자기 현 이력만 쓴 M1x에 더했을 때로 잼
CMP = [('M1', 'M1x'), ('M2x', 'M1x'), ('M3x', 'M1x'), ('M2', 'M1'), ('M4', 'M3'), ('M4x', 'M3x'), ('M2', 'M2x'), ('M1', M0), ('M1x', M0)]
GROUPS = {**{str(Y): [Y] for Y in YEARS}, '2015~2019': PRE, '2021~2023': POST, '2020 제외': EPI, '아홉 해': list(YEARS)}
rows = []
for k, E in EV.items():
    for gname, ys in GROUPS.items():
        d = E[E.eval_year.isin(ys)]
        for a, b in CMP:
            pt, lo, hi = cluster_boot_diff(d, 'pred_' + a, 'pred_' + b)
            # 계절 입력은 모든 현에 같은 값이라 현 단위 구간이 좁게 나옴. 주 묶음 구간도 함께 둠 (한 해만 보면 묶음이 6~7개뿐이라 거칢)
            _, blo, bhi = block_boot_diff(d, 'pred_' + a, 'pred_' + b)
            rows.append({'모델': k, '연도': gname, '비교': f'{short(a)}-{short(b)}', '차이': pt, '하한': lo, '상한': hi, '주 묶음 하한': blo, '주 묶음 상한': bhi})
CE = pd.DataFrame(rows); save(CE, 'calendar_effects.csv')
eff = lambda k, cmp, col='차이': CE[(CE['모델'] == k) & (CE['비교'] == cmp)].set_index('연도')[col]
for k in EV:
    print(k); display(pd.DataFrame({cmp: eff(k, cmp) for cmp in ('M1-M1x', 'M2x-M1x', 'M2-M1', 'M4-M3', 'M4x-M3x', 'M1x-M0')}).round(3))
# 인근 정보의 이득이 계절 입력을 넣든 빼든, 릿지든 부스팅이든 해마다 같은 방향인지
nbr = CE[CE['비교'].isin(['M2x-M1x', 'M2-M1']) & CE['연도'].isin([str(Y) for Y in EPI])]
print('2020년을 뺀 여덟 해 × 모델 둘 × 계절 입력 유무: 음수 %d/%d, 현 단위 구간 상한이 0 아래 %d' % ((nbr['차이'] < 0).sum(), len(nbr), (nbr['상한'] < 0).sum()))

# 그해의 전국 곡선이 과거와 얼마나 닮았는지: 그 전 해들의 같은 주 평균 곡선과의 상관 (log1p 전국 합계, 53주째는 뺌)
# 그해가 끝나야 구할 수 있는 값이라 예측에는 못 쓰고, 결과를 읽는 데만 씀
nat = np.log1p(W.sum(axis=1))
prof = pd.DataFrame({Y: nat[yw.index[(yw.year == Y) & (yw.week <= 52)]].to_numpy() for Y in sorted(yw.year.unique())})
sim = lambda Y, past: np.corrcoef(prof[Y], prof[past].mean(axis=1))[0, 1]
before = lambda Y: [y for y in prof.columns if y < Y]
# 닮음을 다르게 재도 같은지 보려고 세 가지를 더 둠: 과거에서 2020년을 뺌, 직전 3년만, 직전 해만
SIMS = ['과거와 상관', '과거에서 2020년 제외', '직전 3년', '직전 해']
YD = pd.DataFrame([{'연도': Y, SIMS[0]: sim(Y, before(Y)), SIMS[1]: sim(Y, [y for y in before(Y) if y != 2020]), SIMS[2]: sim(Y, before(Y)[-3:]), SIMS[3]: sim(Y, [Y - 1])}
                   for Y in YEARS]).set_index('연도')
ystr = [str(Y) for Y in YEARS]
for k in EV:
    YD['계절 입력 ' + k] = eff(k, 'M1-M1x')[ystr].to_numpy(); YD['인근 ' + k] = eff(k, 'M2x-M1x')[ystr].to_numpy(); YD['M2-M1 ' + k] = eff(k, 'M2-M1')[ystr].to_numpy()
save(YD.reset_index(), 'calendar_years.csv'); display(YD.round(3))
# 2021~2023년은 이 분석을 하게 만든 해라서, 그 해를 뺀 2015~2019년과 2015~2020년만으로도 따로 계산
rows = []
for k in EV:
    for col in SIMS:
        for lab, ys in (('아홉 해', list(YEARS)), ('2020 제외', EPI), ('2015~2019', PRE), ('2015~2020', PRE + [2020])):
            r1, p1 = spearmanr(YD.loc[ys, col], YD.loc[ys, '계절 입력 ' + k]); r2, p2 = spearmanr(YD.loc[ys, col], YD.loc[ys, '인근 ' + k])
            rows.append({'모델': k, '닮음': col, '해': lab, '계절 입력 rho': r1, '계절 입력 p': p1, '인근 rho': r2, '인근 p': p2})
SR = pd.DataFrame(rows); save(SR, 'calendar_similarity.csv'); display(SR[SR['닮음'] == SIMS[0]].round(3))
rho_year = spearmanr(list(YEARS), YD[SIMS[0]])[0]
print('닮음은 뒤의 해일수록 낮음: 연도와 닮음의 순위 상관 %.2f' % rho_year)

# 계절 입력이 오래된 해의 패턴까지 끌고 와서 생긴 일인지: 릿지를 직전 3년의 행만으로 학습해서 다시 봄
EW = pd.concat([evaluate_years(P, W, yw, years=(Y,), models=SP, keep=lambda tr, Y=Y: tr.t_year >= Y - 3) for Y in YEARS], ignore_index=True)
rows = []
for Y in YEARS:
    d = EW[EW.eval_year == Y]; r = {'연도': Y}
    for a, b in (('M1', 'M1x'), ('M2x', 'M1x'), ('M2', 'M1')):
        r[f'{a}-{b}'], r[f'{a}-{b} 하한'], r[f'{a}-{b} 상한'] = cluster_boot_diff(d, 'pred_' + a, 'pred_' + b)
    rows.append(r)
WN = pd.DataFrame(rows); save(WN, 'calendar_window.csv'); display(WN.round(3))
# 이 학습 방식에 맞춰 직전 3년 곡선과의 닮음으로 순위 상관을 다시 구함. 2020년 곡선이 직전 3년 평균에 들어가는 해가 있어서 2020년을 뺀 것도 같이 봄
w3 = WN.set_index('연도')['M1-M1x']
sim3x = pd.Series({Y: sim(Y, [y for y in before(Y)[-3:] if y != 2020]) for Y in YEARS})
win_rho, win_rho_x = spearmanr(YD[SIMS[2]], w3)[0], spearmanr(sim3x, w3)[0]
print('직전 3년 학습: 직전 3년 닮음과 계절 입력 효과의 순위 상관 %.2f | 직전 3년에서 2020년을 빼면 %.2f' % (win_rho, win_rho_x))

# 계절 입력을 sin, cos 한 쌍으로만 둔 탓인지: 조화 항을 하나 더 넣거나 주 번호마다 항을 따로 두고 릿지로 다시 봄
PA = P.copy()
PA['ws2'] = np.sin(4 * np.pi * PA.t_week / 52.18); PA['wc2'] = np.cos(4 * np.pi * PA.t_week / 52.18)
wk = PA.t_week.fillna(0).astype(int).clip(upper=52)  # 53주째는 52주 항에 붙임
PA = pd.concat([PA, pd.DataFrame({f'wk{i}': (wk == i).astype(float) for i in range(1, 53)}, index=PA.index)], axis=1)
ALT = {'조화 항 둘': ['ws', 'wc', 'ws2', 'wc2'], '주 번호 항': [f'wk{i}' for i in range(1, 53)]}
SA = {'M1x': SP['M1x'], **{k + ' M1': SP['M1x'] + cols for k, cols in ALT.items()}, **{k + ' M2': SP['M1x'] + cols + NB for k, cols in ALT.items()}}
EA = evaluate_years(PA, W, yw, years=YEARS, models=SA)
rows = []
for k in ALT:
    for Y in YEARS:
        d = EA[EA.eval_year == Y]; r = {'계절 입력': k, '연도': Y}
        r['계절 입력 효과'], r['계절 입력 하한'], r['계절 입력 상한'] = cluster_boot_diff(d, 'pred_' + k + ' M1', 'pred_M1x')
        r['인근 추가'], r['인근 하한'], r['인근 상한'] = cluster_boot_diff(d, 'pred_' + k + ' M2', 'pred_' + k + ' M1')
        rows.append(r)
AS = pd.DataFrame(rows); save(AS, 'calendar_altseason.csv'); display(AS.round(3))
for k in ALT:
    v = AS[AS['계절 입력'] == k].set_index('연도')['계절 입력 효과']
    print('%s: 과거 닮음과 계절 입력 효과의 순위 상관 %.2f (2020년 제외 %.2f)' % (k, spearmanr(YD[SIMS[0]], v[list(YEARS)])[0], spearmanr(YD.loc[EPI, SIMS[0]], v[EPI])[0]))

# 인근 3현의 효과 가운데 가까운 현이어야 얻는 몫: 현마다 무작위 3현을 넣고 릿지로 다시 계산 (50번)
rows = []
for seed in range(50):
    Er = evaluate_years(build_panel(W, yw, info, nb_w=nb_weights(info, 3, 'random', seed=seed)), W, yw, years=YEARS, models=SP)
    for gname, ys in GROUPS.items():
        d = Er[Er.eval_year.isin(ys)]
        rows.append({'seed': seed, '연도': gname, **{f'{a}-{b}': log_mae(d, a) - log_mae(d, b) for a, b in (('M2x', 'M1x'), ('M2', 'M1'), ('M4', 'M3'), ('M4x', 'M3x'))}})
RD = pd.DataFrame(rows); save(RD, 'calendar_random.csv')
RS = pd.DataFrame({'인근 3현': eff('릿지', 'M2x-M1x'), '무작위 중앙값': RD.groupby('연도')['M2x-M1x'].median(), '무작위 2.5%': RD.groupby('연도')['M2x-M1x'].quantile(0.025),
                   '무작위 97.5%': RD.groupby('연도')['M2x-M1x'].quantile(0.975)}).loc[list(GROUPS)]
RS['무작위가 나은 횟수'] = [int((RD[RD['연도'] == g_]['M2x-M1x'] <= RS.loc[g_, '인근 3현']).sum()) for g_ in RS.index]
print('M2x − M1x, 릿지'); display(RS.round(3))

# 코로나19 이전 다섯 해와 이후 세 해의 차이
def period_gap(E, a, b, pre=PRE, post=POST, B=2000, seed=0):
    # 현 단위로는 같은 현을 두 기간에서 함께 뽑고, 주 묶음으로는 기간마다 묶음을 따로 뽑음
    e = (np.log1p(E['pred_' + a]) - np.log1p(E.y_tgt)).abs() - (np.log1p(E['pred_' + b]) - np.log1p(E.y_tgt)).abs()
    t = pd.DataFrame({'p': E.pref.to_numpy(), 'pre': np.where(E.eval_year.isin(pre), e, np.nan), 'post': np.where(E.eval_year.isin(post), e, np.nan)}).groupby('p').agg(['sum', 'count'])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(t), size=(B, len(t))); full = np.arange(len(t))
    m = lambda k, i: t[(k, 'sum')].to_numpy()[i].sum(-1) / t[(k, 'count')].to_numpy()[i].sum(-1)
    d = m('post', idx) - m('pre', idx)
    key = E.eval_year.astype(int) * 100 + (E.o_week.astype(int) - 1) // 8
    def blocks(ys):
        g = pd.DataFrame({'k': key[E.eval_year.isin(ys)].to_numpy(), 'e': e[E.eval_year.isin(ys)].to_numpy()}).groupby('k').e.agg(['sum', 'count'])
        i = rng.integers(0, len(g), size=(B, len(g)))
        return g['sum'].to_numpy()[i].sum(1) / g['count'].to_numpy()[i].sum(1)
    db = blocks(post) - blocks(pre)
    return m('pre', full), m('post', full), m('post', full) - m('pre', full), np.quantile(d, 0.025), np.quantile(d, 0.975), np.quantile(db, 0.025), np.quantile(db, 0.975)

rows = []
for k, E in EV.items():
    for a, b in (('M1', 'M1x'), ('M2x', 'M1x'), ('M3x', 'M1x'), ('M2', 'M1'), ('M4', 'M3'), ('M4x', 'M3x')):
        pre, post, gap, lo, hi, blo, bhi = period_gap(E, a, b)
        rows.append({'모델': k, '비교': f'{a}-{b}', '2015~2019': pre, '2021~2023': post, '차이': gap, '하한': lo, '상한': hi, '주 묶음 하한': blo, '주 묶음 상한': bhi})
PG = pd.DataFrame(rows); save(PG, 'calendar_prepost.csv'); display(PG.round(3))

# 부스팅은 입력 끝자리에 따라 셋째 자리가 흔들림(6.4절). 계절 입력을 뺀 나머지 입력에 1e-12 크기의 잡음을 더해 5번 다시 돌려서 방향이 그대로인지 봄
NOISY = [c_ for c_ in OWN + NB + NAT if c_ not in ('ws', 'wc')]
rows = []
for seed in range(5):
    Pn = P.copy(); Pn[NOISY] = Pn[NOISY] + np.random.default_rng(seed).normal(0, 1e-12, (len(Pn), len(NOISY)))
    En = evaluate_years(Pn, W, yw, years=YEARS, models=SP, learner=hgb())
    gap = lambda ys, a, b: log_mae(En[En.eval_year.isin(ys)], a) - log_mae(En[En.eval_year.isin(ys)], b)
    for gname, ys in GROUPS.items():
        rows.append({'seed': seed, '연도': gname, **{f'{short(a)}-{short(b)}': gap(ys, a, b) for a, b in CMP}})
NZ = pd.DataFrame(rows); save(NZ, 'calendar_noise.csv')
display(NZ.groupby('연도', sort=False)[['M1-M1x', 'M2x-M1x', 'M2-M1', 'M4-M3', 'M4x-M3x']].agg(['min', 'max']).round(3))
rho_nz = [spearmanr(YD[SIMS[0]], NZ[(NZ.seed == s) & NZ['연도'].isin(ystr)].set_index('연도').loc[ystr, 'M1-M1x'])[0] for s in range(5)]
print('잡음을 더한 5번에서 계절 입력 효과와 과거 닮음의 순위 상관: %.2f ~ %.2f' % (min(rho_nz), max(rho_nz)))
facts('calendar', {'nb_total': len(nbr), 'nb_neg': int((nbr['차이'] < 0).sum()), 'nb_sig': int((nbr['상한'] < 0).sum()), 'noise_n': 5, 'rho_noise_min': min(rho_nz), 'rho_noise_max': max(rho_nz),
                   'rho_year_sim': rho_year, 'win_rho': win_rho, 'win_rho_x': win_rho_x, 'rand_n': int(RD.seed.nunique())})

fig, axes = plt.subplots(1, 2, figsize=(WD, 2.85), sharey=True)
x = np.arange(len(YEARS))
for ax, (k, ttl) in zip(axes, (('릿지', '릿지 회귀'), ('부스팅', '그래디언트 부스팅'))):
    ax.axvspan(YEARS.index(2020) - 0.5, YEARS.index(2020) + 0.5, color='#f0efec', zorder=0)  # 보고가 급감한 해
    for j, (cmp, lab, col) in enumerate((('M1-M1x', '계절 입력을 더했을 때', B2), ('M2x-M1x', '인근 3현을 더했을 때', B1))):
        xs = x + (j - 0.5) * 0.3  # 겹치지 않게
        ax.vlines(xs, eff(k, cmp, '하한')[ystr], eff(k, cmp, '상한')[ystr], color=col, linewidth=1.4)
        ax.plot(xs, eff(k, cmp)[ystr], 'o', color=col, markersize=5, markeredgecolor='white', markeredgewidth=1.0, linestyle='none', label=lab)
    ax.axhline(0, color=INK2, linewidth=0.8); clean(ax); ax.set_xticks(x); ax.set_xticklabels(ystr, fontsize=7); ax.set_title(ttl, fontsize=8.2, pad=4)
axes[0].set_ylabel('로그 MAE 변화 (음수 = 개선)')
axes[0].text(YEARS.index(2020), axes[0].get_ylim()[0] + 0.004, '보고가\n급감한 해', fontsize=6.5, color=INK2, ha='center', va='bottom', linespacing=1.3)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, fontsize=7.4, handletextpad=0.3, columnspacing=1.6)
fig.tight_layout(); fig.subplots_adjust(bottom=0.2); fig.savefig(FIG_DIR / 'fig9_calendar.png'); plt.close(fig)
