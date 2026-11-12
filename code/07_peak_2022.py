# 2022년 정점 직후 과대 예측 (최종보고서 6.1절): 표 5, 표 6, 표 7, 그림 6, 그림 7
from rsv_core import *
from plot_style import *
from scipy.stats import spearmanr

c, W, yw, info = load(DATA_DIR)
P = build_panel(W, yw, info)
E = evaluate_years(P, W, yw)
BIG3 = ['Osaka', 'Aichi', 'Hyogo']
KOR = {'Osaka': '오사카', 'Aichi': '아이치', 'Hyogo': '효고'}

# 1) 입력 묶음이 예측값(로그)에 더한 크기. 기준은 학습 자료의 평균
def contrib(Y, cols, rows):
    tr = train_rows(P, yw, Y)
    m = ridge().fit(tr[cols], tr.tgt_log)
    sc, rg = m.named_steps['standardscaler'], m.named_steps['ridge']
    b = pd.Series(rg.coef_ / sc.scale_, index=cols); mu = pd.Series(sc.mean_, index=cols)
    con = (rows[cols] - mu) * b
    # nb_gap = nb0 - l0 이라 그대로 두면 자기 현 수준이 인근 쪽으로 섞여 들어감. 수준 항으로 다시 나눔
    bg = b.get('nb_gap', 0.0)
    g = pd.DataFrame({'자기 현 수준': con[['l0', 'l1', 'l2', 'l3']].sum(axis=1) - bg * (rows.l0 - mu.l0),
                      '자기 현 4주 변화': con.d4, '계절': con[['ws', 'wc']].sum(axis=1)})
    if 'nb0' in cols:
        g['인근 수준'] = con.nb0 + bg * (rows.nb0 - mu.nb0); g['인근 4주 변화'] = con.nb_d4
    if 'nat0' in cols:
        g['전국 수준'] = con.nat0; g['전국 4주 변화'] = con.nat_d4
    g['절편'] = rg.intercept_
    assert np.allclose(g.sum(axis=1), m.predict(rows[cols]))
    return g

sel = P[(P.o_year == 2022) & P.pref.isin(BIG3) & P.o_week.between(27, 31)]
parts = []
for nm in (M1, M2, M4):
    g = contrib(2022, MODELS[nm], sel)
    g.insert(0, '모델', nm); g.insert(1, 'pref', sel.pref); g.insert(2, 'o_week', sel.o_week)
    g['y_now'] = sel.y_now; g['y_tgt'] = sel.y_tgt; g['예측'] = np.expm1(g.drop(columns=['모델', 'pref', 'o_week', 'y_now', 'y_tgt']).sum(axis=1))
    parts.append(g)
DC = pd.concat(parts, ignore_index=True); save(DC, 'peak2022_decomp.csv')
gcols = ['자기 현 수준', '자기 현 4주 변화', '계절', '인근 수준', '인근 4주 변화', '전국 수준', '전국 4주 변화']
T5 = DC.groupby(['pref', '모델'])[gcols + ['예측', 'y_tgt']].mean().loc[BIG3]
T5['예측/실제'] = DC.assign(r=np.log1p(DC['예측']) - np.log1p(DC.y_tgt)).groupby(['pref', '모델']).r.mean().apply(np.exp)
display(T5.round(3)); save(T5.reset_index(), 'peak2022_decomp_mean.csv')
o = DC[(DC.pref == 'Osaka') & (DC.o_week == 29)].set_index('모델')
print('오사카 29주: M1 %.0f건, M2 %.0f건 | M2의 인근 4주 변화 항 %.3f (%.2f배), M1과 M2의 자기 현 변화 항 %.3f, %.3f' % (
    o.loc[M1, '예측'], o.loc[M2, '예측'], o.loc[M2, '인근 4주 변화'], np.exp(o.loc[M2, '인근 4주 변화']), o.loc[M1, '자기 현 4주 변화'], o.loc[M2, '자기 현 4주 변화']))
nb3 = nearest_k(info)
t22 = yw.index[yw.year == 2022]
pk22 = {p: int(yw.loc[W.loc[t22, p].idxmax(), 'week']) for p in W.columns}
print('2022년 정점 주와 인근 3현의 정점 주:', {p: (pk22[p], {q: pk22[q] for q in nb3[p]}) for p in BIG3})
save(pd.DataFrame([{'pref': p, '정점 주': pk22[p], **{f'인근{i + 1}': q for i, q in enumerate(nb3[p])}, **{f'인근{i + 1} 정점 주': pk22[q] for i, q in enumerate(nb3[p])}} for p in BIG3]), 'peak2022_weeks.csv')

fig, axes = plt.subplots(1, 2, figsize=(WD, 2.5), sharey=True)
for ax, p in zip(axes, ['Osaka', 'Aichi']):
    r = P[(P.o_year == 2022) & (P.pref == p) & P.o_week.between(18, 42)]
    g = contrib(2022, MODELS[M2], r)
    ax.axvline(pk22[p], color=BASE, linewidth=0.9)
    ax.plot(r.o_week, g['자기 현 4주 변화'], color=B1, linewidth=1.3, label='자기 현의 4주 변화')
    ax.plot(r.o_week, g['인근 4주 변화'], color=B2, linewidth=1.3, label='인근 3현의 4주 변화')
    ax.axhline(0, color=INK2, linewidth=0.7); clean(ax); ax.set_xlabel('예측 시점 (2022년 주)'); ax.set_title(KOR[p], fontsize=8.4, pad=4)
    ax.text(pk22[p] + 0.4, 1.12, '정점', fontsize=7, color=INK2, va='top')
axes[0].set_ylabel('로그 예측값에 더한 크기'); axes[1].legend(loc='upper right', fontsize=7.2)
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig6_decomp.png'); plt.close(fig)

# 2) 감소 전환을 잡으려고 넣어 본 입력. 2022년 오차를 보고 고른 것들이라 2021, 2023년 결과를 같이 봄
P['nb_d4_dn'] = P.nb_d4 * (P.pk < 0)  # 자기 현이 최근 8주 최고치에서 내려온 뒤에는 인근 변화의 계수를 따로 둠
lp = np.log(info['pop']); relpop = pd.Series({p: lp[p] - np.mean([lp[q] for q in nb3[p]]) for p in W.columns})
P['nb_d4_pop'] = P.nb_d4 * P.pref.map(relpop)  # 인근보다 인구가 많은 현은 인근 변화를 덜 따르게
SETS = {'기본(M1~M4)': ([], [], []),
        'A 최근 1~2주 변화': ([], ['nb_d1', 'nb_d2'], ['nat_d1', 'nat_d2']),
        'B 최근 8주 최고치 대비 수준': (['pk'], ['nb_pk'], []),
        'C 최근 26주 평균(장기 평균 대비)': (['cum'], [], []),
        'D 자기 현 하락 뒤의 인근 변화': (['pk'], ['nb_d4_dn'], []),
        'E 현의 평소 수준': (['typ'], [], []),  # E, F는 부록 A에만 적음
        'F 인근 대비 인구와 인근 변화의 곱': ([], ['nb_d4_pop'], [])}
ratio = lambda d, m: float(np.exp((np.log1p(d['pred_' + m]) - np.log1p(d.y_tgt)).mean()))
rows = []; EC = None
for nm, (own, nbx, natx) in SETS.items():
    mods = {M1: OWN + own, M2: OWN + own + NB + nbx, M3: OWN + own + NAT + natx, M4: OWN + own + NB + nbx + NAT + natx}
    Ei = evaluate_years(P, W, yw, models=mods)
    if nm.startswith('C'): EC = Ei
    b3 = Ei[(Ei.eval_year == 2022) & Ei.pref.isin(BIG3) & Ei.o_week.between(27, 31)]; d22 = Ei[Ei.eval_year == 2022]
    ef = effect_table(Ei, [(M2, M1), (M4, M3)]); ef = ef[ef['연도'] == '전체'].set_index('비교')
    r = {'입력': nm}
    for m in (M1, M2, M3, M4): r['logMAE ' + m[:2]] = log_mae(Ei, m)
    for m in (M1, M2): r['2022 RMSE ' + m[:2]] = np.sqrt(((d22['pred_' + m] - d22.y_tgt)**2).mean())
    for m in (M1, M2): r['세 현 예측/실제 ' + m[:2]] = ratio(b3, m)
    for k, a in (('M2-M1', f'{M2} - {M1}'), ('M4-M3', f'{M4} - {M3}')):
        r[k] = ef.loc[a, '차이']; r[k + ' 하한'] = ef.loc[a, '하한']; r[k + ' 상한'] = ef.loc[a, '상한']
    rows.append(r)
T6 = pd.DataFrame(rows); save(T6, 'peak2022_inputs.csv'); display(T6.round(3))
display(effect_table(EC, [(M2, M1), (M4, M3)]).round(3))  # C를 넣은 모델끼리의 연도별 차이

# 3) 인근보다 먼저 정점에 이른 현에서 인근 정보가 해가 되는가 (사후 진단. 정점은 지나고 나서야 알 수 있음)
LD = peak_lead(E, W, yw, info); save(LD, 'peak_lead.csv')
x, y = '인근 정점 − 자기 정점', '절대 오차 차이'
rho = {str(Y): spearmanr(d[x], d[y])[0] for Y, d in LD.groupby('연도')}; rho['전체'] = spearmanr(LD[x], LD[y])[0]
rng = np.random.default_rng(0); prefs = LD.pref.unique(); bs = []
for _ in range(2000):  # 한 현이 세 해에 걸쳐 나오니까 현 단위로 뽑음
    s = pd.concat([LD[LD.pref == q] for q in rng.choice(prefs, len(prefs))])
    bs.append(spearmanr(s[x], s[y])[0])
print('현×연도 %d개 | 정점 순서와 절대 오차 차이의 순위 상관: %s | 전체 95%% 구간 [%.2f, %.2f]' % (len(LD), {k: round(float(v), 2) for k, v in rho.items()}, *np.quantile(bs, [0.025, 0.975])))
# 편향 차이는 M2 예측과 M1 예측의 로그 차이라서 실제값이 안 들어감. 모델이 어떻게 움직이는지 보는 용도
rho_bias = spearmanr(LD[x], LD['편향 차이'])[0]
rho_com = spearmanr(LD['인근 무게중심 − 자기 무게중심'], LD[y])[0]  # 최고 주는 잡음이 커서 무게중심으로도 확인
print('정점 순서와 편향 차이의 순위 상관 %.2f | 무게중심 차이로 바꿨을 때 절대 오차 차이와의 순위 상관 %.2f' % (rho_bias, rho_com))
rows = []
for k in ('인근이 1주 넘게 먼저', '차이 1주 이내', '자기 현이 1주 넘게 먼저'):
    d = LD[LD['구분'] == k]; pt, lo, hi = boot_mean(d[y], d.pref)
    rows.append({'구분': k, '현×연도': len(d), 'M1 편향': d['M1 편향'].mean(), 'M2 편향': d['M2 편향'].mean(), y: pt, '하한': lo, '상한': hi})
T7 = pd.DataFrame(rows); display(T7.round(3)); save(T7, 'peak_lead_groups.csv')
# 최고 주가 연초나 연말에 잡혀 차이가 13주를 넘는 조합(그림 7에서 잘라낸 것)을 빼도 같은지
trim = LD[LD[x].abs() <= 13]; tg = trim.groupby('구분')[y].mean()
print('차이 13주 이내 %d개만: %s | 순위 상관 %.2f' % (len(trim), tg.round(3).to_dict(), spearmanr(trim[x], trim[y])[0]))
facts('lead', {'n': len(LD), **{'rho_' + k: v for k, v in rho.items()}, 'rho_lo': np.quantile(bs, 0.025), 'rho_hi': np.quantile(bs, 0.975), 'rho_bias': rho_bias, 'rho_com': rho_com,
               'clip': int((LD[x].abs() > 13).sum()), 'trim_nb': tg['인근이 1주 넘게 먼저'], 'trim_same': tg['차이 1주 이내'], 'trim_own': tg['자기 현이 1주 넘게 먼저'],
               'trim_rho': spearmanr(trim[x], trim[y])[0]})
display(LD.groupby(['연도', '구분']).agg(n=('pref', 'size'), d=(y, 'mean')).round(3).unstack('연도'))
print('세 현(2022):', LD[(LD['연도'] == 2022) & LD.pref.isin(BIG3)][['pref', '정점 주', x, 'M1 편향', 'M2 편향', y]].round(2).to_dict('records'))

fig, ax = plt.subplots(figsize=(WD * 0.62, 3.0))
for Y, col in ((2021, B1), (2022, B2), (2023, B3)):
    d = LD[LD['연도'] == Y]
    ax.plot(d[x], d[y], 'o', color=col, markersize=4.2, markeredgecolor='white', markeredgewidth=0.6, alpha=0.9, label=str(Y), linestyle='none')
b = LD[(LD['연도'] == 2022) & LD.pref.isin(['Osaka', 'Aichi'])]  # 글자를 붙이면 다른 점과 겹쳐서 테두리로만 표시
ax.plot(b[x], b[y], 'o', markersize=6.4, markerfacecolor='none', markeredgecolor=INK, markeredgewidth=1.0, linestyle='none')
ax.axhline(0, color=INK2, linewidth=0.7); ax.axvline(0, color=INK2, linewidth=0.7); clean(ax)
ax.set_xlim(-13, 13)  # 최고 주가 연초나 연말에 잡힌 현은 차이가 수십 주로 나와서 그림에서는 잘라냄
ax.set_xlabel('인근 3현의 평균 정점 주 − 자기 현의 정점 주'); ax.set_ylabel('절대 로그 오차 차이 (M2 − M1)'); ax.legend(loc='upper left', fontsize=7.2, ncol=3, columnspacing=0.8, handletextpad=0.1)
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig7_lead.png'); plt.close(fig)

# 인구가 많은 현이 인근보다 먼저 정점에 이르는지 (부록 A). 그렇다면 인구로 앞서는 현을 미리 가려낼 수 있음
for Y, d in LD.groupby('연도'):
    print(Y, '인근 대비 인구와 (인근 정점 − 자기 정점)의 순위 상관: %.2f (p = %.2f)' % spearmanr(d.pref.map(relpop), d[x]))
save(pd.DataFrame([{'연도': Y, 'rho': spearmanr(d.pref.map(relpop), d[x])[0], 'p': spearmanr(d.pref.map(relpop), d[x])[1]} for Y, d in LD.groupby('연도')]), 'peak_lead_pop.csv')
