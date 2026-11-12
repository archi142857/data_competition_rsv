# 자료 점검 (보고서 2장). 0인 주 비율, 현별 규모, 인근 현까지 거리 등
from rsv_core import *

c, W, yw, info = load(DATA_DIR)
print('보고 수:', W.shape, '| 기간:', tuple(yw.iloc[0]), '~', tuple(yw.iloc[-1]))

hum = pd.read_csv(DATA_DIR / 'humidity.csv', index_col=0)
bad = sorted(set(hum.key) - set(W.columns))
print('현 이름 불일치:', bad, '수정 후:', [FIX[k] for k in bad])
print('기상 자료 마지막 주:', hum.groupby('year').week.max().to_dict())  # 2020년 9주까지라 예측엔 못 씀

z = c.assign(period=np.select([c.year <= 2019, (c.year == 2020) | ((c.year == 2021) & (c.week <= 26))], ['2013~2019', '2020~2021년 26주'], '2021년 27주 이후'))
zero = z.groupby('period').cases.apply(lambda s: (s == 0).mean())
print('보고 수 0인 주의 비율:', zero.round(3).to_dict())

mu = W.loc[yw.index[yw.year <= 2019]].mean().sort_values()
print('2013~2019 주평균 보고 수: 가장 적은 현 %s %.1f건, 가장 많은 현 %s %.1f건' % (mu.index[0], mu.iloc[0], mu.index[-1], mu.iloc[-1]))

D = distance_matrix(info); Dn = D.copy(); np.fill_diagonal(Dn, np.inf)
s3 = np.sort(Dn, axis=1)[:, :3]
nb3 = nearest_k(info)
oki = list(info.index).index('Okinawa')
print('인근 3현까지 거리 중앙값 %.0f km, 최대 %.0f km' % (np.median(s3), s3.max()))
print('오키나와의 인근 3현:', nb3['Okinawa'], '| 가장 가까운 현까지 %.0f km' % Dn[oki].min())
cross = sum(info.island[p] != info.island[q] for p, v in nb3.items() for q in v)
print('다른 섬을 잇는 인근 연결:', cross, '/', 47 * 3)
vals = {'zero_pre': zero['2013~2019'], 'zero_covid': zero['2020~2021년 26주'], 'zero_post': zero['2021년 27주 이후'], 'mean_min': mu.iloc[0], 'mean_max': mu.iloc[-1],
         'nb_dist_median': np.median(s3), 'nb_dist_max': s3.max(), 'okinawa_nearest': Dn[oki].min(), 'cross_island': cross,
         'n_week': len(W), 'cases_max': int(W.to_numpy().max())}
facts('data', vals)
