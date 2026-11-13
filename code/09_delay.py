# 보고 지연 1주, 2주 (최종보고서 6.3절): 표 9
from rsv_core import *

c, W, yw, info = load(DATA_DIR)
E0 = evaluate_years(build_panel(W, yw, info), W, yw)
base = {m: log_mae(E0, m) for m in (M1, M2, M3, M4)}

# 자기 현 값은 예측 시점까지 안다고 두고, 다른 현에서 오는 값만 늦춤
rows = []
for lag in (0, 1, 2):
    Pd = build_panel(W, yw, info, nb_lag=lag, nat_lag=lag); E = evaluate_years(Pd, W, yw)
    ef = effect_table(E, [(M2, M1), (M3, M1), (M4, M1), (M4, M3)]).set_index(['비교', '연도'])
    # M1은 다른 현의 값을 안 쓰지만, 지연을 주면 자료 맨 앞의 학습 행이 47행씩 줄어서 넷째 자리가 달라짐. 차이는 같은 학습 행끼리 구함
    r = {'지연(주)': lag, 'M1': log_mae(E, M1), '학습 행(2021년 평가)': len(train_rows(Pd, yw, 2021))}
    for m in (M2, M3, M4):
        r[m[:2]] = log_mae(E, m)
        r[m[:2] + ' 개선분 유지율'] = (base[M1] - log_mae(E, m)) / (base[M1] - base[m])
    for a, b in ((M2, M1), (M4, M3)):
        k = f'{a} - {b}'
        r[f'{a[:2]}-{b[:2]}'] = ef.loc[(k, '전체'), '차이']; r[f'{a[:2]}-{b[:2]} 하한'] = ef.loc[(k, '전체'), '하한']; r[f'{a[:2]}-{b[:2]} 상한'] = ef.loc[(k, '전체'), '상한']
        for Y in (2021, 2022, 2023): r[f'{a[:2]}-{b[:2]} {Y}'] = ef.loc[(k, Y), '차이']
    rows.append(r)
T10 = pd.DataFrame(rows); save(T10, 'delay.csv'); display(T10.round(3))
print('M1(지연 없음): %.4f' % base[M1])

# 인근만 늦고 전국 합계는 제때 들어오는 경우도 확인 (중간보고서 5.5절과 같은 가정)
for lag in (1, 2):
    E = evaluate_years(build_panel(W, yw, info, nb_lag=lag), W, yw)
    print('인근만 %d주 지연: M2 %.3f (유지율 %.0f%%), M4 %.3f' % (lag, log_mae(E, M2), 100 * (base[M1] - log_mae(E, M2)) / (base[M1] - base[M2]), log_mae(E, M4)))
