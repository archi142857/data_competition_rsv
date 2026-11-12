# 유행 단계별 오차 (보고서 5.4절): 표 4. 끝의 1주 지연 확인은 중간보고서 5.5절
from rsv_core import *
from plot_style import *

c, W, yw, info = load(DATA_DIR)
E = evaluate_years(build_panel(W, yw, info, h=4, k=3), W, yw)

a, b = 'pred_M2 인근 3현 추가', 'pred_M1 자기 현 이력'
# 20%는 그냥 정한 기준. 15%, 25%로 바꾼 결과는 12_sensitivity.py
E['phase'] = np.where(E.y_tgt > E.y_now * 1.2, '상승', np.where(E.y_tgt < E.y_now / 1.2, '하강', '정체'))
lm = lambda x, col: (np.log1p(x[col]) - np.log1p(x.y_tgt)).abs().mean()
rows4 = []
for ph in ['상승', '하강', '정체']:
    for Y in (2021, 2022, 2023):
        x = E[(E.phase == ph) & (E.eval_year == Y)]
        rows4.append({'단계': ph, '연도': Y, '행 수': len(x), 'M1': lm(x, b), 'M2': lm(x, a)})
T4 = pd.DataFrame(rows4).round(3).set_index(['단계', '연도']).unstack('연도').swaplevel(axis=1)
T4 = T4[[(Y, k) for Y in (2021, 2022, 2023) for k in ('행 수', 'M1', 'M2')]].loc[['상승', '하강', '정체']]
display(T4); save(pd.DataFrame(rows4), 'phase_by_year.csv')

E1 = evaluate_years(build_panel(W, yw, info, h=4, k=3, nb_lag=1), W, yw)  # 중간보고서 5.5절. 2주 지연과 전국 합계 지연은 09_delay.py
own = metrics(E, 'pred_M1 자기 현 이력').logMAE
same = metrics(E, 'pred_M2 인근 3현 추가').logMAE
lag1 = metrics(E1, 'pred_M2 인근 3현 추가').logMAE
print(f'세 해 전체 로그 MAE: M1 {own:.3f} | M2 같은 주 {same:.3f} | M2 1주 지연 {lag1:.3f} | 개선분 유지율 {(own - lag1) / (own - same):.0%}')
