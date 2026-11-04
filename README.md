# 일본 RSV 유행의 지역별 관계와 4주 후 보고 수 예측

제1회 과학데이터혁신 경진대회(서울대 과학데이터혁신연구소) 일본 RSV 전파 데이터 분석이다. 인근 현 세 곳의 보고 수를 입력에 넣으면 각 현의 4주 뒤 RSV 보고 수 예측이 나아지는지, 그리고 전국 합계를 넣은 뒤에도 그 효과가 남는지를 봤다.

- `RSV_midterm_analysis.ipynb`: 분석 코드. 중간보고서의 표와 그림, 본문 수치는 모두 여기서 나온다.
- `report/RSV_중간보고서_제출본.pdf`: 중간보고서(뒤쪽에 코드 부록 포함)

## 실행

대회 제공 자료는 저장소에 넣지 않았다. 자료 폴더 `RSV_transmission_in_Japan`(안에 `cases.csv`, `population.csv` 등)을 노트북과 같은 폴더나 `../일본 RSV/` 아래에 두고 위에서부터 실행하면 된다. numpy, pandas, scikit-learn, matplotlib이 필요하다.
