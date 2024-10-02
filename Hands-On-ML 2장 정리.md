## Hands-On-ML 2장 정리
```대부분의 내용들이 코드를 통해서 따라가며 실습하는 것이므로 필요한 개념 위주로 정리```

### 문제정의
- 회사에서는 모델을 어떻게 사용해 "이익"을 얻으려고 할까?
- 목적을 아는 것은 문제를 어떻게 구성하지, 어떤 알고리즘을 선택할지, 모델 평가에 어떤 성능 지표를 사용할지, 모델 튜닝을 위해 얼마 만큼의 노력을 투여할지 결정하는 것이다.
- 이 지점이 명확해야 투자할 가치가 있는 시스템이 되며, 이 결정은 수익과 직결되기 때문에 올바르게 예측해야 한다.

### 주택 가격 예측 프로젝트
- 지도학습 : 레이블 된 훈련 샘플 존재
- 회귀 : 모델이 값을 예측해야함
- 다중 회귀 : 예측에 사용할 특성이 여러 개(구역의 인구, 중간 소득 등)
- 단변량 회귀 : 각 구역마다 하나의 값을 예측

### 성능 지표
1. 평균 제곱근 오차(RMSE) : 오차가 커질수록 값이 커짐. 예측에 얼마나 많은 오차가 있는지 가늠 가능.

![image](https://github.com/user-attachments/assets/2be634b7-c6b7-4bce-985e-3cb77a7f4d67)

2. 평균 절대 오차(MAE) : 이상치로 보이는 구역이 많다면 선호됨.
 
![image](https://github.com/user-attachments/assets/679a1c9d-249e-448b-9709-2cf7044a4f25)

### 테스트 세트
- 데이터 스누핑 편향에 주의 : 테스트 세트를 미리 볼 경우에 테스트 세트에서 드러난 어떤 패턴에 속아 낙관적인 추정을 하는 머신러닝 모델을 선택하게 될 수 있으므로 절대 미리 보기 금지.
- 일반적으로 20% 정도로 train_test_split 함수를 사용
- 계층적 샘플링 : 테스트 세트가 전체 인구를 대표하도록 각 계층을 설정하여 올바른 수의 샘플을 추출.

![image](https://github.com/user-attachments/assets/f3e40321-50c2-42f2-8acf-a3b35bdec90d)

이런식으로 카테고리를 나누고 비율을 보는 형식.

### 데이터 시각화
- 데이터 포인트가 많이 겹치는 경우 투명도 alpha를 낮춰서 겹치는 영역을 더 진하게 표시하여 데이터의 밀집도를 직관적으로 파악함.
- alpha : 0~1로 설정. 숫자 낮을 수록 점들이 더 투명하게 표현됨.

![image](https://github.com/user-attachments/assets/027053c7-a914-4131-a664-39215b679adc)

### 상관관계 조사
- corr를 사용.
- 1에 가까우면 강한 양의 상관관계, -1에 가까우면 강한 음의 상관관계
- 상관관계가 매우 강한 경우 양의 상관관계(양의 기울기를 가짐)

![image](https://github.com/user-attachments/assets/838e211a-a703-4cbd-90b9-9a1983fa1f7b)

이 상태에서 해석을 해보면, median_house_value는 median_income이 올라갈 때 증가, median_house_value는 latitude와 음의 상관관계를 가지므로 북쪽으로 갈수록 주택 가격이 조금씩 내려감.

### One-Hot-Encoding
- 누가봐도 비슷한 특성을 가진 카테고리이지만, 머신러닝 알고리즘이 가까이 있는 두 값을 떨어져 있는 두 값보다 더 비슷하다고 생각하는 문제점을 방지
- 카테고리별 이진 특성을 만들어 해결함.
- 희소행렬을 기본으로 함.
 ```
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```
- 넘파이 배열을 출력하기 위해서는 toarray() 메서드를 사용하거나, OneHotEncoder(sparse_output=False)로 설정해야 함.
- 간단하게 get_dummies()를 사용하여 특정 column에 대한 원-핫을 볼 수도 있음.

### 스케일링
- 머신러닝 알고리즘을 작동시키기 위해 입력된 숫자 특성들의 스케일을 하나로 맞추는 작업
- MinMaxScaler(정규화) : 데이터에서 최솟값을 뺀 후 최댓값과 최솟값의 차이로 나눔.
```
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
```
- StandardScaler(표준화) : 평균을 뺀 후 표준 편차로 나눔. 이상치에 영향을 덜 받음.
```
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

### 파이프라인
- Pipeline
```
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
```
- make_pipeline : 파이프라인의 이름을 짓는 것이 귀찮을 때
```
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
```

### 과소적합과 과대적합
- 과소적합 : 특성들이 좋은 예측을 만들 만큼 충분한 정보를 제공하지 못했거나 모델이 충분히 강력하지 못하다는 사실

![image](https://github.com/user-attachments/assets/45ff8f2d-5a8d-4916-94ba-01565d9b572d)

RMSE를 통해서 체크해본 결과 예측오차가 68000대임.
- 과대적합 : train 데이터에 너무 적합되어서 test 데이터에서 객관적인 예측이 힘든 상태.

![image](https://github.com/user-attachments/assets/7b4610b7-c533-4299-ad13-2b671c2763bd)

RMSE 값이 아예 0.0이 되어버림.
- 교차검증/k-fold 검증을 통해서 정확한 모델 성능을 알 수 있음.
```
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(tree_rmses).describe()
```
cross_val_score를 통해서 교차검증을 쉽게 할 수 있음.

- 오버피팅을 해결하기 위해서는 모델을 단순화, 제한, 더 많은 train 데이터를 모아야함.
- 앙상블 모델링을 위해서 하이퍼파라미터 조정 보다는 여러 종류의 머신러닝 알고리즘을 시도하고 가능성 높은 2~5개 정도의 모델을 선정하는 것이 중요함.

### 모델 튜닝
1. GridSearchCV : 탐색하고자 하는 하이퍼파라미터와 시도해볼 값을 지정하여 자동으로 교차 검증을 사용해 가능한 모든 하이퍼파라미터 조합을 평가함.(곱의 법칙 갯수만큼)
- best_params_ : 최상의 성능을 낼 수 있는 파라미터 조합.
- cv_results_ : 하이퍼파라미터 조합의 점수 보기 가능
2. RnadomizedSearchCV : 가능한 모든 조합을 시도하는 대신 각 반복마다 하이퍼파라미터에 임의의 수를 대입하여 지정한 횟수만큼 평가한다.
- 하이퍼파라미터 값이 연속적이면（또는 이산적이지만 가능한 값이 많다면） 랜덤 서치를 1,000번 실행했을 때 각 하이퍼파라미터마다 1.000개의 다른 값을 탐색한다.
- 그리드 서치에 가능한 값을 10개 추가하면 훈련 시간이 10배 더 늘어나지만, 랜덤 서치에 추가하면 탐색 시간이 늘어나지 않는다.
- 지정한 반복 횟수만큼 실행할 수 있다.

### 최상의 모델과 오차분석
- feature_importances를 뽑아서 중요도 점수를 내림차순으로 정렬하고 중요도를 본다.
- 유용하지 않은 feature들은 제거한다.
- 신뢰구간을 설정하여 오차를 분석하고 교차 검증을 사용해 측정한 것보다 squared_errors 성능이 조금 낮게 나올 수 있지만, 하이퍼파라미터 튜닝을 많이 한 탓이므로 테스트 세트 성능 수치만을 높이기 위한 하이퍼파라미터 튜닝은 오히려 새로운 데이터에 일반화 되지 못하므로 독이 될 수 있다. 지양하자!!!!
