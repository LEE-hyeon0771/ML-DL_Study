## 앙상블 학습과 랜덤 포레스트

- 각 분류기가 weak learner일지라도 앙상블에 있는 weak learner가 충분하게 많고 다양하다면 앙상블은 strong learner가 될 수 있다!!!!!
- 앙상블 방법은 예측기가 가능한 한 서로 독립적일 때 최고의 성능을 발휘!!!!!
- soft voting : 개별 분류기의 예측을 평균 내어 확률이 가장 높은 클래스를 예측 할 수 있다.

![image](https://github.com/user-attachments/assets/600ca74c-66cc-4ad4-a10f-76c577e37198)

![image](https://github.com/user-attachments/assets/91796d2f-ef97-41f0-8f25-38d1c517bc73)

앙상블 한 결과를 살펴보면

![image](https://github.com/user-attachments/assets/a43f2161-4ace-491e-844f-6d05bdffc928)

성능이 더 높아짐.


soft voting을 실시하면, 

![image](https://github.com/user-attachments/assets/9794dffb-bb98-48a7-b7f0-9ce203e04ff7)

확률이 더 높은 투표에 비중을 두기 때문에 soft voting을 사용하면 더 성능이 높다.

### 배깅과 페이스팅
1. 배깅 : 훈련 세트에서 중복을 허용하여 샘플링하는 방식
2. 페이스팅 : 중복을 허용하지 않고 샘플링 하는 방식

- 개별 예측기는 원본 훈련 세트로 훈련시킨 것보다 훨씬 크게 편향되어 있지만, 집계 함수를 통과하면 편향과 분산이 모두 감소한다.
- 일반적으로 앙상블은 원본 데이터셋으로 하나의 예측기를 훈련시킬 때와 비교해 편향은 비슷하지만, 분산은 줄어들게 된다.
- 배깅과 페이스팅은 훈련 세트에서 랜덤 샘플링을 통해 여러개의 예측기를 훈련한다.

![image](https://github.com/user-attachments/assets/59c6b283-df85-48f0-bf46-b5a5acc98290)

![image](https://github.com/user-attachments/assets/dee29479-cbc4-46f8-836b-48db5e2cf409)

이렇게 단일 결정 트리와 배깅을 사용한 결정 트리를 비교하게 되면, 배깅을 사용한 결정 트리에서 훨씬 더 객관적인 결과가 나오는 것을 확인할 수 있다.

### OOB 평가
- 만약 각 예측기에 훈련 샘플의 63% 정도만 샘플링 된다면, 선택되지 않은 나머지 37%를 OOB 샘플이라고 부른다.

![image](https://github.com/user-attachments/assets/3a8b6783-ef73-4545-a121-b9d8ede812ad)

OOB 평가 점수는 89.6% 정도였고, 자동으로 oob_score_에 저장되어 있다.

![image](https://github.com/user-attachments/assets/3f80619b-978c-424f-a733-438613ca6410)

실제 accuracy_score는 92% 정도이고, 이는 oob_score가 2%정도 박하게 나왔다는 것을 보여준다.

### 랜덤 포레스트
- 일반적으로 배깅을 적용한 결정 트리 앙상블
- max_samples를 훈련 세트의 크기로 지정함

![image](https://github.com/user-attachments/assets/b7668f1e-1fa4-468d-8171-f5fa0d282fd1)

RandomForestClassifier는 BaggingClassifier의 매개변수를 모두 가지고 있다. 트리의 노드를 분할할 때 전체 특성 중에서 최선의 특성을 찾는 대신 랜덤으로 선택한 특성 후보 중에서 최적의 특성을 찾는 식으로
무작위성을 더 주입한다. 이를 통해 트리를 더욱 다양하게 만들고 편향을 손해보는 대신 분산을 낮추어 전체적으로 더 훌륭한 모델을 만들어 낸다.

### 특성 중요도(Feature Importance)
- 랜덤포레스트는 특성의 상대적 중요도를 측정하기가 쉽다.

![image](https://github.com/user-attachments/assets/e88486cc-6b73-4239-92bf-72b977210902)

이런 상황이라면, petal width와 petal length가 중요한 feature라는 의미이다.

### 부스팅
- 약한 학습기를 여러 개 연결하여 강한 학습기를 만드는 앙상블 방법
- AdaBoost, GradientBoost, Xgboost 등이 있음.

**1. AdaBoost**
- 이전 모델이 과소적합했던 훈련 샘플의 가중치를 더 높여서 새로운 예측기가 학습하기 어려운 샘플에 점점 더 맞춰지게 되는 것. 샘플의 가중치 업데이트 방식

![image](https://github.com/user-attachments/assets/8efa57f4-f01e-43d5-bf87-4f367f1cee48)

![image](https://github.com/user-attachments/assets/dba27794-7227-43e1-b96a-2ce6938a4996)

처음에 learing rate가 1이라서 잘못 분류된 것들을 learing rate를 0.5로 낮춰서 올바르게 분류했음. 이처럼 샘플의 가중치를 계속 업데이트 해나가는 방식

![image](https://github.com/user-attachments/assets/5949a734-20ee-42f7-b17d-cd2fd9f6c2f3)

![image](https://github.com/user-attachments/assets/cf3049dd-3381-47da-a918-678e053e9549)

![image](https://github.com/user-attachments/assets/22b42a63-5ea9-46c2-a87d-1914fbfe2abf)

![image](https://github.com/user-attachments/assets/22e2c1b8-8ea2-49c8-b434-54144acde6e1)

![image](https://github.com/user-attachments/assets/5341b6bf-9cc9-4801-8ca0-5a730cebe3b0)

**2. GradientBoosting**
- AdaBoost처럼 반복마다 샘플의 가중치를 수정하는 대신 이전 예측기가 만든 잔차에 새로운 예측기를 학습시킴.
- n_iter_no_change = 10 : 훈련 중 마지막 10개의 트리가 도움이 되지 않는 경우 자동으로 트리 추가를 중지시킴.
- subsample=0.25이면 랜덤으로 선택된 25%의 훈련 샘플로 학습됨.(편향이 높아지는 대신 분산이 낮아지고 훈련 속도도 상당히 빨라짐) -> Stochastic Gradient Boosting

![image](https://github.com/user-attachments/assets/f5d94376-f63e-4ed2-9297-38829be80ddb)

![image](https://github.com/user-attachments/assets/d8e60e33-8f87-4532-acd7-874d762643e0)

![image](https://github.com/user-attachments/assets/097eee11-6f80-4089-94ee-1ed041505448)

**3. HistGradientBoosing**
- 대규모 데이터셋에 최적화됨.
- 입력 특성을 구간으로 나누어 정수로 대체하는 방식으로 작동 -> 정수로 작업하면 더 빠르고 메모리 효율적인 데이터 구조 사용 가능,
구간 분할을 사용하여 학습 알고리즘이 평가해야 하는 가능한 임곗값의 수를 크게 줄일 수 있으며, 각 트리를 학습할 때 특성을 정렬할 필요가 없음.
- 계산 복잡도가 O(b*m)으로 낮아짐.(b : 구간의 갯수, m : 훈련 샘플의 갯수)
- 인스턴스 수가 10,000개보다 많으면 조기 종료가 자동으로 활성화됩니다.
- early_stopping 매개변수를 True 또는 False로 설정하여 조기 종료를 항상 켜거나 끌 수 있습니다． 
- subsample 매개변수가 지원되지 않습니다． 
- n_estimator$ 매개변수가 max_iter로 바뀌었음.
- 조정할 수 있는 결정 트리 하이퍼파래기터는 max_leaf_nodes, mmn_samples_leaf, max_depti〕뿐임.

![image](https://github.com/user-attachments/assets/a46ccdf4-7453-4238-8a98-040868b8c5a1)

### 스태킹
-  앙상블에 속한 모든 예측기의 예측을 취합하는 （직접 투표 같은） 간단한 
함수를 사용하는 대신 취합하는 모델을 훈련시키는 기법

![image](https://github.com/user-attachments/assets/b924ae75-6cdb-48a8-ae8b-1a31cba832d2)

![image](https://github.com/user-attachments/assets/d4e04964-eb6e-4424-8804-3392a2ca6067)

![image](https://github.com/user-attachments/assets/cfc11586-a660-4b89-8956-e098542bc9cf)

이전 92%보다 더 높은 수치를 기록.


