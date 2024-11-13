## 결정 트리

### 결정 트리 훈련 모습
![image](https://github.com/user-attachments/assets/f5ebae8c-edfa-4ef8-9cd4-44ae7e63c726)

### 예측 시각화
![image](https://github.com/user-attachments/assets/7b4b6e6b-13c5-471a-8770-f5f624aac1ac)

![image](https://github.com/user-attachments/assets/8ce454ab-65a1-4eea-ace6-37c3a275ce36)

이렇게 max_depth를 3으로 설정하고 깊이 2의 두 노드가 각각 결정 경계를 추가로 만들어 수직 점선이 생기게 된다.

### 계산복잡도
- 예측을 위해서는 결정 트리를 루트 노드에서부터 리프 노드까지 탐색해야 함.
- O(log_2(m))개의 노드를 거쳐야 함.
- 전체 복잡도가 O(log_2(m)) -> 훈련 세트를 다룰 때 예측 속도가 매우 빠르다.
- 각 노드에서 모든 샘플의 모든 특성을 비교하면 훈련 복잡도는 O(n*mlog_2(m))

### 규제 매개변수
- 제한을 두지 않을 경우 트리가 훈련 데이터에 아주 가깝게 맞추려고 해서 오버피팅이 생김
- 모델 파라미터가 보통 많아서 훈련 되기 전에 파라미터 수가 결정 되지 않아 비파라미터 모델임.
- 모델 구조가 고정되지 않고 자유로움
- 선형 모델과 같은 파라미터 모델은 모델 파라미터 수가 미리 정해져 있어서 자유도가 제한되고 과대적합 위험성이 줄어든다.
- 오버피팅을 피하기 위해 결정 트리의 자유도를 제한 : 규제
- max_depth를 줄여서 모델을 규제하고 오버피팅 위험성을 감소시킨다.

```
·max_features: 각 노드에서 분할에 사용할 특성의 최대 수 
·max_leaf_nodes: 리프 노드의 최대 수 
·mmn_samples_split : 분할되기 위해 노드가 가져야 하는 최소 샘플 수 
·min_sa때Dies_leaf: 리프 노드가 생성되기 위해 가지고 있어야 할 최소 샘플 수 
·min_weight_fractionjeaf: mmn_samples_leaf와 같지만 가중치가 부여된 전체 샘플 수에서의 비율
```

![image](https://github.com/user-attachments/assets/24ce2560-df36-46df-9ad9-f67f76e5b182)

![image](https://github.com/user-attachments/assets/43cf660d-8a38-45a2-be6d-2e9449987a51)

왼쪽 모델은 규제가 없고 오버피팅, 오른쪽 모델은 규제를 추가하고 일반화가 잘됨.

### 회귀
![image](https://github.com/user-attachments/assets/80e0529d-9ecc-4e78-9291-84c3bb8399c6)

![image](https://github.com/user-attachments/assets/f00f9720-0d77-4451-99a7-ad481eac3e85)

각 영역의 예측값은 항상 그 영역에 있는 타깃값의 평균이 된다. 예측값과 가능한 한 많은 샘플이 가까이 있도록 영역을 분할하게 됨.

![image](https://github.com/user-attachments/assets/8e7abf67-b8b1-4f83-99e1-6c58ac7c79e8)

오른쪽 그림이 규제를 min_samples_leaf = 10으로 준 모습

### PCA
![image](https://github.com/user-attachments/assets/c74b63db-0539-4226-b3d3-05f51391619d)

- 특성 간의 상관관계를 줄이는 방식으로 데이터를 회전하고, 결정 트리를 더 쉽게 만드는 것.

![image](https://github.com/user-attachments/assets/18d1a7b1-6970-4582-8e97-8a475565296e)

### 결정 트리의 분산 문제
- 결정 트리가 분산이 상당히 커서, 하이퍼파라미터나 데이터를 조금만 변경해도 매우 다른 모델이 생성될 수 있음.
- 문제 해결 : 결정 트리의 예측을 평균하여 분산을 줄인다. -> 결정 트리의 앙상블 : 랜덤 포레스트
