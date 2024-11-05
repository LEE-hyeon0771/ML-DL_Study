## SVM(Support vector machine)

- 매우 강력하며, 비선형 분류, 회귀, 특이치 탐지에 사용하는 다목적 머신러닝 모델. 중소규모의 비선형 데이터셋, 분류 작업에서 좋지만 매우 큰 데이터셋으로는 잘 확장되지 않는다.

### 서포트벡터 분류

**1. 서포트벡터**

![image](https://github.com/user-attachments/assets/7eb39f84-ff4d-4aa1-934d-dd370dcd1f7b)

실선이 SVM 분류기의 결정 경계이며, 이 직선으로 두 개의 클래스를 나누고 있고 제일 가까운 훈련 샘플로부터 가능한 한 멀리 떨어져 있다. SVM 분류기를 클래스 사이에 가장 폭이 넓은 도로를 찾는 것으로 생각할 수 있어서**Large margin classification**이라고 한다.

도로 바깥쪽에 훈련 샘플을 더 추가하더라도 결정 경계에는 영향을 미치지 않고, 도로 경계에 위치한 샘플에 의해 결정된다. 이러한 샘플을 서포트 벡터라고 한다.

**2. Hard margin classification**

- 모든 샘플이 도로 바깥쪽에 올바르게 분류되어 있는 경우
- 데이터가 선형적으로 구분될 수 있어야 제대로 작동, 이상치에 민감

![image](https://github.com/user-attachments/assets/7a1b934d-d113-4d8e-b36f-8725ae3b8d8f)

- 이러한 문제를 피하기 위해 도로의 폭을 가능한 한 넓게 유지하는 것과 margin violation 사이에 적절한 균형이 필요한데, 이를 **soft margin classification**이라고 한다.

![image](https://github.com/user-attachments/assets/9720a76f-9074-439b-9a67-9307dbdbe03e)

- C를 낮게 설정하면 도로는 더 커지고 많은 margin violation이 발생한다. 그 만큼 서포트 샘플이 더 많아져서 오버피팅 위험이 줄어든다. 하지만 너무 많이 줄이면 과소적합될 수 있다.

- 이 그림과 같은 경우에서는 C=100이 더 잘 일반화될 수 있다.

### 비선형 서포트벡터 분류

![image](https://github.com/user-attachments/assets/931ed589-cf7c-4832-b9e6-355e5cc6d2f4)

이런 모양의 비선형 상태에서는 PolynomialFeatures 변환기와 StandardScaler, LinearSVC를 연결하여 파이프라인을 만든다.

![image](https://github.com/user-attachments/assets/5d69b678-9de6-457a-9663-5b3eaba1b277)

**1. 다항식 커널**
**2. 유사도 특성**
**3. 가우스 RBR 커널**

![image](https://github.com/user-attachments/assets/1f9c79e2-8118-4671-b3ee-256a10ab90cc)

- gamma와 C를 바꾸어 훈련시킨다. gamma를 증가시키면 종 모양 그래프가 좁아져 샘플의 영향 범위가 작아진다. 즉, 결정 경계가 조금 더 불규칙해지고 각 샘플을 따라 구불구불하게 휘어지는 것이다.
- gamma 값이 작으면 넓은 종 모양 그래프를 만들게 되고, 샘플이 넓은 범위에 걸쳐 영향을 주기 때문에 결정 경계가 더 부드러워진다.
- gamma : 규제의 역할 -> 과대적합 : 감소시킴, 과소적합 : 증가시킴.

![image](https://github.com/user-attachments/assets/9cb29225-cacb-4811-aa14-75536fe14dd1)

**4. 계산 복잡도**
- LinearSVC 클래스 훈련 시간 복잡도 : O(m*n)
- 정밀도를 높이면 알고리즘 수행 시간 증가
- SVC 훈련 시간 복잡도 : O(m^2 * n), O(m^3 * n) 사이 -> 훈련 샘플 수가 커지면 엄청나게 느려짐.
- 중소규모의 비선형 훈련 세트에 적합

![image](https://github.com/user-attachments/assets/163e2a4e-b7b4-45f6-9e66-25bc4d121add)

### SVM 회귀
- 일정한 마진 오류 안에서 두 클래스 간의 도로 폭이 최대가 되도록 하는 대신, 제한된 마진 오류안에서 도로 안에 가능한 한 많은 샘플이 들어가도록 학습한다.
- epsilon 하이퍼파라미터로 조절

![image](https://github.com/user-attachments/assets/382038fa-3c7d-42bf-a548-ccfe5c7d57e3)

- epsilon을 줄이면 서포트 벡터의 수가 늘어나서 모델이 규제됨.
- 2차 다항 커널에서는 C로 규제하기도 함.

![image](https://github.com/user-attachments/assets/729bfec0-6725-4be5-9d55-c8151b881fce)

### 다항식 커널
![image](https://github.com/user-attachments/assets/121cfa23-43f6-4ec0-85b0-8849b12637c2)

### SVM 이론
![image](https://github.com/user-attachments/assets/a4f6825e-b670-4b3e-87c9-47443af2ff55)

- 왼쪽은 가중치 w1 = 1 이므로 x1 = -1과 +1 -> 마진의 크기 2
- 오른쪽은 가중치 w1 = 0.5 이므로 x1 = -2와 2 -> 마진의 크기 4
- w를 가능한 한 작게 유지해야하며, 편향 b는 마진의 크기에 영향을 미치지 않는다.

선형 SVM 분류기 모델은 단순히 결정 함수 theta^Tx = theta_0x_0 + ... + theta_nx_n 을 계산해서 새로운 샘플 
x의 클래스를 예측합니다（xo은 편향 특성이며 항상 1입니다.) 결괏값이 0보다 크면 예측된 클 
래스 y^hat은 양성 클래스（1）가 됩니다 . 그렇지 않으면 음성 클래스（0）가 됩니다 

