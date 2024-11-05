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

**1. 유사도 특성**


