## 차원 축소

### 1. 차원 축소를 하는 이유
- 훈련 세트의 차원이 클수록 훈련 샘플과 새로운 샘플이 멀리 떨어져 있을 가능성이 높다는 뜻
- 훈련 세트의 차원이 클수록 과대적합 위험이 커진다.
- 해결책 1 : 훈련 샘플의 밀도가 충분히 높아질 때까지 훈련 세트의 크기를 키우는 것. But!!! 훈련 샘플 수는 차원 수가 커짐에 따라 기하급수적으로 늘어나게 됨. 이를 통해 차원을 줄이는 것이 현명한 선택임을 알 수 있음!!!!!

### 2. 차원 축소를 위한 접근법
1) 투영 : 차원을 줄인다

![image](https://github.com/user-attachments/assets/85428a7c-392c-4989-b0cc-5b26f0bfc1cc)

![image](https://github.com/user-attachments/assets/67c2ab74-cf42-4c72-9558-dc35472bfa00)

3차원을 2차원으로 줄인 모습이다.

하지만, 이렇게 단순한 투영을 사용하게 될 경우 스위스 롤 데이터셋처럼 부분 공간이 뒤틀리거나 휘게 된다.
해당 문제를 해결하기 위해 매니폴드 학습을 실시한다.

2) 매니폴드 학습
- 고차원 공간에서 휘어지거나 뒤틀린 저차원 모양
- d차원 매니폴드는 국부적으로 d차원 hyperplane으로 보일 수 있는 n차원 공간의 일부이다.
- 3D에서는 결정 경계가 매우 복잡하지만 펼쳐진 매니폴드 공간인 2D에서는 결정 경계가 단순한 직선이다.

3) 주성분 분석(PCA)
- 투영 시에 분산이 최대로 보존되는 축을 선택하는 것이 정보가 가장 적게 손실되어 합리적이다.
- 원본 데이터셋과 투영된 것 사이의 평균 제곱 거리를 최소화하는 축
- PCA가 훈련 세트에서 분산이 최대인 축을 찾게 된다.

- SVD : 훈련 세트 행렬 X를 세 개의 행렬 곱셈으로 분해해야함.

![image](https://github.com/user-attachments/assets/f0870af9-96fa-4704-8be5-457c5ec8e614)

![image](https://github.com/user-attachments/assets/aaeca14e-9bee-4acb-bfb8-c6fccea084c4)

- hyperplane에 훈련 세트를 투영하고 d차원으로 축소된 X_d-proj를 얻어야함.

![image](https://github.com/user-attachments/assets/202c1678-d587-470e-b203-eb1435fa704a)

![image](https://github.com/user-attachments/assets/1dfc51cc-6d64-44eb-b122-a8c2b0f95281)

이 공식을 통해서 PCA 변환을 완료할 수 있다.

단순하게 sklearn을 사용하여 PCA 하는 방법도 있다.

![image](https://github.com/user-attachments/assets/4467009a-11f8-4e37-a5ab-4ea2866da231)

- 하이퍼파라미터 최적화

![image](https://github.com/user-attachments/assets/8700c920-d6c7-4eeb-a077-bf8b8426a302)

이렇게 randomforestclassifier를 이용하여 784개 차원의 데이터셋을 23차원으로 줄이는 방법도 있다. 이 때 RandomizedSearchCV를 사용하여 최적의 파라미터를 찾았기 때문에 더 좋은 결과를 얻을 수 있었을 것이다.

- PCA 압축 역변환
압축을 풀기 위해서는 PCA 역변환을 적용하여 원본의 차원 수로 되돌리는 방식을 사용할 수 있다.

![image](https://github.com/user-attachments/assets/f41baa6b-3988-494a-9d99-7c01199a8704)

![image](https://github.com/user-attachments/assets/5cdc67b5-dd61-411a-b066-76ff3934ba6e)

### 여러가지 축소 기법
다차원 스케일링(MDS) : 샘플 간의 거리를 보존하면서 차원을 축소, 랜덤 투영은 고차원 데이터에는 적합하지만 저차원 데이터에는 잘 작동하지 않는다.

1. Isomap : 각 샘플을 가장 가까운 이웃과 연결하는 식으로 그래프를 형성. 샘플 간의 gedesic distance를 유지하면서 차원을 축소한다. 그래프에서 두 노드 사이의 geodesic distance는 두 노드 사이의 최단 경로를 이루는 노드 수이다.
2. t-SNE : 비슷한 샘플은 가까이, 비슷하지 않은 샘플은 멀리 떨어지도록 하면서 차원을 축소한다. 시각화에 많이 사용되고, 고차원 공간에 있는 샘플의 군집을 시각화할 때 사용된다.
3. 선형 판별 분석(Linear discriminant analysis) : 선형 분류 알고리즘이다. 훈련 과정에서 클래스 사이를 가장 잘 구분하는 축을 학습한다. 이 알고리즘은 투영을 통해 가능한 한 클래스를 멀리 떨어지게 유지시키므로 다른 분류 알고리즘을 적용하기 전에 차원을 축소시키는 데 매우 좋다.

![image](https://github.com/user-attachments/assets/95010192-819f-4d08-88c4-4558634a950f)


