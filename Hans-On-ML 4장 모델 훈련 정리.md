## 1. 선형회귀

### 1.1 기본 모양

![image](https://github.com/user-attachments/assets/72651d54-fd2e-4b91-a4e7-045b9453c88c)

![image](https://github.com/user-attachments/assets/97c8482b-a849-41d8-a7d8-10c516bd1a1b)


### 1.2 정규방정식

![image](https://github.com/user-attachments/assets/5c52bb24-b884-4a17-8c4d-d3a4fe873a1a)

![image](https://github.com/user-attachments/assets/cd49795d-d63c-4e1b-911c-b51d0200ab08)

정규방정식 형태를 직접 만들어서 inverse()로 사용할 수 있다.

![image](https://github.com/user-attachments/assets/9d5b254e-b6f4-4b74-8bb1-597fff12c365)

![image](https://github.com/user-attachments/assets/60a74acc-9907-4038-a745-4da6ec52429c)

LinearRegression 클래스를 이용해서 linalg.lstsq()로 least squares를 사용할 수도 있다.

단, 유사역행렬 자체는 SVD를 사용해서 계산 가능하다. 

SVD는 훈련 세트 X를 3개의 행렬 곱셈으로 분해 가능하다. (np,linalg.svd()를 사용)

m < n 이거나 어떤 특성이 중복되어 행렬 X^TX의 역행렬이 없다면 정규방정식은 작동하지 않지만, 유사역행렬은 언제나 구할 수 있다.

LinearRegression 클래스가 사용하는 SVD 방법은 O(n^2) 이며, 특성의 개수가 두 배로 늘어나면 계산 시간은 대략 4배가 된다.

선형 회귀 모델은 예측이 매우 빠른 편이고, 예측 계산 복잡도도 샘플 수와 특성 수에 선형적이다. 예측하려는 샘플이 두 배로 늘어나면 걸리는 시간도 두 배 증가.

### 1.3 경사하강법: 특성의 수가 많거나 훈련 샘플이 너무 많아 메모리에 모두 담을 수 없을 때 적합.

비용 함수를 최소화하기 위해 반복해서 파라미터를 조정해나가는 것이 기본 아이디어!!!!!

![image](https://github.com/user-attachments/assets/6e479ba9-9cf7-4dc2-bf63-4144fa9f25f3)

중요한 파라미터는 learning rate 하이퍼파라미터로 결정되며, 너무 작으면 알고리즘이 수렴하기 위해 반복을 많이 진행해야 하므로 시간이 오래 걸린다. 학습률이 너무 크면 알고리즘을 더 큰 값으로 발산하게 만들어 적절한 해법을 찾지 못한다.

![image](https://github.com/user-attachments/assets/429c1949-769c-41ed-8d3b-06206c87630a)

![image](https://github.com/user-attachments/assets/32426ec3-62fe-4981-85ea-6225fddac618)

- **배치 경사 하강법**

![image](https://github.com/user-attachments/assets/e514b645-9501-4260-8b0c-089b09a1fa04)

![image](https://github.com/user-attachments/assets/1db5bf33-437d-4236-8eec-7730c3d37003)

매 스텝에서 전체 훈련 세트 X에 대해 편도함수를 계산함. 매우 큰 훈련 세트에서는 아주 느림. 수십만 개의 특성에서 선현 회귀를 훈련시킬 경우 SVD나 정규방정식보다 경사 하강법이 훨씬 빠름.

![image](https://github.com/user-attachments/assets/3fc895a7-b004-4d85-b569-bc234b3d9a18)

![image](https://github.com/user-attachments/assets/14ec5f8d-eafc-40ca-9ba2-0670f60aa1de)

eta = 0.02의 경우 학습률이 너무 낮은 경우, eta = 0.1은 학습률이 매우 적당, eta = 0.5는 학습률이 너무 높은 경우이다.

적절한 학습률을 찾기 위해 그리드 서치를 사용하지만, 그리드 서치에서 수렴하는 데 너무 오래 걸리는 모델이 제외될 수 있도록 반복 횟수를 제한해야 한다.

반복 횟수를 아주 크게 지정하고, 그레디언트 벡터가 아주 작아지면(벡터의 norm이 어떤 허용 오차보다 작아지면) 경사하강법이 최솟값에 도달한 것이므로 알고리즘을 중지하는 방식을 사용하는 것이 효율적인 반복 횟수 지정방법임.

- **확률적 경사 하강법**

매 스텝에서 한 개의 샘플을 랜덤으로 선택하고 그 하나의 샘플에 대한 그레디언트를 계산한다.

장점 : 한 번에 하나의 샘플을 처리하면 반복 시 다뤄야 할 데이터가 매우 적어서 알고리즘이 확실히 훨씬 빠르다. 또한, 반복마다 하나의 샘플만 메모리에 있으면 되므로 매우 큰 훈련 세트도 훈련이 됨.

단점 : 확률적인 랜덤이므로 알고리즘이 배치 경사 하강법보다 훨씬 불안정하다. 비용 함수가 최솟값에 다다를 때까지 부드럽게 감소해야 하나 그렇지 않고 위아래로 요동치면서 평균적으로 감소한다. 시간이 많이 필요하다.
때문에, 전역 최솟값을 찾을 가능성은 배치 경사 하강법이 더 높다.

해결책 : 학습률을 점진적으로 감소시키는 것. 시작할 때는 학습률을 크게 하고 지역 최솟값에 빠지지 않게 하며, 점차 작게 줄여서 알고리즘이 전역 최솟값에 도달하게 한다.

![image](https://github.com/user-attachments/assets/46d3601a-c52f-49e1-932a-e3e9d65f2ddc)

- **미니 배치 경사 하강법**
미니 배치라고 부르는 임의의 작은 샘플 세트에 대해 그레디언트를 계산한다. 확률적 경사 하강법에 비해 행렬 연산에 최적화된 하드웨어 GPU를 사용해서 성능 향상이 가능함.

장점 : 미니배치를 어느 정도 크게 하면, 알고리즘이 SGD보다 덜 불규칙하게 움직이며, 최솟값에 더 가까이 도달하게 됨.

단점 : 지역 최솟값에서 빠져나오기는 더 힘들다.

![image](https://github.com/user-attachments/assets/0d9893a5-2cdd-43ba-943e-be3420915262)

![image](https://github.com/user-attachments/assets/d627826a-fb5d-4d58-91e3-9a1656806b24)

### 1.4 다항 회귀

비선형 데이터 학습

-> 각 특성의 거듭제곱을 새로운 특성으로 추가하고, 확장된 특성을 포함한 데이터셋에 선형 모델을 훈련시키는 것 

![image](https://github.com/user-attachments/assets/561ae4ea-e1ff-4d11-87d2-cdd292a41d37)

![image](https://github.com/user-attachments/assets/c84c9f57-52df-4677-8a38-48215394a5fb)

LinearRegression이 괜찮게 적용된 것을 확인할 수 있음.

!!!!! 과대 적합에 주의해야한다 !!!!!

1) 릿지 회귀 :
- 규제가 추가된 선형 회귀 버전.
- 규제항이 MSE에 추가됨.
- 모델의 가중치가 가능한 한 작게 유지되도록 한다.
- 규제항은 훈련하는 동안에만 비용함수에 추가된다.
- 모델 훈련이 끝나면 모델의 성능을 규제가 없는 MSE로 평가한다.

![image](https://github.com/user-attachments/assets/6e257572-9737-4c83-8976-184847aade59)

PolynimialFeatures()를 사용해 먼저 데이터를 확장하고 StandardScaler를 사용해서 스케일을 조정한 후 릿지 모델을 적용한 그래프를 살펴보자.

![image](https://github.com/user-attachments/assets/071d879f-eaf2-4209-9629-96a62814a3cf)

alpha가 증가할수록 직선에 가까워지는 것을 확인 가능. 모델의 분산은 줄지만 편향은 커진다.

![image](https://github.com/user-attachments/assets/4d86f04b-6edd-4cf9-b74b-87537aaadbd4)

![image](https://github.com/user-attachments/assets/2506e026-395b-4a93-b5c7-6bb5c645faf5)
- SGDRegressor를 사용한 릿지 회귀, l2 norm을 penalty로 적용

![image](https://github.com/user-attachments/assets/3b4e54ff-bca1-46ad-8bc8-970fcaa24f09)
- 정규 방정식을 사용한 릿지 회귀 적용 예시

2) 라쏘 회귀
- 선형 회귀의 또 다른 규제 버전
- 규제항 : 가중치 벡터의 l1 norm을 사용

![image](https://github.com/user-attachments/assets/2c0e6700-6f59-45d0-a73f-e68dd9315685)

![image](https://github.com/user-attachments/assets/20dfbae0-9c2b-45d7-81d5-91cd3b4f180c)

라쏘 회귀는 덜 중요한 특성의 가중치를 제거하려고 한다. 자동으로 특성 선택을 수행하고 희소 모델을 만들게 된다.

- 라쏘 회귀의 subgradient vector

![image](https://github.com/user-attachments/assets/a6e7f46a-a78c-4614-b0da-1c796071ee8a)

![image](https://github.com/user-attachments/assets/23fe8311-359b-49c0-8f46-3fdd06d2fc59)

3) 엘라스틱넷 회귀
- 릿지 회귀와 라쏘 회귀를 절충한 모델
- 규제항 : 릿지와 회귀의 규제항을 단순히 더한 것, 혼합 정도는 혼합 비율 r을 사용해서 조절한다.
- r = 0 : 릿지 회귀와 같음, r = 1 라쏘 회귀와 같다.

![image](https://github.com/user-attachments/assets/bdf8a2a2-9a36-43b1-b289-018df3356806)

- 릿지가 기본이 되지만 몇 가지 특성만 유용하다고 생각되면 라쏘나 엘라스틱넷 회귀가 나음.
- 피쳐들의 수가 훈련 샘플 수보다 많거나 특성 몇 개가 강하게 연관되어 있을 때는 라쏘 회귀가 문제를 일으키므로 엘라스틱넷이 좋음.

![image](https://github.com/user-attachments/assets/824a6f56-c3a4-435d-b0ff-1ae5f92e3adf)

### 2. Early Stopping
- 검증 오차가 최솟값에 도달하면 바로 훈련을 중지시키는 것.
- RMSE와 검증 세트에 대한 예측 오차가 줄어들다가 다시 상승하게 되면 모델이 훈련 데이터에 오버피팅 되기 시작했다는 것을 의미함.

![image](https://github.com/user-attachments/assets/94c2b06e-1c59-41fb-9018-44cb1f7eff6a)

### 3. LogisticRegression
- 샘플이 특정 클래스에 속할 확률 추정.
- 확률이 주어진 임곗값보다 크면 모델은 그 샘플이 해당 클래스에 속한다고 예측(label = 1 "양성클래스") 아니면 클래스에 속하지 않는다고 예측(label = 0 "음성클래스")
- 이진 분류기

![image](https://github.com/user-attachments/assets/feb654fd-de90-45e9-9428-60ca6ccbf400)

![image](https://github.com/user-attachments/assets/6c696831-4bb4-4a50-bf03-7410707f33fc)

1) 훈련 원리
- 훈련 목적 : 양성 샘플에 대해서는 높은 확률 추정, 음성 샘플에 ㅐ해서는 낮은 확률 추정하는 모델의 파라미터 벡터 theta를 찾는다.

![image](https://github.com/user-attachments/assets/1a3e244f-1875-4271-a1db-7a4c557b02ec)

t가 0에 가까워지면 -log(t)가 매우 커지므로 타당하다고 볼 수 있음. 모델이 양성 샘플을 0에 가까운 확률로 추정하면 비용이 크게 증가한다. 음성 샘플을 1에 가까운 확률로 추정해도 비용이 증가한다.
t가 1에 가까우면 -log(t)는 0에 가까워지므로 음성 샘플의 확률을 0에 가깝게 추정하거나 양성 샘플의 확률을 1에 가깝게 추정하면 비용은 0에 가까워진다.

![image](https://github.com/user-attachments/assets/7c800871-982c-4f7d-806d-a102174ea3d3)

![image](https://github.com/user-attachments/assets/03ef6a45-5e76-42b0-8cc5-54ce0be48b3b)

### Softmax Regression
- 다항 로지스틱 회귀
- 샘플 x가 주어지면 softmax 회귀 모델이 각 클래스 k에 대한 점수 s_k(x)를 계산하고, 그 점수에 소프트맥스 함수를 적용하여 각 클래스의 확률을 추정한다.

![image](https://github.com/user-attachments/assets/211f377a-9cd0-4f9e-a437-7e82c1c2a5aa)

![image](https://github.com/user-attachments/assets/6bb51061-8fa2-4ded-95ff-b1d55ed9cdcb)

![image](https://github.com/user-attachments/assets/c638aede-735e-4dd1-9ea6-46fdfa40687c)

모델 훈련 방법은 cross-entropy 함수를 최소화하는 것은 타깃 클래스에 대해 낮은 확률을 예측하는 모델 만들기

![image](https://github.com/user-attachments/assets/862a7528-66bb-4dcb-a99f-44c4df8fdf51)

![image](https://github.com/user-attachments/assets/d59aaad2-b8f8-448a-a3db-e553138e012e)

클래스에 대한 그에디언트 벡터를 계산하고, 비용 함수를 최소화하기 위한 파라미터 행렬 theta를 찾기 위해 경사하강법을 사용한다.

  
