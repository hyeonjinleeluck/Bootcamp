import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#간단한 집값 예측 머신러닝 모델
def house_price_prediction():
    "집 크기를 기반으로 집값을 예측하는 머신러닝 모델"

    #가상의 집 데이터 생성 (크기 -> 가격)
    np.random.seed(42)
    #house_sizes : 집 크기 
    house_sizes = np.random.normal(100,30,1000) #평균 100v평 , 표준편차30
    print(house_sizes)
    #house_prices : 집 가격
    #크기가 클수록 가격이 높아지는 관계 - 노이즈
    house_prices = house_sizes * 50 * np.random.normal(0,500,1000) + 2000
    print(house_prices)

 #데이터 전처리 
x=house_sizes.reshape(-1,-1)
y=house_prices
 # 훈련용/테스트용 데이터 분할
 x_train , x_test ,y_train , y_test = train_test_split
 x,y test_size=0.2 , random_state = 42


# train_test_split 함수 매개변수
# x : 훈련 데이터
# y : 훈련 데이터의 정답
# test_size : 테스트 데이터 비율
# random_state : 데이터 분할 시 랜덤 시도 설정정  

#머신러닝 모델 생성 및 훈련
model = LinearRegression()
#LinearRegression 클래스의 fit 메서드
#모델 훈련
#x : 훈련 데이터
#y : 훈련 데이터의 정답

model.fit(x_train , y_train)

#예측
y_pred = model.predict(x_test)

#성능 평가 
mse = mean_squared_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)

print(f'평균 제곱 오차(MSE): {mse: 2f}')
print(f'결정 계수 (R*R): {r2:2f}')
print(f"모델 계수 (기울기): {model.coef_[0]:2f}")
print(f"모델 절편: {model.intercept_:2f}")

#새로운 집 크기에 대한 예측
new_house_sizes = [8,120,150]
for size in new_house_sizes:
    predicted_price = model.predict([size])[0]
    print(f"{size}평 집의 예상 가격:{predicted_price:2f}만원")


return model, x_test, y_test, y_pred

#실행
model, x_test, y_test, y_pred = house_price_prediction()






