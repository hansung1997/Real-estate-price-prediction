# Real Estate Price Prediction (인천광역시 부동산 실거래가 예측 프로젝트) 2025.11.6

## 프로젝트 개요
본 프로젝트는 **국토교통부 실거래가 공개시스템**에서 제공하는 아파트 매매 실거래 데이터를 기반으로,  
**주택 가격을 예측하고 지역별 시세 추세를 분석**하기 위해 진행되었다.  
2025년 2월부터 2025년 11월 현재까지의 인천광역시 아파트 실거래 데이터를 수집하여  
머신러닝 회귀모델(Random Forest)을 구축하였으며,  
추가적으로 **단기 시계열 예측(6개월 후 시세 전망)**을 수행하였다.

- 데이터 출처: [국토교통부 실거래가 공개시스템](https://rt.molit.go.kr/)
- 분석 대상: 인천광역시 아파트 매매 실거래 (2025년 2월~10월)
- 데이터 형식: 월별 CSV 파일 (9개 병합)
- 데이터 수: 약 23,000건

---

## 데이터 설명
주요 컬럼 (CSV 원본 기준):

| 컬럼명 | 설명 |
|--------|------|
| `시군구` | 거래 지역 (예: 인천광역시 서구 연희동) |
| `단지명` | 아파트 단지명 |
| `전용면적(㎡)` | 전용면적 |
| `층` | 층수 |
| `건축년도` | 준공년도 |
| `거래금액(만원)` | 실제 거래금액 (만원 단위) |
| `계약년월` | 거래가 발생한 연·월 (시계열 분석용) |

---

## 데이터 전처리 과정

1. **파일 병합**
   
   ```python
   import pandas as pd
   import glob

   files = glob.glob("/content/아파트(매매)_실거래가_25_*.csv")
   df_list = []

   for f in files:
       temp = pd.read_csv(f, encoding='cp949', skiprows=15)
       temp.columns = temp.columns.str.strip()
       df_list.append(temp)

   df = pd.concat(df_list, ignore_index=True)
   print("병합 완료, 총 행 수:", len(df))
   ```
   
2.정재 작업

    # 주요 컬럼 선택
    df_model = df[['시군구', '단지명', '전용면적(㎡)', '층', '건축년도', '거래금액(만원)']].copy()
    
    # 거래금액 문자열 → 숫자 변환
    df_model['거래금액(만원)'] = (
        df_model['거래금액(만원)']
        .astype(str)
        .str.replace(',', '')
        .str.replace(' ', '')
    )
    df_model['거래금액(만원)'] = pd.to_numeric(df_model['거래금액(만원)'], errors='coerce')
    
    # 결측치 제거
    df_model = df_model.dropna()
    print("전처리 후 행 수:", len(df_model))

3.범주형 인수 인코딩

    from sklearn.preprocessing import LabelEncoder

    le_gu = LabelEncoder()
    le_danji = LabelEncoder()
    
    df_model['시군구_enc'] = le_gu.fit_transform(df_model['시군구'])
    df_model['단지명_enc'] = le_danji.fit_transform(df_model['단지명'])

---

##모델링

1. 회귀모델

   ```
      from sklearn.model_selection import train_test_split
      from sklearn.ensemble import RandomForestRegressor
      from sklearn.metrics import mean_absolute_error, r2_score
      
      X = df_model[['시군구_enc', '단지명_enc', '전용면적(㎡)', '층', '건축년도']]
      y = df_model['거래금액(만원)']
      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
      model = RandomForestRegressor(n_estimators=300, random_state=42)
      model.fit(X_train, y_train)
      pred = model.predict(X_test)
      
      print("MAE (평균 절대 오차):", mean_absolute_error(y_test, pred))
      print("R² (설명력):", r2_score(y_test, pred))
   ```
   
2. 단지별 예측 함수

   ```
      import numpy as np

      def predict_price(gu, danji, area, floor, year):
          gu = str(gu).strip()
          danji = str(danji).strip()
      
          try:
              gu_code = le_gu.transform([gu])[0]
          except ValueError:
              print(f"[시군구 오류] '{gu}' 는 학습 데이터에 없습니다.")
              return
      
          try:
              danji_code = le_danji.transform([danji])[0]
          except ValueError:
              print(f"[단지명 오류] '{danji}' 는 학습 데이터에 없습니다.")
              return
      
          X_new = np.array([[gu_code, danji_code, area, floor, year]])
          price = model.predict(X_new)[0]
      
          print(f"{gu} {danji} 예상 거래금액: {price:,.0f}만원 (약 {price/10000:.2f}억 원)")
          return price
   ```

3. 예시 실행
   ```
      predict_price('인천광역시 서구 청라동', '청라한라비발디', 84, 15, 2015)
      # 출력: 인천광역시 서구 청라동 청라한라비발디 예상 거래금액: 85,200만원 (약 8.52억 원
   ```
