# 팜시그널 (FarmSignal)

KAMIS(한국농수산식품유통공사) 일별 도·소매 가격을 매일 자동으로 모아서
**평년 대비 · 변동성 · 4주 예측(80% 범위) · 출하 신호**를 보여주는 생산자용 시세 앱입니다.

```
KAMIS Open API ──▶ 수집기(매일) ──▶ SQLite ──▶ 분석·예측 ──▶ JSON ──▶ 모바일 웹앱
                  GitHub Actions     Release 보관                  GitHub Pages
```

서버 비용 0원으로 시작할 수 있고, 외부 패키지는 `numpy` 하나뿐입니다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `pipeline/farmsignal/wholesale.py` | 공영도매시장 경매 수집 → 일별 품목 경락가(kg당)·반입량 집계, KAMIS 품목 연결 |
| `pipeline/farmsignal/kamis.py` | KAMIS API 클라이언트·응답 파서 (재시도, 인증 오류 즉시 중단, 휴일 날짜 보정) |
| `pipeline/farmsignal/storage.py` | SQLite 저장소 (중복 없이 갱신, 수집 기록으로 중단 후 이어하기) |
| `pipeline/farmsignal/collect.py` | 일일 수집 / 과거 데이터 백필 |
| `pipeline/farmsignal/analytics.py` | 정제(단위 변경·오타값), 이동평균, 변동성, 계절성, 출하 신호 |
| `pipeline/farmsignal/forecast.py` | 3개 모델 앙상블 예측 + 백테스트 기반 예측 범위 |
| `pipeline/farmsignal/evaluate.py` | 홀드아웃 검증(예측이 실제로 얼마나 맞았는지) |
| `pipeline/farmsignal/export.py` | 웹앱용 JSON 내보내기(원자적 교체) |
| `pipeline/farmsignal/demo.py` | KAMIS와 같은 형식의 데모 데이터 생성기 |
| `pipeline/tests/` | 단위·통합 테스트 28개 |
| `web/index.html` | 모바일 웹앱(오늘 / 시세 / 관심 / 분석 방법) |
| `.github/workflows/daily.yml` | 매일 수집 → 분석 → 배포 자동화 |

## 1. 키 없이 바로 돌려보기 (데모)

```bash
cd pipeline
python -m farmsignal demo          # 데모 데이터로 2년치 수집 → 분석 → web/data 생성
python -m unittest discover -s tests
cd ../web && python -m http.server 8000   # 브라우저에서 http://localhost:8000
```

## 2. 실제 데이터 연결

1. **인증키 발급**: kamis.or.kr 회원가입 → 고객센터 → Open API → 이용신청. 인증키와 회원 아이디를 받아둡니다.
2. **GitHub 저장소**에 이 폴더를 올립니다.
3. 저장소 **Settings → Secrets → Actions**에 `KAMIS_CERT_KEY`, `KAMIS_CERT_ID` 등록.
4. **Settings → Pages → Source**를 `GitHub Actions`로.
5. **Actions → 가격 수집·배포 → Run workflow**, `backfill_days`에 `730` 입력(처음 한 번, 약 1시간).
6. 이후로는 매일 KST 18:13, 21:13(월~토)에 자동 실행됩니다.

### 도매시장 경락가·반입량 (선택)

1. 공공데이터포털에서 **'한국농수산식품유통공사_전국 공영도매시장 실시간 경매정보'** 활용신청 (자동승인)
2. 마이페이지 → 개발계정의 **일반 인증키(Decoding)** 를 저장소 시크릿 `DATA_GO_KR_KEY`로 등록
3. 다음 자동 실행부터 가락시장 경매가 매일 쌓여요. 과거분은 실행당 4,500회 호출 한도 안에서 며칠에 걸쳐 채워집니다.

키가 없으면 이 부분만 빠지고 나머지는 그대로 동작해요. 첫 수집 후 Actions 로그에서
`가격 대조 검사` 줄을 확인하세요. 가락 경락가와 KAMIS 도매가(kg당)의 비율이 0.3~3 밖이면 경고가 나오는데,
그러면 단가·단위 해석이 실제 응답과 다른 것이니 로그를 보고 `wholesale.py`를 고쳐야 합니다.

로컬에서 직접 돌릴 때:

```bash
export KAMIS_CERT_KEY=... KAMIS_CERT_ID=...
cd pipeline
python -m farmsignal backfill --days 730   # 처음 한 번 (끊겨도 다시 실행하면 이어서)
python -m farmsignal collect               # 매일
python -m farmsignal wholesale --days 730  # 도매시장 (DATA_GO_KR_KEY 필요)
python -m farmsignal export                # web/data 갱신
python -m farmsignal evaluate              # 예측 성능 확인
```

> 첫 실데이터 수집 후 꼭 확인할 것: KAMIS 응답 필드(`item_code`, `kind_code`, `rank_code`, `day1`, `dpr1~7`)가
> 파서 가정과 맞는지 `python -m farmsignal -v collect`로 로그를 보고, `status`로 저장 건수를 확인하세요.
> 이 코드는 KAMIS에 직접 접속할 수 없는 환경에서 공개된 연동 코드와 문서를 근거로 작성했습니다.

## 3. 분석 방식

- **평년**: KAMIS 제공 값(최근 5년 같은 시기 평균, 최고·최저 제외)
- **변동성**: 최근 20영업일 일별 로그 변동률의 표준편차 × √5 (주간 환산), 품목 자신의 과거 분포 대비 등급
- **예측**: ① 가격 유지 ② 감쇠 추세 Holt ③ 평년 계절 경로 + 평년과의 차이가 AR(1)로 줄어드는 모델.
  최근 약 10개월 롤링 백테스트 오차로 시점별 가중치를 정하고(균등 가중과 50:50 혼합으로 과적합 방지),
  같은 백테스트의 실제 오차 분포로 80% 범위를 만듭니다.
- **제철 품목**(딸기·포도·감귤 등): 21일 넘게 조사가 끊기면 구간을 나눠, 현재 출하 시즌 자료로만 예측합니다.
  14일 넘게 조사가 없으면 예측·신호를 숨기고 ‘조사 중단(비수기)’로 표시합니다.
- **경락가·반입량**: 가락시장 경매 건별 낙찰가를 물량 가중 평균(kg당). 최근 5일 평균 반입량이 4주 평균보다 20% 넘게 많거나 적으면 출하 신호 근거에 표시.
  (아직 예측 모델에 넣지는 않았어요. 실데이터가 쌓이면 반입량을 넣은 모델을 검증한 뒤 반영할 계획)
- **출하 신호**: 평년·1년 가격대 위치 + 4주 예상 변동 + 상승 확률. ‘출하 늦추기’는 저장 가능한 품목에만 냅니다.

### 데모 데이터 기준 검증 결과

| 항목 | 결과 |
|---|---|
| 4주 80% 범위 적중률 | 약 80% (목표 80%) |
| 4주 평균 오차(MAPE) | 약 10% (가격 유지 예측 약 12%) |
| 4주 방향 적중률 | 약 70% |

데모 데이터는 합성이라 이 수치는 **코드가 의도대로 동작한다는 확인**일 뿐입니다.
실데이터가 6개월 이상 쌓이면 `evaluate`를 돌려 `forecast.py`의 `BAND_INFLATE`(현재 1.5)를
80% 적중률에 맞게 다시 조정하세요. 앱의 ‘예측은 얼마나 맞았나’ 카드는 매일 이 검증을 자동으로 다시 계산합니다.

## 4. 유료 서비스로 가려면 (다음 단계)

1. **이용 조건 확인 먼저**: KAMIS Open API 이용약관의 상업적 이용 가능 범위와 출처 표시 의무를 확인하세요.
   원자료 재판매가 아니라 *분석·예측*에 대해 돈을 받는 구조가 안전합니다.
2. **로그인·결제**: 지금은 누구나 볼 수 있는 정적 사이트입니다. 유료 기능(예측·신호·알림)은
   Supabase(로그인 + DB) + 토스페이먼츠 정기결제로 막는 구성이 1인 운영에 현실적입니다.
3. **알림**: 관심 품목이 신호를 바꾸거나 급등락할 때 카카오 알림톡/이메일 — 생산자가 돈을 내는 이유는 ‘매일 들어와서 보는 것’보다 ‘중요할 때 알려주는 것’입니다.
4. **지역별 가격**: `p_countycode`로 지역 단위 수집을 추가하면 산지 판매 판단에 더 직접적입니다.
5. **검증 공개**: 품목별 예측 정확도를 공개하는 것이 다른 시세 앱과의 차별점이 됩니다.

## 주의

예측과 출하 신호는 과거 가격 패턴에 기반한 통계 추정입니다. 기상 이변, 수입 물량, 정책 등 외부 요인은 반영하지 않으며 판매 결과를 보장하지 않습니다.
