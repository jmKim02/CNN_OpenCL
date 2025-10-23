# Multicore CNN Optimization using OpenCL

### 프로젝트 환경 기준 스펙
Number of platforms: 1

platform: 0

- CL_PLATFORM_NAME :NVIDIA CUDA
- CL_PLATFORM_VENDOR :NVIDIA Corporation
Number of devices: 1

device: 0

- CL_DEVICE_TYPE : CL_DEVICE_TYPE_GPU
- CL_DEVICE_NAME : NVIDIA GeForce RTX 3060
- CL_DEVICE_VENDOR : NVIDIA Corporation
- CL_DEVICE_VERSION : OpenCL 3.0 CUDA
- CL_DEVICE_MAX_CLOCK_FREQUENCY : 1837MHz
- CL_DEVICE_MAX_COMPUTE_UNITS : 28
- CL_DEVICE_MAX_WORK_GROUP_SIZE : 1024
- CL_DEVICE_GLOBAL_MEM_SIZE : 4294311936
- CL_DEVICE_LOCAL_MEM_SIZE : 49152
- CL_DEVICE_QUEUE_PROPERTIES : CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE CL_QUEUE_PROFILING_ENABLE

---
- 2024-2학기 멀티코어 프로그래밍: CNN 병렬 최적화 팀프로젝트
- 수행 기간: 2024.11 ~ 2024.12 (2개월)
- 팀 구성: 3인 1조
  - 본인: Tiling, Ping-Pong버퍼, 통신-계산 중첩 코드, Vector, 메모리 coalescing
  - 팀원: fp16 자료형, im2col 시도

---
<br/>

## 프로젝트 개요
이 프로젝트는 멀티코어 환경에서 CNN(Convolutional Neural Network)을 최적화하여 고속 이미지 분류를 수행하기 위한 것입니다.<br/>
기존 순차적 C 코드로 작성된 CNN을 OpenCL 기반 병렬 처리로 가속화하여, 최종적으로 약 450배 이상의 성능 향상을 달성하였습니다.<br/>

### 📊 최종 성능:
- 순차 코드 실행 시간: 약 720초
- OpenCL 최적화 실행 시간: 약 1.5초

---
<br/>

## 🧬 CNN이란?
CNN(Convolutional Neural Network)은 이미지 같은 고차원 데이터를 다룰 때 공간 구조를 보존하면서 특징을 추출하고,<br/>
이를 바탕으로 분류를 수행하는 딥러닝 모델입니다.<br/>

- **Convolution Layer**: 이미지에서 필터(커널)를 통해 특징을 추출
- **Pooling Layer**: 특징맵의 크기를 줄여 계산량을 감소시키고, 중요 특징을 보존
- **Fully Connected Layer**: 추출된 특징들을 기반으로 최종 클래스를 분류

- CNN 구조

<img width="940" height="198" alt="image" src="https://github.com/user-attachments/assets/62268e38-4776-45ec-8499-d09aa75b78a5" />

<img width="940" height="198" alt="image" src="https://github.com/user-attachments/assets/1e2d9f8d-5196-48dd-8f02-34b9be7dbcc7" />


---
<br/>

## ⚙️ 적용한 최적화 기법
### ✅ 1. Batch 처리
- 여러 이미지를 한 번에 처리하여 메모리 접근 오버헤드를 줄이고 연산 효율 극대화
- 최적 배치 크기: 256

### ✅ 2. Ping-Pong 버퍼 전략
- 전체 레이어마다 버퍼를 새로 만들지 않고, 두 개의 버퍼를 교대로 사용하여 메모리 사용 최소화
- `layer_buf[0] ↔ layer_buf[1]`

### ✅ 3. 커널 분할 (convolution_large / convolution_small)
- 특징맵 크기에 따라 두 종류의 커널을 분리 작성
  - large: 8x8 이상
  - small: 8x8 미만
- 각각에 맞는 최적화 기법 적용

### ✅ 4. 벡터화 및 내장 함수 활용
- `float4`, `vload4`, `fma`, `dot` 등의 벡터 자료형과 내장 함수로 연산 단축
- ReLU, 내적, max 연산 등에서 SIMD 최적화

### ✅ 5. 통신-계산 중첩
- 두 개의 Command Queue (transfer_queue, compute_queue)를 분리하여
  Host-Device 간 데이터 전송과 커널 연산을 병렬 수행

### ✅ 6. 메모리 접근 패턴 최적화
- NCHW 기반 인덱싱 + 2차원 워크스페이스 설정
  → 메모리 코얼레싱(coalescing) 극대화로 캐시 효율 개선

---
<br/>

## 시도했지만 최종 선택하지 않았던 기법들: Tiling, im2col, fp16 자료형
프로젝트 중 다양한 최적화 기법들을 실험해 보았으며, 일부는 실제 환경에서 성능 향상에 기여하지 못하거나 오히려 성능 저하를 유발하여 최종 코드에서는 제외하였습니다.

### ❌ Tiling
- 개념: 입력 데이터를 일정 크기 타일(예: 8×8, 16×16)로 잘라 로컬 메모리에 올려 반복 재사용하며 메모리 접근을 줄이는 기법
- 시도 결과: 일정 수준까지는 성능 향상(720초 → 3.6초)을 보였지만, 더 고도화된 벡터화 기반 구조에 비해 효율이 떨어졌고, 작은 특징맵에서는 오히려 메모리 오버헤드로 성능 저하 발생
- 결론: 큰 특징맵에서만 제한적으로 유효, 하지만 최종 구조에서는 제외

### ❌ im2col
- 개념: 3차원 입력 데이터를 2차원 행렬로 변환하여 Convolution을 행렬 곱으로 바꾸는 방식
- 시도 결과: 연산 자체는 효율적이었으나, global memory 접근이 두 번 필요해 오히려 전체적인 메모리 병목 발생 → 성능 저하 (6.3초)
- 결론: 변환 오버헤드로 인해 제외

### ❌ FP16 (Half Precision)
- 개념: float 대신 half-precision(16비트 부동소수점)을 사용해 연산량과 메모리 사용량을 절감
- 시도 결과: 일부 환경에서는 성능 이점을 줄 수 있지만, 정확도 손실과 OpenCL 지원 이슈, 그리고 최종 테스트 환경에서는 눈에 띄는 개선이 없어 제외
- 결론: 실험적 가능성은 있지만, 이번 프로젝트에서는 미적용
  
---
<br/>

## 성능 향상 과정 요약

| 단계        | 주요 기법                        | 실행 시간 (초)        |
| --------- | ---------------------------- | ---------------- |
| 초기 순차 실행  | 순수 C                         | 720            |
| 기본 병렬화    | OpenCL                       | 18.5             |
| 타일링 적용    | 타일 크기 8x8            | 10         |
| 타일링 크기에 따른 분할 적용 | 타일 크기 8x8 / 16x 16 | 4.0 ~ 3.6 |
| im2col 시도 | (성능 저하)                      | 6.3              |
| 최적 구조 도출  | 벡터화, ping-pong, 2D pattern 등 | **1.53 \~ 1.57** |

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/814a651c-a8d9-4913-93fb-9972ca1ef7f1" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/fb6f182d-7114-4ffd-b3d3-0750032390ca" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/ff01aecd-00f4-479e-963b-c76387bb5c7c" />

<br/>
  
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/326067a6-93bb-4948-ada3-f7c6ad3524cf" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/cd8f4038-2708-4b73-8a19-20b9952d1440" />

---
<br/>

## 👨‍💻 내 역할 및 느낀 점
이번 프로젝트에서 저는 전체 최종 구조의 핵심 최적화 기법 대부분을 직접 설계하고 구현하였습니다.<br/>
초기 아이디어 단계부터 OpenCL 커널 작성과 구조 개선, 성능 테스트에 이르기까지 병렬화의 전체 흐름을 경험하며 실제 성능을 끌어올리는 과정에 집중했습니다.

### 주요 기여 사항
- ✅ Ping-Pong 버퍼 구조 도입으로 메모리 사용량 최소화
- ✅ Batch 처리 구조화 및 최적 배치 크기 설정 (256)
- ✅ Convolution 커널 분할 (large/small) 및 각 레이어별 연산 구조 최적화
- ✅ 벡터 자료형(float4) 및 내장 함수(dot, fma 등) 활용을 통한 연산 벡터화
- ✅ Command Queue 이중화를 통한 통신-계산 중첩
- ✅ 메모리 접근 패턴 분석 및 2D 워크스페이스 기반 코얼레싱 최적화
- 🔍 다양한 워크 그룹 및 글로벌 워크 사이즈 실험을 통한 최적 파라미터 도출

### 느낀 점
이번 프로젝트를 통해 단순히 복잡한 이론이나 고급 기법(Tiling, im2col 등)을 적용한다고 해서 무조건 성능이 향상되는 것은 아니라는 점을 체감했습니다. 오히려 하드웨어 구조와 메모리 계층, 캐시 동작, 연산과 전송의 병목 등을 정확히 이해하고, 그에 맞는 전략을 선택하는 것이 더 중요하다는 사실을 배웠습니다.

또한 최적화를 진행하며 자연스럽게 메모리 오버헤드와 연산 성능 간의 트레이드오프, 메모리 접근 방식(NCHW), 워크스페이스 차원 구성 등을 고려하게 되었고, 이러한 요소들이 실제 소프트웨어 구조 설계에서도 중요한 영향을 준다는 점을 깊이 느꼈습니다.

비록 제가 앞으로 하드웨어 레벨의 커널을 직접 다루는 역할을 맡지는 않더라도, 이번 경험은 앞으로의 소프트웨어 개발과 최적화 방향성에 있어 보다 하드웨어 친화적인 사고와 시스템 전체 흐름에 대한 통찰력을 키울 수 있는 매우 유의미한 경험이었습니다.
