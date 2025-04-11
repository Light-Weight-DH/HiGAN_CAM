# HiGAN_CAM
CAM representation with HiGAN


## 📌 Introduction

최근 Diffusion 모델은 마스크 지정이나 텍스트 프롬프트를 통한 이미지 조작에 널리 활용되고 있다. 하지만 사용자가 조작하고자 하는 속성의 위치나 대상이 이미지 상 어디에 반영되어야 하는지를 **사전에 파악할 수 있다면**, 더욱 직관적이고 효율적인 조작이 가능할 것이다.

GAN은 학습을 통해 이미지의 구조적 특성을 레이어별로 학습하고, latent space나 파라미터 조작을 통해 특정 속성을 변경하거나 추가할 수 있는 특징을 가진다. 본 프로젝트는 이러한 **GAN의 구조적 조작 가능성**과 **Grad-CAM을 활용한 해석 기법**을 결합하여, 조작 대상의 위치를 시각적으로 식별하고, 이를 향후 Diffusion 기반 이미지 생성에 활용할 수 있는 가능성을 모색한다.   

<br>



## 📚 관련/참고 연구

### 📄 *Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis (HiGAN)*

- **핵심 요약**:  
  이 연구는 실내 장면 생성에 사용된 **StyleGAN 기반 Generator**에서 **계층적 의미 구조(Semantic Hierarchy)**가 자연스럽게 형성된다는 점에 주목하였다.  
  Latent space에서 **특정 객체(예: bed)의 존재 유무를 기준으로 SVM을 학습**하고, 이를 통해 생성자의 구조적 특성에 대한 **의미 있는 조작 및 해석**이 가능함을 보였다.

- **핵심 구조(예시)**:  
  - 특정 환경(실내외 공간 ex: 침실,교회 등)을 기반으로 한 생성 모델(PGGAN ,BigGAN, StyleGAN) 학습  
  - ResNet18을 활용한 특성(ex: 조명, 나무 등) 유무 판별하는 모델 학습  
  - 학습된 ResNet18로 GAN 이미지의 특정 속성을 Positive, Negative 데이터로 분류
  - SVM을 통해 해당 속성의 Latent Boundary 추출
  - 해당 boundary를 기준으로 latent vector 조작하여, 특정 속성이 변경된 이미지를 생성

- **활용**:  
  - 해당 연구에서 사용한 Generator, 해당 Generator를 기반으로 학습시킨 SVM 모델을 사용하였음 
  

- **GitHub**: [https://github.com/soccz/HIGAN](https://github.com/soccz/HIGAN)

<br>

## 💡 활용 프로젝트 (Applied Project)

### 조명 추천 및 적용 시스템

본 프로젝트의 기법과 **Diffusion 모델**을 결합하여 실내 공간의 조명 배치 위치를 추천하고,  
추천된 위치에 조명 효과를 사실적으로 **합성하는 종합적 시스템** 개발에 활용하였다. 

프로젝트 상세 내용은 링크를 통해 확인 바람. <br>
- **GitHub**: [Lumterior](https://github.com/jihyeyoo/inisw)

<br>


## 🔍 Core Idea

- **StyleGAN**을 기반으로, 특정 속성이 존재하는 이미지와 그렇지 않은 이미지의 Latent Vector를 분리하여 학습
- **SVM**을 통해 속성 존재 유무를 기준으로 Latent Space에서의 분류 경계(Boundary)를 학습
- **Grad-CAM**을 활용하여 생성된 이미지에서 조작 속성에 해당하는 영역을 시각적으로 해석
- 이를 통해 사용자가 조작하고자 하는 **속성이 반영되어야 할 위치를 사전 식별**할 수 있음
- 향후 **Diffusion 모델과 연계하여 마스크 또는 guide로 활용 가능**   

<br>

## 🛠️ 기술 스택

- `StyleGAN` – 이미지 생성 및 latent 조작
- `SVM (Linear)` – Latent Space 분류 경계 학습
- `Grad-CAM` – 속성 강조 영역 시각화
- `ResNet18` – 조명 유무 분류에 활용된 pretrained 모델   

<br>

## ⚙️ HiGAN_CAM 흐름

본 프로젝트는 생성 모델의 조작 가능성과 해석 기법을 결합하여, **StyleGAN 기반 이미지 생성**, **Latent Space 조작**, **속성 시각화**, **공간 추출**으로 이어지는 과정을 통해 이미지 상 특정 속성의 공간적 반영 위치를 도출한다. 전반적인 구성은 다음과 같다.

![Image](https://github.com/user-attachments/assets/a01b7748-e8b5-498d-8a12-32881de7c694)
<br>
### 1. StyleGAN을 활용한 이미지 생성
본 프로젝트는 **Ceyuan Yang and Yujun Shen의 연구(HIGAN)**에서 공개한 코드와 체크포인트를 기반으로 하여, StyleGAN을 활용한 이미지 생성을 수행하였다. 사용된 모델은 **LSUN Bedroom 데이터셋**(해상도 256×256, 약 50,000장)을 기반으로 학습된 것으로, 다양한 조명 조건과 구조적 특성이 반영되어 있어 생성 이미지의 조작 및 분석에 적합하다.

### 2. Latent Space 조작을 위한 SVM 적용
Latent Space 조작 또한 HIGAN 연구에서 제안된 방식과 공개된 코드를 참고하여 구현되었다.  
생성된 이미지는 하나의 잠재 벡터(latent vector)로부터 생성되며, 각 벡터는 특정 속성 정보를 포함한다. 본 프로젝트에서는 **조명 강도**를 기준으로 이미지에 라벨을 부여하고, 이를 바탕으로 SVM 모델을 학습하였다.

- StyleGAN으로부터 50,000장의 이미지를 생성  
- 각 이미지에 대해 **ResNet18 (Place365 pre-trained)**을 사용하여 조명 강도를 0–3으로 측정  
- 상위 2,000개 이미지를 **Positive (강한 조명)**, 하위 2,000개를 **Negative (약한 조명)**으로 라벨링  
- SVM을 통해 두 클래스의 경계를 학습한 후, Latent Space에서 조명 속성이 극대화/최소화된 두 벡터를 생성

### 3. Feature Map 기반 시각적 해석 (Grad-CAM)
SVM으로 생성된 Positive / Negative 벡터는 각각 StyleGAN의 Generator를 통해 이미지로 복원된다.  
이후, 각 생성 과정에서 Layer별 Feature Map을 추출하고 **Grad-CAM**을 적용하여 **속성에 반응한 시각적 영역(heatmap)**을 생성한다.

- Grad-CAM을 Positive / Negative 각각에 적용  
- 각 Layer별 heatmap의 차이를 계산하여 속성이 반영된 특징 차이 누적  
- 전체 레이어에서 차이값을 누적하여 최종적으로 속성이 강조된 공간 영역 도출

### 4. 클러스터링 기반 주요 영역 추출
시각화된 heatmap 차이의 누적값이 일정 임계값 이상인 지점들만을 대상으로 클러스터링을 수행

- 클러스터링은 **DBSCAN** 알고리즘 사용 (eps=4, min_samples=40)  
- 클러스터별 평균 누적값을 기준으로 상위 3개 클러스터를 최종 대상 위치로 선정  
- 각 클러스터의 중요도는 클러스터 내 heatmap 평균값과 점 개수를 곱한 값으로 계산

<br>

## :arrow_forward: **Colab QuickStart**

[[예시데이터 생성](https://colab.research.google.com/drive/1gdmoDPuyOJKogsPuvj9BmficmRiDbomB)] <br>
[[HiGAN_CAM](https://colab.research.google.com/drive/1U4E21tIwgUvybyZYvlB4m4CnWCxLP0ex)] <br>


## 🛠 실행 방법 (How to Run)

1. 예제 이미지 준비  
   - 코랩(Colab)을 통해 생성한 **GAN 이미지 예제**를 `input_data/` 폴더에 저장  

2. 아래 모델 파일을 다운로드하여 `higan/models/pretrain/pytorch` 경로에 저장  
   - 🔗 [StyleGAN Bedroom Checkpoint 다운로드](https://www.dropbox.com/s/h1w7ld4hsvte5zf/stylegan_bedroom256_generator.pth)

3. 프로젝트 루트 디렉토리에서 `Higan_CAM.ipynb` 실행

<br> 

## 📌 예시 결과 (Result)







