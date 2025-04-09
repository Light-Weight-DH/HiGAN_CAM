# HiGAN_CAM
CAM representation with HiGAN


## 📌 Introduction

최근 Diffusion 모델은 마스크 지정이나 텍스트 프롬프트를 통한 이미지 조작에 널리 활용되고 있다. 하지만 사용자가 조작하고자 하는 속성의 위치나 대상이 이미지 상 어디에 반영되어야 하는지를 **사전에 파악할 수 있다면**, 더욱 직관적이고 효율적인 조작이 가능할 것이다.

GAN은 학습을 통해 이미지의 구조적 특성을 레이어별로 학습하고, latent space나 파라미터 조작을 통해 특정 속성을 변경하거나 추가할 수 있는 특징을 가진다. 본 프로젝트는 이러한 **GAN의 구조적 조작 가능성**과 **Grad-CAM을 활용한 해석 기법**을 결합하여, 조작 대상의 위치를 시각적으로 식별하고, 이를 향후 Diffusion 기반 이미지 생성에 활용할 수 있는 가능성을 모색한다.

---

## 📚 관련/참고 연구

### 📄 *Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis*

- **핵심 요약**:  
  이 연구는 실내 장면 생성에 사용된 **StyleGAN 기반 Generator**에서 **계층적 의미 구조(Semantic Hierarchy)**가 자연스럽게 형성된다는 점에 주목하였다.  
  Latent space에서 **특정 객체(예: bed)의 존재 유무를 기준으로 SVM을 학습**하고, 이를 통해 생성자의 구조적 특성에 대한 **의미 있는 조작 및 해석**이 가능함을 보였다.

- **핵심 구조(예시)**:  
  - 특정 환경(실내외 공간 ex: 침실,교회 등)을 기반으로 한 생성 모델(PGGAN ,BigGAN, StyleGAN) 학습  
  - ResNet18을 활용한 특성(ex: 조명, 나무 등) 유무 판별하는 모델 학습  
  - SVM을 통해 객체 유무를 분류하고, 해당 속성의 Latent Boundary 추출

- **활용**:  
  - 해당 연구에서 사용한 Generator, 해당 Generator를 기반으로 학습시킨 SVM 모델을 사용하였음 
  

- **GitHub**: [https://github.com/soccz/HIGAN](https://github.com/soccz/HIGAN)

--

## 🔍 Core Idea

- **StyleGAN**을 기반으로, 특정 속성이 존재하는 이미지와 그렇지 않은 이미지의 Latent Vector를 분리하여 학습
- **SVM**을 통해 속성 존재 유무를 기준으로 Latent Space에서의 분류 경계(Boundary)를 학습
- **Grad-CAM**을 활용하여 생성된 이미지에서 조작 속성에 해당하는 영역을 시각적으로 해석
- 이를 통해 사용자가 조작하고자 하는 **속성이 반영되어야 할 위치를 사전 식별**할 수 있음
- 향후 **Diffusion 모델과 연계하여 마스크 또는 guide로 활용 가능**

---

## 🛠️ 기술 스택

- `StyleGAN2` – 이미지 생성 및 latent 조작
- `SVM (Linear)` – Latent Space 분류 경계 학습
- `Grad-CAM` – 속성 강조 영역 시각화
- `ResNet18` – 조명 유무 분류에 활용된 pretrained 모델

---

## 📁 폴더 구성


