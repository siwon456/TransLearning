'''
TransLearning

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 혼동 행렬 시각화를 위해 추가
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model 

1. 모델 예측 및 평가 지표 계산
print("--- 모델 성능 평가 시작 ---")

실제 레이블과 예측 레이블 추출
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

클래스 이름 가져오기
class_labels = list(test_generator.class_indices.keys())
num_classes = len(class_labels)


2. 분류 리포트 출력
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

3. F1-score 출력
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
print(f"\nMacro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

4. 혼동 행렬 시각화
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

fig, ax = plt.subplots(figsize=(num_classes // 2, num_classes // 2)) # 클래스 수에 따라 크기 동적 조절
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical') # x축 레이블 회전
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('final_confusion_matrix.png')
plt.show()
plt.close()

5. 각 클래스별 샘플 예측 시각화
print("\n--- 각 클래스별 샘플 예측 시각화 시작 ---")

각 클래스마다 1개씩 이미지를 저장할 리스트
selected_images = []
selected_true_labels_idx = [] # 실제 레이블 (정수 인덱스)
selected_file_paths = [] # 원본 파일 경로 (디버깅용)

이미지를 이미 선택했는지 추적하기 위한 딕셔너리
found_class_samples = {label_idx: False for label_idx in range(num_classes)}
found_count = 0

for i, filename in enumerate(test_generator.filenames):
    true_class_idx = test_generator.classes[i] # 해당 파일의 실제 클래스 인덱스

    if not found_class_samples[true_class_idx]: # 아직 이 클래스의 샘플을 찾지 못했다면
        full_img_path = os.path.join(test_dir, filename) # 전체 파일 경로

        try:
            img = load_img(full_img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0 # 모델 입력 형식으로 변환

            selected_images.append(img_array)
            selected_true_labels_idx.append(true_class_idx)
            selected_file_paths.append(full_img_path)

            found_class_samples[true_class_idx] = True
            found_count += 1
            if found_count == num_classes: # 모든 클래스에서 샘플을 찾았다면 중단
                break
        except Exception as e:
            print(f"경고: 이미지 로드 중 오류 발생 - {full_img_path}: {e}")
            continue

선택된 이미지가 없거나 클래스 수보다 적을 경우 처리
if not selected_images:
    print("오류: 시각화할 이미지를 찾을 수 없습니다. 테스트 디렉토리와 파일들을 확인해주세요.")
else:
    NumPy 배열로 변환
    selected_images_np = np.array(selected_images)
    selected_true_labels_np = np.array(selected_true_labels_idx)

    선택된 이미지들에 대한 예측 수행
    predictions_for_display = model.predict(selected_images_np)
    predicted_classes_for_display = np.argmax(predictions_for_display, axis=1)

    시각화 (25개 이미지를 한 줄에 표시하기 어려울 수 있으므로 여러 줄로 나눕니다)
    images_per_row = 5 # 한 줄에 표시할 이미지 수
    num_rows = (len(selected_images) + images_per_row - 1) // images_per_row

    plt.figure(figsize=(images_per_row * 3.5, num_rows * 4)) # Figure 사이즈 조정

    for i in range(len(selected_images)):
        plt.subplot(num_rows, images_per_row, i + 1)
        plt.imshow(selected_images[i])
        true_label_name = class_labels[selected_true_labels_np[i]]
        pred_label_name = class_labels[predicted_classes_for_display[i]]
        color = "green" if true_label_name == pred_label_name else "red"
        plt.title(f"True: {true_label_name}\nPred: {pred_label_name}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions_per_class.png')
    plt.show()

print("\n--- 모든 평가 및 시각화 작업 완료 ---")
'''
