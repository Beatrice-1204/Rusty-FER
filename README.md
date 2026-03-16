# Face Processing Pipeline  
## Recunoașterea expresiilor faciale prin metode ale viziunii computerizate

---

## 1. Definirea problemei

Recunoașterea expresiilor faciale este o problemă fundamentală în domeniul viziunii computerizate, având aplicații în:
- interacțiune om–calculator,
- analiză comportamentală,
- sisteme inteligente adaptive.

Problema principală constă în:
- variația iluminării,
- poziționarea feței,
- zgomotul imaginii,
- fluctuații temporale între frame-uri.

Proiectul propune o soluție completă de procesare și stabilizare a predicției emoțiilor în timp real.

---

## 2. Obiectivele proiectului

Proiectul își propune:

- Detectarea feței în timp real folosind MediaPipe.
- Extracția regiunii de interes (ROI).
- Aplicarea unui pipeline de procesare a imaginii:
  - conversie grayscale
  - histogram equalization
  - redimensionare la 48x48
  - normalizare
- Clasificarea expresiei faciale utilizând un model CNN.
- Stabilizarea predicției prin vot majoritar pe 5 frame-uri consecutive.
- Implementarea unui sistem interactiv pentru activarea/dezactivarea etapelor de procesare.

---

## 3. Fundament teoretic

### 3.1 Grayscale Conversion

Reducerea imaginii la un singur canal elimină informația cromatică irelevantă și reduce complexitatea computațională.

### 3.2 Histogram Equalization

Tehnică de îmbunătățire a contrastului prin redistribuirea intensităților pixelilor.
Ajută la evidențierea trăsăturilor faciale în condiții de iluminare variabilă.

### 3.3 Redimensionare 48x48

Modelul CNN este antrenat pe imagini 48x48 pixeli (specific datasetului FER2013).
Reducerea rezoluției:
- scade costul computațional
- păstrează caracteristicile relevante pentru expresie

### 3.4 Normalizare

Pixelii sunt scalați în intervalul [0,1], ceea ce:
- stabilizează antrenarea rețelei
- îmbunătățește convergența

### 3.5 Stabilizare temporală (5-frame voting)

Pentru reducerea fluctuațiilor între frame-uri consecutive, se aplică vot majoritar pe ultimele 5 predicții.
Se calculează media încrederii pentru emoția dominantă.


---

## 4. Descrierea soluției

Soluția este structurată modular:

- camera_stream → captură video
- face_processor → detectare față
- image_preprocessor → procesare imagine
- emotion_predictor → clasificare CNN
- app.py → integrarea și controlul aplicației

Pipeline-ul este executat în timp real pentru fiecare frame.

---

## 5. Scenariu de utilizare

1. Utilizatorul pornește aplicația.
2. Camera detectează fața.
3. Sistemul aplică pipeline-ul de procesare.
4. Emoția este afișată pe ecran.
5. Utilizatorul poate activa/dezactiva:
   - histogram equalization
   - stabilizarea pe 5 frame-uri

---

## 6. Implementarea tehnică

### 6.1 MediaPipe
Utilizat pentru detectarea landmark-urilor faciale și extragerea ROI.

### 6.2 OpenCV
Utilizat pentru:
- captură video
- procesare imagine
- UI minimalist (trackbar-uri)

### 6.3 TensorFlow / Keras
Utilizat pentru:
- încărcarea modelului CNN
- predicția emoțiilor

### 6.4 Dataset utilizat – FER2013

Modelul CNN a fost antrenat utilizând dataset-ul public FER2013 (Facial Expression Recognition 2013).

Dataset-ul conține imagini:

- rezoluție 48x48 pixeli
- format grayscale
- organizate în directoare separate pentru fiecare emoție

Versiunea utilizată a dataset-ului conținea 7 clase inițiale.
În cadrul acestui proiect au fost selectate 5 clase:

- Angry
- Happy
- Neutral
- Sad
- Surprise

Două clase au fost eliminate pentru a simplifica problema de clasificare și a concentra modelul pe emoțiile principale.

Datele erau deja împărțite în:

- set de antrenare (train)
- set de test (test)

Nu a fost necesară o preprocesare manuală suplimentară a dataset-ului, deoarece imaginile erau deja în format 48x48 grayscale.


### 6.5 Structura proiectului

Rusty/
│
├── data/ # Dataset-uri sau fișiere auxiliare
├── models/ # Modele CNN antrenate
│
├── src/
│ ├── camera/ # Captură video în timp real
│ ├── emotion_recognition/ # Predicția emoțiilor (CNN inference)
│ ├── face_detection/ # Detectarea feței (MediaPipe)
│ ├── image_processing/ # Pipeline de procesare imagine
│ ├── interaction/ # Logică de control și stabilizare
│ ├── init.py
│ └── app.py # Entry point aplicație
│
├── train_emotion_model.py # Script pentru antrenarea modelului
├── requirements.txt # Dependențe proiect
└── README.md # Documentație

---

## 7. UI și Interactivitate

Interfața utilizează:

- cv2.imshow pentru afișare video
- cv2.createTrackbar pentru control:
  - Equalization ON/OFF
  - Voting ON/OFF
- Debug mode pentru afișarea etapelor intermediare

---


## 8. Posibile îmbunătățiri viitoare

- Filtrare temporală bazată pe medie ponderată.
- Imbunatatirea modelului pentru o acuratete ridicata 
- Adaugarea de reactii ale robotului in functie de emotiea detectata 
- Integrare într-un sistem robotic
- Optimizare pentru Raspberry Pi