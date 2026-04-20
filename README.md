# Rusty - recunoasterea emotiilor faciale in timp real

## Descriere generala

Rusty este un proiect pentru materia **Inteligenta Artificiala**. Aplicatia foloseste camera web pentru a detecta fata utilizatorului si pentru a estima emotia faciala in timp real.

Proiectul combina doua modele ONNX:

- **YuNet**, folosit pentru detectarea fetei;
- **FERPlus**, folosit pentru recunoasterea emotiei.

Rezultatul este afisat direct peste imaginea live de la camera. Aplicatia marcheaza fata detectata, afiseaza punctele importante ale fetei si scrie emotia prezisa impreuna cu procentul de incredere.

## Problema rezolvata

Recunoasterea emotiilor faciale este o problema din zona de computer vision si inteligenta artificiala. Scopul este ca un sistem sa primeasca o imagine cu o fata si sa estimeze expresia faciala afisata.

In cazul acestui proiect, problema este rezolvata pe un flux video live, nu pe o singura imagine statica. Asta inseamna ca aplicatia trebuie sa faca mai multe lucruri rapid, pentru fiecare cadru:

- sa citeasca imaginea de la camera;
- sa gaseasca fata;
- sa extraga corect zona fetei;
- sa pregateasca imaginea pentru modelul de emotii;
- sa ruleze predictia;
- sa afiseze rezultatul fara intarzieri mari.

## Obiective

- detectarea fetei in timp real folosind YuNet;
- extragerea regiunii fetei din cadrul video;
- alinierea fetei folosind pozitia ochilor;
- pregatirea imaginii pentru modelul FERPlus;
- clasificarea emotiei folosind ONNX Runtime;
- afisarea emotiei si a increderii predictiei;
- filtrarea detectiilor slabe sau instabile;
- pastrarea unei emotii stabile cand predictia curenta nu este suficient de sigura;
- masurarea performantelor prin timpi de procesare si FPS.

## Tehnologii folosite

Proiectul este scris in **Python** si foloseste:

- **OpenCV / opencv-contrib-python** pentru camera, procesare de imagine, detectorul YuNet si afisare;
- **ONNX Runtime** pentru rularea modelului FERPlus;
- **NumPy** pentru calcule pe vectori, imagini si probabilitati.

Dependentele sunt trecute in `requirements-runtime.txt`:

```text
numpy
opencv-contrib-python
onnxruntime
```

## Modele folosite

### Detectarea fetei

Pentru detectarea fetei se foloseste modelul:

```text
models/YuNet/face_detection_yunet.onnx
```

Modelul este incarcat prin `cv2.FaceDetectorYN.create`. Detectorul intoarce:

- bounding box-ul fetei;
- pozitia ochiului stang;
- pozitia ochiului drept;
- pozitia nasului;
- colturile gurii;
- scorul detectiei.

In proiect se pastreaza fata cu scorul cel mai mare, deoarece aplicatia este gandita pentru o singura persoana principala in cadru.

### Recunoasterea emotiei

Pentru emotii se foloseste modelul:

```text
models/ONNX/emotion-ferplus-7.onnx
```

Acesta este un model FERPlus rulat cu ONNX Runtime. Modelul lucreaza cu o imagine grayscale de `64x64`, in format NCHW:

```text
1 x 1 x 64 x 64
```

Clasele brute ale modelului sunt:

- neutral;
- happiness;
- surprise;
- sadness;
- anger;
- disgust;
- fear;
- contempt.

In aplicatie, rezultatul brut este mapat la un set mai simplu de etichete:

- `neutral` -> `neutral`;
- `happiness` -> `happy`;
- `surprise` -> `surprise`;
- `sadness` -> `sad`;
- `anger` -> `angry`;
- `disgust` -> `neutral`;
- `fear` -> `sad`;
- `contempt` -> `neutral`.

## Pipeline-ul aplicatiei

Aplicatia ruleaza o bucla continua in `src/app.py`.

Pentru fiecare cadru video:

1. Se citeste un frame din camera.
2. YuNet detecteaza fata.
3. Detectia este validata.
4. Daca detectia este buna, se extrage zona fetei.
5. ROI-ul fetei este aliniat simplu dupa pozitia ochilor.
6. Se face un crop intern al fetei.
7. Imaginea este convertita in grayscale.
8. Imaginea este redimensionata la `64x64`.
9. Imaginea este transformata in tensor `float32` cu forma `1 x 1 x 64 x 64`.
10. Modelul FERPlus produce scoruri pentru clasele de emotii.
11. Scorurile sunt transformate in probabilitati cu softmax.
12. Se alege emotia cu probabilitatea cea mai mare.
13. Rezultatul este afisat in fereastra video.

## Fundament teoretic

### Detectarea fetei

Detectarea fetei este prima etapa a proiectului. Fara aceasta etapa, modelul de emotii ar primi prea multa informatie inutila din imagine: fundal, haine, lumina sau alte obiecte.

YuNet localizeaza fata si ofera si puncte importante ale fetei. Aceste puncte sunt utile pentru ca aplicatia poate verifica daca detectia este plauzibila si poate face o aliniere simpla a fetei.

### Regiunea de interes

Dupa detectare, aplicatia extrage doar regiunea fetei. Aceasta regiune se numeste ROI, de la "Region of Interest". In proiect, ROI-ul este extras cu un padding, ca sa nu se piarda parti importante din fata.

### Alinierea fetei

Fata poate fi usor inclinata in fata camerei. Pentru a reduce aceasta problema, proiectul foloseste pozitia celor doi ochi si roteste ROI-ul astfel incat fata sa fie mai bine aliniata.

Aceasta etapa ajuta modelul de emotii, deoarece expresiile sunt mai usor de analizat cand fata are o pozitie mai constanta.

### Pregatirea imaginii pentru FERPlus

Modelul FERPlus folosit in proiect asteapta o imagine grayscale de `64x64`. De aceea, ROI-ul final este:

- convertit din BGR in grayscale;
- optional egalizat cu `cv2.equalizeHist`, daca setarea este activata;
- redimensionat la `64x64`;
- convertit la `float32`;
- transformat in format NCHW.

In configuratia actuala, egalizarea este dezactivata:

```text
onnx_use_equalization = False
```

### Softmax

Modelul returneaza scoruri brute pentru fiecare clasa. Aceste scoruri sunt transformate in probabilitati folosind softmax. Dupa softmax, suma probabilitatilor este 1, iar clasa cu probabilitatea cea mai mare este aleasa ca predictie.

## Structura proiectului

```text
Rusty/
|-- models/
|   |-- ONNX/
|   |   `-- emotion-ferplus-7.onnx
|   `-- YuNet/
|       `-- face_detection_yunet.onnx
|-- src/
|   |-- app.py
|   |-- runtime_config.py
|   |-- runtime_support.py
|   |-- camera/
|   |   `-- camera_stream.py
|   |-- emotion_recognition/
|   |   `-- onnx_emotion_predictor.py
|   |-- face_detection/
|   |   `-- yunet_face_detector.py
|   |-- image_processing/
|   |   `-- image_preprocessor.py
|   `-- interaction/
|       `-- response_engine.py
`-- requirements-runtime.txt
```

## Descrierea modulelor

### `src/app.py`

Este fisierul principal. Aici sunt create camera, detectorul YuNet, preprocesorul, predictorul FERPlus, validatorul de fata si tracker-ul de performanta.

Tot aici se afla bucla principala care citeste cadrele, detecteaza fata, ruleaza predictia si afiseaza rezultatul.

### `src/camera/camera_stream.py`

Modul simplu pentru accesul la camera. Deschide camera cu `cv2.VideoCapture`, seteaza rezolutia la `640x480` si returneaza frame-urile citite.

### `src/face_detection/yunet_face_detector.py`

Contine clasa `YuNetFaceDetector`. Aceasta incarca modelul YuNet si transforma rezultatul primit de la OpenCV intr-o structura mai usor de folosit in restul proiectului.

Structura `YuNetFaceDetection` contine bounding box-ul, cele cinci puncte faciale si scorul detectiei.

### `src/image_processing/image_preprocessor.py`

Se ocupa de pregatirea fetei pentru modelul de emotii:

- extrage ROI-ul fetei;
- adauga padding;
- aliniaza fata folosind ochii;
- aplica un crop intern;
- returneaza imaginea finala care merge mai departe la FERPlus.

### `src/emotion_recognition/onnx_emotion_predictor.py`

Incarca modelul `emotion-ferplus-7.onnx` cu ONNX Runtime si face predictia emotiei.

In acest modul se gasesc:

- lista claselor brute FERPlus;
- maparea claselor brute la etichetele folosite in aplicatie;
- conversia la grayscale;
- redimensionarea la `64x64`;
- calculul softmax;
- pastrarea ultimelor rezultate, inclusiv top 3 predictii.

### `src/runtime_config.py`

Contine setarile principale ale aplicatiei. Este util pentru prezentare, pentru ca arata clar ce praguri si parametri sunt folositi.

Exemple:

- camera: `640x480`;
- prag YuNet: `score_threshold = 0.6`;
- NMS YuNet: `nms_threshold = 0.3`;
- detectie pentru o singura fata: `top_k = 1`;
- latime maxima pentru detectie: `max_detection_width = 320`;
- prag minim pentru emotie: `min_confidence = 0.45`;
- pastrarea ultimei emotii stabile: `hold_last_stable = True`.

### `src/runtime_support.py`

Contine clase ajutatoare:

- `DetectionSmoother`, pentru netezirea bounding box-ului intre cadre;
- `FaceQualityValidator`, pentru eliminarea detectiilor slabe;
- `PerfTracker`, pentru masurarea timpilor si a FPS-ului.

## Stabilizare si validare

Proiectul include verificari pentru a evita rezultate slabe.

Detectia fetei este respinsa daca:

- nu exista fata detectata;
- fata este prea mica;
- fata este prea aproape de marginea imaginii;
- punctele faciale ies din cadru;
- distanta dintre ochi este prea mica;
- linia ochilor este prea inclinata.

Bounding box-ul poate fi netezit intre cadre cu `DetectionSmoother`, folosind parametrul `alpha = 0.35`.

Pentru emotii, aplicatia verifica:

- increderea predictiei;
- diferenta dintre top 1 si top 2, daca este folosita;
- ultima emotie stabila, care poate ramane afisata daca predictia noua este slaba.

Exista si suport pentru vot pe mai multe cadre (`voting_window = 5`), dar in configuratia curenta acesta este dezactivat:

```text
enable_voting = False
```

## Interfata si vizualizare

Interfata este facuta cu OpenCV. Aplicatia deschide o fereastra numita:

```text
Rusty - Emotion Recognition
```

In aceasta fereastra se afiseaza:

- imaginea live de la camera;
- dreptunghiul fetei detectate;
- punctele faciale oferite de YuNet;
- emotia prezisa;
- procentul de incredere;
- eticheta bruta FERPlus, daca `show_raw_label = True`.

Comenzi:

- `q` sau `Esc` inchide aplicatia;
- `d` porneste sau opreste vizualizarea pipeline-ului de debug, daca este folosita.

## Performanta

Aplicatia masoara timpul pentru mai multe etape:

- citirea frame-ului;
- detectarea fetei;
- validarea si post-procesarea detectiei;
- preprocesarea ROI-ului;
- predictia emotiei;
- timpul total pe frame.

La fiecare 30 de frame-uri, daca `show_perf = True`, consola afiseaza un rezumat cu timpii medii si FPS-ul.

## Directia proiectului

Acest proiect este gandit ca baza pentru un robot interactiv. Varianta finala este planificata sa ruleze pe un **Raspberry Pi 5**, folosind o camera **Raspberry Pi Camera Module 2 Wide**.

Ideea este ca robotul sa poata observa expresia faciala a persoanei din fata lui si sa reactioneze in functie de emotia detectata. In forma actuala, proiectul se ocupa de partea de perceptie: citirea imaginii, detectarea fetei si recunoasterea emotiei. Pe baza acestui rezultat se pot adauga mai departe raspunsuri ale robotului, cum ar fi mesaje, miscari sau alte forme de interactiune.

Folosirea modelelor ONNX ajuta si pentru aceasta directie, deoarece aplicatia poate rula local, fara sa depinda de o conexiune la internet. In plus, setarile precum rezolutia camerei, reducerea imaginii pentru detectie si masurarea FPS-ului sunt importante pentru rularea pe un dispozitiv cu resurse mai limitate decat un laptop.


## Ce demonstreaza proiectul

Proiectul demonstreaza folosirea practica a unor concepte importante din inteligenta artificiala:

- detectare de fete cu un model dedicat;
- clasificare de emotii cu un model neuronal;
- rulare de modele ONNX local;
- preprocesarea datelor inainte de predictie;
- folosirea probabilitatilor pentru alegerea clasei finale;
- filtrarea predictiilor nesigure;
- analiza unui flux video in timp real.

## Limitari

- Aplicatia este gandita pentru o singura fata principala in cadru.
- Rezultatul depinde de lumina, pozitia fetei si calitatea camerei.
- Modelul poate confunda emotii apropiate.
- Predictia reprezinta o estimare dupa expresia faciala, nu o citire sigura a starii emotionale reale.

## Posibile imbunatatiri

- detectarea si clasificarea mai multor fete in acelasi cadru;
- afisarea unui grafic cu probabilitatile pentru toate clasele FERPlus;
- salvarea unui istoric al predictiilor;
- testarea pe mai multe exemple si conditii de lumina;


## Concluzie

Rusty este o aplicatie de inteligenta artificiala care foloseste YuNet pentru detectarea fetei si FERPlus pentru recunoasterea emotiei. Pipeline-ul este complet: camera, detectie, validare, preprocesare, predictie, stabilizare si afisare.

Proiectul este structurat modular, iar fiecare componenta are un rol clar. Din acest motiv, codul poate fi explicat usor la prezentare si poate fi extins mai departe pentru proiectul final.
