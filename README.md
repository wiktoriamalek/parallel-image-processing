# Masowo-równoległe przetwarzanie obrazu

**Porównanie CPU (1 wątek / N wątków) z GPU (OpenCL)**

**Autor:** Wiktoria Malek

---

## 1. Opis programu

`img_proc` wykonuje cztery operacje na obrazach w formacie **RGBA (8 bitów na kanał)**.

| Polecenie | Wynik           | Algorytm CPU               | Algorytm GPU (OpenCL) |
| --------- | --------------- | -------------------------- | --------------------- |
| `invert`  | PNG – negatyw   | `p = 255 − p`              | kernel 1D             |
| `edge`    | PNG – krawędzie | Sobel 3×3                  | kernel 2D             |
| `blur`    | PNG – rozmycie  | Box Blur 3×3 (40 iteracji) | kernel 2D             |
| `hist`    | CSV – histogram | 3×256 binów                | kernel 1D + redukcja  |

### Składnia

```bash
img_proc [invert|edge|blur|hist] in.png out [gpu|cpu] [wątki]
```

**Tryby pracy:**

* `gpu` – OpenCL (domyślnie wszystkie dostępne urządzenia)
* `cpu` – implementacja jednowątkowa lub wielowątkowa
* `[wątki]` – liczba wątków dla trybu CPU (domyślnie 1)

---

Pliki OpenCL (`*.cl`) nie są kompilowane przez C++. Program wczytuje je dynamicznie z katalogu `kernels/` podczas uruchamiania.

---

## 2. Wymagania

* Kompilator zgodny z **C++17**

  * clang++
  * g++
  * MSVC 19.3+
* OpenCL ≥ 1.2
* macOS – systemowy framework OpenCL
* Linux – `ocl-icd-opencl-dev` lub sterowniki GPU
* Windows – OpenCL SDK lub sterowniki GPU
* (opcjonalnie) CMake ≥ 3.16

---

## 3. Kompilacja

### 3.1 macOS

```bash
clang++ -std=c++17 -O2 -pthread \
-Iinclude src/*.cpp \
-framework OpenCL \
-o img_proc
```

W przypadku Xcode należy dodać pliki `.cpp` do:

```text
Build Phases → Compile Sources
```

Framework OpenCL jest linkowany automatycznie.

---

### 4.2 Linux

```bash
sudo apt install build-essential ocl-icd-opencl-dev

g++ -std=c++17 -O2 -pthread \
-Iinclude src/*.cpp \
-lOpenCL \
-o img_proc
```

---

### 4.3 Windows

Po zainstalowaniu sterownika lub SDK OpenCL:

```text
Project → Properties

C/C++
└── Additional Include Directories
    └── $(ProjectDir)include

Linker
├── Additional Dependencies
│   └── OpenCL.lib
└── Additional Library Directories
    └── <OpenCL>\lib
```

Przykładowa kompilacja:

```bash
g++ -std=c++17 -O2 ^
-Iinclude ^
-IC:\OpenCL\include ^
src\*.cpp ^
-LC:\OpenCL\lib ^
-lOpenCL ^
-o img_proc.exe
```

---

### 4.4 Kompilacja przy użyciu CMake

Projekt zawiera plik `CMakeLists.txt`.

Przykład (Visual Studio 2022):

```bash
cd C:\Users\<user>\Downloads\ParallelComputing

mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

Powstanie:

```text
build\Release\img_proc.exe
```

CMake automatycznie dobiera sposób linkowania OpenCL:

* macOS → `-framework OpenCL`
* Linux → `-lOpenCL`
* Windows → `OpenCL.lib`

### Ważne

Katalog `kernels/` musi znajdować się obok pliku wykonywalnego.

Przykład:

```bash
xcopy /E /I ..\kernels Release\kernels
```

Można również otworzyć projekt bezpośrednio w Visual Studio:

```text
File → Open → CMake...
```

i wskazać plik `CMakeLists.txt`.

---

## 5. Przykłady uruchomienia

```bash
img_proc invert test_images/flower.png out_inv.png gpu

img_proc blur test_images/flower.png blur.png cpu 8

img_proc edge test_images/flower.png edge.png gpu

img_proc hist test_images/flower.png hist.csv gpu
```

---

## 6. Obsługa różnych platform

Nagłówek OpenCL jest dołączany warunkowo:

```cpp
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
```

Dzięki temu wystarczy poprawnie podlinkować bibliotekę:

| System  | Biblioteka          |
| ------- | ------------------- |
| macOS   | `-framework OpenCL` |
| Linux   | `-lOpenCL`          |
| Windows | `OpenCL.lib`        |

---

## 7. Biblioteki zewnętrzne

Projekt wykorzystuje:

* `stb_image.h`
* `stb_image_write.h`

Autor: **Sean Barrett**

Licencja: **Public Domain / MIT**
