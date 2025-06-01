# Masowo-równoległe przetwarzanie obrazu 
Porównanie CPU (1 wątek / N wątków) z GPU (OpenCL)

Autor : Wiktoria Malek

## 1. Opis programu

img_proc wykonuje cztery operacje na obrazach w formacie RGBA (8 bit na kanał):

| polecenie | wynik (plik)    | algorytm CPU           | algorytm GPU (OpenCL) |
|-----------|-----------------|------------------------|-----------------------|
| `invert`  | PNG – negatyw   | p = 255 − p            | kernel 1-D            |
| `edge`    | PNG – krawędzie | Sobel 3 × 3            | kernel 2-D            |
| `blur`    | PNG – rozmycie  | Box 3 × 3 (40 iteracji)| kernel 2-D            |
| `hist`    | CSV – histogram | 3 × 256 binów          | kernel 1-D + reszta   |

Składnia:

img_proc  [invert|edge|blur|hist]  in.png  out  [gpu|cpu]  [wątki]

gpu → OpenCL (domyślnie wszystkie dostępne jednostki GPU/CPU)
cpu → implementacja jednowątkowa lub wielowątkowa
[wątki] podawane tylko w trybie cpu (domyślnie 1)

## 2. Struktura katalogów

├─ include/               # nagłówki C++
│  ├─ cpu_ops.hpp
│  ├─ gpu_ops.hpp
│  ├─ image_io.hpp
│  ├─ stb_image.h
│  └─ stb_image_write.h
├─ src/                   # implementacje
│  ├─ cpu_ops.cpp
│  ├─ gpu_ops.cpp
│  ├─ image_io.cpp
│  └─ main.cpp
├─ kernels/               # pliki OpenCL ładowane w runtime
│  ├─ invert.cl
│  ├─ edge.cl
│  ├─ blur.cl
│  └─ hist.cl
└─ README.md

Pliki .cl nie są kompilowane przez C++; program wczytuje je w czasie
wykonywania z podkatalogu kernels.

## 3. Wymagania
Kompilator C++17 (clang++, g++, MSVC 19.3+)
Środowisko OpenCL ≥ 1.2
macOS – systemowy framework OpenCL
Linux – pakiet ocl-icd-opencl-dev lub sterownik GPU
Windows – sterownik GPU + plik OpenCL.lib / OpenCL.dll
(opcjonalnie) CMake ≥ 3.16 – ułatwia budowanie na dowolnym OS

## 4. Kompilacja
### 4.1 macOS

clang++ -std=c++17 -O2 -pthread \
        -Iinclude src/*.cpp \
        -framework OpenCL -o img_proc
        
Xcode: trzeba dodać pliki .cpp do zakładki Build Phases → Compile Sources
(Framework OpenCL linkuje się automatycznie).

### 4.2 Linux 
sudo apt install build-essential ocl-icd-opencl-dev
g++ -std=c++17 -O2 -pthread \
    -Iinclude src/*.cpp \
    -lOpenCL -o img_proc
    
### 4.3 Windows
Visual Studio; Zainstalować sterownik lub SDK OpenCL.
Utworzyć projekt, dodać pliki z src/.
Project → Properties
C/C++ → Additional Include Dirs → $(ProjectDir)include; <OpenCL>\include
Linker → Additional Dependencies → OpenCL.lib
Linker → Additional Library Dirs → <OpenCL>\lib


g++ -std=c++17 -O2 -Iinclude -IC:\OpenCL\include ^
    src\*.cpp -LC:\OpenCL\lib -lOpenCL -o img_proc.exe
    
# 4.4 Uniwersalne – CMake
Plik CMakeLists.txt znajduje się w repozytorium projektu.
W Command Prompt, np. w Visual Studio:
Przejdź do folderu projektu, np. poprzez:

cd C:\Users\<user>\Downloads\ParallelComputing

Następnie:

mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

Powstanie build\Release\img_proc.exe.

CMake automatycznie użyje:
macOS → -framework OpenCL
Linux/Windows → -lOpenCL lub OpenCL.lib


Pliki z katalogu kernels\ muszą leżeć obok EXE, więc skopiuj:
xcopy /E /I ..\kernels Release\kernels

Można także:
Uruchomić Visual Studio 2022 → File → Open → CMake…
i wskazać plik CMakeLists.txt.

VS automatycznie wykryje konfigurację: „x64-Debug” i „x64-Release”.
Następnie zbudować i uruchomić projekt.
EXE znajduje się w out\build\x64-Release\img_proc.exe


## 5. Przykłady uruchomienia

Testowanie polega na:
cd Release                    :: katalog z exe i podfolderem kernels
img_proc invert ..\test_images\flower.png out_inv.png gpu
img_proc blur   ..\test_images\flower.png blur.png    cpu 8
img_proc edge   ..\test_images\flower.png edge.png    gpu
img_proc hist   ..\test_images\flower.png hist.csv    gpu


## 6.Nagłówek OpenCL dołączany jest warunkowo:

## #if defined(__APPLE__)
## #include <OpenCL/opencl.h>
## #else
##  #include <CL/cl.h>
## #endif

Dzięki temu wystarczy poprawnie podlinkować bibliotekę:

-framework OpenCL (macOS)
-lOpenCL (Linux)
OpenCL.lib (Windows)

## 7. Pliki dodatkowe, z githuba
Pliki stb_image.h / stb_image_write.h – Public Domain / MIT (Sean Barrett).
