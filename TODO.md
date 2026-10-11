# TODO

- [ ] Histogramy lokalne na GPU (`__local uint hist[256]` w każdej grupie roboczej, potem sumowanie do globalnego) -> dzielić obraz na części i zrobić kilka histogramów
- [ ] 256 wątków w grupie roboczej (jawny local size, każdy wątek odpowiada za jeden bin)
- [ ] (opcjonalnie) Próg Otsu liczony równolegle na GPU — 256 wątków, każdy sprawdza jeden próg
- [ ] GPU z danymi zostawionymi na karcie (wysłać raz, bez odczytu wyników pośrednich) -> rozdzielić z kopiowaniem i bez kopiowania na GPU zostawić na karcie (można)
- [ ] Zmiana rozmiaru jako algorytm (najbliższy sąsiad / interpolacja dwuliniowa) — dopytać, czy o to chodziło
- [ ] Interpolacja dwuliniowa zamiast najbliższego sąsiada w korekcji dystorsji
- [ ] Sortowanie równoległe (sieć sortująca dla 9 elementów w kernelu / bitonic sort na GPU)
- [ ] Szukanie kilku progów na histogramie (multi-Otsu / doliny między szczytami)
- [ ] (opcjonalnie) Niblack / Sauvola do porównania
- [ ] Segmentacja przez rozrost obszaru (CPU: BFS z kolejką; GPU: iteracje aż do braku zmian)
- [ ] Detektor Canny’ego: Gauss → Sobel (moduł i kierunek) → tłumienie niemaksymalne → podwójny próg → histereza (śledzenie ścieżki krawędzi) -> Cannego (ściezka, do edge)

## Moje notatki
* może by tak zrobić więcej modelów dystorsji?
* compare also many CPU threads to GPU
* może dodać negatyw?
* może by tak jeszcze zrobić porównanie z Metal
* upewnij się, że w latex masz wyniki z najnowszych testów
* porównaj własne progi może