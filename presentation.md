# Skrypt prezentacji (wideo 5–6 min)

> **Lokalizacja Markowa dla robotów mobilnych w środowiskach dynamicznych**
> D. Fox, W. Burgard, S. Thrun — *Journal of AI Research* 11 (1999), 391–427.
> Projekt SIWR — Probabilistyczne Modele Grafowe w Robotyce.

Każdy slajd ma **Na slajdzie** (co widać) i **Narracja** (co mówi prelegent).
Czas docelowy ≈ 6 min (35 + 60 + 60 + 70 + 80 + 55 s).

---

## Wizualizacje — które, gdzie, co pokazują (do dopracowania)

> Wszystkie pliki już są w `results/`. Poniżej tylko brief — co dana animacja/wykres ma przedstawiać.

| Slajd | Plik w `results/` | Co ma pokazywać |
|---|---|---|
| 1 (tytuł) | `global_localization.gif` *(lub zdjęcie robota)* | hero: belief startuje rozmyty (jasny wszędzie) i **zbiega do prawdziwej pozy** w miarę jazdy — „robot znajduje siebie na mapie" |
| 2 (problem) | `global_multimodal.gif` *(+ `global_multimodal.png`)* | symetryczny korytarz: belief pozostaje **wielomodalny** (kilka jasnych plam naraz) — niejednoznaczność, z którą unimodalny Kalman sobie nie radzi |
| 3 (model) | *diagram HMM na slajdzie* (+ opcjonalnie klatka heatmapy belief) | główny wizual to graf; opcjonalnie still heatmapy marginalnego rozkładu `(x,y)` — „belief = rozkład na siatce" |
| 4 (założenia) | `motion_banana.png` | model ruchu: **punktowy** belief po jednym ruchu rozmywa się w kształt **„banana"** — krok predykcji + szum rosnący z przebytą drogą |
| 5 (filtry + kod) | `filters_crowd.gif` | robot jedzie **wśród ludzi** (obiekty spoza mapy), większość wiązek zasłonięta; belief **pozostaje skupiony** na prawdziwej pozie dzięki filtrowi *(idealnie: porównanie bez filtra vs z filtrem)* |
| 6 (wyniki) | `filters_error.png` *(+ `kidnapped_error.png`)* | wykres **błędu w czasie**: bez filtra rośnie/rozbiega się (~5 m), oba filtry zostają nisko. `kidnapped_error.png`: po porwaniu filtr entropii **blokuje** odzyskanie, filtr odległości odzyskuje |

---

## Slajd 1 — Tytuł i kontekst  (~35 s)

**Na slajdzie:**
- Tytuł: **Lokalizacja Markowa dla robotów mobilnych w środowiskach dynamicznych**
- Autorzy: D. Fox, W. Burgard, S. Thrun — *Journal of AI Research* 11 (1999), 391–427
- Jedno zdanie: *„Robot ustala swoją pozycję na znanej mapie — nawet gdy tłum ludzi zasłania mu czujniki."*
- **Wizual:** `results/global_localization.gif` — belief zbiega do prawdziwej pozy *(lub zdjęcie robota RHINO; zob. tabela wizualizacji)*
- Stopka: SIWR — Probabilistyczne Modele Grafowe w Robotyce

**Narracja:**
> Wybrany artykuł to „Markov Localization for Mobile Robots in Dynamic Environments" Foxa, Burgarda
> i Thruna z 1999 roku. Opisuje metodę, dzięki której robot mobilny ustala, gdzie znajduje się na
> znanej mapie — używając wyłącznie zaszumionej odometrii i odczytów czujników odległości. Co
> najważniejsze, metoda działa w środowiskach dynamicznych: na korytarzach pełnych ludzi
> zasłaniających czujniki. Autorzy przetestowali ją na prawdziwym robocie-przewodniku RHINO w Muzeum
> Niemieckim w Bonn. Zaimplementowałem tę metodę od zera w Pythonie i pokażę ją na własnych symulacjach.

---

## Slajd 2 — Jaki problem rozwiązujemy?  (~60 s)

**Na slajdzie:**
- Nagłówek: **Jaki problem rozwiązujemy?**
- *Lokalizacja globalna:* „gdzie jestem na mapie?" — start z **nieznanej** pozycji
- Dane wejściowe: zaszumiona **odometria** + zaszumiony **skan odległości**; mapa **znana**
- Filtr Kalmana zakłada: jeden gaussowski pik (**unimodalność**) + **statyczny świat**
- ...więc zawodzi przy: lokalizacji globalnej, **porwanym robocie**, **symetrii/niejednoznaczności**
- ⭐ Główne wyzwanie artykułu: **środowisko dynamiczne** — ludzie i obiekty spoza mapy
- **Wizual:** `results/global_multimodal.gif` — wielomodalny belief w symetrycznym korytarzu *(zob. tabela wizualizacji)*

**Narracja:**
> Robot ma znaną mapę, ale nie wie, gdzie na niej stoi. Ma tylko dwa źródła informacji: odometrię —
> zaszumiony pomiar własnego ruchu — oraz skan odległości. Pytanie brzmi „gdzie jestem?", i to nawet
> gdy startujemy z całkowicie nieznanej pozycji — to tzw. lokalizacja globalna.
>
> Klasyczny filtr Kalmana przyjmuje dwa upraszczające założenia: że nasza wiedza o pozycji to jeden
> gaussowski pik — rozkład unimodalny — oraz że świat jest statyczny. Przez to nie radzi sobie z
> trzema sytuacjami: lokalizacją globalną, problemem porwanego robota — gdy ktoś nagle przestawia
> robota — oraz środowiskami symetrycznymi, gdzie wiele miejsc wygląda identycznie.
>
> Ale najważniejsze wyzwanie tego artykułu to środowisko dynamiczne: korytarze pełne ludzi i
> obiektów, których nie ma na mapie. To one zasłaniają czujniki i psują lokalizację — i to jest
> sedno, które metoda ma rozwiązać.

---

## Slajd 3 — Jak modelujemy problem? Graf, zmienne, zależności  (~60 s)

**Na slajdzie:**
- Nagłówek: **Model: ukryty model Markowa (HMM / dynamiczna sieć Bayesa)**
- Diagram grafu (centralny element):
  ```
        a_1        a_2        a_T       a = odometria (sterowanie / wejście)
   L_0 → L_1  →   L_2  → ... → L_T      L = poza (x, y, θ)        [ukryta]
          │        │            │        s = skan odległości       [obserwacja]
          ▼        ▼            ▼
         s_1      s_2          s_T
  ```
- Stan ukryty: **L_t = (x, y, θ)** — nieobserwowany bezpośrednio
- Zależności: **L_{t-1} → L_t** (ruch) i **L_t → s_t** (percepcja)
- *Belief:* **Bel(L_t) = P(L_t | s_1..t, a_1..t)** — pełny rozkład na siatce 3D `Bel[ix, iy, iθ]`
- ⭐ Rozkład może być **wielomodalny** → rozwiązuje lokalizację globalną i symetrię
- **Wizual:** diagram HMM = element centralny; opcjonalnie klatka heatmapy belief *(zob. tabela wizualizacji)*

**Narracja:**
> Lokalizację modelujemy jako estymację stanu w ukrytym modelu Markowa — w języku modeli grafowych
> to dynamiczna sieć Bayesa. Spójrzmy na graf.
>
> Zmienne ukryte, których nie obserwujemy wprost, to L_t — poza robota: pozycja x, y i orientacja
> theta w chwili t. Wejściem jest odometria a_t. To, co faktycznie mierzymy, to obserwacja s_t —
> skan odległości.
>
> Graf ma dwa typy krawędzi, czyli dwie zależności. Pozioma: L_{t-1} prowadzi do L_t — to model
> ruchu, mówi jak poza zmienia się pod wpływem odometrii. Pionowa: L_t prowadzi do s_t — to model
> percepcji, mówi jakiego odczytu spodziewamy się z danej pozy.
>
> Naszą wiedzę o pozycji nazywamy belief — Bel(L_t) — i jest to pełny rozkład prawdopodobieństwa nad
> wszystkimi pozami, reprezentowany na trójwymiarowej siatce x, y, theta. I tu jest kluczowa różnica
> wobec Kalmana: ten rozkład może być wielomodalny — mieć wiele pików naraz. Dzięki temu potrafi
> reprezentować „mogę być tu albo tu" i rozwiązuje lokalizację globalną oraz środowiska symetryczne.

---

## Slajd 4 — Jakie założenia probabilistyczne?  (~70 s)

**Na slajdzie:**
- Nagłówek: **Założenia probabilistyczne — i dlaczego są kluczowe**
- **Założenie Markowa / statycznego świata:** `P(s_t | L_t, przeszłość) = P(s_t | L_t)`
- **Niezależność wiązek** (warunkowa, przy danej pozie): `P(s | l) = Πₖ P(sᵏ | l)`
- Model ruchu: odometria + szum gaussowski (wariancja rośnie z drogą)
- Model percepcji: mieszanka — *trafienie* (Gauss) + *obiekt nieznany* (krótki odczyt) + *max-zasięg* + *jednostajny*
- Element centralny — **rekurencyjny filtr Bayesa** (z `localizer.py`):
  ```
  predykcja:  Bel⁻(l) = Σ_l'  P(l | l', a) · Bel(l')        # predict(odom)
  korekcja:   Bel(l)  ∝  P(s | l) · Bel⁻(l)  → normalizacja  # correct(scan)
  ```
- ⭐ To **założenie statycznego świata** łamią ludzie → następny slajd to naprawia
- **Wizual:** `results/motion_banana.png` — punktowy belief → kształt „banana" po ruchu *(zob. tabela wizualizacji)*

**Narracja:**
> Cała metoda stoi na założeniu Markowa — inaczej: założeniu statycznego świata. Mówi ono, że przy
> znanej bieżącej pozie odczyt czujnika zależy tylko od tej pozy i od mapy, a nie od przeszłości.
> Drugie założenie to warunkowa niezależność wiązek skanu: prawdopodobieństwo całego skanu jest
> iloczynem prawdopodobieństw pojedynczych wiązek — pod warunkiem, że znamy pozę.
>
> Te założenia pozwalają aktualizować belief rekurencyjnie, filtrem Bayesa, w dwóch krokach.
> Predykcja: bierzemy model ruchu i „rozmywamy" belief zgodnie z odometrią i jej szumem. Korekcja:
> mnożymy przez model percepcji P(s|l) i normalizujemy. To dokładnie algorytm forward w HMM — w
> kodzie to metody `predict` i `correct`.
>
> Dwa modele warunkowe. Model ruchu to odometria plus szum gaussowski, którego wariancja rośnie z
> przebytą drogą. Model percepcji to mieszanka: gaussowskie trafienie w spodziewaną przeszkodę,
> człon na obiekt nieznany — czyli odczyt krótszy niż przewiduje mapa — pik na maksymalnym zasięgu i
> mały człon jednostajny. Ten jednostajny człon trzyma niezerowe prawdopodobieństwo w każdej komórce
> — i dzięki niemu robot potrafi wyjść z porwania.
>
> I rzecz najważniejsza dla tego artykułu: to właśnie założenie statycznego świata łamią ludzie. Ich
> odczytów nie ma w mapie, więc psują belief. Jak to naprawić — pokazuję na następnym slajdzie.

---

## Slajd 5 — Wkład pracy: filtry dla środowisk dynamicznych + implementacja  (~80 s)

**Na slajdzie:**
- Nagłówek: **Wkład artykułu: filtry dla środowisk dynamicznych**
- Struktura repo (skrót):
  ```
  src/markov_loc/
    occ_map.py    mapa + ray-casting + cache spodziewanych odległości
    belief.py     siatka belief: init / normalize / entropia / MAP
    motion.py     krok predykcji            (§3.1)
    sensor.py     model percepcji — mieszanka wiązek (§3.2)
    filters.py    filtr entropii + filtr odległości (§3.3)
    localizer.py  pętla filtra Bayesa: predict / correct
  ```
- Decyzja **per wiązka**: użyć odczytu czy odrzucić?
  - **Filtr entropii:** `ΔH = H(L|s) − H(L)`; **ZACHOWAJ gdy ΔH ≤ 0**, odrzuć gdy ΔH > 0 (odczyt zwiększa niepewność → obiekt spoza mapy)
  - **Filtr odległości:** odrzuć wiązkę gdy `P_short(d) > γ` (γ = 0.99) — odczyt prawie na pewno **krótszy** niż przewiduje mapa
- **Wizual:** `results/filters_crowd.gif` — robot lokalizuje się w tłumie *(zob. tabela wizualizacji)*

**Narracja:**
> Tu zaczyna się właściwy wkład artykułu. W tłumie większość wiązek trafia w ludzi, nie w mapę —
> więc trzeba zdecydować, których odczytów użyć. Autorzy proponują dwa filtry, działające per
> pojedyncza wiązka.
>
> Pierwszy to filtr entropii. Entropia mierzy niepewność beliefu. Dla każdej wiązki sprawdzamy, jak
> zmieniłaby entropię, gdybyśmy ją uwzględnili. Jeśli odczyt potwierdza nasze przekonanie — entropia
> nie rośnie, delta H jest mniejsza lub równa zero — zachowujemy go. Jeśli odczyt zwiększa
> niepewność, delta H większe od zera, odrzucamy: prawdopodobnie pochodzi od obiektu spoza mapy, na
> przykład od człowieka. Drobna ciekawostka: w samym artykule jest tu literówka w znaku — w kodzie
> zaimplementowałem intencję autorów, nie dosłowny zapis.
>
> Drugi to filtr odległości, dla czujników zakresowych: jeśli uśredniona po beliefie wiązka jest
> niemal na pewno krótsza niż przewiduje mapa, odrzucamy ją — coś bliżej stanęło. Próg gamma to 0,99.
>
> Po lewej struktura kodu — całość napisana od zera w Pythonie z numpy. Sercem jest `localizer.py`,
> czyli pętla predict/correct, którą widzieliśmy. Statyczny cache spodziewanych odległości liczę raz
> i współdzielę między modelem percepcji a oboma filtrami. Na animacji po prawej robot lokalizuje się
> w tłumie.

---

## Slajd 6 — Wyniki i podsumowanie  (~55 s)

**Na slajdzie:**
- Nagłówek: **Wyniki**
- Tabela — lokalizacja w tłumie (~50% wiązek zasłoniętych przez ludzi):

  | warunek | średni błąd pozycji |
  |---|---|
  | **bez filtra** | **5,2 m** — rozbiega się, blokuje na złej pozie |
  | filtr entropii | **0,10 m** |
  | filtr odległości | **0,13 m** |

- **Wizual:** `results/filters_error.png` (+ `kidnapped_error.png`) — błąd pozycji w czasie *(zob. tabela wizualizacji)*
- Subtelność (porwany robot): filtr **odległości** odzyskuje pozycję, filtr **entropii** ją **blokuje** (belief-based vs geometry-based)
- Podsumowanie: grid-based Markov localization = solidna probabilistyczna alternatywa dla Kalmana; krok ku Monte Carlo Localization
- GitHub: `<wstaw link do repo>`

**Narracja:**
> Wyniki. W scenariuszu, gdzie około połowa wiązek trafia w ludzi: bez żadnego filtra średni błąd
> pozycji to ponad pięć metrów — belief się rozbiega i blokuje na złej pozie. Z filtrem entropii błąd
> spada do dziesięciu centymetrów, z filtrem odległości — trzynastu. To ilościowy dowód tezy artykułu.
>
> Ciekawa subtelność dotyczy porwanego robota. Po teleportacji filtr odległości szybko odzyskuje
> poprawną pozycję, ale filtr entropii ją blokuje — bo prawidłowe odczyty z nowego miejsca przeczą
> staremu beliefowi, więc zwiększają entropię i zostają odrzucone. Filtr entropii opiera się na
> beliefie i nie umie sam wyjść z błędnej blokady; filtr odległości jest geometryczny i potrafi.
> Artykuł wprost opisuje ten przypadek.
>
> Podsumowując: grid-based Markov localization to solidna, probabilistyczna alternatywa dla filtru
> Kalmana — radzi sobie z lokalizacją globalną, symetrią i tłumem. Była też krokiem milowym ku Monte
> Carlo Localization i całej probabilistycznej robotyce. Cały kod jest na GitHubie. Dziękuję.

---

## Skrót prezentacji — jedno zdanie na slajd

1. **Tytuł i kontekst** — przedstawiam artykuł Foxa, Burgarda i Thruna (1999) o lokalizacji Markowa, który pozwala robotowi ustalić swoją pozę na znanej mapie nawet w tłumie zasłaniającym czujniki.
2. **Problem** — robot ma znaną mapę, ale nie wie, gdzie stoi, a klasyczny filtr Kalmana (unimodalny, statyczny świat) zawodzi przy lokalizacji globalnej, symetrii i — najważniejsze — w środowisku dynamicznym.
3. **Model** — problem modelujemy jako ukryty model Markowa (dynamiczną sieć Bayesa), gdzie ukryta poza L_t generuje obserwacje s_t, a belief jest pełnym, potencjalnie wielomodalnym rozkładem na siatce 3D.
4. **Założenia probabilistyczne** — założenie Markowa i warunkowa niezależność wiązek pozwalają aktualizować belief rekurencyjnym filtrem Bayesa (predykcja + korekcja), ale to właśnie założenie statycznego świata łamią ludzie.
5. **Wkład pracy + implementacja** — autorzy naprawiają to dwoma filtrami decydującymi per wiązka (filtr entropii i filtr odległości), które odrzucają odczyty od obiektów spoza mapy; całość zaimplementowałem od zera w Pythonie.
6. **Wyniki i podsumowanie** — w tłumie filtry redukują błąd z ponad 5 m do ok. 10 cm, a różnica w problemie porwanego robota (odległość odzyskuje, entropia blokuje) pokazuje, że grid-based Markov localization to solidna alternatywa dla Kalmana i krok ku Monte Carlo Localization.
