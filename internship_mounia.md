# Internship Context – Mounia Baddou (FlameTrack 2026)

> Diese Datei dient als Kontext-Übergabe zwischen Claude-Sessions.
> Einfach diese Datei in eine neue Session hochladen und weitermachen.

## Eckdaten

- **Praktikantin:** Mounia Baddou
- **Zeitraum:** 01.07.2026 – 25.09.2026 (3 Monate)
- **Betreuer:** Marc Fehr (m.fehr@fz-juelich.de)
- **Institution:** Forschungszentrum Jülich GmbH
- **Hochschule:** University Mohammed VI Polytechnic
- **Vor-Ort:** Ja (On-site)
- **Praktikumsvertrag (PDF):** /Users/mfehr/Documents/project1.pdf

## Projekt: FlameTrack

- **Repo:** https://github.com/FireDynamics/FlameTrack
- **Lokaler Pfad:** /Users/mfehr/Documents/3)Software/FlameTrack
- **Stack:** Python, PySide6, NumPy/SciPy, OpenCV, HDF5, pytest
- **Docs:** ReadTheDocs (Sphinx)

## Die 6 Assignments (aus dem Praktikumsvertrag)

### Phase 1 – Region-Based Emissivity Tools

1. **Interactive Region Tools** – GUI-basierte Tools (Rechteck, Fenster, Polygon) zum Definieren von ROIs auf dewarped IR-Bildern
2. **Region-Based Emissivity Correction** – Emissivitätswerte pro Region zuweisen und Korrekturen im Processing-Workflow anwenden
3. **ROI Data Extraction & Analysis** – Statistische Kennzahlen und Zeitreihen aus ROIs extrahieren
4. **Integration & Validation** – Alles in bestehende FlameTrack-GUI integrieren, mit synthetischen und echten Daten validieren

### Phase 2 – ML Flame Edge Detection

5. **ML-basierte Flame Edge Detection** – Trainingsdaten vorbereiten, ML-Ansatz (z.B. U-Net, SAM) prototypen und evaluieren (IR + optische Bilder)
6. **Evaluation & Integration** – Klassisch vs. ML vergleichen, beste Methode in FlameTrack-Pipeline integrieren

## Macs Vorbereitungs-To-Dos (in TickTick mit #FlameTrack)

- [x] TickTick-Tasks angelegt (Vorbereitung + Mounia-Zeitplan als Vorlage)
- [ ] Codebase aufräumen (Debug-Kommentare, conftest.py Fix, linting) → bis 2026-05-15
- [ ] Dokumentation & Onboarding verbessern (CONTRIBUTING.md, Docstrings, Architektur) → bis 2026-06-01
- [ ] Tests stabilisieren (main_window Smoke Test, frischer Checkout muss laufen) → bis 2026-06-15
- [ ] Infrastruktur: Beispieldatensatz, Datenzugang, GitHub Issues erstellen → bis 2026-06-15
- [ ] Fachliche Vorbereitung (Emissivitäts-Einführung, ROI-Datenstruktur skizzieren, ML vorrecherchieren) → bis 2026-06-25
- [ ] Übergabe-Tag vorbereiten (README Getting Started, Weekly-Termin) → 2026-07-01

## GitHub Issues (vorbereitet, noch nicht erstellt)

Issues für alle 6 Assignments sind als `gh issue create`-Befehle vorbereitet.
Siehe: noch einzufügen (nächste Session oder nach Erstellung).

## Mounia-Zeitplan (Vorlage – ohne genaue Termine)

Termine werden festgelegt sobald der genaue Starttermin bekannt ist.
Struktur steht bereits in TickTick unter #Mounia.

- Onboarding (ca. 2 Wochen): Umgebung, Codebase, Literatur
- Phase 1 (ca. 6 Wochen): ROI-Tools → Emissivität → Auswertung → Integration
- Phase 2 (ca. 3 Wochen): ML-Recherche → Prototyp → Integration
- Abschluss (ca. 1 Woche): Cleanup, Bericht, Präsentation, PR

## Stand / letzte Session

- Datum: 2026-04-25
- Was besprochen: Praktikumsvertrag analysiert, Vorbereitungsplan für Marc in TickTick strukturiert, Mounia-Zeitplan als Vorlage erstellt, GitHub Issues vorbereitet (gh-Befehle, noch nicht ausgeführt)
- Nächste Schritte: gh issue create Befehle einfügen / ausführen, Terminplanung für Mounia wenn Startdatum fix
