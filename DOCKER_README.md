# Docker Setup – AC-CFM-GNN

## Voraussetzungen

| Tool | Zweck |
|---|---|
| Docker Desktop / Docker Engine | Container-Laufzeit |
| docker compose (v2) | Multi-Container-Orchestrierung |
| NVIDIA Container Toolkit *(nur GPU)* | GPU-Zugriff aus Containern |

---

## Dateien in dieses Repo kopieren

Lege `Dockerfile` und `docker-compose.yml` ins **Root-Verzeichnis** des Repos:

```
AC-CFM-GNN/
├── Dockerfile          ← hier ablegen
├── docker-compose.yml  ← hier ablegen
├── configurations/
│   └── local_env.yml
├── src/
├── raw/
└── ...
```

---

## CPU – Quickstart

```bash
# 1. Image bauen (dauert beim ersten Mal ~5–10 min wegen Conda)
docker compose build ac-cfm-cpu

# 2. Code mit aktueller configuration starten
docker compose up ac-cfm-cpu
```

---

## GPU – Quickstart

> Voraussetzung: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) ist installiert.

```bash
# Image bauen (GPU-Variante)
docker compose --profile gpu build ac-cfm-gpu

# Training starten
docker compose --profile gpu up ac-cfm-gpu
```

---

## Nützliche Kommandos

```bash
# In laufenden Container einsteigen (Debugging)
docker compose exec ac-cfm-cpu bash

# Einmaligen Befehl ausführen ohne dauerhaften Container
docker compose run --rm ac-cfm-cpu python src/scripts/normalize.py

# Logs verfolgen
docker compose logs -f ac-cfm-cpu

# Container + Image aufräumen
docker compose down --rmi all
```

---

## Volumes (persistente Daten)

Folgende Ordner werden als **Bind Mounts** ins Container-Dateisystem eingehängt.
Änderungen sind sofort auf dem Host sichtbar und bleiben nach dem Container-Stop erhalten:

| Host-Pfad | Container-Pfad | Inhalt |
|---|---|---|
| `./raw` | `/app/raw` | Rohdaten (.mat-Dateien) |
| `./processed` | `/app/processed` | Verarbeitete (nicht normalisierte) pytorch Dateien |
| `./normalized`| `/app/normalized`| Normalisierte pytorch Dateien
| `./results` | `/app/results` | Trainingsresultate, Plots, Modelle |
| `./configurations` | `/app/configurations` | Konfigurationsdateien |

Die Konfiguration (`configuration.json`) kann also direkt auf dem Host bearbeitet
werden — beim nächsten `docker compose up` wird sie automatisch verwendet.

---

## Build-Argument: USE_GPU

Das Dockerfile enthält ein Build-Argument um CPU/GPU zu steuern:

```bash
# Explizit CPU
docker build --build-arg USE_GPU=0 -t ac-cfm-gnn:cpu .

# Explizit GPU (ersetzt pytorch=*=cpu* in der env.yml)
docker build --build-arg USE_GPU=1 -t ac-cfm-gnn:gpu .
```

---

## Tipps für den Bewerbungsprozess

- **Multi-stage / Build-Args**: Das Dockerfile nutzt `ARG`-basierte Multi-Base-Images – ein gutes Beispiel für flexible, wiederverwendbare Images.
- **Volumes vs. COPY**: Rohdaten werden als Volume eingehängt statt ins Image kopiert → Image bleibt klein und universell.
- **Profiles**: `docker compose --profile gpu` zeigt, wie man Services bedingt aktiviert.
- **Layer Caching**: `local_env.yml` wird *vor* dem restlichen Code kopiert, damit der teure Conda-Install-Layer gecacht bleibt solange sich die Dependencies nicht ändern.
