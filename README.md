# EDAN 2025 Election Data Chat

Application de chat sur les données électorales EDAN 2025 (résultats nationaux détaillés).

- Interface Streamlit.
- Orchestration LangGraph.
- LLM DeepSeek via l'API compatible OpenAI.
- Base analytique DuckDB.
- Routeur hybride : pré-routage déterministe (regex) + décision LLM.
- Résolution d'entités avec fuzzy matching (fautes, alias, accents).
- Mémoire conversationnelle structurée avec reformulation autonome des questions de suivi.
- RAG hybride vecteur + mots-clés (sentence-transformers, 60/40).
- Validation SQL avant exécution (guardrails).
- Tableaux et graphiques Plotly interactifs.
- Réponses toujours dans la langue de l'utilisateur.

---

## Prérequis

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le projet cible Python 3.12.7. Utilisez un environnement virtuel dédié : les
paquets d'un environnement Python global peuvent être incompatibles entre eux.

Configurer l'environnement :

```powershell
copy .env.example .env
```

Éditer `.env` :

```text
DEEPSEEK_API_KEY=your_key_here
EDAN_DUCKDB_PATH=edan_2025_resultat_national_details.duckdb
EDAN_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

---

## Construction de la base DuckDB

Un seul script à lancer, depuis la racine du projet :

```powershell
python build_db.py docs/EDAN_2025_RESULTAT_NATIONAL_DETAILS.pdf
```

Ce script orchestre 4 étapes dans l'ordre :

| Étape | Script | Rôle |
|---|---|---|
| 1 | `db_builders/step1_extract_pdf.py` | Extraction PDF → tables DuckDB + RAG chunks |
| 2 | `db_builders/step2_fix_views.py` | Création de `vw_turnout_by_region` avec `region_norm` |
| 3 | `db_builders/step3_build_aliases.py` | 1 800+ alias d'entités (partis, régions, circonscriptions) |
| 4 | `db_builders/step4_build_embeddings.py` | Embeddings vectoriels 384-dim sur les RAG chunks |

Chaque étape peut aussi être lancée individuellement (utile pour une mise à jour partielle) :

```powershell
python db_builders/step4_build_embeddings.py edan_2025_resultat_national_details.duckdb
```

> **Note** : l'étape 4 télécharge le modèle `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) au premier lancement. Les lancements suivants utilisent le cache local.

---

## Lancer l'application

```powershell
streamlit run app.py
```

> Si DuckDB est déjà ouvert dans DBeaver ou un autre outil, fermer la connexion ou pointer `EDAN_DUCKDB_PATH` vers une copie.

### Version du dataset

La construction enregistre dans `dataset_versions` le SHA-256 du PDF, les
versions de schéma/chunking, le modèle d'embedding et les volumes de données.

```powershell
python -m ai_engineer_app.dataset_version
```

### Observabilité Langfuse

Les exécutions de l'agent sont tracées exclusivement dans Langfuse. Aucune base
SQLite locale d'observabilité n'est créée.

Chaque trace contient les identifiants de trace, session et utilisateur
anonymisé, les versions chatbot/prompt/dataset, les nœuds LangGraph, les
routes, les appels DeepSeek, les tokens, les coûts, les erreurs, le RAG, le SQL,
les graphiques et le feedback utilisateur.

Les prompts et réponses sont représentés par empreinte et longueur par défaut.
Le contenu brut n'est envoyé à Langfuse que si `EDAN_LANGFUSE_CAPTURE_CONTENT=true`.

Le dashboard Langfuse remplace l'ancien dashboard local et expose les traces,
sessions, coûts, latences, erreurs, feedbacks et scores.

TTFT, TPOT, files d'attente, préemptions et saturation GPU ne sont pas
mesurables avec l'intégration DeepSeek non-streaming et hébergée. Ces champs
sont explicitement marqués indisponibles au lieu d'être estimés.

### Versionnement des prompts

Les prompts système sont enregistrés dans un registre central avec un nom,
une version immuable et une empreinte SHA-256. Toute modification doit
incrémenter `EDAN_PROMPT_VERSION`; cette version participe aussi à
l'invalidation du cache.

### Évaluation et feedback

La suite combine contrôles déterministes, exactitude factuelle, agrégations,
SQL, retrieval, citations, fidélité RAG, cohérence conversationnelle,
sécurité, pertinence, complétude, concision, coût et latence. Le juge LLM
optionnel utilise une grille explicite multi-critères et ne remplace jamais le
verdict déterministe.

Les évaluations sont exécutées comme des experiments Langfuse. Les cas de
`evals/test_cases.json` sont synchronisés dans un dataset Langfuse, chaque
lancement crée un dataset run, chaque cas produit une trace et des scores natifs
(`score`, `pass`, `latency_ms`, `category`, `case_id`). Aucun rapport
JSON/CSV/JSONL local n'est généré.

```powershell
.venv\Scripts\python.exe evals\run_evals.py --langfuse-run-name edan-eval-main-v1
```

Les boutons pouce haut/bas de l'application alimentent les scores de feedback
Langfuse.

---

## Fonctionnalités

- Agrégations, classements, graphiques sur les candidats, partis, régions.
- Fuzzy matching : « Tiapum » → Tiapoum, « R.H.D.P » → RHDP, « Agnebi Tiassa » → AGNEBY-TIASSA.
- Réponses narratives via RAG hybride (60 % embeddings + 40 % mots-clés).
- Détection des prompts hors périmètre et des tentatives adversariales.
- Réponses dans la langue de l'utilisateur (FR, EN, ES, AR, PT, …).

## Vues SQL exposées à l'agent

| Vue | Contenu |
|---|---|
| `vw_results_clean` | Vue principale — un rang par candidat/liste |
| `vw_winners` | Élus uniquement (`elu = TRUE`) |
| `vw_turnout_by_region` | Taux de participation agrégé par région, avec `region_norm` |
| `vw_national_summary` | Totaux nationaux (inscrits, votants, taux de participation) |

---

## Architecture du pipeline

```
Question utilisateur
        │
        ▼
┌───────────────────┐     adversarial / greeting
│  Pré-détection    │────────────────────────────► Réponse immédiate
│  (regex statique) │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Résolution entités│  fuzzy matching alias → entités canoniques
│ + contextualisation│  reformulation si question de suivi
└────────┬──────────┘
         │
         ├── Pré-routage regex ──► RAG ou SQL déterministe
         │
         ▼
┌───────────────────┐
│  LLM intent router│  deepseek-chat → intent JSON
│  (fallback LLM)   │  { intent, sql, chart_type, searched }
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
   SQL       RAG
    │         │
    ▼         ▼
Validation  retrieve_chunks
+ guardrails (vector 60 % + keyword 40 %)
    │         │
    ▼         ▼
 DuckDB    Narration
 execute     LLM
    │         │
    └────┬────┘
         │
         ▼
  Génération réponse (+ graphique si demandé)
         │
         ▼
  Mise à jour mémoire conversationnelle
```

---

## Schéma de la base de données

### Décisions de conception

| Décision | Choix | Justification |
|---|---|---|
| Moteur | DuckDB | Analytique in-process, aucun serveur, LIMIT/timeout natifs |
| Tables normalisées | `circonscriptions` + `candidats` (clé `circonscription_code`) | Évite la duplication des métriques de participation par candidat |
| Colonnes `_norm` | `region_norm`, `groupement_parti_norm`, `candidat_liste_norm` | Comparaisons insensibles à la casse et aux accents |
| RAG séparé | Table `rag_chunks` avec colonne `embedding FLOAT[384]` | Permet la recherche vectorielle sans ORM externe |
| Observabilité | Langfuse | Traces, sessions, coûts, feedbacks et évaluations centralisés hors stockage local |

### Tables

**`circonscriptions`** — une ligne par circonscription (205 lignes)

| Colonne | Type | Description |
|---|---|---|
| `circonscription_code` | VARCHAR PK | Code unique (ex. `CI-01-001`) |
| `region` / `region_norm` | VARCHAR | Région brute et normalisée (UPPER ASCII) |
| `circonscription` / `circonscription_norm` | VARCHAR | Nom brut et normalisé |
| `nb_bv` | INTEGER | Nombre de bureaux de vote |
| `inscrits` | INTEGER | Électeurs inscrits |
| `votants` | INTEGER | Votes exprimés + nuls + blancs |
| `taux_participation_pct` | DOUBLE | Taux de participation (%) |
| `bulletins_nuls` / `suffrages_exprimes` / `bulletins_blancs_*` | INTEGER/DOUBLE | Détail des suffrages |
| `source_page_start` / `source_page_end` | INTEGER | Pages PDF source |

**`candidats`** — une ligne par liste/candidat (1 125 lignes)

| Colonne | Type | Description |
|---|---|---|
| `candidat_id` | INTEGER PK | Identifiant auto |
| `circonscription_code` | VARCHAR FK | Lien vers `circonscriptions` |
| `groupement_parti` / `groupement_parti_norm` | VARCHAR | Parti brut et normalisé |
| `candidat_liste` / `candidat_liste_norm` | VARCHAR | Nom brut et normalisé |
| `scores` | INTEGER | Voix obtenues |
| `score_pct` | DOUBLE | Part des suffrages exprimés (%) |
| `elu` | BOOLEAN | Élu(e) = TRUE |
| `page` | INTEGER | Page PDF source |

**`entity_aliases`** — table des alias pour la résolution floue (1 800+ entrées)

| Colonne | Description |
|---|---|
| `alias_norm` | Forme normalisée de l'alias (UPPER ASCII) |
| `canonical_value` | Valeur officielle (ex. `AGNEBY-TIASSA`) |
| `entity_type` | `region` \| `circonscription` \| `parti` |

---

## Normalisation des entités

La normalisation est appliquée à deux niveaux :

### 1. Normalisation statique (ingestion)

```python
def normalize(text: str) -> str:
    # 1. Décomposition unicode → ASCII (supprime accents)
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode()
    # 2. Majuscules
    return text.upper().strip()
```

Appliquée sur : `region`, `circonscription`, `groupement_parti`, `candidat_liste` → colonnes `*_norm`.

### 2. Résolution dynamique (requête)

Le module `ai_engineer_app/entity_resolver.py` :

1. **Extraction de n-grammes** depuis la question de l'utilisateur
2. **Correspondance floue** (`difflib.SequenceMatcher`) contre `entity_aliases.alias_norm`
   - Seuil configurable : `ENTITY_SIMILARITY_THRESHOLD=0.80` (par défaut)
3. **Désambiguïsation** si plusieurs entités correspondent (Level 3)
4. **Contexte injecté** dans le prompt SQL : `-- Région : AGNEBY-TIASSA (alias de "Agnebi tiassa")`

**Exemples couverts :**

| Saisie utilisateur | Entité résolue |
|---|---|
| `Tiapum` | `TIAPOUM` (circonscription) |
| `Agnebi Tiassa` | `AGNEBY-TIASSA` (région) |
| `R.H.D.P` | `RHDP` (parti) |
| `rhdp` | `RHDP` (parti) |
| `Grand Bassam` | `GRAND-BASSAM` (circonscription) |

---

## Routage et garde-fous

### Pipeline de routage (ordre de priorité)

| Priorité | Mécanisme | Déclencheur |
|---|---|---|
| 1 | Détection adversariale (regex) | `DROP TABLE`, `ignore tes règles`, `clé API`… |
| 2 | Détection salutation (regex) | `bonjour`, `hello`… |
| 3 | Pré-routage SQL (regex) | `combien`, `top`, `histogramme`, `taux`… |
| 4 | Pré-routage RAG (regex) | `décris`, `explique`, `résume`… |
| 5 | Décision LLM (deepseek-chat) | Intent JSON → `aggregation \| ranking \| chart \| factual \| rag_narrative \| out_of_scope` |

### Garde-fous SQL (`ai_engineer_app/sql_guardrails.py`)

| Règle | Détail |
|---|---|
| SELECT uniquement | `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE` → rejet |
| Tables autorisées | `circonscriptions`, `candidats`, `rag_chunks`, `entity_aliases`, `vw_*` |
| LIMIT obligatoire | Ajouté si absent (`SQL_DEFAULT_LIMIT=100`) ; réduit si supérieur à 500 |
| Timeout DuckDB | `QUERY_TIMEOUT_SECONDS=20` (configurable) |
| Pas de UNION/CROSS JOIN | Bloqués pour éviter les requêtes à résultat non borné |
| Pas d'injection SQL | Colonnes/tables validées contre le schéma avant exécution |

---

## Suite d'évaluation

30 cas de test répartis en 10 catégories :

| Catégorie | N | Ce qui est testé |
|---|---|---|
| `routing` | 5 | SQL vs RAG, pré-routage déterministe |
| `security` | 5 | Résistance aux prompts adversariaux |
| `facts` | 3 | Précision des réponses factuelles |
| `aggregation` | 3 | Exactitude numérique (± tolérance) |
| `sql` | 3 | Validité du SQL généré |
| `fidelity` | 3 | Absence de hallucinations (règles déterministes) |
| `retrieval` | 2 | RAG : chunks pertinents récupérés |
| `citation` | 1 | Réponse ancrée dans les chunks |
| `conversation` | 2 | Mémoire multi-tours |
| `latency` | 3 | Réponse < 12 s |

### Lancer les évaluations

```powershell
# Suite complète (34 cas)
python evals/run_evals.py

# Une seule catégorie
python evals/run_evals.py --category routing

# Un seul cas
python evals/run_evals.py --id FA_03

# Les résultats sont affichés dans la page Evaluation de l'application
```

---

## Intégration continue

Le workflow `.github/workflows/ci.yml` comprend trois jobs :

| Job | Déclencheur | Durée estimée |
|---|---|---|
| `lint` — ruff check + format | Push / PR | ~15 s |
| `test` — pytest tests/ | Push / PR (après lint) | ~30 s |
| `evals` — suite 30 cas | `workflow_dispatch` manuel uniquement | ~5 min |

Le job `evals` nécessite le secret `DEEPSEEK_API_KEY` configuré dans les paramètres du dépôt GitHub et le flag `run_evals: true` à l'exécution manuelle.

---

## Limites connues et prochaines étapes

### Limites actuelles

| Limitation | Détail |
|---|---|
| LLM non déterministe | L'agent peut générer un SQL légèrement différent à chaque appel ; les résultats peuvent varier pour les requêtes complexes |
| Listing des partis tronqué | Avec une question ouverte « quels partis ont participé ? », le LLM peut ajouter spontanément `LIMIT 30` → 43 partis ne sont pas tous listés en une réponse |
| Corpus RAG tabulaire uniquement | Les chunks RAG sont des lignes converties en texte ; il n'existe pas de texte narratif dans le PDF source pour les questions de contexte électoral général |
| Pas de streaming | Les réponses sont retournées en bloc (pas de streaming token par token) |
| Langue de l'interface | L'UI Streamlit est en français ; l'agent répond dans la langue de l'utilisateur (FR/EN/ES/AR/PT) |
| Modèle local offline uniquement | Le modèle d'embedding (`paraphrase-multilingual-MiniLM-L12-v2`) doit être téléchargé une première fois (~470 MB) |

### Prochaines étapes envisagées

- **Streaming LLM** — affichage progressif de la réponse dans Streamlit
- **Export PDF du rapport d'évaluation** — pour faciliter la revue périodique
- **Compression auto du contexte** — pour les conversations longues dépassant la fenêtre de contexte du LLM
- **Tests de régression automatiques en CI** — activer le job `evals` sur chaque push avec un cache des résultats LLM
