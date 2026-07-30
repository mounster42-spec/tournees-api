"""Couche isolee d'appel au binaire VROOM LOCAL, par subprocess.

Ce module ne connait rien des strategies de tournees. Il ne fait qu'une chose :
transformer un payload VROOM en solution VROOM, ou echouer proprement, en
consommant un budget compte.

Deux invariants gouvernent tout le fichier :

1. AUCUN RESEAU. Le binaire est compile avec USE_ROUTING=false : il ne sait pas
   parler a OSRM, a ORS ni a Valhalla. La matrice est toujours fournie par
   l'appelant. Ce module n'importe meme pas `requests`.

2. AUCUNE RESSOURCE ORPHELINE. Tout subprocess lance est attendu ou tue avec
   son groupe de processus, et tout fichier temporaire ecrit est supprime,
   y compris sur timeout, exception ou payload invalide.

Le module reste importable meme quand l'experimentation est desactivee et
meme quand le binaire est absent : app.py doit pouvoir demarrer partout.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

try:                                  # POSIX seulement : absent sous Windows.
    import fcntl
except ImportError:                   # pragma: no cover - chemin Windows local
    fcntl = None


# =========================================================================
# 1. CODES D'ERREUR
# =========================================================================
# Liste fermee : un diagnostic renvoye a Google Sheets doit toujours tomber
# dans l'un de ces cas, jamais dans un message libre.

ERR_DISABLED = "local_vroom_disabled"
ERR_BINARY_MISSING = "local_vroom_binary_missing"
ERR_TIMEOUT = "local_vroom_timeout"
ERR_INVALID_JSON = "local_vroom_invalid_json"
ERR_INVALID_SOLUTION = "local_vroom_invalid_solution"
ERR_RESOURCE_ERROR = "local_vroom_resource_error"
ERR_GLOBAL_TIME_LIMIT = "local_vroom_global_time_limit"
ERR_BUDGET_EXHAUSTED = "local_vroom_budget_exhausted"
ERR_BUSY = "local_vroom_busy"

LOCAL_VROOM_ERROR_CODES = (
    ERR_DISABLED,
    ERR_BINARY_MISSING,
    ERR_TIMEOUT,
    ERR_INVALID_JSON,
    ERR_INVALID_SOLUTION,
    ERR_RESOURCE_ERROR,
    ERR_GLOBAL_TIME_LIMIT,
    ERR_BUDGET_EXHAUSTED,
    ERR_BUSY,
)


class LocalVroomError(Exception):
    """Echec d'une resolution locale, porteur d'un code de la liste fermee."""

    def __init__(self, code, message="", detail=None):
        if code not in LOCAL_VROOM_ERROR_CODES:
            raise ValueError("code d'erreur local_vroom inconnu: %r" % (code,))
        self.code = code
        self.detail = detail
        super().__init__(message or code)


# =========================================================================
# 2. CONFIGURATION
# =========================================================================

def _env_int(name, default, lo, hi):
    """Meme contrat que dans app.py : une valeur illisible ou hors bornes
    retombe sur le defaut prudent plutot que de faire exploser le budget."""
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if lo <= value <= hi else default


def _env_float(name, default, lo, hi):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if lo <= value <= hi else default


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class LocalVroomConfig:
    """Budgets et plafonds, tous surchargeables par variable d'environnement.

    Les defauts sont volontairement prudents : ils sont calibres pour une
    instance Render Free (0,1 CPU, 512 Mo). Monter LOCAL_VROOM_MAX_SOLVES a 16
    et LOCAL_VROOM_FINALIST_SOLVES a 12 ne demande AUCUNE modification de code,
    seulement des variables d'environnement -- c'est la raison d'etre de cette
    classe."""

    def __init__(self):
        self.enabled = _env_flag("LOCAL_VROOM_EXPERIMENT_ENABLED", False)
        self.binary = os.environ.get("LOCAL_VROOM_BINARY", "/usr/local/bin/vroom")
        self.tmpdir = os.environ.get("LOCAL_VROOM_TMPDIR") or tempfile.gettempdir()

        # --- budgets de temps ---
        # Calibres sur la mesure CI sous 512 Mo / 0,10 CPU : une resolution
        # conjointe de 60 taches y prend 4,0 s. Le timeout par resolution est
        # donc porte a 8 s (deux fois la mesure) et le minimum requis pour en
        # DEMARRER une a 9 s, marge d'une seconde comprise.
        self.total_soft_limit_s = _env_float("LOCAL_VROOM_TOTAL_SOFT_LIMIT_S", 58.0, 1.0, 600.0)
        self.per_solve_timeout_s = _env_float("LOCAL_VROOM_PER_SOLVE_TIMEOUT_S", 8.0, 0.2, 300.0)
        self.min_remaining_to_start_s = _env_float("LOCAL_VROOM_MIN_REMAINING_TO_START_S", 9.0, 0.0, 600.0)
        self.route_first_budget_s = _env_float("LOCAL_VROOM_ROUTE_FIRST_BUDGET_S", 3.0, 0.0, 600.0)
        self.alns_budget_s = _env_float("LOCAL_VROOM_ALNS_BUDGET_S", 6.0, 0.0, 600.0)

        # --- budgets de resolutions ---
        # Quatre et non huit : 4 x 8 s de timeout laissent 26 s pour la matrice
        # ORS, route-first et l'ALNS a l'interieur des 58 s. Monter a 8 reste
        # possible par variable d'environnement, mais seulement quand Render
        # aura montre qu'il en a le temps.
        self.max_solves = _env_int("LOCAL_VROOM_MAX_SOLVES", 4, 0, 64)
        self.direct_solves = _env_int("LOCAL_VROOM_DIRECT_SOLVES", 1, 0, 8)
        self.nucleus_solves = _env_int("LOCAL_VROOM_NUCLEUS_SOLVES", 1, 0, 16)
        self.finalist_solves = _env_int("LOCAL_VROOM_FINALIST_SOLVES", 2, 0, 32)
        self.max_concurrent_solves = _env_int("LOCAL_VROOM_MAX_CONCURRENT_SOLVES", 1, 1, 1)

        # --- reglages du binaire ---
        # -t 1 est impose : sur 0,1 CPU, plusieurs threads VROOM se disputent un
        # dixieme de coeur et degradent le temps de reponse au lieu de l'ameliorer.
        self.threads = _env_int("LOCAL_VROOM_THREADS", 1, 1, 4)
        self.explore = _env_int("LOCAL_VROOM_EXPLORE", 5, 0, 5)

        # --- plafonds de securite sur les entrees/sorties ---
        self.max_jobs = _env_int("LOCAL_VROOM_MAX_JOBS", 200, 1, 5000)
        self.max_vehicles = _env_int("LOCAL_VROOM_MAX_VEHICLES", 4, 1, 64)
        self.max_matrix_dim = _env_int("LOCAL_VROOM_MAX_MATRIX_DIM", 256, 2, 4096)
        self.max_payload_bytes = _env_int("LOCAL_VROOM_MAX_PAYLOAD_BYTES", 4 * 1024 * 1024, 1024, 64 * 1024 * 1024)
        self.max_output_bytes = _env_int("LOCAL_VROOM_MAX_OUTPUT_BYTES", 8 * 1024 * 1024, 1024, 64 * 1024 * 1024)
        self.max_stderr_chars = _env_int("LOCAL_VROOM_MAX_STDERR_CHARS", 500, 0, 20000)

        # Delai laisse au processus pour mourir sur SIGTERM avant le SIGKILL.
        self.kill_grace_s = _env_float("LOCAL_VROOM_KILL_GRACE_S", 1.0, 0.05, 30.0)

    def as_diagnostics(self):
        return {
            "local_vroom_enabled": self.enabled,
            "local_vroom_max_solves": self.max_solves,
        }


_CONFIG = None
_CONFIG_LOCK = threading.Lock()


def get_config(refresh=False):
    """Config partagee, relue a la demande. `refresh=True` sert aux tests et a
    toute relecture explicite apres modification de l'environnement."""
    global _CONFIG
    with _CONFIG_LOCK:
        if _CONFIG is None or refresh:
            _CONFIG = LocalVroomConfig()
        return _CONFIG


# =========================================================================
# 3. VERSION DU BINAIRE
# =========================================================================

_VERSION_CACHE = {}


def binary_available(config=None):
    config = config or get_config()
    path = config.binary
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def binary_version(config=None):
    """Retourne la version rapportee par le binaire, ou None s'il est absent.

    Le resultat est memoise : appeler le binaire a chaque diagnostic couterait
    un fork par requete pour une information immuable."""
    config = config or get_config()
    path = config.binary
    if path in _VERSION_CACHE:
        return _VERSION_CACHE[path]
    version = None
    if binary_available(config):
        try:
            completed = subprocess.run(
                [path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            if completed.returncode == 0:
                version = completed.stdout.decode("utf-8", "replace").strip() or None
        except (OSError, subprocess.SubprocessError):
            version = None
    _VERSION_CACHE[path] = version
    return version


# =========================================================================
# 4. VERROU D'EXECUTION, A L'ECHELLE DU CONTENEUR
# =========================================================================
# Un threading.Lock ne protege qu'UN processus. Gunicorn peut en lancer
# plusieurs : le jour ou GUNICORN_WORKERS passe a 2, un verrou en memoire
# laisserait deux optimisations lourdes demarrer cote a cote sur 512 Mo.
# On combine donc les deux niveaux :
#   - un verrou memoire, qui couvre les threads d'un meme worker ;
#   - un verrou fichier flock, qui couvre tous les workers du conteneur et que
#     le noyau libere automatiquement si le processus meurt (y compris sur
#     SIGKILL du timeout Gunicorn).
# Les deux sont non bloquants : une seconde optimisation est refusee tout de
# suite, elle n'attend pas.

_THREAD_LOCKS = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path):
    """Un seul verrou memoire par chemin, partage par tout le processus.

    Sans ce partage, deux objets LocalVroomRunLock construits dans deux threads
    du meme worker auraient chacun leur verrou et ne s'excluraient pas."""
    key = os.path.abspath(path)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


class LocalVroomRunLock:

    def __init__(self, path=None):
        self._path = path or os.path.join(
            os.environ.get("LOCAL_VROOM_TMPDIR") or tempfile.gettempdir(),
            "local_vroom.lock",
        )
        self._thread_lock = _thread_lock_for(self._path)
        self._fd = None
        self._held = False

    @property
    def path(self):
        return self._path

    def acquire(self):
        """Tentative non bloquante. Retourne True si le verrou est pris."""
        if self._held:
            return False
        if not self._thread_lock.acquire(blocking=False):
            return False
        self._held = True
        if fcntl is None:                 # pragma: no cover - Windows local
            return True                   # le verrou memoire suffit hors conteneur
        try:
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:               # deja tenu par un autre worker
                os.close(fd)
                self._held = False
                self._thread_lock.release()
                return False
            self._fd = fd
            return True
        except OSError:
            # Systeme de fichiers non verrouillable : on ne bloque pas le
            # service pour autant, le verrou memoire reste actif.
            self._fd = None
            return True

    def release(self):
        if not self._held:
            return
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self._held = False
        self._thread_lock.release()

    def __enter__(self):
        if not self.acquire():
            raise LocalVroomError(
                ERR_BUSY,
                "une optimisation experimentale est deja en cours sur cette instance",
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


RUN_LOCK = LocalVroomRunLock()


# =========================================================================
# 5. LEDGER
# =========================================================================

class LocalVroomLedger:
    """Compteur unique de toutes les resolutions VROOM locales d'une requete.

    Regle centrale : UNE requete VROOM = UNE resolution, quel que soit le
    nombre de vehicules qu'elle contient. Envoyer deux vehicules dans un seul
    appel coute 1, pas 2.

    Le plafond est verifie AVANT le lancement du subprocess (`can_attempt`),
    pas apres : verifier apres laisserait passer une resolution de trop."""

    def __init__(self, max_solves=None, soft_limit_s=None, started_at=None, config=None):
        config = config or get_config()
        self.max_solves = config.max_solves if max_solves is None else int(max_solves)
        self.soft_limit_s = config.total_soft_limit_s if soft_limit_s is None else float(soft_limit_s)
        self.started_at = time.monotonic() if started_at is None else float(started_at)

        self.planned = 0
        self.attempted = 0
        self.succeeded = 0
        self.failed = 0
        self.timed_out = 0
        self.skipped_for_time = 0
        self.reused = 0
        self.elapsed_ms = 0
        self.last_error = None
        self.stop_reason = None

    # --- horloge globale -------------------------------------------------

    @property
    def deadline(self):
        return self.started_at + self.soft_limit_s

    def remaining_s(self):
        return max(0.0, self.deadline - time.monotonic())

    def soft_limit_reached(self):
        return time.monotonic() >= self.deadline

    # --- budget de resolutions -------------------------------------------

    def plan(self, count):
        self.planned += int(count)
        return self.planned

    def budget_left(self):
        return max(0, self.max_solves - self.attempted)

    def can_attempt(self):
        """Vrai s'il reste a la fois du budget de resolutions ET du temps."""
        if self.budget_left() <= 0:
            return False
        config = get_config()
        return self.remaining_s() >= config.min_remaining_to_start_s

    # --- enregistrement --------------------------------------------------

    def record_attempt(self):
        self.attempted += 1

    def record_success(self, elapsed_ms):
        self.succeeded += 1
        self.elapsed_ms += int(elapsed_ms)

    def record_failure(self, code, elapsed_ms=0):
        self.failed += 1
        self.elapsed_ms += int(elapsed_ms)
        self.last_error = code
        if code == ERR_TIMEOUT:
            self.timed_out += 1

    def record_skip_for_time(self, code=ERR_GLOBAL_TIME_LIMIT):
        self.skipped_for_time += 1
        self.last_error = code
        if self.stop_reason is None:
            self.stop_reason = code

    def record_reuse(self):
        self.reused += 1

    def stop(self, reason):
        if self.stop_reason is None:
            self.stop_reason = reason

    # --- diagnostics -----------------------------------------------------

    def as_diagnostics(self):
        return {
            "local_vroom_planned": self.planned,
            "local_vroom_attempted": self.attempted,
            "local_vroom_succeeded": self.succeeded,
            "local_vroom_failed": self.failed,
            "local_vroom_timed_out": self.timed_out,
            "local_vroom_reused": self.reused,
            "local_vroom_skipped_for_time": self.skipped_for_time,
            "local_vroom_elapsed_ms": self.elapsed_ms,
            "local_vroom_stop_reason": self.stop_reason,
            "local_vroom_last_error": self.last_error,
            "local_vroom_max_solves": self.max_solves,
        }


# =========================================================================
# 6. CONSTRUCTION DU PAYLOAD CONJOINT
# =========================================================================

def build_joint_payload(job_ids, durations, start_index, end_index,
                        max_tasks_per_vehicle, service_times=None,
                        job_location_index=None, vehicle_ids=(1, 2),
                        job_skills=None, vehicle_skills=None,
                        profile="car"):
    """Construit une requete VROOM a deux vehicules sur matrice personnalisee.

    Le plafond `max_tasks_per_vehicle` n'est pas un objectif d'equilibrage :
    c'est une contrainte de capacite. Chaque tache pese 1, chaque vehicule
    porte au plus `max_tasks_per_vehicle`. Avec 60 taches et 2 vehicules a 30,
    la seule affectation admissible est 30/30 ; avec 58 taches et 2 vehicules
    a 29, c'est 29/29. La cardinalite est donc imposee par le modele, pas
    esperee du solveur.

    `job_skills` / `vehicle_skills` servent aux variantes a noyaux : un job
    portant la skill 1 ne peut etre servi que par un vehicule portant aussi la
    skill 1. Les points de frontiere ne portent aucune skill et restent libres.

    Aucune geometrie n'est demandee : `-g` n'est jamais passe au binaire, donc
    VROOM ne calcule ni polyligne ni distance detaillee.
    """
    job_ids = list(job_ids)
    if not job_ids:
        raise LocalVroomError(ERR_INVALID_SOLUTION, "aucune tache a affecter")
    if len(set(job_ids)) != len(job_ids):
        raise LocalVroomError(ERR_INVALID_SOLUTION, "identifiants de taches dupliques")

    n = len(durations)
    for row in durations:
        if len(row) != n:
            raise LocalVroomError(ERR_RESOURCE_ERROR, "matrice de durees non carree")

    if job_location_index is None:
        job_location_index = {jid: i for i, jid in enumerate(job_ids)}

    vehicle_ids = list(vehicle_ids)
    jobs = []
    for jid in job_ids:
        loc = job_location_index[jid]
        if not 0 <= loc < n:
            raise LocalVroomError(ERR_RESOURCE_ERROR,
                                  "index de localisation hors matrice")
        job = {
            "id": int(jid),
            "location_index": int(loc),
            # Chaque tache consomme une unite de la capacite du vehicule :
            # c'est ainsi que le plafond par vehicule est rendu dur.
            "delivery": [1],
        }
        if service_times:
            job["service"] = int(service_times.get(jid, 0))
        if job_skills and jid in job_skills:
            job["skills"] = [int(s) for s in job_skills[jid]]
        jobs.append(job)

    vehicles = []
    for vid in vehicle_ids:
        vehicle = {
            "id": int(vid),
            "profile": profile,
            "start_index": int(start_index),
            "end_index": int(end_index),
            "capacity": [int(max_tasks_per_vehicle)],
        }
        if vehicle_skills and vid in vehicle_skills:
            vehicle["skills"] = [int(s) for s in vehicle_skills[vid]]
        vehicles.append(vehicle)

    return {
        "vehicles": vehicles,
        "jobs": jobs,
        "matrices": {profile: {"durations": [[int(v) for v in row] for row in durations]}},
    }


# =========================================================================
# 7. VALIDATION STRICTE
# =========================================================================

def validate_joint_solution(solution, expected_job_ids, vehicle_ids,
                            max_tasks_per_vehicle=None,
                            start_index=None, end_index=None):
    """Verifie qu'une solution VROOM est utilisable. Toute anomalie leve.

    Une solution partielle n'est jamais "presque bonne" : une tache non
    affectee, un doublon ou une route vide rendent la reponse inexploitable
    pour une tournee reelle. On refuse, on garde l'incumbent, on ne bricole pas.

    Retourne {vehicle_id: [job_id, ...]} dans l'ordre de visite.
    """
    if not isinstance(solution, dict):
        raise LocalVroomError(ERR_INVALID_SOLUTION, "solution non dictionnaire")

    code = solution.get("code")
    if code not in (0, None):
        raise LocalVroomError(ERR_INVALID_SOLUTION,
                              "vroom a retourne le code %r" % (code,))

    unassigned = solution.get("unassigned") or []
    if unassigned:
        raise LocalVroomError(
            ERR_INVALID_SOLUTION,
            "%d tache(s) non affectee(s)" % len(unassigned),
            detail={"unassigned": len(unassigned)},
        )

    routes = solution.get("routes")
    if not isinstance(routes, list) or not routes:
        raise LocalVroomError(ERR_INVALID_SOLUTION, "aucune route dans la solution")

    expected = list(expected_job_ids)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise LocalVroomError(ERR_INVALID_SOLUTION, "taches attendues dupliquees")

    wanted_vehicles = [int(v) for v in vehicle_ids]
    if len(routes) != len(wanted_vehicles):
        raise LocalVroomError(
            ERR_INVALID_SOLUTION,
            "%d route(s) pour %d vehicule(s)" % (len(routes), len(wanted_vehicles)),
        )

    sequences = {}
    seen = []
    for route in routes:
        if not isinstance(route, dict):
            raise LocalVroomError(ERR_INVALID_SOLUTION, "route non dictionnaire")
        vid = route.get("vehicle")
        if vid is None or int(vid) not in wanted_vehicles:
            raise LocalVroomError(ERR_INVALID_SOLUTION,
                                  "vehicule inattendu: %r" % (vid,))
        vid = int(vid)
        if vid in sequences:
            raise LocalVroomError(ERR_INVALID_SOLUTION,
                                  "vehicule %d present deux fois" % vid)

        steps = route.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            raise LocalVroomError(ERR_INVALID_SOLUTION,
                                  "etapes manquantes sur le vehicule %d" % vid)
        if steps[0].get("type") != "start":
            raise LocalVroomError(ERR_INVALID_SOLUTION,
                                  "le vehicule %d ne demarre pas au depart" % vid)
        if steps[-1].get("type") != "end":
            raise LocalVroomError(ERR_INVALID_SOLUTION,
                                  "le vehicule %d ne finit pas a l'arrivee" % vid)
        if start_index is not None and steps[0].get("location_index") not in (None, start_index):
            raise LocalVroomError(ERR_INVALID_SOLUTION,
                                  "depart incorrect sur le vehicule %d" % vid)
        if end_index is not None and steps[-1].get("location_index") not in (None, end_index):
            raise LocalVroomError(ERR_INVALID_SOLUTION,
                                  "arrivee incorrecte sur le vehicule %d" % vid)

        sequence = []
        for step in steps:
            if step.get("type") != "job":
                continue
            jid = step.get("id")
            if jid is None:
                jid = step.get("job")
            if jid is None:
                raise LocalVroomError(ERR_INVALID_SOLUTION, "etape job sans identifiant")
            sequence.append(int(jid))

        if not sequence:
            raise LocalVroomError(ERR_INVALID_SOLUTION,
                                  "route vide sur le vehicule %d" % vid)
        if max_tasks_per_vehicle is not None and len(sequence) > int(max_tasks_per_vehicle):
            raise LocalVroomError(
                ERR_INVALID_SOLUTION,
                "vehicule %d : %d taches pour un plafond de %d"
                % (vid, len(sequence), int(max_tasks_per_vehicle)),
            )

        sequences[vid] = sequence
        seen.extend(sequence)

    if len(seen) != len(set(seen)):
        raise LocalVroomError(ERR_INVALID_SOLUTION, "une tache apparait plusieurs fois")
    if set(seen) != expected_set:
        missing = sorted(expected_set - set(seen))
        extra = sorted(set(seen) - expected_set)
        raise LocalVroomError(
            ERR_INVALID_SOLUTION,
            "union incomplete: %d manquante(s), %d inconnue(s)" % (len(missing), len(extra)),
            detail={"missing": missing[:10], "extra": extra[:10]},
        )

    return sequences


# =========================================================================
# 8. NETTOYAGE DES MESSAGES
# =========================================================================

_SECRET_HINTS = ("api_key", "apikey", "authorization", "ors_key", "token", "bearer")


def _scrub(text, max_chars):
    """Tronque stderr et neutralise toute ligne susceptible de porter un secret.

    Le binaire local ne recoit aucune cle, mais une regle de journalisation ne
    doit pas dependre de cette hypothese."""
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        low = line.lower()
        if any(hint in low for hint in _SECRET_HINTS):
            lines.append("[redacted]")
        else:
            lines.append(line)
    cleaned = "\n".join(lines).strip()
    if max_chars >= 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "...[tronque]"
    return cleaned


# =========================================================================
# 9. ARRET DU SUBPROCESS ET DE SES ENFANTS
# =========================================================================

def _terminate_tree(process, grace_s):
    """Tue le processus ET tous ses descendants, puis attend leur disparition.

    `start_new_session=True` place VROOM dans son propre groupe de processus :
    on peut donc signaler le groupe entier, ce qu'un simple process.kill() ne
    ferait pas. Sans cela, un timeout laisserait des enfants vivants qui
    continueraient a consommer le CPU deja rare de l'instance."""
    if process.poll() is not None:
        return
    killed_group = False
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            killed_group = True
        except (OSError, ProcessLookupError):
            killed_group = False
    if not killed_group:
        try:
            process.terminate()
        except OSError:
            pass

    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
    else:                                  # pragma: no cover - Windows local
        try:
            process.kill()
        except OSError:
            pass

    try:
        # Ce wait final est ce qui evite les zombies : sans lui, le processus
        # reste dans la table des processus jusqu'a la mort du worker.
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass


# =========================================================================
# 10. RESOLUTION LOCALE
# =========================================================================

def solve_vroom_local(payload, timeout_s=None, ledger=None,
                      cancellation_deadline=None, config=None):
    """Execute une resolution VROOM locale et retourne la solution parsee.

    Ordre des verifications, volontairement du moins cher au plus cher :
      1. experimentation activee ;
      2. binaire present et executable ;
      3. budget de resolutions restant (AVANT tout fork) ;
      4. temps global restant suffisant ;
      5. taille et forme du payload ;
      6. et seulement alors, le subprocess.

    Leve LocalVroomError en cas d'echec. L'appelant conserve son incumbent :
    ce module ne fabrique jamais de solution de remplacement.
    """
    config = config or get_config()

    if not config.enabled:
        if ledger is not None:
            ledger.stop(ERR_DISABLED)
            ledger.last_error = ERR_DISABLED
        raise LocalVroomError(ERR_DISABLED,
                              "LOCAL_VROOM_EXPERIMENT_ENABLED n'est pas actif")

    if not binary_available(config):
        if ledger is not None:
            ledger.record_failure(ERR_BINARY_MISSING)
            ledger.stop(ERR_BINARY_MISSING)
        raise LocalVroomError(ERR_BINARY_MISSING,
                              "binaire vroom introuvable ou non executable")

    # --- budget de resolutions, verifie AVANT le fork --------------------
    if ledger is not None and ledger.budget_left() <= 0:
        ledger.stop(ERR_BUDGET_EXHAUSTED)
        ledger.last_error = ERR_BUDGET_EXHAUSTED
        raise LocalVroomError(
            ERR_BUDGET_EXHAUSTED,
            "plafond de %d resolutions atteint" % ledger.max_solves,
        )

    # --- temps global restant --------------------------------------------
    remaining = None
    if cancellation_deadline is not None:
        remaining = cancellation_deadline - time.monotonic()
    elif ledger is not None:
        remaining = ledger.remaining_s()

    if remaining is not None and remaining < config.min_remaining_to_start_s:
        if ledger is not None:
            ledger.record_skip_for_time(ERR_GLOBAL_TIME_LIMIT)
        raise LocalVroomError(
            ERR_GLOBAL_TIME_LIMIT,
            "il reste %.1fs, moins que le minimum de %.1fs pour demarrer"
            % (max(0.0, remaining), config.min_remaining_to_start_s),
        )

    # --- forme et taille du payload --------------------------------------
    hard_timeout = float(timeout_s if timeout_s is not None else config.per_solve_timeout_s)
    if remaining is not None:
        hard_timeout = min(hard_timeout, max(0.1, remaining))

    _check_payload_limits(payload, config)
    try:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LocalVroomError(ERR_RESOURCE_ERROR,
                              "payload non serialisable: %s" % exc) from exc
    if len(encoded) > config.max_payload_bytes:
        raise LocalVroomError(
            ERR_RESOURCE_ERROR,
            "payload de %d octets au-dela du plafond de %d"
            % (len(encoded), config.max_payload_bytes),
        )

    if ledger is not None:
        ledger.record_attempt()

    started = time.monotonic()
    workdir = None
    process = None
    try:
        os.makedirs(config.tmpdir, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="vroom-", dir=config.tmpdir)
        input_path = os.path.join(workdir, "input.json")
        output_path = os.path.join(workdir, "output.json")
        with open(input_path, "wb") as handle:
            handle.write(encoded)

        # -l laisse VROOM s'arreter tout seul un peu avant notre couperet :
        # une solution rendue vaut mieux qu'un processus tue.
        internal_limit = max(0.2, hard_timeout - 1.0)
        command = [
            config.binary,
            "-i", input_path,
            "-o", output_path,
            "-t", str(config.threads),
            "-x", str(config.explore),
            "-l", "%.3f" % internal_limit,
        ]

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "cwd": workdir,
            # Environnement minimal : le binaire n'a besoin d'aucune variable
            # du service, et surtout pas de ORS_KEY.
            "env": {"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C"},
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        # shell=False est le defaut de Popen et la commande est une LISTE :
        # aucun argument ne peut etre reinterprete par un shell.
        process = subprocess.Popen(command, **popen_kwargs)

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=hard_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_tree(process, config.kill_grace_s)
            try:
                stdout, stderr = process.communicate(timeout=config.kill_grace_s)
            except (subprocess.TimeoutExpired, ValueError):
                stdout, stderr = b"", b""

        elapsed_ms = int((time.monotonic() - started) * 1000)
        stderr_text = _scrub(stderr.decode("utf-8", "replace") if stderr else "",
                             config.max_stderr_chars)

        if timed_out:
            if ledger is not None:
                ledger.record_failure(ERR_TIMEOUT, elapsed_ms)
            raise LocalVroomError(
                ERR_TIMEOUT,
                "vroom tue apres %.1fs" % hard_timeout,
                detail={"stderr": stderr_text, "elapsed_ms": elapsed_ms},
            )

        if process.returncode != 0:
            if ledger is not None:
                ledger.record_failure(ERR_RESOURCE_ERROR, elapsed_ms)
            raise LocalVroomError(
                ERR_RESOURCE_ERROR,
                "vroom a termine avec le code %s" % process.returncode,
                detail={"stderr": stderr_text, "elapsed_ms": elapsed_ms},
            )

        raw = _read_output(output_path, stdout, config)
        try:
            solution = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            if ledger is not None:
                ledger.record_failure(ERR_INVALID_JSON, elapsed_ms)
            raise LocalVroomError(
                ERR_INVALID_JSON,
                "sortie vroom illisible: %s" % exc,
                detail={"stderr": stderr_text},
            ) from exc

        if not isinstance(solution, dict):
            if ledger is not None:
                ledger.record_failure(ERR_INVALID_JSON, elapsed_ms)
            raise LocalVroomError(ERR_INVALID_JSON, "sortie vroom non dictionnaire")

        if ledger is not None:
            ledger.record_success(elapsed_ms)
        solution["_local_vroom_elapsed_ms"] = elapsed_ms
        return solution

    except LocalVroomError:
        raise
    except OSError as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if ledger is not None:
            ledger.record_failure(ERR_RESOURCE_ERROR, elapsed_ms)
        raise LocalVroomError(ERR_RESOURCE_ERROR,
                              "echec systeme: %s" % exc) from exc
    finally:
        # Aucun chemin de sortie ne laisse de processus ni de fichier derriere
        # lui : ni le succes, ni le timeout, ni une exception inattendue.
        if process is not None and process.poll() is None:
            _terminate_tree(process, config.kill_grace_s)
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)


def _check_payload_limits(payload, config):
    if not isinstance(payload, dict):
        raise LocalVroomError(ERR_RESOURCE_ERROR, "payload non dictionnaire")

    jobs = payload.get("jobs") or []
    vehicles = payload.get("vehicles") or []
    if len(jobs) > config.max_jobs:
        raise LocalVroomError(
            ERR_RESOURCE_ERROR,
            "%d taches au-dela du plafond de %d" % (len(jobs), config.max_jobs),
        )
    if not vehicles:
        raise LocalVroomError(ERR_RESOURCE_ERROR, "payload sans vehicule")
    if len(vehicles) > config.max_vehicles:
        raise LocalVroomError(
            ERR_RESOURCE_ERROR,
            "%d vehicules au-dela du plafond de %d" % (len(vehicles), config.max_vehicles),
        )

    matrices = payload.get("matrices") or {}
    for name, matrix in matrices.items():
        durations = matrix.get("durations") or []
        if len(durations) > config.max_matrix_dim:
            raise LocalVroomError(
                ERR_RESOURCE_ERROR,
                "matrice %s de dimension %d au-dela du plafond de %d"
                % (name, len(durations), config.max_matrix_dim),
            )


def _read_output(output_path, stdout, config):
    """Lit la solution, en preferant le fichier -o au flux stdout."""
    if os.path.isfile(output_path):
        size = os.path.getsize(output_path)
        if size > config.max_output_bytes:
            raise LocalVroomError(
                ERR_RESOURCE_ERROR,
                "sortie de %d octets au-dela du plafond de %d"
                % (size, config.max_output_bytes),
            )
        with open(output_path, "rb") as handle:
            return handle.read()

    if stdout is None:
        raise LocalVroomError(ERR_INVALID_JSON, "aucune sortie vroom")
    if len(stdout) > config.max_output_bytes:
        raise LocalVroomError(
            ERR_RESOURCE_ERROR,
            "sortie de %d octets au-dela du plafond de %d"
            % (len(stdout), config.max_output_bytes),
        )
    return stdout


# =========================================================================
# 11. DIAGNOSTICS
# =========================================================================

def diagnostics(ledger=None, config=None):
    """Bloc de diagnostic pour la fin du Benchmark Google Sheets."""
    config = config or get_config()
    out = {
        "local_vroom_enabled": config.enabled,
        "local_vroom_version": binary_version(config),
        "local_vroom_binary_present": binary_available(config),
        "local_vroom_max_solves": config.max_solves,
    }
    if ledger is not None:
        out.update(ledger.as_diagnostics())
    return out


def healthz():
    """Payload de l'endpoint /healthz. Volontairement sans appel systeme
    couteux : il ne doit jamais forker ni toucher au reseau."""
    config = get_config()
    return {
        "status": "ok",
        "python": "%d.%d.%d" % sys.version_info[:3],
        "local_vroom_enabled": config.enabled,
        "local_vroom_binary_present": binary_available(config),
    }
