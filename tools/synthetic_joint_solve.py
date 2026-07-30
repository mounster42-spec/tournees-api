"""Resolution VROOM locale synthetique : 60 taches, 2 vehicules, 30/30.

Ce script est le banc d'essai execute DANS le conteneur. Il ne touche ni le
reseau, ni ORS, ni Render : la matrice est fabriquee sur place a partir de
coordonnees deterministes, donc deux executions produisent exactement la meme
instance et les mesures sont comparables.

Il mesure aussi ce que le Dockerfile ne peut pas prouver seul :
  - le temps reel de resolution ;
  - la memoire maximale du processus VROOM (ru_maxrss des enfants) ;
  - l'absence de processus enfant ou de zombie apres la resolution ;
  - l'absence de fichier temporaire residuel.

Usage :
    python tools/synthetic_joint_solve.py --mode solve   --output rapport.json
    python tools/synthetic_joint_solve.py --mode timeout --output rapport.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_vroom  # noqa: E402

try:
    import resource
except ImportError:                        # pragma: no cover - Windows local
    resource = None


# =========================================================================
# INSTANCE SYNTHETIQUE DETERMINISTE
# =========================================================================
# Un generateur congruentiel explicite plutot que le module random : son
# comportement ne depend d'aucune version de Python, donc l'instance est
# identique sur le runner CI, dans le conteneur et sur un poste local.

def _lcg(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def synthetic_points(n_tasks=60, seed=20260731):
    """Depot en index 0, puis n_tasks points repartis en deux grappes.

    Deux grappes et non un nuage uniforme : une instance sans structure
    territoriale ne dirait rien du probleme reel, ou les deux tournees se
    partagent un terrain geographiquement separable."""
    rand = _lcg(seed)
    depot = (48.8566, 2.3522)              # centre arbitraire, jamais geocode
    points = [depot]
    for i in range(n_tasks):
        cluster_lat = depot[0] + (0.045 if i % 2 == 0 else -0.045)
        cluster_lon = depot[1] + (0.060 if i % 2 == 0 else -0.060)
        points.append((
            cluster_lat + (next(rand) - 0.5) * 0.055,
            cluster_lon + (next(rand) - 0.5) * 0.075,
        ))
    return points


def haversine_m(a, b):
    radius = 6371000.0
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    x = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(x))


def synthetic_matrix(points, speed_kmh=28.0):
    """Matrice de durees entieres, symetrique, diagonale nulle.

    Elle joue le role de la matrice ORS reelle. Les valeurs n'ont pas besoin
    d'etre realistes : ce qui est teste ici est la mecanique d'appel, pas la
    qualite routiere."""
    speed_ms = speed_kmh * 1000.0 / 3600.0
    size = len(points)
    matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            seconds = int(round(haversine_m(points[i], points[j]) / speed_ms))
            matrix[i][j] = seconds
            matrix[j][i] = seconds
    return matrix


# =========================================================================
# OBSERVATION DES PROCESSUS ET DE LA MEMOIRE
# =========================================================================

def child_processes():
    """Enfants directs encore presents, avec leur etat ('Z' = zombie)."""
    if not os.path.isdir("/proc"):         # pragma: no cover - hors Linux
        return []
    me = os.getpid()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry, "rb") as handle:
                raw = handle.read().decode("utf-8", "replace")
        except OSError:
            continue
        # Le nom de commande peut contenir espaces et parentheses : on repart
        # de la DERNIERE parenthese fermante pour lire etat et ppid.
        cut = raw.rfind(")")
        if cut < 0:
            continue
        fields = raw[cut + 2:].split()
        if len(fields) < 2:
            continue
        if int(fields[1]) == me:
            found.append({"pid": int(entry), "state": fields[0]})
    return found


def child_peak_rss_kb():
    if resource is None:                   # pragma: no cover - Windows local
        return None
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def self_peak_rss_kb():
    if resource is None:                   # pragma: no cover - Windows local
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def child_cpu_times():
    if resource is None:                   # pragma: no cover - Windows local
        return (None, None)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (usage.ru_utime, usage.ru_stime)


def tmp_entries(config):
    try:
        return sorted(os.listdir(config.tmpdir))
    except OSError:
        return []


# =========================================================================
# SCENARIOS
# =========================================================================

def run_solve(config, n_tasks, max_tasks):
    points = synthetic_points(n_tasks)
    durations = synthetic_matrix(points)
    job_ids = list(range(1, n_tasks + 1))
    # Le depot est en index 0 ; la tache i est en index i.
    payload = local_vroom.build_joint_payload(
        job_ids=job_ids,
        durations=durations,
        start_index=0,
        end_index=0,
        max_tasks_per_vehicle=max_tasks,
        service_times={jid: 120 for jid in job_ids},
        vehicle_ids=(1, 2),
    )

    ledger = local_vroom.LocalVroomLedger(config=config)
    ledger.plan(1)

    tmp_before = tmp_entries(config)
    wall_start = time.monotonic()
    solution = local_vroom.solve_vroom_local(
        payload,
        timeout_s=config.per_solve_timeout_s,
        ledger=ledger,
        cancellation_deadline=ledger.deadline,
        config=config,
    )
    wall_s = time.monotonic() - wall_start

    sequences = local_vroom.validate_joint_solution(
        solution,
        expected_job_ids=job_ids,
        vehicle_ids=(1, 2),
        max_tasks_per_vehicle=max_tasks,
        start_index=0,
        end_index=0,
    )

    sizes = {str(vid): len(seq) for vid, seq in sorted(sequences.items())}
    visited = [jid for seq in sequences.values() for jid in seq]
    user_s, sys_s = child_cpu_times()

    return {
        "mode": "solve",
        "ok": True,
        "n_tasks": n_tasks,
        "max_tasks_per_vehicle": max_tasks,
        "vehicles": 2,
        "sizes": sizes,
        "cardinality_ok": sorted(visited) == job_ids,
        "duplicates": len(visited) - len(set(visited)),
        "missing": sorted(set(job_ids) - set(visited)),
        "unassigned": len(solution.get("unassigned") or []),
        "routes_non_empty": all(len(seq) > 0 for seq in sequences.values()),
        "vroom_reported_duration_s": (solution.get("summary") or {}).get("duration"),
        "solve_wall_s": round(wall_s, 3),
        "solve_elapsed_ms": solution.get("_local_vroom_elapsed_ms"),
        "child_user_s": None if user_s is None else round(user_s, 3),
        "child_sys_s": None if sys_s is None else round(sys_s, 3),
        "child_peak_rss_kb": child_peak_rss_kb(),
        "self_peak_rss_kb": self_peak_rss_kb(),
        "ledger": ledger.as_diagnostics(),
        "tmp_before": tmp_before,
        "tmp_after": tmp_entries(config),
        "children_after": child_processes(),
    }


def run_timeout(config, n_tasks, max_tasks):
    """Timeout volontaire, avec un binaire de substitution qui engendre un
    enfant puis dort.

    Le binaire VROOM reel resout 60 taches trop vite pour qu'un timeout soit
    reproductible : on ne testerait alors qu'une course. Ce qui doit etre
    prouve ici n'est pas la lenteur de VROOM mais la mise a mort du GROUPE de
    processus, l'absence de zombie et le nettoyage des fichiers temporaires --
    et cela se teste de facon deterministe avec un processus qui refuse de
    finir."""
    stub = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slow_stub.py")
    config.binary = stub
    os.chmod(stub, 0o755)

    points = synthetic_points(n_tasks)
    durations = synthetic_matrix(points)
    job_ids = list(range(1, n_tasks + 1))
    payload = local_vroom.build_joint_payload(
        job_ids=job_ids,
        durations=durations,
        start_index=0,
        end_index=0,
        max_tasks_per_vehicle=max_tasks,
        vehicle_ids=(1, 2),
    )

    ledger = local_vroom.LocalVroomLedger(config=config)
    tmp_before = tmp_entries(config)

    wall_start = time.monotonic()
    error_code = None
    try:
        local_vroom.solve_vroom_local(
            payload,
            timeout_s=2.0,
            ledger=ledger,
            cancellation_deadline=time.monotonic() + 120.0,
            config=config,
        )
    except local_vroom.LocalVroomError as exc:
        error_code = exc.code
    wall_s = time.monotonic() - wall_start

    # Laisser au noyau le temps de faire disparaitre les processus signales
    # avant de conclure qu'il en reste.
    time.sleep(0.5)
    remaining = child_processes()

    return {
        "mode": "timeout",
        "ok": (error_code == local_vroom.ERR_TIMEOUT
               and not remaining
               and tmp_entries(config) == tmp_before),
        "error_code": error_code,
        "expected_error_code": local_vroom.ERR_TIMEOUT,
        "wall_s": round(wall_s, 3),
        "children_after": remaining,
        "zombies_after": [c for c in remaining if c.get("state") == "Z"],
        "tmp_before": tmp_before,
        "tmp_after": tmp_entries(config),
        "tmp_clean": tmp_entries(config) == tmp_before,
        "ledger": ledger.as_diagnostics(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("solve", "timeout"), default="solve")
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--max-tasks", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    # Ce script est un banc d'essai : il active explicitement l'experimentation
    # pour lui-meme, sans rien changer au defaut du service.
    os.environ["LOCAL_VROOM_EXPERIMENT_ENABLED"] = "true"
    config = local_vroom.get_config(refresh=True)

    # Version lue AVANT le scenario : le mode timeout substitue le binaire, et
    # le rapport doit citer la version du vrai VROOM, pas celle du stub.
    real_version = local_vroom.binary_version(config)

    started = time.time()
    try:
        if args.mode == "solve":
            report = run_solve(config, args.tasks, args.max_tasks)
        else:
            report = run_timeout(config, args.tasks, args.max_tasks)
    except local_vroom.LocalVroomError as exc:
        report = {"mode": args.mode, "ok": False,
                  "error_code": exc.code, "error": str(exc)}
    except Exception as exc:               # noqa: BLE001 - le rapport doit sortir
        report = {"mode": args.mode, "ok": False,
                  "error_code": "unexpected", "error": repr(exc)}

    report["vroom_version"] = real_version
    report["total_wall_s"] = round(time.time() - started, 3)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
