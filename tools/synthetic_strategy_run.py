"""Execute la strategie hybride complete, avec le VRAI binaire VROOM.

Difference avec synthetic_joint_solve.py : ce script ne teste plus une
resolution isolee mais toute la chaine A -> B -> C, telle que le service
l'executerait, y compris la selection finale.

La matrice ORS est remplacee par une matrice synthetique deterministe. Ce
n'est pas une commodite de test : c'est la garantie que ce banc d'essai ne
peut PAS appeler ORS, meme par accident, et que ses mesures sont
reproductibles. Les compteurs d'appels sortants du service sont verifies a
zero en fin de course.

Usage :
    python tools/synthetic_strategy_run.py --output rapport.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_joint_solve import (                              # noqa: E402
    child_peak_rss_kb,
    child_processes,
    haversine_m,
    self_peak_rss_kb,
)

DEPOT = (48.8566, 2.3522)
STEP = 0.004
COLUMNS = 6


def territorial_points(n_tasks=60):
    """Deux blocs adjacents en grille reguliere, separes d'un pas.

    Le nuage a deux grappes de synthetic_joint_solve.py convient a une
    resolution isolee, pas a une strategie TERRITORIALE : il faut ici une
    instance ou une partition connexe existe reellement, sinon le banc
    d'essai mesurerait l'echec de l'instance et non celui du code. Les deux
    blocs se touchent, donc il existe une vraie frontiere a redessiner.
    """
    points = [DEPOT]
    half = n_tasks - n_tasks // 2
    for i in range(n_tasks):
        second = i >= half
        rank = i - half if second else i
        lat = DEPOT[0] + 0.02 + (rank // COLUMNS) * STEP
        lon = DEPOT[1] + (rank % COLUMNS) * STEP
        if second:
            lon += COLUMNS * STEP + STEP
        points.append((lat, lon))
    return points


class SyntheticMatrix:
    """Remplace _build_full_matrix_chunked et compte ses invocations."""

    def __init__(self, points, speed_kmh=28.0):
        speed = speed_kmh * 1000.0 / 3600.0
        size = len(points)
        self.dur = [[0] * size for _ in range(size)]
        self.dist = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                metres = int(round(haversine_m(points[i], points[j])))
                self.dur[i][j] = self.dur[j][i] = int(round(metres / speed))
                self.dist[i][j] = self.dist[j][i] = metres
        self.calls = 0

    def __call__(self, points, headers):
        self.calls += 1
        # 62 x 62 depasse le plafond de routes du palier gratuit : le vrai
        # constructeur decoupe en deux blocs, donc deux appels Matrix.
        meta = {"n": len(points), "calls": 2, "blocks": 2,
                "cached": False, "nulls": 0}
        return self.dur, self.dist, meta, None


def run(n_tasks=60):
    import app                                                   # noqa: PLC0415
    import local_vroom                                           # noqa: PLC0415

    coords = territorial_points(n_tasks)
    points = [{"id": "DEPOT" if i == 0 else "P%02d" % i,
               "lat": lat, "lon": lon}
              for i, (lat, lon) in enumerate(coords)]

    matrix = SyntheticMatrix(coords)
    app._build_full_matrix_chunked = matrix
    app._reset_api_stats()

    config = local_vroom.get_config(refresh=True)
    capacity = app._exact_capacity(n_tasks, 2)

    started = time.monotonic()
    groups, err, meta = app.hybrid_local_vroom_territorial(
        points, 2, capacity, 0, 0, {"Authorization": "dummy"})
    wall_s = time.monotonic() - started

    diag = (meta or {}).get("hybrid") or {}
    routes = (meta or {}).get("hybrid_routes") or []
    sizes = sorted(len(g) for g in groups) if groups else []
    visited = sorted(p for r in routes for p in r[1:-1])
    expected = list(range(1, n_tasks + 1))

    report = {
        "n_tasks": n_tasks,
        "capacity": capacity,
        "error": err,
        "sizes": sizes,
        "expected_sizes": [n_tasks // 2, n_tasks - n_tasks // 2],
        "cardinality_ok": visited == expected,
        "duplicates": len(visited) - len(set(visited)),
        "missing": sorted(set(expected) - set(visited)),
        "routes_start_end_ok": all(r[0] == 0 and r[-1] == 0 for r in routes),
        "components": diag.get("joint_selected_components"),
        "wall_s": round(wall_s, 3),

        # --- appels sortants reels du service ---
        "api_vroom_calls": app._API_STATS["vroom"],
        "api_matrix_calls": app._API_STATS["matrix"],
        "matrix_builder_invocations": matrix.calls,
        "declared_matrix_calls": diag.get("hybrid_matrix_calls"),

        # --- resolutions locales ---
        "solves_attempted": diag.get("local_vroom_attempted"),
        "solves_succeeded": diag.get("local_vroom_succeeded"),
        "solves_failed": diag.get("local_vroom_failed"),
        "solves_timed_out": diag.get("local_vroom_timed_out"),
        "solves_reused": diag.get("local_vroom_reused"),
        "solves_skipped_for_time": diag.get("local_vroom_skipped_for_time"),
        "max_solves": diag.get("local_vroom_max_solves"),
        "vroom_version": diag.get("local_vroom_version"),

        # --- resultats par bloc ---
        "joint_direct_valid": diag.get("joint_direct_valid"),
        "joint_direct_duration_s": diag.get("joint_direct_duration_s"),
        "joint_direct_sizes": diag.get("joint_direct_sizes"),
        "joint_nucleus_attempted": diag.get("joint_nucleus_attempted"),
        "joint_nucleus_valid": diag.get("joint_nucleus_valid"),
        "joint_nucleus_best_duration_s": diag.get("joint_nucleus_best_duration_s"),
        "route_first_cycles": diag.get("route_first_cycles"),
        "route_first_unique": diag.get("route_first_unique"),
        "route_first_best_duration_s": diag.get("route_first_best_duration_s"),
        "joint_alns_iterations": diag.get("joint_alns_iterations"),
        "joint_alns_accepted": diag.get("joint_alns_accepted"),
        "joint_alns_seed": diag.get("joint_alns_seed"),
        "joint_alns_best_duration_s": diag.get("joint_alns_best_duration_s"),
        "joint_finalists": diag.get("joint_finalists"),
        "joint_finalists_local_vroom_solved":
            diag.get("joint_finalists_local_vroom_solved"),
        "joint_finalists_reused": diag.get("joint_finalists_reused"),

        # --- selection ---
        "selected_source": diag.get("joint_selected_source"),
        "selected_duration_s": diag.get("joint_selected_duration_s"),
        "selected_distance_m": diag.get("joint_selected_distance_m"),
        "selected_enclaves": diag.get("joint_selected_enclaves"),
        "solutions_considered": diag.get("joint_solutions_considered"),

        # --- discipline temporelle ---
        "stages": diag.get("hybrid_stages"),
        "total_elapsed_ms": diag.get("total_elapsed_ms"),
        "soft_limit_reached": diag.get("soft_limit_reached"),
        "soft_limit_s": config.total_soft_limit_s,

        # --- ressources ---
        # ATTENTION a la lecture de child_peak_rss_kb : Python cree le
        # subprocess par fork() puis exec(). Le pic de RSS attribue a l'enfant
        # inclut donc la photographie en copie-sur-ecriture du parent, ici un
        # service charge de numpy, scikit-learn et OR-Tools. Ce chiffre suit la
        # taille du PARENT, il ne mesure pas l'empreinte propre de VROOM. La
        # seule mesure decisive est celle du conteneur entier sous 512 Mo.
        "child_peak_rss_kb": child_peak_rss_kb(),
        "child_peak_rss_note": ("inclut la photographie copie-sur-ecriture du "
                                "parent au fork ; ce n'est pas l'empreinte "
                                "propre de vroom"),
        "service_peak_rss_kb": self_peak_rss_kb(),
        "children_after": child_processes(),
        # Le fichier de verrou vit dans ce repertoire par construction et doit
        # y rester : ce qui ne doit PAS survivre, ce sont les repertoires de
        # travail d'une resolution.
        "tmp_after": sorted(name for name in os.listdir(config.tmpdir)
                            if name != "local_vroom.lock")
        if os.path.isdir(config.tmpdir) else [],
    }

    problems = []
    if err:
        problems.append("la strategie a echoue: %s" % err)
    else:
        if sizes != report["expected_sizes"]:
            problems.append("cardinalites %s au lieu de %s"
                            % (sizes, report["expected_sizes"]))
        if not report["cardinality_ok"]:
            problems.append("taches manquantes ou dupliquees")
        if not report["routes_start_end_ok"]:
            problems.append("depart ou arrivee incorrects")
        if report["components"] != [1, 1]:
            problems.append("territoires non connexes: %s" % report["components"])
    if report["api_vroom_calls"]:
        problems.append("%d appel(s) VROOM public" % report["api_vroom_calls"])
    if report["api_matrix_calls"]:
        problems.append("%d appel(s) Matrix reel(s)" % report["api_matrix_calls"])
    if (report["declared_matrix_calls"] or 0) > 2:
        problems.append("%s appels Matrix declares, plus de 2"
                        % report["declared_matrix_calls"])
    if (report["solves_attempted"] or 0) > (report["max_solves"] or 0):
        problems.append("%s resolutions pour un plafond de %s"
                        % (report["solves_attempted"], report["max_solves"]))
    if report["children_after"]:
        problems.append("processus enfants restants: %s" % report["children_after"])
    if report["tmp_after"]:
        problems.append("fichiers temporaires residuels: %s" % report["tmp_after"])

    report["problems"] = problems
    report["ok"] = not problems
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default="default")
    args = parser.parse_args(argv)

    os.environ["LOCAL_VROOM_EXPERIMENT_ENABLED"] = "true"
    os.environ.setdefault("ORS_KEY", "dummy")
    # Repertoire temporaire DEDIE : sans cela le wrapper ecrirait dans le /tmp
    # du systeme et le controle des fichiers residuels y listerait tout ce qui
    # traine, ce qui ne prouverait rien.
    if not os.environ.get("LOCAL_VROOM_TMPDIR"):
        os.environ["LOCAL_VROOM_TMPDIR"] = tempfile.mkdtemp(prefix="lvrun-")

    try:
        report = run(args.tasks)
    except Exception as exc:               # noqa: BLE001 - le rapport doit sortir
        report = {"ok": False, "problems": [repr(exc)]}
    report["label"] = args.label

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")

    if not report.get("ok"):
        for problem in report.get("problems", []):
            print("ECHEC:", problem)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
