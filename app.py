from flask import Flask, request, jsonify
import hashlib
import json
import math
import os
import time
import numpy as np
import requests
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# OR-Tools est optionnel. Son absence (echec de build Render, wheel trop
# lourde) ne doit jamais empecher le demarrage ni degrader la strategie
# kmeans : elle desactive seulement les strategies ortools_*, qui repondent
# alors 501 au lieu de retomber silencieusement sur K-Means.
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

app = Flask(__name__)


# =========================
# 1. HAVERSINE
# =========================
def haversine(a, b):
    R = 6371
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(x))


# =========================
# 2. ORS / VROOM CONFIG
# =========================
ORS_KEY = os.environ.get("ORS_KEY", "")
ORS_VROOM_URL = "https://api.heigit.org/vroom/v0"
ORS_MATRIX_URL = "https://api.heigit.org/openrouteservice/v2/matrix/driving-car"


# =========================
# 2b. INSTRUMENTATION (compteur d'appels API + strategies)
# =========================
VALID_STRATEGIES = ("kmeans", "ortools_haversine", "ortools_ors_matrix",
                    "ortools_ors_matrix_connected")
IMPLEMENTED_STRATEGIES = {"kmeans"}
if ORTOOLS_AVAILABLE:
    IMPLEMENTED_STRATEGIES.add("ortools_haversine")
    IMPLEMENTED_STRATEGIES.add("ortools_ors_matrix")
    IMPLEMENTED_STRATEGIES.add("ortools_ors_matrix_connected")

# Matrice ORS complete (strategie ortools_ors_matrix).
# 62 x 62 = 3844 routes depasse le plafond free tier : on decoupe par blocs
# de sources. Les n locations sont envoyees a chaque appel, sources et
# destinations servant d'index dans ce tableau.
ORS_MATRIX_MAX_ROUTES = 3500      # plafond free tier : sources x destinations
ORS_MATRIX_MAX_LOCATIONS = 0      # 0 = aucun plafond connu sur le nb de locations
ORS_MATRIX_RETRIES = 3            # tentatives sur 429 / 5xx
ORS_MATRIX_BACKOFF_S = 2          # 2s, puis 4s, puis 8s
ORS_MATRIX_NULL_SPEED_KMH = 30    # vitesse de repli pour une paire inatteignable

_MATRIX_CACHE = {}
_MATRIX_CACHE_MAX = 8

# Criteres d'arret du solveur OR-Tools.
# solution_limit est le critere REEL : il est deterministe, contrairement a
# time_limit qui depend de la charge CPU. time_limit n'est qu'un garde-fou ;
# s'il se declenche, le determinisme est perdu et on le journalise.
ORTOOLS_SOLUTION_LIMIT = 75
ORTOOLS_TIME_LIMIT_S = 25

# Post-traitement des swaps (lot D-3), surchargeables par requete.
SWAP_MAX_CANDIDATES = 50           # 0 desactive entierement les swaps
SWAP_MAX_CONSECUTIVE_FAILS = 0     # 0 desactive l'arret anticipe

# Valeurs autorisees de swap_stop_reason.
SWAP_STOP_REASONS = ("disabled", "candidate_limit", "consecutive_failures",
                     "convergence", "no_border_points", "vroom_error", "completed",
                     "territorial_partition_locked", "connected_partition_locked")

_API_STATS = {"vroom": 0, "matrix": 0}


def _reset_api_stats():
    _API_STATS["vroom"] = 0
    _API_STATS["matrix"] = 0


def _api_calls_total():
    return _API_STATS["vroom"] + _API_STATS["matrix"]


def _post_vroom(payload, headers, timeout):
    """Unique point de sortie vers l'endpoint Vroom/Optimization (compte les appels)."""
    _API_STATS["vroom"] += 1
    return requests.post(ORS_VROOM_URL, json=payload, headers=headers, timeout=timeout)


def _post_matrix(payload, headers, timeout):
    """Unique point de sortie vers l'endpoint ORS Matrix (compte les appels)."""
    _API_STATS["matrix"] += 1
    return requests.post(ORS_MATRIX_URL, json=payload, headers=headers, timeout=timeout)


def _strict_int_param(data, name, lo, hi, default):
    """Lit un entier STRICT dans le corps de requete.
    Refuse chaines, booleens et decimaux, et ne tronque JAMAIS : 12.7 est une
    erreur, pas un 12 silencieux. isinstance(True, int) valant True en Python,
    le cas booleen est teste en premier.
    Retourne (valeur, None) ou (None, message_erreur)."""
    raw = data.get(name)
    if raw is None:
        return default, None
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, (f"{name} must be a strict integer "
                      f"(string, boolean and decimal values are rejected)")
    if not (lo <= raw <= hi):
        return None, f"{name} out of range ({lo}..{hi})"
    return raw, None


def _terr_get(matrix_meta, key, default):
    """Lit un champ du certificat territorial. Les autres strategies ne
    produisent pas ce bloc : la valeur par defaut s'applique alors."""
    if not matrix_meta:
        return default
    terr = matrix_meta.get("territorial")
    if not terr:
        return default
    return terr.get(key, default)


def _flatten_counts(mapping):
    """Compteurs par source, aplatis en une chaine STABLE pour Benchmark :
    "kmedoids=4;mst=6;sweep=21". Triee par cle : deux runs comparables donnent
    deux chaines comparables, ce qu'un dictionnaire ne garantit pas."""
    if not isinstance(mapping, dict) or not mapping:
        return ""
    return ";".join("%s=%s" % (k, mapping[k]) for k in sorted(mapping))


def _flatten_per_source(mapping):
    """Bilan par source aplati pour Benchmark, dans un ordre stable :
    "sweep:r=180,d=150,s=0,x=8,f=2,u=28". r=brutes, d=doublons, s=taille
    invalide, x=deconnectees, f=reparations echouees, u=uniques conservees."""
    if not isinstance(mapping, dict) or not mapping:
        return ""
    parts = []
    for src in sorted(mapping):
        b = mapping[src] or {}
        parts.append("%s:r=%s,d=%s,s=%s,x=%s,f=%s,u=%s"
                     % (src, b.get("raw", 0), b.get("duplicates", 0),
                        b.get("invalid_size", 0), b.get("disconnected", 0),
                        b.get("repair_failed", 0), b.get("unique", 0)))
    return ";".join(parts)


def _conn_get(matrix_meta, key, default):
    """Lit un champ du certificat de connexite. Les autres strategies ne
    produisent pas ce bloc : la valeur par defaut s'applique alors."""
    if not matrix_meta:
        return default
    conn = matrix_meta.get("connected")
    if not conn:
        return default
    return conn.get(key, default)


def _points_signature(points):
    """Empreinte courte du jeu de points (IDs tries). Prouve que deux runs
    portent sur exactement les memes donnees."""
    ids = sorted(str(p.get("id", "")) for p in points)
    return hashlib.md5("|".join(ids).encode("utf-8")).hexdigest()[:8]


# =========================
# 3. VROOM MULTI-VEHICULES (affectation + sequencement simultanes)
# =========================
def _call_vroom_multi(jobs, vehicles, headers):
    """Un appel Vroom multi-vehicules. Retourne (routes_by_vehicle, total_duration) ou (None, err)."""
    try:
        response = _post_vroom(
            {"jobs": jobs, "vehicles": vehicles},
            headers,
            timeout=15
        )
        data = response.json()

        if "routes" not in data:
            err = data.get("error", data)
            if isinstance(err, dict):
                err = err.get("message", str(err))
            return None, str(err)

        # Verifier les jobs non assignes
        unassigned = data.get("unassigned", [])
        if unassigned:
            print(f"  Vroom: {len(unassigned)} jobs non assignes!", flush=True)

        routes_by_vehicle = {}
        total_duration = 0

        for route in data["routes"]:
            vid = route["vehicle"]
            ordered = []
            for step in route["steps"]:
                if step["type"] == "job":
                    ordered.append(step["id"])
            routes_by_vehicle[vid] = ordered
            total_duration += route.get("duration", 0)

        return (routes_by_vehicle, total_duration), None

    except Exception as e:
        return None, str(e)


def _auto_num_runs(n_jobs, num_vehicles):
    """Ajuste automatiquement le nombre de runs Vroom.
    1 vehicule : Vroom est exact (LKH3), 1 run suffit.
    Multi-vehicules : l'affectation est heuristique, plus de runs = meilleure solution.
    """
    if num_vehicles == 1:
        return 1
    if n_jobs <= 40:
        return 5
    else:
        return 3


def optimize_with_vroom(points, num_vehicles, max_per_vehicle, start_idx, end_idx):
    """Appelle Vroom avec tous les vehicules pour affectation + sequencement simultanes.
       Nombre de runs auto-ajuste selon la taille : plus le probleme est petit, plus on teste d'ordres."""

    # ORS free tier : max 3500 routes = ~59 locations
    # locations = nb_jobs + nb_depots_uniques (start et/ou end)
    unique_locations = len({
        (round(float(p["lon"]), 6), round(float(p["lat"]), 6))
        for p in points
    })
    if unique_locations > 59:
        print(f"Vroom multi-vehicules: {unique_locations} locations distinctes > 59 (limite ORS 3500), skip", flush=True)
        return None, False, "ORS limit: too many locations for multi-vehicle"

    start_coord = [points[start_idx]["lon"], points[start_idx]["lat"]]
    end_coord = [points[end_idx]["lon"], points[end_idx]["lat"]]

    headers = {
        "Authorization": ORS_KEY,
        "Content-Type": "application/json"
    }

    # Points hors depot
    depot_indices = {start_idx, end_idx}
    delivery_indices = [i for i in range(len(points)) if i not in depot_indices]

    # Vehicules avec capacite
    vehicles = []
    for v in range(num_vehicles):
        vehicles.append({
            "id": v,
            "profile": "driving-car",
            "start": start_coord,
            "end": end_coord,
            "capacity": [max_per_vehicle]
        })

    # Nombre de runs auto-ajuste
    n_runs = _auto_num_runs(len(delivery_indices), num_vehicles)
    print(f"Vroom multi-vehicules: {len(delivery_indices)} jobs -> {n_runs} runs", flush=True)

    # 5 ordres de jobs possibles (tronques a n_runs)
    base_jobs = [
        {"id": idx, "location": [points[idx]["lon"], points[idx]["lat"]], "delivery": [1]}
        for idx in delivery_indices
    ]
    depot_lat = points[start_idx]["lat"]
    depot_lon = points[start_idx]["lon"]

    all_orderings = [
        ("normal",    base_jobs),
        ("reverse",   list(reversed(base_jobs))),
        ("lat_asc",   sorted(base_jobs, key=lambda j: points[j["id"]]["lat"])),
        ("lon_asc",   sorted(base_jobs, key=lambda j: points[j["id"]]["lon"])),
        ("dist_depot",sorted(base_jobs, key=lambda j: haversine(
                          (depot_lat, depot_lon),
                          (points[j["id"]]["lat"], points[j["id"]]["lon"])))),
    ]
    orderings_to_run = all_orderings[:n_runs]

    # Lancer les runs et garder le meilleur
    best = None
    best_duration = float("inf")
    best_name = ""
    last_err = None

    consecutive_failures = 0
    for name, jobs in orderings_to_run:
        if consecutive_failures >= 2:
            print(f"  Abandon apres 2 echecs consecutifs", flush=True)
            break
        if name != orderings_to_run[0][0]:
            time.sleep(1.5)
        print(f"  Run '{name}'...", flush=True)
        result, err = _call_vroom_multi(jobs, vehicles, headers)
        if result:
            routes, dur = result
            print(f"    -> {dur}s", flush=True)
            consecutive_failures = 0
            if dur < best_duration:
                best_duration = dur
                best = routes
                best_name = name
        else:
            last_err = err
            consecutive_failures += 1
            print(f"    -> erreur: {err}", flush=True)

    if best:
        all_routes = []
        for v in range(num_vehicles):
            route = [start_idx] + best.get(v, []) + [end_idx]
            all_routes.append(route)

        print(f"Vroom meilleur run: '{best_name}' ({best_duration}s)", flush=True)
        for v in range(num_vehicles):
            print(f"  Vehicule {v+1}: {len(best.get(v, []))} pts", flush=True)

        return all_routes, True, None

    print(f"Vroom multi-vehicules: tous les runs ont echoue ({last_err})", flush=True)
    return None, False, last_err


# =========================
# 4. FALLBACK : K-MEANS + VROOM PAR VEHICULE
# =========================
def _balance_groups(groups, points, max_per_vehicle):
    """Equilibre les groupes pour respecter la capacite max par vehicule."""
    k = len(groups)
    for _ in range(100):
        changed = False
        for g in range(k):
            while len(groups[g]) > max_per_vehicle:
                # Trouver le point le plus eloigne du centroide
                c_lat = np.mean([points[i]["lat"] for i in groups[g]])
                c_lon = np.mean([points[i]["lon"] for i in groups[g]])
                dists = [(idx, haversine((c_lat, c_lon), (points[idx]["lat"], points[idx]["lon"])))
                         for idx in groups[g]]
                dists.sort(key=lambda x: -x[1])
                furthest_idx = dists[0][0]

                # Trouver le groupe le plus proche avec capacite
                best_group = None
                best_dist = float("inf")
                for g2 in range(k):
                    if g2 != g and len(groups[g2]) < max_per_vehicle:
                        c2_lat = np.mean([points[i]["lat"] for i in groups[g2]])
                        c2_lon = np.mean([points[i]["lon"] for i in groups[g2]])
                        d = haversine((c2_lat, c2_lon), (points[furthest_idx]["lat"], points[furthest_idx]["lon"]))
                        if d < best_dist:
                            best_dist = d
                            best_group = g2

                if best_group is not None:
                    groups[g].remove(furthest_idx)
                    groups[best_group].append(furthest_idx)
                    changed = True
                else:
                    break
        if not changed:
            break
    return groups


def _create_sub_clusters(coords, delivery_indices, k_sub):
    """Cree k_sub sous-clusters K-Means. Retourne la liste des groupes (indices points)."""
    k_sub = min(k_sub, len(delivery_indices))
    km = KMeans(n_clusters=k_sub, n_init=10, random_state=42)
    labels = km.fit_predict(coords)
    sub_groups = [[] for _ in range(k_sub)]
    for i, label in enumerate(labels):
        sub_groups[label].append(delivery_indices[i])
    return [g for g in sub_groups if g]  # enlever groupes vides


def _enumerate_partitions(sub_groups, num_vehicles, points, max_per_vehicle, max_partitions=50):
    """Enumere TOUTES les facons d'assigner les sous-clusters aux vehicules.
    Pour num_vehicles=2 : teste chaque combinaison valide de sous-clusters.
    Garantit des partitions genuinement differentes (pas de recurrence sur K-Means centroides)."""
    k_sub = len(sub_groups)
    partitions = []
    seen = set()

    if num_vehicles == 2:
        # Enumerer : vehicle 0 prend 'size' sous-clusters, vehicle 1 prend le reste
        for size in range(1, k_sub):
            for combo in combinations(range(k_sub), size):
                rest = [i for i in range(k_sub) if i not in combo]
                pts0 = [p for i in combo for p in sub_groups[i]]
                pts1 = [p for i in rest for p in sub_groups[i]]
                groups = _balance_groups([list(pts0), list(pts1)], points, max_per_vehicle)
                key = frozenset(frozenset(g) for g in groups)
                if key not in seen:
                    seen.add(key)
                    partitions.append(groups)
                if len(partitions) >= max_partitions:
                    return partitions
    else:
        # Pour >2 vehicules : K-Means sur centroides avec plusieurs seeds
        sub_coords = np.array([
            [np.mean([points[p]["lat"] for p in g]),
             np.mean([points[p]["lon"] for p in g])]
            for g in sub_groups
        ])
        for seed in [42, 0, 7, 13, 99]:
            km_merge = KMeans(n_clusters=num_vehicles, n_init=10, random_state=seed)
            v_labels = km_merge.fit_predict(sub_coords)
            groups = [[] for _ in range(num_vehicles)]
            for sub_g, v_label in enumerate(v_labels):
                groups[v_label].extend(sub_groups[sub_g])
            groups = _balance_groups(groups, points, max_per_vehicle)
            key = frozenset(frozenset(g) for g in groups)
            if key not in seen:
                partitions.append(groups)
    return partitions


def _sequence_groups(points, groups, start_idx, end_idx, headers):
    """Sequence chaque groupe avec Vroom. Retourne (routes, total_dur, vroom_ok, vroom_error)."""
    start_coord = [points[start_idx]["lon"], points[start_idx]["lat"]]
    end_coord = [points[end_idx]["lon"], points[end_idx]["lat"]]

    all_routes = []
    total_dur = 0
    vroom_ok = True
    vroom_error = None

    for v, group in enumerate(groups):
        if not group:
            all_routes.append([start_idx, end_idx])
            continue

        vehicle = {
            "id": 0,
            "profile": "driving-car",
            "start": start_coord,
            "end": end_coord
        }
        jobs = [{"id": idx, "location": [points[idx]["lon"], points[idx]["lat"]]} for idx in group]

        try:
            time.sleep(0.5)  # delay between Vroom calls
            response = _post_vroom(
                {"jobs": jobs, "vehicles": [vehicle]},
                headers,
                timeout=20
            )
            data = response.json()
            if "routes" in data:
                ordered = [start_idx]
                for step in data["routes"][0]["steps"]:
                    if step["type"] == "job":
                        ordered.append(step["id"])
                ordered.append(end_idx)
                all_routes.append(ordered)
                dur = data["routes"][0].get("duration", 0)
                total_dur += dur
            else:
                vroom_ok = False
                err = data.get("error", data)
                if isinstance(err, dict):
                    err = err.get("message", str(err))
                vroom_error = str(err)
                all_routes.append(_nearest_neighbor_route(points, group, start_idx, end_idx))
        except Exception as e:
            vroom_ok = False
            vroom_error = str(e)
            all_routes.append(_nearest_neighbor_route(points, group, start_idx, end_idx))

    return all_routes, total_dur, vroom_ok, vroom_error


def _find_best_k_silhouette(coords, k_min, k_max):
    """Trouve le k optimal via silhouette score (mesure la coherence des clusters)."""
    best_k = k_min
    best_score = -1.0
    n = len(coords)
    for k in range(k_min, k_max + 1):
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(coords)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(coords, labels)
        print(f"  Silhouette k={k}: {score:.3f}", flush=True)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def kmeans_partition(points, num_vehicles, max_per_vehicle, start_idx, end_idx):
    """Multi-strategie : K-Means + splits par axe, sequence chacune avec Vroom, garde la meilleure."""
    depot_indices = {start_idx, end_idx}
    delivery_indices = [i for i in range(len(points)) if i not in depot_indices]

    if not delivery_indices:
        return [[start_idx, end_idx]] * num_vehicles, False, "no delivery points"

    headers = {
        "Authorization": ORS_KEY,
        "Content-Type": "application/json"
    }

    # Auto-detection du k optimal via silhouette score
    coords = np.array([[points[i]["lat"], points[i]["lon"]] for i in delivery_indices])
    n = len(delivery_indices)
    k_min = num_vehicles
    k_max = min(num_vehicles * 8, n // 2)
    print(f"Recherche k optimal silhouette (k={k_min}..{k_max})...", flush=True)
    best_k, best_score = _find_best_k_silhouette(coords, k_min, k_max)
    print(f"  -> k optimal = {best_k} (silhouette={best_score:.3f})", flush=True)

    # Valeurs de k_sub a tester : k direct, intermediaire, k optimal
    k_sub_values = sorted({num_vehicles, best_k})
    if best_k > num_vehicles + 2:
        k_sub_values.append((num_vehicles + best_k) // 2)
    k_sub_values = sorted(set(k_sub_values))

    # Enumerer toutes les partitions uniques
    all_partitions = []
    seen_keys = set()
    for k_sub in k_sub_values:
        sub_groups = _create_sub_clusters(coords, delivery_indices, k_sub)
        candidates = _enumerate_partitions(sub_groups, num_vehicles, points, max_per_vehicle)
        for groups in candidates:
            key = frozenset(frozenset(g) for g in groups)
            if key not in seen_keys:
                seen_keys.add(key)
                all_partitions.append((f"k={k_sub}", groups))

    print(f"{len(all_partitions)} partitions uniques enumerees", flush=True)

    # Pre-scoring haversine (0 appel API) : classe les partitions par distance estimee
    def _hav_cost(groups):
        total = 0.0
        for group in groups:
            route = _nearest_neighbor_route(points, group, start_idx, end_idx)
            total += _compute_route_distance(points, route)
        return total

    all_partitions.sort(key=lambda x: _hav_cost(x[1]))
    for i, (name, groups) in enumerate(all_partitions[:6]):
        print(f"  #{i+1} '{name}': {[len(g) for g in groups]} pts, hav={_hav_cost(groups):.2f}km", flush=True)

    # Appel Vroom sur les 4 meilleures partitions (8 appels max, 0 rate limit)
    TOP_VROOM = 4
    print(f"Vroom sur top {TOP_VROOM} partitions...", flush=True)

    best_routes = None
    best_dur = float("inf")
    best_ok = False
    best_err = None
    best_name = ""

    for name, groups in all_partitions[:TOP_VROOM]:
        pts_str = [len(g) for g in groups]
        print(f"Partition '{name}': pts={pts_str}", flush=True)

        routes, dur, ok, err = _sequence_groups(points, groups, start_idx, end_idx, headers)
        print(f"  -> duree={dur}s, vroom_ok={ok}", flush=True)

        if dur < best_dur:
            best_dur = dur
            best_routes = routes
            best_ok = ok
            best_err = err
            best_name = name

    print(f"Meilleure partition: '{best_name}' ({best_dur}s)", flush=True)

    # Completer avec des routes vides si necessaire
    while len(best_routes) < num_vehicles:
        best_routes.append([start_idx, end_idx])

    return best_routes, best_ok, best_err


# =========================
# 4b. PARTITION OR-TOOLS (affectation seule, le sequencement reste a Vroom)
# =========================
def _build_haversine_matrix(points):
    """Matrice n x n de distances haversine en METRES entiers.
    OR-Tools exige des couts entiers. Symetrique, diagonale nulle."""
    n = len(points)
    coords = [(float(p["lat"]), float(p["lon"])) for p in points]
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = int(round(haversine(coords[i], coords[j]) * 1000))
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def _ortools_status(routing):
    """Statut du solveur, defensif : l'API a change selon les versions
    d'OR-Tools et ce libelle ne sert qu'aux logs. Il ne doit jamais faire
    echouer une resolution reussie."""
    try:
        return routing.status()
    except Exception:
        return "n/a"


def _ortools_solver_stats(routing):
    """Compteurs du solveur CP sous-jacent, lus defensivement.
    Leurs noms ont change selon les versions d'OR-Tools et ces valeurs ne
    servent qu'aux logs : une lecture impossible vaut -1 et ne doit jamais
    faire echouer une tournee. Ne leve aucune exception.

    solutions    : nombre de solutions trouvees. Egal a solution_limit =>
                   l'arret vient du critere deterministe. Inferieur =>
                   c'est time_limit qui a tranche.
    branches     : points de decision explores.
    failures     : retours arriere.
    wall_time_ms : temps solveur mesure par OR-Tools lui-meme.
    """
    stats = {"solutions": -1, "branches": -1, "failures": -1, "wall_time_ms": -1}
    try:
        solver = routing.solver()
    except Exception:
        return stats

    for key, getter in (("solutions", "Solutions"),
                        ("branches", "Branches"),
                        ("failures", "Failures"),
                        ("wall_time_ms", "WallTime")):
        try:
            stats[key] = getattr(solver, getter)()
        except Exception:
            pass

    return stats


def _solve_cvrp_ortools(cost_matrix, num_vehicles, capacity, start_idx, end_idx,
                        solution_limit=None):
    """Resout un CVRP localement et retourne UNIQUEMENT l'affectation.

    L'ordre trouve par OR-Tools est volontairement jete : c'est Vroom qui
    sequence ensuite, sur le reseau routier reel. On ne compare donc que la
    qualite de la partition.

    Entree : matrice de couts entiers, nb de vehicules, capacite par vehicule,
             index du depot de depart, index du depot d'arrivee.
    Retour : (groups, None) ou (None, message_erreur).
             groups = [[indices vehicule 0], [indices vehicule 1], ...]
    """
    if not ORTOOLS_AVAILABLE:
        return None, "ortools not installed"

    # LOT 4.1-C : solution_limit pilotable par requete pour l'experimentation.
    # None -> ORTOOLS_SOLUTION_LIMIT, donc un appel sans l'argument reproduit
    # exactement le comportement actuel.
    effective_limit = solution_limit if solution_limit else ORTOOLS_SOLUTION_LIMIT

    n = len(cost_matrix)
    depots = {start_idx, end_idx}

    try:
        # Depart == arrivee -> forme mono-depot. Sinon, depart et arrivee
        # distincts par vehicule (supporte nativement par OR-Tools).
        if start_idx == end_idx:
            manager = pywrapcp.RoutingIndexManager(n, num_vehicles, start_idx)
        else:
            manager = pywrapcp.RoutingIndexManager(
                n, num_vehicles,
                [start_idx] * num_vehicles,
                [end_idx] * num_vehicles
            )

        routing = pywrapcp.RoutingModel(manager)

        def transit_callback(from_index, to_index):
            return cost_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_idx = routing.RegisterTransitCallback(transit_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Capacite : 1 point = 1 unite, les depots ne consomment rien.
        def demand_callback(from_index):
            return 0 if manager.IndexToNode(from_index) in depots else 1

        demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx,
            0,                              # pas de marge
            [capacity] * num_vehicles,      # capacite par vehicule
            True,                           # cumul demarre a zero
            "Capacity"
        )

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        params.solution_limit = effective_limit
        params.time_limit.FromSeconds(ORTOOLS_TIME_LIMIT_S)
        params.log_search = False

        t0 = time.time()
        solution = routing.SolveWithParameters(params)
        elapsed = time.time() - t0

        if solution is None:
            return None, f"no solution (status={_ortools_status(routing)})"

        # Le garde-fou temps ne doit jamais trancher : s'il tranche, le
        # resultat depend de la machine et n'est plus reproductible.
        if elapsed >= ORTOOLS_TIME_LIMIT_S - 0.5:
            print(f"  ATTENTION OR-Tools: arret sur time_limit ({elapsed:.1f}s), "
                  f"resultat NON deterministe", flush=True)

        groups = [[] for _ in range(num_vehicles)]
        for v in range(num_vehicles):
            index = routing.Start(v)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node not in depots:
                    groups[v].append(node)
                index = solution.Value(routing.NextVar(index))

        # Objectif sans unite : cette fonction recoit une matrice d'entiers et
        # ne peut pas savoir s'il s'agit de metres (haversine) ou de secondes
        # (durees ORS). Le "m" qui figurait ici etait faux pour ortools_ors_matrix.
        print(f"  OR-Tools: status={_ortools_status(routing)}, objectif={solution.ObjectiveValue()}, "
              f"{elapsed:.1f}s, solution_limit={effective_limit}", flush=True)

        stats = _ortools_solver_stats(routing)
        print(f"  OR-Tools stats: solutions={stats['solutions']} "
              f"branches={stats['branches']} failures={stats['failures']} "
              f"wall_time_ms={stats['wall_time_ms']}", flush=True)

        return groups, None

    except Exception as e:
        return None, f"ortools error: {e}"


def ortools_partition_haversine(points, num_vehicles, max_per_vehicle, start_idx, end_idx):
    """Affectation des points aux vehicules par OR-Tools, cout = haversine.
    Aucun appel ORS. Retourne (groups, None) ou (None, message_erreur)."""
    depot_indices = {start_idx, end_idx}
    delivery_indices = [i for i in range(len(points)) if i not in depot_indices]

    if not delivery_indices:
        return [[] for _ in range(num_vehicles)], None

    if num_vehicles * max_per_vehicle < len(delivery_indices):
        return None, (f"capacity too small: {num_vehicles} x {max_per_vehicle} "
                      f"< {len(delivery_indices)} points")

    print(f"OR-Tools haversine: {len(delivery_indices)} points, {num_vehicles} vehicules, "
          f"capacite {max_per_vehicle}", flush=True)

    matrix = _build_haversine_matrix(points)
    groups, err = _solve_cvrp_ortools(
        matrix, num_vehicles, max_per_vehicle, start_idx, end_idx
    )
    if groups is None:
        return None, err

    print(f"  Partition: {[len(g) for g in groups]} pts", flush=True)
    return groups, None


# =========================
# 4c. MATRICE ORS COMPLETE DECOUPEE (strategie ortools_ors_matrix)
# =========================
def _matrix_cache_key(points, profile="driving-car"):
    """Cle de cache : liste ORDONNEE des coordonnees arrondies. L'ordre compte,
    la matrice etant indexee par position."""
    coords = ";".join(
        f"{round(float(p['lon']), 6)},{round(float(p['lat']), 6)}" for p in points
    )
    raw = f"{profile}|duration+distance|{coords}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _matrix_cache_put(key, value):
    """Cache borne, eviction FIFO (les dict Python conservent l'ordre d'insertion)."""
    if key in _MATRIX_CACHE:
        return
    if len(_MATRIX_CACHE) >= _MATRIX_CACHE_MAX:
        _MATRIX_CACHE.pop(next(iter(_MATRIX_CACHE)))
    _MATRIX_CACHE[key] = value


def _post_matrix_retry(payload, headers, timeout=30):
    """Appel ORS Matrix avec retry sur 429 et 5xx.
    CHAQUE tentative passe par _post_matrix et est donc comptee : le compteur
    doit refleter la consommation reelle de quota, pas les appels reussis.
    Un 4xx autre que 429 n'est pas rejouable : le message ORS brut est remonte
    tel quel (c'est lui qui revelera un eventuel plafond sur les locations).
    Retourne (data, None) ou (None, message)."""
    last_err = None

    for attempt in range(ORS_MATRIX_RETRIES):
        if attempt > 0:
            wait = ORS_MATRIX_BACKOFF_S * (2 ** (attempt - 1))
            print(f"    Matrix retry {attempt + 1}/{ORS_MATRIX_RETRIES} "
                  f"dans {wait}s ({last_err})", flush=True)
            time.sleep(wait)

        try:
            response = _post_matrix(payload, headers, timeout=timeout)

            if response.status_code == 429 or response.status_code >= 500:
                last_err = f"HTTP {response.status_code}"
                continue

            data = response.json()

            if response.status_code != 200:
                err = data.get("error", data)
                if isinstance(err, dict):
                    err = err.get("message", str(err))
                return None, f"HTTP {response.status_code}: {err}"

            return data, None

        except Exception as e:
            last_err = str(e)

    return None, f"failed after {ORS_MATRIX_RETRIES} attempts: {last_err}"


def _matrix_content_hash(dur_matrix, dist_matrix):
    """Empreinte courte du CONTENU des deux matrices ORS.

    Deux runs qui affichent la meme empreinte ont resolu le meme probleme
    routier ; deux empreintes differentes expliquent a elles seules un
    resultat different, sans qu'aucun bug de selection soit en cause.
    """
    h = hashlib.md5()
    for matrix in (dur_matrix, dist_matrix):
        if not matrix:
            h.update(b"|none|")
            continue
        for row in matrix:
            h.update((",".join(str(v) for v in row) + ";").encode("utf-8"))
    return h.hexdigest()[:12]


def _matrix_block_plan(n):
    """Decoupe les n sources en blocs de lignes respectant le plafond de routes.
    Retourne (blocs, None) ou (None, message)."""
    if n <= 0:
        return [], None

    if ORS_MATRIX_MAX_LOCATIONS and n > ORS_MATRIX_MAX_LOCATIONS:
        # Chaque appel envoie les n locations (obtenir cout(i,j) exige i et j
        # dans le meme tableau). Un plafond sur les locations imposerait un
        # decoupage en damier lignes x colonnes, non implemente ici.
        return None, (f"{n} locations > ORS_MATRIX_MAX_LOCATIONS="
                      f"{ORS_MATRIX_MAX_LOCATIONS}: checkerboard split required")

    rows_max = max(1, ORS_MATRIX_MAX_ROUTES // n)
    n_calls = math.ceil(n / rows_max)
    block = math.ceil(n / n_calls)
    return [(s, min(s + block, n)) for s in range(0, n, block)], None


def _build_full_matrix_chunked(points, headers):
    """Matrice complete n x n des durees (s) et distances (m) ORS, en entiers.
    Retourne (dur_matrix, dist_matrix, meta, None) ou (None, None, meta, err)."""
    n = len(points)
    meta = {"n": n, "calls": 0, "blocks": 0, "cached": False, "nulls": 0}

    key = _matrix_cache_key(points)
    if key in _MATRIX_CACHE:
        dur_m, dist_m, nulls = _MATRIX_CACHE[key]
        meta.update({"cached": True, "nulls": nulls})
        print(f"  Matrice ORS {n}x{n}: cache HIT ({key[:8]}), 0 appel", flush=True)
        return dur_m, dist_m, meta, None

    blocks, err = _matrix_block_plan(n)
    if blocks is None:
        return None, None, meta, err

    meta["blocks"] = len(blocks)
    locations = [[float(p["lon"]), float(p["lat"])] for p in points]
    all_dest = list(range(n))
    calls_before = _API_STATS["matrix"]

    print(f"  Matrice ORS {n}x{n}: {len(blocks)} appel(s), blocs de "
          f"{[b[1] - b[0] for b in blocks]} lignes x {n} colonnes "
          f"({[(b[1] - b[0]) * n for b in blocks]} routes)", flush=True)

    dur_rows = []
    dist_rows = []

    for (s0, s1) in blocks:
        payload = {
            "locations": locations,
            "sources": list(range(s0, s1)),
            "destinations": all_dest,
            "metrics": ["distance", "duration"],
        }
        data, err = _post_matrix_retry(payload, headers)
        if data is None:
            meta["calls"] = _API_STATS["matrix"] - calls_before
            return None, None, meta, f"block {s0}-{s1}: {err}"

        durs = data.get("durations")
        dists = data.get("distances")

        # Un bloc manquant ou de forme inattendue = echec global. Assembler une
        # matrice partiellement a zero produirait une partition absurde sans
        # le moindre signal.
        if not durs or len(durs) != (s1 - s0):
            meta["calls"] = _API_STATS["matrix"] - calls_before
            return None, None, meta, (f"block {s0}-{s1}: unexpected shape "
                                      f"({len(durs) if durs else 0} rows for {s1 - s0})")

        dur_rows.extend(durs)
        dist_rows.extend(dists if dists else [[None] * n for _ in range(s1 - s0)])

    meta["calls"] = _API_STATS["matrix"] - calls_before

    if len(dur_rows) != n:
        return None, None, meta, f"assembled {len(dur_rows)} rows, expected {n}"

    # Conversion en entiers. Une cellule nulle (paire inatteignable) est
    # remplacee par une estimation haversine, comptee et remontee : echouer sur
    # une seule paire serait fragile, le faire en silence serait pire.
    coords = [(float(p["lat"]), float(p["lon"])) for p in points]
    dur_matrix = [[0] * n for _ in range(n)]
    dist_matrix = [[0] * n for _ in range(n)]
    nulls = 0

    for i in range(n):
        row_d = dur_rows[i]
        row_k = dist_rows[i] if i < len(dist_rows) and dist_rows[i] else [None] * n
        for j in range(n):
            d = row_d[j] if j < len(row_d) else None
            m = row_k[j] if j < len(row_k) else None
            if d is None:
                nulls += 1
                hav_km = haversine(coords[i], coords[j])
                d = hav_km / ORS_MATRIX_NULL_SPEED_KMH * 3600.0
                if m is None:
                    m = hav_km * 1000.0
            dur_matrix[i][j] = int(round(d))
            dist_matrix[i][j] = int(round(m)) if m is not None else 0

    for i in range(n):
        dur_matrix[i][i] = 0
        dist_matrix[i][i] = 0

    meta["nulls"] = nulls
    if nulls:
        print(f"  Matrice ORS: {nulls} cellule(s) nulle(s) remplacee(s) par "
              f"une estimation haversine a {ORS_MATRIX_NULL_SPEED_KMH} km/h", flush=True)

    # Empreinte du CONTENU de la matrice, pas de ses entrees. La signature des
    # points ne prouve rien : deux runs sur le meme jeu peuvent recevoir des
    # durees routieres differentes -- mise a jour du reseau ORS, trafic, ou
    # cellules nulles remplacees. Sans cette empreinte, comparer deux runs
    # revient a comparer deux problemes qu'on suppose identiques.
    meta["content_hash"] = _matrix_content_hash(dur_matrix, dist_matrix)

    _matrix_cache_put(key, (dur_matrix, dist_matrix, nulls))
    print(f"  Matrice ORS {n}x{n} assemblee en {meta['calls']} appel(s), "
          f"empreinte contenu={meta['content_hash']}, "
          f"mise en cache ({key[:8]})", flush=True)

    return dur_matrix, dist_matrix, meta, None


# =========================
# 4d. PARTITION TERRITORIALE (strategie ortools_ors_matrix)
# =========================
# Pourquoi remplacer la partition OR-Tools : _solve_cvrp_ortools minimise le
# cout d'arc total sous contrainte de capacite. NI l'objectif NI les
# contraintes n'imposent la moindre contiguite spatiale. Avec un depot commun,
# deux tournees en rayons entrelaces coutent souvent moins qu'un decoupage en
# deux blocs compacts : le solveur produit donc legitimement des territoires
# imbriques. Le probleme n'est pas la resolution, c'est le modele.
#
# CERTIFICAT retenu : separabilite lineaire. Deux groupes sont declares separes
# s'il existe une droite telle que tous les points d'un groupe sont
# STRICTEMENT d'un cote et tous ceux de l'autre groupe strictement de l'autre.
# Ce n'est ni une distance entre centroides, ni une penalite de dispersion :
# c'est une propriete verifiable, recomptee independamment de la construction.

TERRITORIAL_TOP_REFINE = 12       # candidates affinees en phase 2
TERRITORIAL_MAX_SAMPLES = 6000    # garde-fou sur le nombre d'angles balayes


def _finite_coords(p):
    """Coordonnees exploitables : deux nombres finis dans les bornes terrestres."""
    try:
        lat, lon = float(p["lat"]), float(p["lon"])
    except (TypeError, ValueError, KeyError):
        return False
    return (lat == lat and lon == lon                      # ecarte les NaN
            and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
            and abs(lat) != float("inf") and abs(lon) != float("inf"))


def _local_xy(points, indices):
    """Projette lat/lon en metres dans un plan local equirectangulaire.
    Suffisant a l'echelle d'une agglomeration, et seule la geometrie RELATIVE
    compte pour separer deux groupes. Retourne {index_global: (x_m, y_m)}."""
    if not indices:
        return {}
    lat0 = sum(float(points[i]["lat"]) for i in indices) / len(indices)
    lon0 = sum(float(points[i]["lon"]) for i in indices) / len(indices)
    kx = 111320.0 * math.cos(math.radians(lat0))
    return {i: ((float(points[i]["lon"]) - lon0) * kx,
                (float(points[i]["lat"]) - lat0) * 111320.0)
            for i in indices}


def _critical_angles(xy, indices):
    """Angles ou l'ordre des projections peut changer.

    En faisant tourner la direction de projection u(theta), l'ordre des points
    ne peut changer qu'au moment ou deux d'entre eux se projettent au meme
    endroit, c'est-a-dire quand u est PERPENDICULAIRE au segment qui les joint.
    Ces angles decoupent [0, pi) en intervalles ou la partition est constante.
    Deux points confondus ne definissent aucun angle et sont ignores.
    """
    crit = set()
    idx = list(indices)
    for a in range(len(idx)):
        xa, ya = xy[idx[a]]
        for b in range(a + 1, len(idx)):
            xb, yb = xy[idx[b]]
            dx, dy = xb - xa, yb - ya
            if dx == 0.0 and dy == 0.0:
                continue
            crit.add(round((math.atan2(dy, dx) + math.pi / 2.0) % math.pi, 10))
    return sorted(crit)


def _sample_angles(crit):
    """Un angle par intervalle entre deux angles critiques consecutifs.

    Prendre le MILIEU de chaque intervalle atteint chaque ordre de projection
    distinct au moins une fois : l'enumeration des partitions separables par
    une droite est donc EXHAUSTIVE, pas heuristique.
    """
    if not crit:
        return [0.0]
    n = len(crit)
    out = []
    for k in range(n):
        a2 = crit[k + 1] if k + 1 < n else crit[0] + math.pi
        out.append((crit[k] + a2) / 2.0)
    return out


def _split_by_angle(xy, indices, angle, group_size):
    """Trie les points par projection sur u(angle) et coupe en deux groupes.

    Retourne (group_a, group_b, margin_m), ou (None, None, 0.0) si la coupure
    n'est pas STRICTE : une marge nulle signifie que deux points de part et
    d'autre se projettent au meme endroit, donc qu'aucune droite ne les separe.
    Le tri est departage par index global, le resultat est donc deterministe.
    """
    if group_size <= 0 or group_size >= len(indices):
        return None, None, 0.0
    ux, uy = math.cos(angle), math.sin(angle)
    proj = sorted((xy[i][0] * ux + xy[i][1] * uy, i) for i in indices)
    margin = proj[group_size][0] - proj[group_size - 1][0]
    if margin <= 0.0:
        return None, None, 0.0
    return ([i for _, i in proj[:group_size]],
            [i for _, i in proj[group_size:]],
            margin)


def _territorial_certificate(xy, group_a, group_b, angle):
    """Recompte INDEPENDAMMENT les violations du certificat.

    Ne fait aucune confiance a la construction : reprojette les deux groupes,
    place la frontiere au milieu de l'intervalle qui les separe et compte les
    points qui ne sont pas STRICTEMENT du bon cote. Une marge nulle produit
    donc au moins une violation, comme il se doit.
    Retourne (violations, margin_m, boundary).
    """
    if not group_a or not group_b:
        return len(group_a) + len(group_b), 0.0, 0.0
    ux, uy = math.cos(angle), math.sin(angle)
    pa = [xy[i][0] * ux + xy[i][1] * uy for i in group_a]
    pb = [xy[i][0] * ux + xy[i][1] * uy for i in group_b]
    hi_a, lo_b = max(pa), min(pb)
    boundary = (hi_a + lo_b) / 2.0
    violations = (sum(1 for v in pa if v >= boundary)
                  + sum(1 for v in pb if v <= boundary))
    return violations, lo_b - hi_a, boundary


def _partition_key(group_a, group_b):
    """Cle canonique d'une partition NON ordonnee : le groupe contenant le
    plus petit index sert de reference. Deux balayages symetriques donnent
    donc la meme cle et la partition n'est comptee qu'une fois."""
    ta, tb = tuple(sorted(group_a)), tuple(sorted(group_b))
    return ta if (ta and tb and ta[0] < tb[0]) else tb


def enumerate_territorial_partitions(points, delivery_indices, group_size,
                                     max_samples=TERRITORIAL_MAX_SAMPLES):
    """Enumere les partitions separables par une droite, dedupliquees.

    Retourne (candidates, stats). Chaque candidate porte son certificat :
    {group_a, group_b, angle, margin_m, violations}.
    Purement geometrique : aucune matrice, aucun appel reseau.
    """
    stats = {"generated": 0, "unique": 0, "rejected_margin": 0, "samples": 0}
    if len(delivery_indices) < 2 or group_size <= 0:
        return [], stats

    xy = _local_xy(points, delivery_indices)
    angles = _sample_angles(_critical_angles(xy, delivery_indices))
    if len(angles) > max_samples:
        step = len(angles) / float(max_samples)
        angles = [angles[int(k * step)] for k in range(max_samples)]
    stats["samples"] = len(angles)

    seen = {}
    for ang in angles:
        ga, gb, margin = _split_by_angle(xy, delivery_indices, ang, group_size)
        stats["generated"] += 1
        if ga is None:
            stats["rejected_margin"] += 1
            continue
        viol, cert_margin, _ = _territorial_certificate(xy, ga, gb, ang)
        if viol != 0 or cert_margin <= 0.0:
            stats["rejected_margin"] += 1
            continue
        key = _partition_key(ga, gb)
        # A cle egale, on garde la plus grande marge : la separation la plus nette.
        if key not in seen or cert_margin > seen[key]["margin_m"]:
            seen[key] = {"group_a": ga, "group_b": gb, "angle": ang,
                         "margin_m": cert_margin, "violations": viol}

    # Ordre stable, independant de l'ordre d'insertion du dictionnaire.
    candidates = sorted(seen.values(), key=lambda c: (_partition_key(c["group_a"], c["group_b"]),))
    stats["unique"] = len(candidates)
    return candidates, stats


def _nn_route_matrix(matrix, group, start_idx, end_idx):
    """Nearest-neighbour sur la matrice, en index GLOBAUX.
    La cle de comparaison inclut l'index : les egalites sont departagees de
    facon stable et le resultat ne depend pas de l'ordre d'iteration."""
    remaining = list(group)
    route = [start_idx]
    cur = start_idx
    while remaining:
        nxt = min(remaining, key=lambda j: (matrix[cur][j], j))
        route.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    route.append(end_idx)
    return route


def _estimate_group_cost(matrix, group, start_idx, end_idx, refine):
    """Cout estime d'un groupe sur la matrice fournie, en index globaux.
    Nearest-neighbour puis, si refine, Or-opt et 2-opt matriciels — les memes
    que ceux du pipeline. Local et deterministe, aucun appel reseau."""
    if not group:
        return 0.0, [start_idx, end_idx]
    route = _nn_route_matrix(matrix, group, start_idx, end_idx)
    if refine:
        route = _or_opt_matrix(matrix, route)
        route = _two_opt_matrix(matrix, route)
    return _matrix_route_cost(matrix, route), route


def select_territorial_partition(candidates, dur_matrix, dist_matrix,
                                 start_idx, end_idx,
                                 top_refine=TERRITORIAL_TOP_REFINE):
    """Choisit la meilleure partition territoriale.

    Ordre lexicographique impose : violations, puis DUREE ORS totale, puis
    distance ORS totale, puis cle canonique. Aucun terme d'equilibrage n'entre
    dans le cout : une tournee peut etre bien plus longue que l'autre.

    Deux etages pour tenir le temps de calcul : estimation rapide de toutes
    les candidates, puis affinage des meilleures seulement.
    """
    stats = {"scored": 0, "refined": 0}
    if not candidates:
        return None, stats

    rough = []
    for c in candidates:
        da, _ = _estimate_group_cost(dur_matrix, c["group_a"], start_idx, end_idx, False)
        db, _ = _estimate_group_cost(dur_matrix, c["group_b"], start_idx, end_idx, False)
        rough.append((da + db, c))
        stats["scored"] += 1

    rough.sort(key=lambda t: (t[0], _partition_key(t[1]["group_a"], t[1]["group_b"])))

    best = None
    for _, c in rough[:max(1, top_refine)]:
        da, _ = _estimate_group_cost(dur_matrix, c["group_a"], start_idx, end_idx, True)
        db, _ = _estimate_group_cost(dur_matrix, c["group_b"], start_idx, end_idx, True)
        dur_total = da + db
        if dist_matrix:
            ka, _ = _estimate_group_cost(dist_matrix, c["group_a"], start_idx, end_idx, False)
            kb, _ = _estimate_group_cost(dist_matrix, c["group_b"], start_idx, end_idx, False)
            dist_total = ka + kb
        else:
            dist_total = 0.0
        stats["refined"] += 1

        key = (c["violations"], dur_total, dist_total,
               _partition_key(c["group_a"], c["group_b"]))
        if best is None or key < best[0]:
            best = (key, c, dur_total, dist_total)

    chosen = dict(best[1])
    chosen["est_duration_s"] = best[2]
    chosen["est_distance_m"] = best[3]
    return chosen, stats


def ortools_partition_ors_matrix(points, num_vehicles, max_per_vehicle,
                                 start_idx, end_idx, headers,
                                 solution_limit=None):
    """Partition TERRITORIALE sur les durees routieres reelles ORS.

    Priorites, dans cet ordre strict : deux territoires separables par une
    droite, exactement group_size points par tournee, aucune perte ni doublon,
    puis duree ORS totale minimale. Aucun objectif d'equilibrage.

    Retourne (groups, err, meta). meta porte le certificat territorial.
    """
    t0 = time.time()
    meta = {"territorial": {
        "territorial_partition": False,
        "territorial_method": "sweep_line_projection",
        "territorial_membership_locked": False,
        "territorial_candidates_generated": 0,
        "territorial_candidates_unique": 0,
        "territorial_candidates_scored": 0,
        "territorial_side_violations": None,
        "territorial_separator_angle_deg": None,
        "territorial_separator_margin_m": None,
        "territorial_overlap_status": "unknown",
        "territorial_fallback_used": False,
        "territorial_error": "",
        "territorial_enum_ms": 0,
        "territorial_score_ms": 0,
    }}
    terr = meta["territorial"]

    depot_indices = {start_idx, end_idx}
    delivery_indices = [i for i in range(len(points)) if i not in depot_indices]

    if not delivery_indices:
        terr["territorial_overlap_status"] = "no_points"
        return [[] for _ in range(num_vehicles)], None, meta

    if num_vehicles != 2:
        terr["territorial_error"] = (f"territorial partition requires 2 vehicles, "
                                     f"got {num_vehicles}")
        terr["territorial_overlap_status"] = "not_applicable"
        return None, terr["territorial_error"], meta

    n = len(delivery_indices)
    if num_vehicles * max_per_vehicle < n:
        terr["territorial_error"] = (f"capacity too small: {num_vehicles} x "
                                     f"{max_per_vehicle} < {n} points")
        return None, terr["territorial_error"], meta

    # Coordonnees inexploitables : on refuse plutot que de partitionner a l'aveugle.
    bad = [points[i].get("id", i) for i in delivery_indices
           if not _finite_coords(points[i])]
    if bad:
        terr["territorial_error"] = (f"{len(bad)} point(s) with invalid coordinates: "
                                     f"{bad[:5]}")
        terr["territorial_overlap_status"] = "invalid_coordinates"
        return None, terr["territorial_error"], meta

    # Repartition la plus egale que la capacite autorise. Pour 60 points, 30/30.
    group_size = n // 2
    if group_size > max_per_vehicle or (n - group_size) > max_per_vehicle:
        terr["territorial_error"] = (f"cannot split {n} points into two groups "
                                     f"under capacity {max_per_vehicle}")
        return None, terr["territorial_error"], meta

    print(f"Partition territoriale: {n} points -> {group_size}/{n - group_size}, "
          f"capacite {max_per_vehicle}", flush=True)

    dur_matrix, dist_matrix, mmeta, err = _build_full_matrix_chunked(points, headers)
    meta.update(mmeta)
    meta["territorial"] = terr
    if dur_matrix is None:
        terr["territorial_error"] = f"ORS matrix failed: {err}"
        return None, terr["territorial_error"], meta

    t_enum = time.time()
    candidates, cstats = enumerate_territorial_partitions(
        points, delivery_indices, group_size)
    terr["territorial_enum_ms"] = int((time.time() - t_enum) * 1000)
    terr["territorial_candidates_generated"] = cstats["generated"]
    terr["territorial_candidates_unique"] = cstats["unique"]

    print(f"  {cstats['samples']} angles balayes, {cstats['generated']} coupures, "
          f"{cstats['unique']} partitions uniques certifiees "
          f"({cstats['rejected_margin']} rejetees pour marge nulle) "
          f"en {terr['territorial_enum_ms']}ms", flush=True)

    if not candidates:
        # Aucune separation stricte : on NE retombe PAS sur l'ancienne partition
        # imbriquee, on echoue de facon explicite et diagnostiquee.
        terr["territorial_error"] = ("no strictly separating line found for "
                                     f"{group_size}/{n - group_size}")
        terr["territorial_overlap_status"] = "no_separator"
        return None, terr["territorial_error"], meta

    t_score = time.time()
    chosen, sstats = select_territorial_partition(
        candidates, dur_matrix, dist_matrix, start_idx, end_idx)
    terr["territorial_score_ms"] = int((time.time() - t_score) * 1000)
    terr["territorial_candidates_scored"] = sstats["scored"]

    if chosen is None:
        terr["territorial_error"] = "no candidate could be scored"
        terr["territorial_overlap_status"] = "scoring_failed"
        return None, terr["territorial_error"], meta

    groups = [chosen["group_a"], chosen["group_b"]]

    # Verification finale, independante de la construction ET de la selection.
    xy = _local_xy(points, delivery_indices)
    viol, margin, _ = _territorial_certificate(
        xy, chosen["group_a"], chosen["group_b"], chosen["angle"])

    union = set(chosen["group_a"]) | set(chosen["group_b"])
    inter = set(chosen["group_a"]) & set(chosen["group_b"])
    sizes_ok = (len(chosen["group_a"]) == group_size
                and len(chosen["group_b"]) == n - group_size)
    complete = (union == set(delivery_indices)) and not inter

    if viol != 0 or margin <= 0.0 or not sizes_ok or not complete:
        terr["territorial_side_violations"] = viol
        terr["territorial_separator_margin_m"] = round(margin, 2)
        terr["territorial_overlap_status"] = "overlapping"
        terr["territorial_error"] = (
            f"final check failed: violations={viol}, margin={margin:.2f}m, "
            f"sizes_ok={sizes_ok}, complete={complete}")
        return None, terr["territorial_error"], meta

    terr["territorial_partition"] = True
    terr["territorial_membership_locked"] = True
    terr["territorial_side_violations"] = 0
    terr["territorial_separator_angle_deg"] = round(math.degrees(chosen["angle"]) % 180.0, 3)
    terr["territorial_separator_margin_m"] = round(margin, 2)
    terr["territorial_overlap_status"] = "separated"
    terr["territorial_fallback_used"] = False
    terr["territorial_est_duration_min"] = round(chosen["est_duration_s"] / 60.0, 1)

    print(f"  Retenue: {[len(g) for g in groups]} pts, angle "
          f"{terr['territorial_separator_angle_deg']}deg, marge "
          f"{terr['territorial_separator_margin_m']}m, violations 0, "
          f"duree estimee {terr['territorial_est_duration_min']}min "
          f"({sstats['refined']} affinees en {terr['territorial_score_ms']}ms)", flush=True)
    print(f"  Partition: {[len(g) for g in groups]} pts "
          f"(total {int((time.time() - t0) * 1000)}ms)", flush=True)

    return groups, None, meta


# =========================
# 4e. PARTITION CONNEXE (strategie ortools_ors_matrix_connected)
# =========================
# La separation lineaire du mode sweep garantit des territoires nets, mais
# elle interdit des decoupages parfaitement acceptables : une vallee en U, un
# territoire en croissant autour d'un autre. Ici la contrainte est plus faible
# et plus proche du terrain : chaque territoire doit former UN SEUL bloc
# connexe dans un graphe de voisinage geographique. Aucune droite n'est exigee.

def _env_int(name, default, lo=0, hi=10 ** 7):
    """Budget surchargeable par variable d'environnement, borne des deux cotes.
    Une valeur illisible ou hors bornes retombe sur le defaut prudent : un
    reglage errone ne doit jamais faire exploser le temps de calcul."""
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if lo <= value <= hi else default


CONNECTED_KNN_START = 4          # k initial du graphe de voisinage
CONNECTED_KNN_MAX = 12           # au-dela, l'arbre couvrant prend le relais
CONNECTED_LOCAL_ITERS = 60       # bornes de la recherche locale
CONNECTED_REPAIR_ITERS = 200
CONNECTED_TIE_SECONDS = 30.0     # ecart de duree sous lequel on departage au km

# --- budgets de diversification ---
# La generation est LOCALE : elle ne consomme ni Matrix ni Vroom. Ses seules
# ressources sont le temps de calcul et la memoire, d'ou des plafonds explicites
# plutot qu'une exploration ouverte.
CONNECTED_TARGET_UNIQUE_CANDIDATES = _env_int("CONNECTED_TARGET_UNIQUE_CANDIDATES", 60, 2, 5000)
CONNECTED_MAX_UNIQUE_CANDIDATES = _env_int("CONNECTED_MAX_UNIQUE_CANDIDATES", 100, 2, 5000)
CONNECTED_MAX_RAW_CANDIDATES = _env_int("CONNECTED_MAX_RAW_CANDIDATES", 2000, 2, 100000)
CONNECTED_MAX_PERTURBATIONS = _env_int("CONNECTED_MAX_PERTURBATIONS", 250, 0, 20000)
CONNECTED_MAX_CHAIN_LENGTH = _env_int("CONNECTED_MAX_CHAIN_LENGTH", 4, 1, 12)
CONNECTED_MAX_REPAIRS = _env_int("CONNECTED_MAX_REPAIRS", 400, 1, 20000)
CONNECTED_LOCAL_SEARCH_SEEDS = _env_int("CONNECTED_LOCAL_SEARCH_SEEDS", 6, 0, 100)
# Places reservees aux candidates du generateur historique parmi les douze
# finalistes. Une seule ne suffit pas : le score proxy classe mal ces
# partitions -- sur un jeu reel, la meilleure au proxy rendait 11456 s apres
# OR-Tools quand une autre, classee neuvieme, en rendait 11347. Reserver la
# moitie des places laisse l'autre moitie a la diversite.
CONNECTED_LEGACY_FINALIST_SLOTS = _env_int("CONNECTED_LEGACY_FINALIST_SLOTS", 6, 1, 12)
CONNECTED_MAX_GENERATION_S = _env_int("CONNECTED_MAX_GENERATION_S", 25, 1, 600)
CONNECTED_ORS_NEIGHBOR_K = _env_int("CONNECTED_ORS_NEIGHBOR_K", 3, 0, 20)

# Plafonds d'appels payants. Ils ne bougent PAS avec la diversification :
# davantage de candidates locales, autant d'appels reseau.
CONNECTED_ORTOOLS_FINALISTS = _env_int("CONNECTED_ORTOOLS_FINALISTS", 12, 1, 60)
CONNECTED_VROOM_FINALISTS = _env_int("CONNECTED_VROOM_FINALISTS", 3, 1, 12)
CONNECTED_TOP_ORTOOLS = CONNECTED_ORTOOLS_FINALISTS   # candidates au solveur (niveau 2)
CONNECTED_TOP_VROOM = CONNECTED_VROOM_FINALISTS       # candidates a Vroom (niveau 3)

# Part des 12 finalistes reservee a la diversite d'appartenance. Les places
# restantes vont aux meilleurs scores, quelle que soit leur ressemblance.
CONNECTED_DIVERSE_SHARE = 0.5


def _symmetrised_ors(matrix, i, j):
    """Cout ORS symetrise entre deux points.

    La matrice ORS est DIRIGEE : d(i,j) et d(j,i) different, et l'une des deux
    peut manquer. Moyenne des deux quand elles sont finies, sinon la seule
    finie, sinon None -- la paire est alors declaree indisponible plutot que
    remplacee par une valeur inventee.
    """
    if not matrix:
        return None
    try:
        a, b = matrix[i][j], matrix[j][i]
    except (IndexError, TypeError):
        return None
    fa = a if isinstance(a, (int, float)) and math.isfinite(a) else None
    fb = b if isinstance(b, (int, float)) and math.isfinite(b) else None
    if fa is not None and fb is not None:
        return (fa + fb) / 2.0
    return fa if fa is not None else fb


def build_geo_graph(points, indices, k=CONNECTED_KNN_START, dur_matrix=None,
                    ors_k=CONNECTED_ORS_NEIGHBOR_K):
    """Graphe de voisinage non oriente sur les points de collecte.

    Trois apports, en UNION -- aucune arete n'est retiree :
      - k plus proches voisins au sens haversine, k augmente tant que le
        graphe global n'est pas connexe ;
      - ors_k plus proches voisins au sens de la DUREE ORS symetrisee, quand
        la matrice deja chargee est fournie. Deux points separes par une
        riviere sont proches a vol d'oiseau et lointains par la route : sans
        ces aretes, le graphe declare voisins des points que la voirie ne
        relie pas, et l'inverse ;
      - un arbre couvrant minimal, filet de securite : il garantit la
        connexite globale meme pour un point tres isole, ce que les kNN seuls
        ne donnent jamais.
    Retourne (adjacency, meta). Aucune dependance, AUCUN appel reseau : la
    matrice recue est celle deja en memoire.
    """
    n = len(indices)
    adjacency = {i: set() for i in indices}
    if n <= 1:
        return adjacency, {"k": 0, "mst_edges": 0, "connected": True,
                           "ors_k": 0, "ors_edges": 0, "tree_edges": [],
                           "method": "knn_haversine_mst", "edges": 0}

    dist = {}
    for a in range(n):
        ia = indices[a]
        for b in range(a + 1, n):
            ib = indices[b]
            d = haversine((float(points[ia]["lat"]), float(points[ia]["lon"])),
                          (float(points[ib]["lat"]), float(points[ib]["lon"])))
            dist[(ia, ib)] = d
            dist[(ib, ia)] = d

    def knn_edges(kk):
        edges = set()
        for i in indices:
            # tri par (distance, index) : les egalites sont departagees de
            # facon stable, le graphe est donc deterministe.
            near = sorted((j for j in indices if j != i),
                          key=lambda j: (dist[(i, j)], j))[:kk]
            for j in near:
                edges.add((min(i, j), max(i, j)))
        return edges

    used_k = min(k, max(1, n - 1))
    edges = knn_edges(used_k)
    while used_k < min(CONNECTED_KNN_MAX, n - 1):
        adj = {i: set() for i in indices}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        if _graph_connected(indices, adj):
            break
        used_k += 1
        edges = knn_edges(used_k)

    # Voisins ROUTIERS : ors_k plus proches au sens de la duree ORS symetrisee.
    # Les paires dont la matrice ne dit rien sont ignorees, jamais devinees.
    ors_added = 0
    if dur_matrix and ors_k > 0:
        for i in indices:
            costs = []
            for j in indices:
                if j == i:
                    continue
                c = _symmetrised_ors(dur_matrix, i, j)
                if c is not None:
                    costs.append((c, j))
            for _, j in sorted(costs)[:ors_k]:
                key = (min(i, j), max(i, j))
                if key not in edges:
                    edges.add(key)
                    ors_added += 1

    # Arbre couvrant minimal (Prim), en union : filet de securite. Ses aretes
    # sont conservees a part : couper l'une d'elles est une source de
    # partitions naturellement connexes.
    mst_added = 0
    tree_edges = []
    in_tree = {indices[0]}
    rest = set(indices[1:])
    while rest:
        best = min(((u, v) for u in in_tree for v in rest),
                   key=lambda e: (dist[e], e[0], e[1]))
        u, v = best
        key = (min(u, v), max(u, v))
        tree_edges.append(key)
        if key not in edges:
            edges.add(key)
            mst_added += 1
        in_tree.add(v)
        rest.discard(v)

    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    method = ("knn_haversine_ors_mst" if (dur_matrix and ors_k > 0)
              else "knn_haversine_mst")
    return adjacency, {"k": used_k, "mst_edges": mst_added,
                       "connected": _graph_connected(indices, adjacency),
                       "edges": len(edges),
                       "ors_k": ors_k if (dur_matrix and ors_k > 0) else 0,
                       "ors_edges": ors_added,
                       "tree_edges": sorted(tree_edges),
                       "method": method}


def _graph_connected(nodes, adjacency):
    """Le graphe induit par nodes est-il d'un seul tenant ?"""
    nodeset = set(nodes)
    if not nodeset:
        return True
    start = min(nodeset)
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in adjacency.get(cur, ()):
            if nb in nodeset and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(nodeset)


def is_connected_partition(group_ids, adjacency):
    """Connexite d'un groupe, evaluee sur le SOUS-GRAPHE INDUIT par ses seuls
    points. Retourne {connected, component_count, component_sizes, components}.
    Les tailles sont triees decroissantes, la sortie est donc deterministe."""
    nodeset = set(group_ids)
    seen = set()
    components = []
    for node in sorted(nodeset):
        if node in seen:
            continue
        comp = {node}
        stack = [node]
        seen.add(node)
        while stack:
            cur = stack.pop()
            for nb in adjacency.get(cur, ()):
                if nb in nodeset and nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c[0]))
    return {
        "connected": len(components) <= 1,
        "component_count": len(components),
        "component_sizes": [len(c) for c in components],
        "components": components,
    }


def boundary_metrics(ga, gb, adjacency, points):
    """Mesures geographiques secondaires : elles departagent, elles ne
    remplacent pas l'objectif de duree ORS."""
    sa, sb = set(ga), set(gb)
    cut_edges = 0
    cut_len = 0.0
    cross = 0
    enclaves = 0
    for group, other in ((sa, sb), (sb, sa)):
        for i in group:
            nbs = adjacency.get(i, set())
            foreign = sum(1 for j in nbs if j in other)
            cross += foreign
            if nbs and foreign > len(nbs) / 2.0:
                enclaves += 1
    for i in sa:
        for j in adjacency.get(i, ()):
            if j in sb:
                cut_edges += 1
                cut_len += haversine(
                    (float(points[i]["lat"]), float(points[i]["lon"])),
                    (float(points[j]["lat"]), float(points[j]["lon"]))) * 1000.0
    return {"cut_edges": cut_edges, "cut_length_m": round(cut_len, 1),
            "cross_neighbors": cross, "enclave_points": enclaves}


def canonical_partition_key(group_a, group_b):
    """Cle canonique d'une partition en DEUX groupes, insensible a l'echange.

    T1 = A, T2 = B et T1 = B, T2 = A designent la meme decoupe du terrain :
    seule l'etiquette du vehicule change. La cle retenue est donc la plus
    petite des deux ecritures ordonnees. Elle ne contient QUE l'appartenance :
    deux ordres de visite differents sur les memes groupes donnent la meme cle
    et ne comptent que pour UNE partition.
    """
    side_a = tuple(sorted(group_a))
    side_b = tuple(sorted(group_b))
    return min((side_a, side_b), (side_b, side_a))


def _source_family(seed):
    """Famille d'une source : "legacy:sweep_3" -> "legacy".

    La diversite reserve des places aux sources encore absentes. Sans ce
    regroupement, les onze graines historiques comptent pour onze sources
    differentes et rafleraient toutes les places au titre de la nouveaute,
    au detriment des geometries reellement distinctes.
    """
    return str(seed or "").split(":", 1)[0]


def _short_key(partition_key):
    """Empreinte courte et stable d'une partition, pour les journaux.
    Comparer deux runs a l'oeil sur trente index est illisible ; huit
    caracteres suffisent a dire "meme partition" ou "partition differente"."""
    if not partition_key:
        return "--------"
    raw = "|".join(",".join(str(i) for i in side) for side in partition_key)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def partition_difference(key_a, key_b):
    """Distance d'appartenance entre deux partitions canoniques.

    Nombre minimal de points qui changent de cote, l'echange des deux groupes
    etant pris en compte : deux partitions identiques a l'etiquette pres sont
    a distance 0. Sert a mesurer la diversite des finalistes, jamais a scorer
    une tournee.
    """
    a0, a1 = set(key_a[0]), set(key_a[1])
    b0, b1 = set(key_b[0]), set(key_b[1])
    direct = len(a0 - b0) + len(a1 - b1)
    swapped = len(a0 - b1) + len(a1 - b0)
    return min(direct, swapped)


def validate_partition(group_a, group_b, indices, target_a, adjacency):
    """Certificat complet d'une candidate, recompte sans faire confiance a sa
    construction. Retourne (ok, reason, info)."""
    sa, sb = set(group_a), set(group_b)
    allset = set(indices)
    if len(sa) != len(group_a) or len(sb) != len(group_b) or (sa & sb):
        return False, "duplicate", None
    if (sa | sb) != allset:
        return False, "lost_points", None
    if len(sa) != target_a or len(sb) != len(allset) - target_a:
        return False, "invalid_size", None
    ia = is_connected_partition(group_a, adjacency)
    ib = is_connected_partition(group_b, adjacency)
    if not (ia["connected"] and ib["connected"]):
        return False, "disconnected", (ia, ib)
    return True, "ok", (ia, ib)


def _move_cost_delta(dur_matrix, group_from, group_to, node, start_idx, end_idx):
    """Variation de duree estimee si node passe d'un groupe a l'autre.
    Estimation nearest-neighbour, locale et deterministe."""
    a0, _ = _estimate_group_cost(dur_matrix, group_from, start_idx, end_idx, False)
    b0, _ = _estimate_group_cost(dur_matrix, group_to, start_idx, end_idx, False)
    a1, _ = _estimate_group_cost(dur_matrix, [x for x in group_from if x != node],
                                 start_idx, end_idx, False)
    b1, _ = _estimate_group_cost(dur_matrix, list(group_to) + [node],
                                 start_idx, end_idx, False)
    return (a1 + b1) - (a0 + b0)


# Regles de reparation. Une partition ORS morcelee n'a pas UNE reparation
# naturelle mais plusieurs, toutes legitimes et deterministes : les faire
# toutes produit des territoires reellement differents a partir d'une seule
# source, au lieu d'un unique compromis arbitraire.
CONNECTED_COMPONENT_RULES = ("smallest", "farthest", "cheapest_ors")
CONNECTED_MOVE_RULES = ("ors_delta", "connectivity", "cross", "haversine")


def _component_choice(rule, comps, src, dst, points, dur_matrix):
    """Composante secondaire a absorber, selon la regle demandee."""
    secondary = comps[1:]
    if not secondary:
        return None
    if rule == "smallest":
        return min(secondary, key=lambda c: (len(c), c[0]))
    main = comps[0]
    if rule == "farthest":
        cx = sum(float(points[i]["lat"]) for i in main) / len(main)
        cy = sum(float(points[i]["lon"]) for i in main) / len(main)

        def far(c):
            ax = sum(float(points[i]["lat"]) for i in c) / len(c)
            ay = sum(float(points[i]["lon"]) for i in c) / len(c)
            return (-haversine((cx, cy), (ax, ay)), c[0])
        return min(secondary, key=far)
    if rule == "cheapest_ors" and dur_matrix:
        def absorb_cost(c):
            best = []
            for i in c:
                costs = [_symmetrised_ors(dur_matrix, i, j) for j in dst]
                costs = [v for v in costs if v is not None]
                best.append(min(costs) if costs else float("inf"))
            return (sum(best) / len(best), c[0])
        return min(secondary, key=absorb_cost)
    return min(secondary, key=lambda c: (len(c), c[0]))


def _move_choice_key(rule, node, src, dst, adjacency, points, dur_matrix,
                     start_idx, end_idx):
    """Cout d'un deplacement de frontiere, selon la regle demandee. Le noeud
    est toujours inclus dans la cle : les egalites restent deterministes."""
    dstset = set(dst)
    if rule == "connectivity":
        return (-sum(1 for nb in adjacency.get(node, ()) if nb in dstset), node)
    if rule == "cross":
        srcset = set(src) - {node}
        gain = (sum(1 for nb in adjacency.get(node, ()) if nb in srcset)
                - sum(1 for nb in adjacency.get(node, ()) if nb in dstset))
        return (gain, node)
    if rule == "haversine":
        cx = sum(float(points[i]["lat"]) for i in dst) / len(dst)
        cy = sum(float(points[i]["lon"]) for i in dst) / len(dst)
        return (haversine((cx, cy), (float(points[node]["lat"]),
                                     float(points[node]["lon"]))), node)
    return (_move_cost_delta(dur_matrix, src, dst, node, start_idx, end_idx), node)


def repair_to_connected_ex(ga, gb, adjacency, points, dur_matrix,
                           start_idx, end_idx, target_a,
                           component_rule="smallest", move_rule="ors_delta"):
    """Rend deux groupes connexes en conservant la cardinalite exacte.

    Deux temps : absorber les composantes secondaires -- iles et enclaves --
    dans le groupe voisin, puis retablir la cardinalite en deplacant des points
    de FRONTIERE, et seulement s'ils laissent les deux groupes connexes.
    Les deux regles de choix sont parametrees : la meme partition morcelee
    donne donc plusieurs reparations valides et distinctes.
    Retourne (ga, gb, ok, moves).
    """
    ga, gb = list(ga), list(gb)
    moves = 0

    for _ in range(CONNECTED_REPAIR_ITERS):
        ia = is_connected_partition(ga, adjacency)
        ib = is_connected_partition(gb, adjacency)
        if ia["connected"] and ib["connected"]:
            break
        # Cote le plus morcele d'abord.
        if ia["component_count"] > 1:
            src, dst, comps = ga, gb, ia["components"]
        else:
            src, dst, comps = gb, ga, ib["components"]
        chosen = _component_choice(component_rule, comps, src, dst, points,
                                   dur_matrix)
        if chosen is None:
            return ga, gb, False, moves
        for node in chosen:
            src.remove(node)
            dst.append(node)
            moves += 1
        ga, gb = (src, dst) if src is ga else (dst, src)
    else:
        return ga, gb, False, moves

    # Retablissement de la cardinalite, sans jamais casser la connexite.
    for _ in range(CONNECTED_REPAIR_ITERS):
        if len(ga) == target_a:
            break
        if len(ga) > target_a:
            src, dst = ga, gb
        else:
            src, dst = gb, ga

        best = None
        dstset = set(dst)
        for node in sorted(src):
            # candidat de frontiere uniquement : il doit toucher l'autre groupe
            if not any(nb in dstset for nb in adjacency.get(node, ())):
                continue
            new_src = [x for x in src if x != node]
            new_dst = list(dst) + [node]
            if not is_connected_partition(new_src, adjacency)["connected"]:
                continue
            if not is_connected_partition(new_dst, adjacency)["connected"]:
                continue
            key = _move_choice_key(move_rule, node, src, dst, adjacency, points,
                                   dur_matrix, start_idx, end_idx)
            if best is None or key < best[0]:
                best = (key, new_src, new_dst)
        if best is None:
            return ga, gb, False, moves
        _, new_src, new_dst = best
        moves += 1
        if src is ga:
            ga, gb = new_src, new_dst
        else:
            gb, ga = new_src, new_dst
    else:
        return ga, gb, False, moves

    ok = (len(ga) == target_a
          and is_connected_partition(ga, adjacency)["connected"]
          and is_connected_partition(gb, adjacency)["connected"])
    return sorted(ga), sorted(gb), ok, moves


def repair_to_connected(ga, gb, adjacency, points, dur_matrix,
                        start_idx, end_idx, target_a):
    """Reparation par defaut : plus petite composante absorbee d'abord, points
    de frontiere choisis pour degrader le moins possible la duree ORS."""
    return repair_to_connected_ex(ga, gb, adjacency, points, dur_matrix,
                                  start_idx, end_idx, target_a,
                                  component_rule="smallest",
                                  move_rule="ors_delta")


def connected_local_search(ga, gb, adjacency, dur_matrix, start_idx, end_idx,
                           max_iters=CONNECTED_LOCAL_ITERS):
    """Echanges 1 contre 1 sur les points de frontiere.

    Un echange n'est retenu que s'il ameliore la duree estimee ET laisse les
    deux territoires connexes. La cardinalite est invariante par construction.
    Parcours trie : la recherche est deterministe.
    """
    ga, gb = list(ga), list(gb)
    base, _ = _estimate_group_cost(dur_matrix, ga, start_idx, end_idx, False)
    base += _estimate_group_cost(dur_matrix, gb, start_idx, end_idx, False)[0]
    swaps = 0

    for _ in range(max_iters):
        sa, sb = set(ga), set(gb)
        border_a = sorted(i for i in ga if any(nb in sb for nb in adjacency.get(i, ())))
        border_b = sorted(i for i in gb if any(nb in sa for nb in adjacency.get(i, ())))
        best = None
        for i in border_a:
            for j in border_b:
                na = [x for x in ga if x != i] + [j]
                nb_ = [x for x in gb if x != j] + [i]
                if not is_connected_partition(na, adjacency)["connected"]:
                    continue
                if not is_connected_partition(nb_, adjacency)["connected"]:
                    continue
                c = _estimate_group_cost(dur_matrix, na, start_idx, end_idx, False)[0]
                c += _estimate_group_cost(dur_matrix, nb_, start_idx, end_idx, False)[0]
                if c < base - 1e-9 and (best is None or (c, i, j) < (best[0], best[1], best[2])):
                    best = (c, i, j, na, nb_)
        if best is None:
            break
        base, _, _, ga, gb = best
        swaps += 1

    return sorted(ga), sorted(gb), base, swaps


def _farthest_pair(xy, indices):
    """Les deux points les plus eloignes, egalites departagees par les index."""
    seed_a, seed_b, best = indices[0], indices[min(1, len(indices) - 1)], -1.0
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            ia, ib = indices[a], indices[b]
            d = math.hypot(xy[ia][0] - xy[ib][0], xy[ia][1] - xy[ib][1])
            if (d, -ia, -ib) > (best, -seed_a, -seed_b):
                best, seed_a, seed_b = d, ia, ib
    return seed_a, seed_b


def _two_means_partition(points, indices, target_a, seeds=None):
    """2-moyennes local et deterministe.

    Germes par defaut : les deux points les plus eloignes. En fournir d'autres
    -- extremes nord/sud, est/ouest, extremites de l'arbre couvrant -- fait
    converger l'algorithme vers des bassins differents : c'est une source de
    diversite gratuite, sans dependance ni appel reseau.
    """
    if len(indices) < 2:
        return list(indices), []
    xy = _local_xy(points, indices)
    if seeds is None:
        seed_a, seed_b = _farthest_pair(xy, indices)
    else:
        seed_a, seed_b = seeds
        if seed_a == seed_b or seed_a not in xy or seed_b not in xy:
            return [], []
    ca, cb = xy[seed_a], xy[seed_b]
    for _ in range(25):
        scored = sorted(
            ((math.hypot(xy[i][0] - ca[0], xy[i][1] - ca[1])
              - math.hypot(xy[i][0] - cb[0], xy[i][1] - cb[1]), i)
             for i in indices))
        ga = [i for _, i in scored[:target_a]]
        gb = [i for _, i in scored[target_a:]]
        na = (sum(xy[i][0] for i in ga) / len(ga), sum(xy[i][1] for i in ga) / len(ga))
        nb = (sum(xy[i][0] for i in gb) / len(gb), sum(xy[i][1] for i in gb) / len(gb)) if gb else cb
        if na == ca and nb == cb:
            break
        ca, cb = na, nb
    return sorted(ga), sorted(gb)


def _normalize_sizes(ga, gb, indices, target_a, xy):
    """Ramene les groupes a la cardinalite exacte en deplacant les points les
    plus proches du centroide oppose. Deterministe."""
    ga, gb = list(ga), list(gb)
    allset = set(indices)
    ga = [i for i in ga if i in allset]
    gb = [i for i in gb if i in allset and i not in set(ga)]
    missing = sorted(allset - set(ga) - set(gb))
    gb.extend(missing)
    while len(ga) > target_a and gb is not None:
        cb = (sum(xy[i][0] for i in gb) / len(gb), sum(xy[i][1] for i in gb) / len(gb)) if gb else (0, 0)
        move = min(ga, key=lambda i: (math.hypot(xy[i][0] - cb[0], xy[i][1] - cb[1]), i))
        ga.remove(move)
        gb.append(move)
    while len(ga) < target_a:
        ca = (sum(xy[i][0] for i in ga) / len(ga), sum(xy[i][1] for i in ga) / len(ga)) if ga else (0, 0)
        move = min(gb, key=lambda i: (math.hypot(xy[i][0] - ca[0], xy[i][1] - ca[1]), i))
        gb.remove(move)
        ga.append(move)
    return sorted(ga), sorted(gb)


# =========================
# 4f. SOURCES DE PARTITIONS CONNEXES
# =========================
# Une seule source produit peu de decoupages reellement differents : le
# balayage rend des tranches paralleles, les 2-moyennes un unique bassin. Or la
# meilleure partition connexe d'un terrain reel n'est presque jamais celle que
# suggere une seule geometrie. D'ou plusieurs sources INDEPENDANTES, toutes
# deterministes et toutes locales : aucune ne declenche le moindre appel
# reseau, elles se contentent de la matrice ORS deja chargee.
#
# Chaque source rend des APPARTENANCES brutes (deux listes d'index). La
# validation, la reparation et la deduplication canonique sont communes et
# appliquees ensuite, une seule fois.


def _sweep_membership_candidates(points, indices, target_a, budget):
    """SOURCE 1 -- balayage angulaire enrichi.

    Les angles critiques sont ceux ou l'ordre des projections change : entre
    deux d'entre eux, la partition est constante. Les parcourir tous donne
    toutes les decoupes separables par une droite, et non quelques angles
    fixes. Les coupes legerement decalees autour de la taille cible s'ajoutent
    a l'ensemble : reparees ensuite, elles menent a des territoires que la
    coupe exacte ne produit jamais.
    """
    out = []
    if len(indices) < 2 or target_a <= 0:
        return out
    xy = _local_xy(points, indices)
    angles = _sample_angles(_critical_angles(xy, indices))
    if len(angles) > TERRITORIAL_MAX_SAMPLES:
        step = len(angles) / float(TERRITORIAL_MAX_SAMPLES)
        angles = [angles[int(k * step)] for k in range(TERRITORIAL_MAX_SAMPLES)]
    offsets = [0, 1, -1, 2, -2]
    # Deduplication LOCALE, indispensable : deux angles critiques consecutifs
    # ne changent l'ordre que de deux points, souvent loin de la coupure, et
    # rendent donc la meme appartenance. Sans ce filtre, le budget se depense
    # entierement sur les premiers degres de rotation et les decoupes des
    # autres directions ne sont jamais atteintes.
    seen = set()
    for ang in angles:
        ux, uy = math.cos(ang), math.sin(ang)
        proj = sorted((xy[i][0] * ux + xy[i][1] * uy, i) for i in indices)
        for off in offsets:
            size = target_a + off
            if not (0 < size < len(indices)):
                continue
            ga = [i for _, i in proj[:size]]
            gb = [i for _, i in proj[size:]]
            key = canonical_partition_key(ga, gb)
            if key in seen:
                continue
            seen.add(key)
            out.append(("sweep", ga, gb))
            if len(out) >= budget:
                return out
    return out


def _mst_cut_candidates(points, indices, target_a, tree_edges, budget):
    """SOURCE 2 -- coupures de l'arbre couvrant.

    Retirer UNE arete d'un arbre le scinde exactement en deux composantes,
    toutes deux connexes par construction. La taille de ces composantes est
    imposee par l'arbre : on ne garde que les coupures assez proches de la
    cible pour que le rattrapage reste marginal. Une coupure qui demanderait
    de deplacer la moitie des points ne decrit plus le terrain, elle decrit la
    reparation -- elle est rejetee.
    """
    out = []
    n = len(indices)
    if n < 2 or not tree_edges:
        return out
    tree = {i: set() for i in indices}
    for u, v in tree_edges:
        if u in tree and v in tree:
            tree[u].add(v)
            tree[v].add(u)
    # Au-dela de ce decalage, la reparation dominerait la coupure.
    max_gap = max(2, n // 6)
    scored = []
    for u, v in sorted(tree_edges):
        if u not in tree or v not in tree:
            continue
        tree[u].discard(v)
        tree[v].discard(u)
        seen = {u}
        stack = [u]
        while stack:
            cur = stack.pop()
            for nb in tree[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        tree[u].add(v)
        tree[v].add(u)
        side_a = sorted(seen)
        side_b = sorted(set(indices) - seen)
        if not side_a or not side_b:
            continue
        gap = min(abs(len(side_a) - target_a), abs(len(side_b) - target_a))
        if gap > max_gap:
            continue
        scored.append((gap, (u, v), side_a, side_b))
    scored.sort(key=lambda t: (t[0], t[1]))
    for _gap, _edge, side_a, side_b in scored[:budget]:
        out.append(("mst", side_a, side_b))
    return out


def _seed_pairs(points, indices, adjacency, dur_matrix, tree_edges):
    """Paires de germes deterministes et geographiquement contrastees."""
    xy = _local_xy(points, indices)
    pairs = []

    def push(a, b):
        if a is not None and b is not None and a != b:
            pairs.append((min(a, b), max(a, b)))

    push(*_farthest_pair(xy, indices))
    lat = sorted(indices, key=lambda i: (float(points[i]["lat"]), i))
    lon = sorted(indices, key=lambda i: (float(points[i]["lon"]), i))
    push(lat[0], lat[-1])                      # extremes nord / sud
    push(lon[0], lon[-1])                      # extremes est / ouest
    # extremes selon plusieurs directions de projection
    for deg in (30, 60, 120, 150):
        ang = math.radians(deg)
        ux, uy = math.cos(ang), math.sin(ang)
        proj = sorted(((xy[i][0] * ux + xy[i][1] * uy), i) for i in indices)
        push(proj[0][1], proj[-1][1])
    # extremites de l'arbre couvrant : ses feuilles les plus eloignees
    if tree_edges:
        degree = {i: 0 for i in indices}
        for u, v in tree_edges:
            if u in degree:
                degree[u] += 1
            if v in degree:
                degree[v] += 1
        leaves = sorted(i for i in indices if degree.get(i, 0) <= 1)
        if len(leaves) >= 2:
            push(*_farthest_pair(xy, leaves))
    # paire la plus eloignee au sens de la DUREE ORS, pas du vol d'oiseau
    if dur_matrix:
        best, pair = -1.0, None
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                c = _symmetrised_ors(dur_matrix, indices[a], indices[b])
                if c is not None and c > best:
                    best, pair = c, (indices[a], indices[b])
        if pair:
            push(*pair)
    # paires purement deterministes fondees sur les identifiants
    ordered_ids = sorted(indices)
    push(ordered_ids[0], ordered_ids[-1])
    push(ordered_ids[0], ordered_ids[len(ordered_ids) // 2])
    seen, uniq = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _grow_two_regions(indices, target_a, adjacency, points, dur_matrix,
                      seed_a, seed_b, rule, alternate):
    """SOURCE 3 -- croissance de deux regions depuis deux germes.

    Les deux groupes n'absorbent que des VOISINS DU GRAPHE : chacun reste donc
    connexe a chaque etape, sans aucune verification a posteriori. La
    cardinalite est atteinte par construction. Plusieurs regles de choix et
    deux politiques de tour produisent des frontieres nettement differentes
    pour une meme paire de germes.
    """
    xy = _local_xy(points, indices)
    remaining = set(indices) - {seed_a, seed_b}
    ga, gb = [seed_a], [seed_b]
    target_b = len(indices) - target_a

    def cost(group, node):
        if rule == "ors" and dur_matrix:
            vals = [_symmetrised_ors(dur_matrix, node, i) for i in group]
            vals = [v for v in vals if v is not None]
            return min(vals) if vals else float("inf")
        if rule == "continuity":
            gset = set(group)
            return -sum(1 for nb in adjacency.get(node, ()) if nb in gset)
        if rule == "centroid":
            cx = sum(xy[i][0] for i in group) / len(group)
            cy = sum(xy[i][1] for i in group) / len(group)
            return math.hypot(xy[node][0] - cx, xy[node][1] - cy)
        return min(math.hypot(xy[node][0] - xy[i][0],
                              xy[node][1] - xy[i][1]) for i in group)

    turn = 0
    while remaining:
        if alternate:
            first = ga if turn % 2 == 0 else gb
        else:
            # priorite au groupe auquel il reste le plus de places
            slack_a, slack_b = target_a - len(ga), target_b - len(gb)
            first = ga if (slack_a, 0) >= (slack_b, 1) else gb
        order = [first, gb if first is ga else ga]
        moved = False
        for group in order:
            limit = target_a if group is ga else target_b
            if len(group) >= limit:
                continue
            gset = set(group)
            frontier = sorted(node for node in remaining
                              if any(nb in gset for nb in adjacency.get(node, ())))
            if not frontier:
                continue
            pick = min(frontier, key=lambda node: (cost(group, node), node))
            group.append(pick)
            remaining.discard(pick)
            moved = True
            turn += 1
            break
        if not moved:
            # Les deux groupes sont enclaves : les points restants ne touchent
            # plus aucun des deux. Le reliquat part au groupe non sature et la
            # candidate est marquee INCOMPLETE -- la reparation commune s'en
            # chargera. L'abandonner rendrait la source totalement muette sur
            # les terrains en impasse, ce qui est exactement ce qu'on observait.
            leftovers = sorted(remaining)
            (ga if len(ga) < target_a else gb).extend(leftovers)
            remaining.clear()
            return sorted(ga), sorted(gb), False
    return sorted(ga), sorted(gb), True


def _region_growing_candidates(points, indices, target_a, adjacency,
                               dur_matrix, tree_edges, budget):
    """SOURCE 3 -- toutes les paires de germes croisees avec toutes les regles.

    Deux etiquettes, pour que le bilan par source reste lisible :
      - "region_growing" : croissance menee a terme uniquement par voisinage,
        donc cardinalite exacte et une seule composante par groupe, CERTIFIEES
        ici meme ;
      - "region_growing_repaired" : terrain en impasse, les deux regions se
        sont enclavees avant d'atteindre la cible. La candidate part quand meme
        vers la reparation commune. La rejeter rendait la source entierement
        muette sur certains jeux, ce qui se lisait a tort comme une source qui
        ne produit rien.
    """
    out = []
    if len(indices) < 2:
        return out
    for seed_a, seed_b in _seed_pairs(points, indices, adjacency, dur_matrix,
                                      tree_edges):
        for rule in ("haversine", "ors", "continuity", "centroid"):
            for alternate in (True, False):
                ga, gb, complete = _grow_two_regions(
                    indices, target_a, adjacency, points, dur_matrix, seed_a,
                    seed_b, rule, alternate)
                if not ga or not gb:
                    continue
                if complete:
                    ok, _reason, _info = validate_partition(ga, gb, indices,
                                                            target_a, adjacency)
                    if not ok:
                        continue
                    out.append(("region_growing", ga, gb))
                else:
                    out.append(("region_growing_repaired", ga, gb))
                if len(out) >= budget:
                    return out
    return out


def _two_means_candidates(points, indices, target_a, adjacency, dur_matrix,
                          tree_edges, budget):
    """SOURCE 4a -- 2-moyennes, une execution par paire de germes."""
    out = []
    for seeds in [None] + _seed_pairs(points, indices, adjacency, dur_matrix,
                                      tree_edges):
        ga, gb = _two_means_partition(points, indices, target_a, seeds=seeds)
        if ga and gb:
            out.append(("two_means", ga, gb))
        if len(out) >= budget:
            break
    return out


def _kmedoids_candidates(points, indices, target_a, adjacency, dur_matrix,
                         tree_edges, budget, max_iters=12):
    """SOURCE 4b -- K-Medoids sur la DUREE ORS symetrisee.

    Les centres sont de VRAIS points, pas des barycentres : sur un reseau
    routier, le milieu geometrique de deux points tombe souvent la ou aucune
    route ne passe. Le cout est la duree ORS symetrisee, deja en memoire --
    aucun appel supplementaire, aucune dependance lourde.
    """
    out = []
    if not dur_matrix or len(indices) < 2:
        return out

    def cost(i, j):
        c = _symmetrised_ors(dur_matrix, i, j)
        return c if c is not None else float("inf")

    for seeds in _seed_pairs(points, indices, adjacency, dur_matrix, tree_edges):
        ma, mb = seeds
        ga, gb = [], []
        for _ in range(max_iters):
            ranked = sorted(((cost(i, ma) - cost(i, mb), i) for i in indices))
            ga = [i for _, i in ranked[:target_a]]
            gb = [i for _, i in ranked[target_a:]]
            if not ga or not gb:
                break
            new_a = min(ga, key=lambda c: (sum(cost(c, i) for i in ga), c))
            new_b = min(gb, key=lambda c: (sum(cost(c, i) for i in gb), c))
            if (new_a, new_b) == (ma, mb):
                break
            ma, mb = new_a, new_b
        if ga and gb:
            out.append(("kmedoids", sorted(ga), sorted(gb)))
        if len(out) >= budget:
            break
    return out


def _unconstrained_partitions(indices, dur_matrix, hav_matrix, start_idx,
                              end_idx):
    """Les deux partitions NON CONTRAINTES du solveur : sur les durees ORS et
    sur le vol d'oiseau.

    Calculees UNE SEULE FOIS et partagees par la source historique et par les
    reparations multiples. Chaque appel a _solve_cvrp_ortools coute plusieurs
    secondes du budget de generation ; les dupliquer priverait les sources
    suivantes de leur temps.
    Retourne [(label, ga, gb), ...], eventuellement vide.
    """
    out = []
    if not ORTOOLS_AVAILABLE:
        return out
    g, _err = _solve_cvrp_ortools(dur_matrix, 2, len(indices), start_idx, end_idx)
    if g and len(g) == 2:
        out.append(("ors_unconstrained", list(g[0]), list(g[1])))
    gh, _err = _solve_cvrp_ortools(hav_matrix, 2, len(indices), start_idx, end_idx)
    if gh and len(gh) == 2:
        out.append(("haversine", list(gh[0]), list(gh[1])))
    return out


def legacy_connected_candidates(points, indices, target_a, adjacency, dur_matrix,
                                start_idx, end_idx, unconstrained=None):
    """SOURCE 0 -- reproduction EXACTE du generateur d'avant diversification.

    Cette source n'est pas une redite : le generateur d'origine terminait
    chaque candidate par connected_local_search(), une montee de colline par
    echanges 1 contre 1 sur la frontiere. L'appartenance qu'elle rendait
    n'etait donc PAS l'appartenance brute du balayage, et aucune des nouvelles
    sources ne la reproduit. En la retirant, la diversification avait
    silencieusement supprime l'incumbent historique du jeu de candidates : sur
    la signature 42dd749a, la solution a 6641 s n'etait tout simplement plus
    proposee au solveur.

    Ses candidates sont etiquetees "legacy:<seed>" et beneficient d'une place
    reservee parmi les finalistes : la diversification ajoute des solutions,
    elle n'en retire jamais.
    Retourne [(source, ga, gb), ...].
    """
    out = []
    xy = _local_xy(points, indices)
    raw = list(unconstrained or [])

    # Balayage lineaire : les huit premieres coupes strictement separables,
    # dans l'ordre historique (cle canonique croissante).
    sweep, _sstats = enumerate_territorial_partitions(points, indices, target_a)
    for rank, c in enumerate(sweep[:8]):
        raw.append(("sweep_%d" % rank, c["group_a"], c["group_b"]))

    ka, kb = _two_means_partition(points, indices, target_a)
    if ka and kb:
        raw.append(("two_means", ka, kb))

    for seed, ga, gb in raw:
        ga, gb = _normalize_sizes(ga, gb, indices, target_a, xy)
        ga, gb, ok, _moves = repair_to_connected(
            ga, gb, adjacency, points, dur_matrix, start_idx, end_idx, target_a)
        if not ok:
            continue
        # L'etape que la diversification avait perdue.
        ga, gb, _est, _swaps = connected_local_search(
            ga, gb, adjacency, dur_matrix, start_idx, end_idx)
        out.append(("legacy:" + seed, sorted(ga), sorted(gb)))
    return out


def _local_search_candidates(seeds, adjacency, dur_matrix, start_idx, end_idx,
                             budget):
    """Optimum local des meilleures candidates NOUVELLES.

    Meme montee de colline que la source historique, appliquee cette fois aux
    partitions issues des nouvelles sources : une appartenance brute et son
    optimum local sont deux territoires distincts, et c'est le second qui
    gagnait historiquement.
    """
    out = []
    for ga, gb in seeds[:budget]:
        na, nb, _est, swaps = connected_local_search(ga, gb, adjacency,
                                                     dur_matrix, start_idx,
                                                     end_idx)
        if swaps:
            out.append(("local_search", sorted(na), sorted(nb)))
    return out


def _ors_repair_candidates(points, indices, target_a, adjacency, dur_matrix,
                           hav_matrix, start_idx, end_idx, budget,
                           unconstrained=None):
    """SOURCE 5 -- reparations multiples de la partition ORS non contrainte.

    La partition que le solveur trouve sans contrainte de connexite est la
    meilleure en duree ; elle est simplement morcelee. Elle n'a pas UNE
    reparation naturelle : chaque regle d'absorption et chaque regle de
    deplacement en donnent une differente, toutes valides. Les produire toutes
    conserve la performance de la source tout en explorant plusieurs
    territoires.
    """
    out = []
    labels = {"ors_unconstrained": "ors_repair", "haversine": "haversine_repair"}
    seeds = [(labels.get(lbl, lbl), ga, gb)
             for lbl, ga, gb in (unconstrained or [])]
    if not seeds:
        return out

    xy = _local_xy(points, indices)
    for label, ga0, gb0 in seeds:
        for comp_rule in CONNECTED_COMPONENT_RULES:
            for move_rule in CONNECTED_MOVE_RULES:
                ga, gb = _normalize_sizes(ga0, gb0, indices, target_a, xy)
                ga, gb, ok, _moves = repair_to_connected_ex(
                    ga, gb, adjacency, points, dur_matrix, start_idx, end_idx,
                    target_a, component_rule=comp_rule, move_rule=move_rule)
                if ok:
                    out.append((label, ga, gb))
                if len(out) >= budget:
                    return out
    return out


def _boundary_nodes(ga, gb, adjacency):
    """Points de chaque groupe qui touchent l'autre. Tries : deterministe."""
    sa, sb = set(ga), set(gb)
    ba = sorted(i for i in ga if any(nb in sb for nb in adjacency.get(i, ())))
    bb = sorted(i for i in gb if any(nb in sa for nb in adjacency.get(i, ())))
    return ba, bb


def _is_articulation(group, node, adjacency):
    """Retirer ce point deconnecte-t-il son groupe ? Test exact et bon marche
    a cette taille : un simple parcours du sous-graphe induit."""
    rest = [x for x in group if x != node]
    if len(rest) <= 1:
        return False
    return not is_connected_partition(rest, adjacency)["connected"]


def _perturbation_candidates(seeds, indices, target_a, adjacency, budget,
                             max_chain=CONNECTED_MAX_CHAIN_LENGTH):
    """SOURCE 6 -- voisines connexes des meilleures candidates.

    Echanges 1 contre 1, 2 contre 2 et chaines frontalieres. Seuls les points
    de FRONTIERE sont candidats : deplacer un point du coeur d'un territoire
    le deconnecte presque toujours et ne produit rien d'exploitable. Les
    points d'articulation sont ecartes d'emblee, puis la connexite est
    verifiee EXACTEMENT avant de conserver une variante -- le pre-filtre
    accelere, il ne decide pas.
    """
    out = []
    tried = 0
    for ga0, gb0 in seeds:
        ba, bb = _boundary_nodes(ga0, gb0, adjacency)
        safe_a = [i for i in ba if not _is_articulation(ga0, i, adjacency)]
        safe_b = [i for i in bb if not _is_articulation(gb0, i, adjacency)]

        # 1 contre 1
        for i in safe_a:
            for j in safe_b:
                if tried >= budget:
                    return out
                tried += 1
                na = sorted([x for x in ga0 if x != i] + [j])
                nb = sorted([x for x in gb0 if x != j] + [i])
                out.append(("perturbation", na, nb))

        # 2 contre 2 : uniquement des paires de points de frontiere adjacents,
        # sinon le nombre de combinaisons explose sans rien apporter.
        for a1 in range(len(safe_a)):
            for a2 in range(a1 + 1, len(safe_a)):
                i1, i2 = safe_a[a1], safe_a[a2]
                if i2 not in adjacency.get(i1, ()):
                    continue
                for b1 in range(len(safe_b)):
                    for b2 in range(b1 + 1, len(safe_b)):
                        j1, j2 = safe_b[b1], safe_b[b2]
                        if j2 not in adjacency.get(j1, ()):
                            continue
                        if tried >= budget:
                            return out
                        tried += 1
                        na = sorted([x for x in ga0 if x not in (i1, i2)] + [j1, j2])
                        nb = sorted([x for x in gb0 if x not in (j1, j2)] + [i1, i2])
                        out.append(("perturbation", na, nb))

        # chaines frontalieres : un chemin du graphe, compense par un chemin
        # de meme longueur pris dans l'autre groupe.
        for length in range(2, max_chain + 1):
            for chain_a in _border_chains(ga0, safe_a, adjacency, length):
                for chain_b in _border_chains(gb0, safe_b, adjacency, length):
                    if tried >= budget:
                        return out
                    tried += 1
                    na = sorted([x for x in ga0 if x not in chain_a] + list(chain_b))
                    nb = sorted([x for x in gb0 if x not in chain_b] + list(chain_a))
                    out.append(("perturbation", na, nb))
    return out


def _border_chains(group, border, adjacency, length, max_chains=8):
    """Chemins de `length` points adjacents, tous pris dans le groupe et
    demarrant sur la frontiere. Exploration bornee et triee : aucune
    combinatoire ouverte."""
    gset = set(group)
    chains = []
    for start in border:
        stack = [(start,)]
        while stack and len(chains) < max_chains:
            path = stack.pop()
            if len(path) == length:
                chains.append(path)
                continue
            for nb in sorted(adjacency.get(path[-1], ())):
                if nb in gset and nb not in path:
                    stack.append(path + (nb,))
        if len(chains) >= max_chains:
            break
    return chains


def generate_connected_candidates(points, indices, target_a, adjacency,
                                  dur_matrix, hav_matrix, start_idx, end_idx,
                                  tree_edges=None, deadline=None):
    """Partitions initiales diversifiees, puis reparees jusqu'a la connexite.

    Six sources independantes -- balayage enrichi, coupures de l'arbre
    couvrant, croissance de deux regions, 2-moyennes multi-germes, K-Medoids
    ORS, reparations multiples de la partition ORS -- puis des perturbations
    des meilleures. Chaque appartenance est dedupliquee AVANT la reparation
    quand elle est deja connue, validee, reparee si besoin, puis dedupliquee a
    nouveau. Aucune requete reseau : la matrice ORS recue est celle deja en
    memoire.
    Retourne (candidates, stats).
    """
    stats = {
        "generated": 0, "valid": 0, "sources": {},
        "raw": 0, "unique": 0, "duplicates": 0, "invalid_size": 0,
        "disconnected": 0, "repair_failed": 0, "repairs": 0,
        "by_source": {}, "perturbations": 0, "timeout": False,
        # Bilan DETAILLE par source. Un simple compteur d'uniques ne dit pas
        # si une source est muette parce qu'elle ne produit rien, parce que
        # tout est deja connu, ou parce que ses variantes sont morcelees.
        "per_source": {}, "legacy_keys": [], "expired_after": None,
    }
    xy = _local_xy(points, indices)
    tree_edges = tree_edges or []
    if deadline is None:
        deadline = time.time() + CONNECTED_MAX_GENERATION_S

    def expired(stage=None):
        if time.time() >= deadline:
            stats["timeout"] = True
            if stats["expired_after"] is None:
                stats["expired_after"] = stage
            return True
        return False

    def bucket(source):
        return stats["per_source"].setdefault(
            source, {"raw": 0, "duplicates": 0, "invalid_size": 0,
                     "disconnected": 0, "repair_failed": 0, "unique": 0})

    # Les deux resolutions non contraintes servent A LA FOIS a la source
    # historique et aux reparations multiples : on les calcule une seule fois.
    unconstrained = _unconstrained_partitions(indices, dur_matrix, hav_matrix,
                                              start_idx, end_idx)

    # --- collecte des appartenances brutes, source par source ---
    raw_budget = CONNECTED_MAX_RAW_CANDIDATES

    # SOURCE 0 : les candidates du generateur historique, hors entrelacement.
    # Elles passent AVANT tout le reste et ne peuvent donc jamais etre evincees
    # par le plafond de partitions uniques.
    legacy = legacy_connected_candidates(points, indices, target_a, adjacency,
                                         dur_matrix, start_idx, end_idx,
                                         unconstrained=unconstrained)

    batches = []
    batches.append(_ors_repair_candidates(points, indices, target_a, adjacency,
                                          dur_matrix, hav_matrix, start_idx,
                                          end_idx, raw_budget,
                                          unconstrained=unconstrained))
    if not expired("mst"):
        batches.append(_mst_cut_candidates(points, indices, target_a, tree_edges,
                                           raw_budget))
    if not expired("region_growing"):
        batches.append(_region_growing_candidates(points, indices, target_a,
                                                  adjacency, dur_matrix,
                                                  tree_edges, raw_budget))
    if not expired("two_means"):
        batches.append(_two_means_candidates(points, indices, target_a, adjacency,
                                             dur_matrix, tree_edges, raw_budget))
    if not expired("kmedoids"):
        batches.append(_kmedoids_candidates(points, indices, target_a, adjacency,
                                            dur_matrix, tree_edges, raw_budget))
    if not expired("sweep"):
        batches.append(_sweep_membership_candidates(points, indices, target_a,
                                                    raw_budget))

    # Entrelacement round-robin. Concatener les sources laisserait la premiere
    # -- le balayage, qui produit a lui seul des milliers de coupes -- remplir
    # le quota de partitions uniques avant que les autres aient ete essayees :
    # on aurait beaucoup de candidates et une seule geometrie.
    raw = []
    for rank in range(max((len(b) for b in batches), default=0)):
        for batch in batches:
            if rank < len(batch):
                raw.append(batch[rank])
        if len(raw) >= raw_budget:
            break
    raw = raw[:raw_budget]

    seen = {}
    seen_raw = set()
    repairs_used = 0

    def consider(source, ga, gb, allow_repair=True):
        """Valide, repare si necessaire, deduplique. Retourne True si une
        NOUVELLE partition connexe a ete retenue."""
        nonlocal repairs_used
        b = bucket(source)
        b["raw"] += 1
        stats["raw"] += 1
        stats["generated"] += 1
        # Deduplication AVANT toute reparation : reparer deux fois la meme
        # appartenance brute coute cher et rend exactement le meme resultat.
        raw_key = canonical_partition_key(ga, gb)
        if raw_key in seen_raw:
            stats["duplicates"] += 1
            b["duplicates"] += 1
            return False
        seen_raw.add(raw_key)

        ok, reason, _info = validate_partition(ga, gb, indices, target_a,
                                               adjacency)
        if not ok and allow_repair:
            if reason in ("invalid_size", "disconnected", "lost_points",
                          "duplicate"):
                if repairs_used >= CONNECTED_MAX_REPAIRS or expired("repair"):
                    stats["repair_failed"] += 1
                    b["repair_failed"] += 1
                    return False
                repairs_used += 1
                stats["repairs"] += 1
                ga, gb = _normalize_sizes(ga, gb, indices, target_a, xy)
                ga, gb, done, _moves = repair_to_connected(
                    ga, gb, adjacency, points, dur_matrix, start_idx, end_idx,
                    target_a)
                if not done:
                    stats["repair_failed"] += 1
                    b["repair_failed"] += 1
                    return False
                ok, reason, _info = validate_partition(ga, gb, indices,
                                                       target_a, adjacency)
        if not ok:
            if reason == "disconnected":
                stats["disconnected"] += 1
                b["disconnected"] += 1
            elif reason == "invalid_size":
                stats["invalid_size"] += 1
                b["invalid_size"] += 1
            else:
                stats["repair_failed"] += 1
                b["repair_failed"] += 1
            return False

        # Deduplication APRES reparation : deux sources differentes convergent
        # tres souvent vers la meme partition reparee.
        key = canonical_partition_key(ga, gb)
        if key in seen:
            stats["duplicates"] += 1
            b["duplicates"] += 1
            return False
        est = (_estimate_group_cost(dur_matrix, ga, start_idx, end_idx, False)[0]
               + _estimate_group_cost(dur_matrix, gb, start_idx, end_idx, False)[0])
        seen[key] = {"group_a": ga, "group_b": gb, "seed": source,
                     "est_duration_s": est, "partition_key": key,
                     "legacy": source.startswith("legacy:")}
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        b["unique"] += 1
        stats["valid"] += 1
        return True

    # Les candidates historiques d'abord, SANS aucun plafond ni garde-fou de
    # temps : leur nombre est borne par construction (onze au plus) et elles
    # portent l'incumbent qu'il ne faut jamais perdre.
    for source, ga, gb in legacy:
        if consider(source, ga, gb):
            stats["legacy_keys"].append(canonical_partition_key(ga, gb))

    for source, ga, gb in raw:
        if len(seen) >= CONNECTED_MAX_UNIQUE_CANDIDATES or expired("consider"):
            break
        consider(source, ga, gb)

    # --- optimum local des meilleures nouvelles candidates ---
    # Meme montee de colline que la source historique : c'est elle qui
    # transformait une coupe de balayage en incumbent.
    if not expired("local_search") and len(seen) < CONNECTED_MAX_UNIQUE_CANDIDATES:
        fresh = sorted((c for c in seen.values() if not c["legacy"]),
                       key=lambda c: (c["est_duration_s"], c["partition_key"]))
        for source, ga, gb in _local_search_candidates(
                [(c["group_a"], c["group_b"]) for c in fresh], adjacency,
                dur_matrix, start_idx, end_idx, CONNECTED_LOCAL_SEARCH_SEEDS):
            if len(seen) >= CONNECTED_MAX_UNIQUE_CANDIDATES or expired("local_search"):
                break
            consider(source, ga, gb, allow_repair=False)

    # --- perturbations des meilleures candidates ---
    # Elles ne partent QUE de partitions deja valides : leurs voisines sont
    # donc presque toutes valides elles aussi, ce qui evite de depenser le
    # budget de reparation sur des variantes sans avenir.
    if (CONNECTED_MAX_PERTURBATIONS > 0
            and len(seen) < CONNECTED_TARGET_UNIQUE_CANDIDATES
            and not expired("perturbation")):
        best = sorted(seen.values(),
                      key=lambda c: (c["est_duration_s"], c["partition_key"]))[:4]
        variants = _perturbation_candidates(
            [(c["group_a"], c["group_b"]) for c in best], indices, target_a,
            adjacency, CONNECTED_MAX_PERTURBATIONS)
        stats["perturbations"] = len(variants)
        for source, ga, gb in variants:
            if len(seen) >= CONNECTED_MAX_UNIQUE_CANDIDATES or expired("perturbation"):
                break
            # Une voisine n'est retenue que si elle est valide TELLE QUELLE :
            # la reparer effacerait justement la perturbation.
            consider(source, ga, gb, allow_repair=False)

    stats["unique"] = len(seen)
    stats["sources"] = dict(stats["by_source"])
    candidates = sorted(seen.values(),
                        key=lambda c: (c["est_duration_s"], c["partition_key"]))
    return candidates, stats


def _fallback_connected_candidates(points, indices, target_a, adjacency,
                                   dur_matrix, hav_matrix, start_idx, end_idx):
    """Repli minimal si la diversification echoue : balayage et 2-moyennes,
    normalises puis repares. Peu de candidates, mais la strategie tient."""
    stats = {"generated": 0, "valid": 0, "sources": {}, "raw": 0, "unique": 0,
             "duplicates": 0, "invalid_size": 0, "disconnected": 0,
             "repair_failed": 0, "by_source": {}, "timeout": False}
    xy = _local_xy(points, indices)
    seen = {}
    raw = []
    sweep, _s = enumerate_territorial_partitions(points, indices, target_a)
    for c in sweep[:8]:
        raw.append(("sweep", c["group_a"], c["group_b"]))
    ka, kb = _two_means_partition(points, indices, target_a)
    if ka and kb:
        raw.append(("two_means", ka, kb))
    stats["generated"] = stats["raw"] = len(raw)

    for source, ga, gb in raw:
        ga, gb = _normalize_sizes(ga, gb, indices, target_a, xy)
        ga, gb, ok, _moves = repair_to_connected(
            ga, gb, adjacency, points, dur_matrix, start_idx, end_idx, target_a)
        if not ok:
            stats["repair_failed"] += 1
            continue
        key = canonical_partition_key(ga, gb)
        if key in seen:
            stats["duplicates"] += 1
            continue
        est = (_estimate_group_cost(dur_matrix, ga, start_idx, end_idx, False)[0]
               + _estimate_group_cost(dur_matrix, gb, start_idx, end_idx, False)[0])
        seen[key] = {"group_a": ga, "group_b": gb, "seed": source,
                     "est_duration_s": est, "partition_key": key}
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        stats["valid"] += 1
    stats["unique"] = len(seen)
    stats["sources"] = dict(stats["by_source"])
    return sorted(seen.values(),
                  key=lambda c: (c["est_duration_s"], c["partition_key"])), stats


def select_diverse_finalists(scored, limit,
                             diverse_share=CONNECTED_DIVERSE_SHARE,
                             protected_keys=()):
    """Choisit les finalistes en melangeant score et diversite d'appartenance.

    Prendre les `limit` meilleurs scores donne souvent douze variantes du meme
    decoupage a un point pres : le solveur les resequence toutes pour
    quasiment le meme resultat, et une decoupe franchement differente -- mais
    classee treizieme -- n'est jamais essayee. On reserve donc une partie des
    places a la distance d'appartenance et aux sources encore absentes.

    Deux garanties, dans cet ordre :
      - les partitions PROTEGEES entrent d'office. C'est la place reservee a
        la meilleure candidate du generateur historique : un proxy defavorable
        ne doit pas suffire a l'ecarter du banc d'essai OR-Tools, sinon la
        diversification peut degrader le resultat au lieu de l'enrichir ;
      - la MEILLEURE candidate au score est prise ensuite, sans condition.
    La diversite ne remplit que les places restantes.
    Retourne (finalistes, min_difference).
    """
    ordered = sorted(scored, key=_selection_key)
    if limit <= 0 or not ordered:
        return [], 0
    if len(ordered) <= limit:
        return ordered, _min_pairwise_difference(ordered)

    protected = set(protected_keys or ())
    chosen = [c for c in ordered if c["partition_key"] in protected][:limit]
    chosen_keys = [c["partition_key"] for c in chosen]

    quota = max(1, limit - int(limit * diverse_share))
    for cand in ordered:                        # les meilleurs scores ensuite
        if len(chosen) >= quota:
            break
        if cand["partition_key"] not in chosen_keys:
            chosen.append(cand)
            chosen_keys.append(cand["partition_key"])
    used_sources = {_source_family(c.get("seed")) for c in chosen}
    rank_of = {c["partition_key"]: r for r, c in enumerate(ordered)}

    while len(chosen) < limit:
        best, best_key = None, None
        for cand in ordered:
            if cand["partition_key"] in chosen_keys:
                continue
            diff = min((partition_difference(cand["partition_key"], k)
                        for k in chosen_keys), default=1)
            if diff == 0:
                continue
            # source encore absente d'abord, puis ecart d'appartenance, puis
            # rang de score : une candidate lointaine mais mediocre ne passe
            # jamais devant une candidate lointaine et bien classee.
            novel = 0 if _source_family(cand.get("seed")) not in used_sources else 1
            key = (novel, -diff, rank_of[cand["partition_key"]],
                   cand["partition_key"])
            if best_key is None or key < best_key:
                best, best_key = cand, key
        if best is None:
            break
        chosen.append(best)
        chosen_keys.append(best["partition_key"])
        used_sources.add(_source_family(best.get("seed")))

    # Complement si la diversite n'a pas trouve assez de candidates distinctes.
    for cand in ordered:
        if len(chosen) >= limit:
            break
        if cand["partition_key"] not in chosen_keys:
            chosen.append(cand)
            chosen_keys.append(cand["partition_key"])
    chosen.sort(key=_selection_key)
    return chosen, _min_pairwise_difference(chosen)


def _min_pairwise_difference(cands):
    """Plus petit ecart d'appartenance entre deux finalistes. Zero signale des
    partitions identiques, donc un banc d'essai redondant."""
    keys = [c["partition_key"] for c in cands]
    if len(keys) < 2:
        return 0
    return min(partition_difference(keys[a], keys[b])
               for a in range(len(keys)) for b in range(a + 1, len(keys)))


def _rescore(dur_matrix, dist_matrix, route_a, route_b):
    """Rescore DEUX ordres avec la MEME matrice ORS.

    Indispensable : les durees rendues par Vroom et celles calculees par
    OR-Tools ne sortent pas du meme estimateur. Les comparer directement
    fausserait le classement. Ici tout repasse par la matrice.
    """
    d = _matrix_route_cost(dur_matrix, route_a) + _matrix_route_cost(dur_matrix, route_b)
    k = (_matrix_route_cost(dist_matrix, route_a) + _matrix_route_cost(dist_matrix, route_b)
         ) if dist_matrix else 0.0
    return d, k


def _selection_key(cand):
    """Ordre de PRESELECTION rapide. L'equilibre entre tournees n'y figure pas.

    Cette cle CLASSE les candidates -- pour choisir les 12 envoyees a OR-Tools
    puis les 3 envoyees a Vroom -- elle ne DESIGNE PAS le gagnant final. Le
    gagnant sort de select_best_solution(), appliquee a l'ensemble complet des
    solutions. L'ancienne version arrondissait la duree au palier de tolerance
    (round(duree / 30)) : deux solutions distantes d'une seconde pouvaient
    tomber dans deux paliers differents alors que trois solutions distantes de
    60s pouvaient s'enchainer dans le meme, ce qui rendait la comparaison pair
    a pair non transitive. La duree exacte supprime le probleme.
    """
    return (0 if cand["connected"] else 1,
            0 if cand["cardinality_ok"] else 1,
            cand["components_total"],
            cand["duration_s"],
            cand["distance_m"],
            cand["boundary"]["cut_edges"] + cand["boundary"]["enclave_points"],
            _partition_key(cand["group_a"], cand["group_b"]))


# Rang de departage des sequenceurs. Il n'intervient QUE sur une egalite
# parfaite -- meme duree ORS, meme distance ORS, meme qualite de frontiere,
# meme partition -- c'est-a-dire quand les deux ordres sont interchangeables.
# Aucun sequenceur n'est donc avantage dans la comparaison des metriques : ce
# rang evite seulement qu'un ordre issu d'un solveur soit annonce sous
# l'etiquette "heuristic" alors qu'un solveur l'a bel et bien produit.
_SEQUENCER_RANK = {"ortools": 0, "vroom": 1, "heuristic": 2}


def _solution_tiebreak(sol):
    """Departage DANS la fenetre de tolerance : distance ORS exacte d'abord,
    puis qualite geographique, puis criteres purement deterministes."""
    boundary = sol.get("boundary") or {}
    sequencer = sol.get("sequencer") or ""
    return (
        sol["distance_m"],
        boundary.get("cut_edges", 0) + boundary.get("enclave_points", 0),
        sol.get("partition_key") or (),
        _SEQUENCER_RANK.get(sequencer, 9),
        sequencer,
        tuple(sol.get("route_a") or ()) + tuple(sol.get("route_b") or ()),
    )


def select_best_solution(solutions, tie_seconds=CONNECTED_TIE_SECONDS):
    """Selection FINALE, sur l'ensemble complet des solutions.

    Une comparaison pair a pair avec tolerance n'est pas transitive : A ~ B et
    B ~ C n'impliquent pas A ~ C, et l'ordre d'examen changeait alors le
    gagnant. Ici la fenetre est calculee UNE FOIS, a partir du minimum global :

      1. ne retenir que les solutions valides (connexes, cardinalite exacte) ;
      2. best = min(duration_s) ;
      3. fenetre = {s : s.duration_s <= best + tie_seconds} ;
      4. dans la fenetre, distance ORS totale minimale ;
      5. egalite -> departage deterministe stable.

    Les durees et distances comparees sont les SECONDES et METRES ORS exacts,
    recalcules depuis les ordres reellement produits. Jamais des minutes
    arrondies, jamais un cout interne OR-Tools, jamais une duree brute Vroom.
    Retourne la solution gagnante, ou None si la liste est vide.
    """
    if not solutions:
        return None
    valid = [s for s in solutions
             if s.get("connected") and s.get("cardinality_ok")]
    pool = valid if valid else list(solutions)
    best_duration = min(s["duration_s"] for s in pool)
    window = [s for s in pool if s["duration_s"] <= best_duration + tie_seconds]
    return min(window, key=_solution_tiebreak)


# Une seule structure de solution circule du scoring jusqu'a la reponse : les
# routes, les metriques et le sequenceur voyagent ensemble et ne peuvent donc
# plus etre desynchronises par une variable ecrasee plus loin.
_SELECTION_REASONS = {
    "heuristic": "level1_heuristic",
    "ortools": "level2_ortools",
    "vroom": "level3_vroom",
}


def _make_connected_solution(base, route_a, route_b, dur_matrix, dist_matrix,
                             sequencer):
    """Fabrique une solution complete a partir d'une candidate et de DEUX
    ordres. Les metriques sont systematiquement rescorees sur la MEME matrice
    ORS : c'est la seule facon de comparer OR-Tools et Vroom sans biais."""
    dur, dist = _rescore(dur_matrix, dist_matrix, route_a, route_b)
    sol = dict(base)
    sol.update({
        "route_a": list(route_a),
        "route_b": list(route_b),
        "duration_s": dur,
        "distance_m": dist,
        "sequencer": sequencer,
        "selection_reason": _SELECTION_REASONS.get(sequencer, sequencer),
        "partition_key": base.get("partition_key")
        or canonical_partition_key(base["group_a"], base["group_b"]),
    })
    return sol


def ortools_partition_ors_matrix_connected(points, num_vehicles, max_per_vehicle,
                                           start_idx, end_idx, headers,
                                           solution_limit=None):
    """Partition en deux territoires CONNEXES sur les durees routieres ORS.

    Contrat : cardinalite exacte, aucune perte ni doublon, chaque territoire
    d'un seul tenant dans le graphe de voisinage, puis duree ORS minimale.
    Aucun objectif d'equilibrage. Retourne (groups, err, meta).
    """
    t0 = time.time()
    diag = {
        "connected_partition": False,
        "connected_method": "knn_graph_repair_local_search",
        "connected_membership_locked": False,
        "connected_target_sizes": None,
        "connected_components_t1": None,
        "connected_components_t2": None,
        "connected_component_sizes_t1": None,
        "connected_component_sizes_t2": None,
        "connected_candidates_generated": 0,
        "connected_candidates_valid": 0,
        "connected_candidates_scored": 0,
        "connected_candidates_ortools": 0,
        "connected_candidates_vroom": 0,
        "connected_cut_edges": None,
        "connected_cut_length_m": None,
        "connected_cross_neighbors": None,
        "connected_enclave_points": None,
        "connected_selected_seed": None,
        "connected_fallback_used": False,
        "connected_error": "",
        "connected_graph_k": None,
        "connected_vroom_calls": 0,
        "selected_sequencer": None,
        "final_selection_reason": "",
        "ortools_total_duration_s": None,
        "ortools_total_distance_m": None,
        "vroom_total_duration_s": None,
        "vroom_total_distance_m": None,
        "connected_enum_ms": 0,
        "connected_score_ms": 0,
        # --- selection symetrique OR-Tools / Vroom ---
        "connected_solutions_considered": 0,
        "connected_selection_window_s": CONNECTED_TIE_SECONDS,
        "connected_selected_duration_s": None,
        "connected_selected_distance_m": None,
        "connected_vroom_cache_hits": 0,
        "connected_vroom_error": "",
        # --- diversification des partitions connexes ---
        "connected_candidates_raw": 0,
        "connected_candidates_unique": 0,
        "connected_candidates_duplicates": 0,
        "connected_candidates_invalid_size": 0,
        "connected_candidates_disconnected": 0,
        "connected_candidates_repair_failed": 0,
        "connected_candidates_by_source": {},
        "connected_candidates_sweep": 0,
        "connected_candidates_mst": 0,
        "connected_candidates_region_growing": 0,
        "connected_candidates_two_means": 0,
        "connected_candidates_kmedoids": 0,
        "connected_candidates_ors_repair": 0,
        "connected_candidates_perturbation": 0,
        "connected_candidate_min_difference": 0,
        "connected_candidates_selected_diverse": 0,
        "connected_graph_method": None,
        "connected_ors_neighbor_k": CONNECTED_ORS_NEIGHBOR_K,
        "connected_diversity_error": "",
        # --- protection de l'incumbent historique ---
        "connected_candidates_legacy": 0,
        "connected_legacy_protected": False,
        "connected_legacy_proxy_rank": None,
        "connected_legacy_finalist_slots": 0,
        "connected_legacy_finalists": 0,
        "connected_legacy_in_finalists": False,
        "connected_legacy_is_winner": False,
        "connected_legacy_duration_s": None,
        "connected_legacy_distance_m": None,
        "connected_legacy_seed": None,
        "connected_candidates_local_search": 0,
        "connected_per_source": {},
        "connected_generation_expired_after": None,
        "connected_matrix_hash": None,
    }
    meta = {"connected": diag}

    depots = {start_idx, end_idx}
    indices = [i for i in range(len(points)) if i not in depots]
    if not indices:
        diag["connected_error"] = "no delivery points"
        return [[] for _ in range(num_vehicles)], None, meta

    if num_vehicles != 2:
        diag["connected_error"] = f"connected partition requires 2 vehicles, got {num_vehicles}"
        return None, diag["connected_error"], meta

    bad = [points[i].get("id", i) for i in indices if not _finite_coords(points[i])]
    if bad:
        diag["connected_error"] = f"{len(bad)} point(s) with invalid coordinates: {bad[:5]}"
        return None, diag["connected_error"], meta

    n = len(indices)
    target_a = n // 2                     # N pair -> N/2 ; N impair -> ecart de 1
    if target_a > max_per_vehicle or (n - target_a) > max_per_vehicle:
        diag["connected_error"] = (f"cannot split {n} points under capacity "
                                   f"{max_per_vehicle}")
        return None, diag["connected_error"], meta
    diag["connected_target_sizes"] = [target_a, n - target_a]

    print(f"Partition connexe: {n} points -> {target_a}/{n - target_a}", flush=True)

    dur_matrix, dist_matrix, mmeta, err = _build_full_matrix_chunked(points, headers)
    meta.update(mmeta)
    meta["connected"] = diag
    if dur_matrix is None:
        diag["connected_error"] = f"ORS matrix failed: {err}"
        return None, diag["connected_error"], meta

    # Graphe HYBRIDE : voisins a vol d'oiseau, voisins routiers issus de la
    # matrice deja chargee, et arbre couvrant. Aucun appel Matrix de plus.
    adjacency, gmeta = build_geo_graph(points, indices, dur_matrix=dur_matrix)
    diag["connected_graph_k"] = gmeta["k"]
    diag["connected_graph_method"] = gmeta["method"]
    diag["connected_ors_neighbor_k"] = gmeta["ors_k"]
    print(f"  Graphe: k={gmeta['k']}, {gmeta['edges']} aretes, "
          f"{gmeta['ors_edges']} ajoutees par la duree ORS (k={gmeta['ors_k']}), "
          f"{gmeta['mst_edges']} ajoutees par l'arbre couvrant, "
          f"methode={gmeta['method']}, connexe={gmeta['connected']}", flush=True)

    hav_matrix = _build_haversine_matrix(points)

    t_gen = time.time()
    try:
        cands, cstats = generate_connected_candidates(
            points, indices, target_a, adjacency, dur_matrix, hav_matrix,
            start_idx, end_idx, tree_edges=gmeta.get("tree_edges"),
            deadline=time.time() + CONNECTED_MAX_GENERATION_S)
    except Exception as exc:
        # La diversification est un ENRICHISSEMENT : si elle echoue, la
        # strategie doit continuer avec ce que la source de base sait faire,
        # pas rendre une erreur 500.
        diag["connected_diversity_error"] = str(exc)[:200]
        print(f"  Diversification en echec ({exc}), repli sur la source de base",
              flush=True)
        cands, cstats = _fallback_connected_candidates(
            points, indices, target_a, adjacency, dur_matrix, hav_matrix,
            start_idx, end_idx)
    diag["connected_enum_ms"] = int((time.time() - t_gen) * 1000)
    diag["connected_candidates_generated"] = cstats["generated"]
    diag["connected_candidates_valid"] = cstats["valid"]
    diag["connected_candidates_raw"] = cstats.get("raw", cstats["generated"])
    diag["connected_candidates_unique"] = cstats.get("unique", cstats["valid"])
    diag["connected_candidates_duplicates"] = cstats.get("duplicates", 0)
    diag["connected_candidates_invalid_size"] = cstats.get("invalid_size", 0)
    diag["connected_candidates_disconnected"] = cstats.get("disconnected", 0)
    diag["connected_candidates_repair_failed"] = cstats.get("repair_failed", 0)
    by_source = dict(cstats.get("by_source") or {})
    diag["connected_candidates_by_source"] = by_source
    diag["connected_candidates_sweep"] = by_source.get("sweep", 0)
    diag["connected_candidates_mst"] = by_source.get("mst", 0)
    diag["connected_candidates_region_growing"] = (
        by_source.get("region_growing", 0)
        + by_source.get("region_growing_repaired", 0))
    diag["connected_candidates_two_means"] = by_source.get("two_means", 0)
    diag["connected_candidates_kmedoids"] = by_source.get("kmedoids", 0)
    diag["connected_candidates_ors_repair"] = (by_source.get("ors_repair", 0)
                                               + by_source.get("haversine_repair", 0))
    diag["connected_candidates_perturbation"] = by_source.get("perturbation", 0)
    diag["connected_candidates_local_search"] = by_source.get("local_search", 0)
    diag["connected_candidates_legacy"] = sum(
        v for k, v in by_source.items() if k.startswith("legacy:"))
    diag["connected_per_source"] = dict(cstats.get("per_source") or {})
    diag["connected_generation_expired_after"] = cstats.get("expired_after")
    diag["connected_matrix_hash"] = mmeta.get("content_hash")
    if cstats.get("timeout"):
        diag["connected_diversity_error"] = (diag["connected_diversity_error"]
                                             or "generation budget reached after %s"
                                             % cstats.get("expired_after"))
    print(f"  Candidates: {diag['connected_candidates_raw']} brutes, "
          f"{diag['connected_candidates_unique']} uniques, "
          f"{diag['connected_candidates_duplicates']} doublons, "
          f"{diag['connected_candidates_repair_failed']} reparations echouees "
          f"en {diag['connected_enum_ms']}ms | matrice={diag['connected_matrix_hash']}",
          flush=True)
    # Bilan par source : brutes / doublons / taille KO / deconnectees /
    # reparations echouees / uniques conservees. C'est la seule facon de savoir
    # si une source est muette parce qu'elle ne produit rien ou parce que tout
    # ce qu'elle produit est deja connu.
    for src in sorted(diag["connected_per_source"]):
        b = diag["connected_per_source"][src]
        print("    %-22s brutes=%-4d dup=%-4d tailleKO=%-3d deconn=%-4d "
              "repKO=%-3d uniques=%d"
              % (src, b["raw"], b["duplicates"], b["invalid_size"],
                 b["disconnected"], b["repair_failed"], b["unique"]),
              flush=True)
    if cstats.get("expired_after"):
        print(f"    budget de generation epuise a l'etape "
              f"'{cstats['expired_after']}' : les sources suivantes n'ont pas "
              f"tourne", flush=True)

    if not cands:
        diag["connected_error"] = "no connected partition could be built"
        return None, diag["connected_error"], meta

    # --- niveau 1 : heuristique locale sur la matrice ORS ---
    # TOUTES les solutions produites -- heuristique, OR-Tools, Vroom -- vont
    # dans une seule liste. Le gagnant en sort a la toute fin, par
    # select_best_solution(). Aucun sequenceur n'ecrase l'autre en chemin.
    t_score = time.time()
    solutions = []
    rough = []
    allset = set(indices)
    for c in cands:
        ga, gb = c["group_a"], c["group_b"]
        ia = is_connected_partition(ga, adjacency)
        ib = is_connected_partition(gb, adjacency)
        # Score RAPIDE : plus proche voisin sur la matrice ORS, sans affinage.
        # Il ne sert qu'a presélectionner ; la decision finale se fait sur les
        # ordres reellement produits par OR-Tools et Vroom.
        ra = _estimate_group_cost(dur_matrix, ga, start_idx, end_idx, False)[1]
        rb = _estimate_group_cost(dur_matrix, gb, start_idx, end_idx, False)[1]
        dur, dist = _rescore(dur_matrix, dist_matrix, ra, rb)
        rough.append({
            "group_a": ga, "group_b": gb, "seed": c["seed"],
            "partition_key": c.get("partition_key")
            or canonical_partition_key(ga, gb),
            "connected": ia["connected"] and ib["connected"],
            "cardinality_ok": (len(ga) == target_a and len(gb) == n - target_a
                               and set(ga) | set(gb) == allset
                               and not (set(ga) & set(gb))),
            "components_total": ia["component_count"] + ib["component_count"] - 2,
            "comp_a": ia, "comp_b": ib,
            "boundary": boundary_metrics(ga, gb, adjacency, points),
            "duration_s": dur, "distance_m": dist,
            "legacy": bool(c.get("legacy")),
        })
    diag["connected_candidates_scored"] = len(rough)

    # Rang proxy de chaque candidate, pour le journal de traçabilite.
    proxy_rank = {c["partition_key"]: r
                  for r, c in enumerate(sorted(rough, key=_selection_key))}

    # --- places reservees aux candidates historiques ---
    # Les candidates du generateur d'avant diversification entrent d'office
    # dans les finalistes, dans la limite de CONNECTED_LEGACY_FINALIST_SLOTS.
    # Sans cette reserve, un proxy defavorable suffit a les ecarter du banc
    # d'essai OR-Tools : c'est exactement ce qui a fait perdre la solution a
    # 6641 s sur 42dd749a. Et une seule place ne suffit pas -- le proxy classe
    # mal ces partitions, la meilleure au proxy n'est pas la meilleure apres
    # sequencement.
    legacy_rough = sorted((c for c in rough if c["legacy"]), key=_selection_key)
    protected_keys = tuple(c["partition_key"]
                           for c in legacy_rough[:CONNECTED_LEGACY_FINALIST_SLOTS])
    if legacy_rough:
        best_legacy = legacy_rough[0]
        diag["connected_legacy_protected"] = True
        diag["connected_legacy_proxy_rank"] = proxy_rank[best_legacy["partition_key"]]
        diag["connected_legacy_seed"] = best_legacy["seed"]
        diag["connected_legacy_finalist_slots"] = len(protected_keys)

    # --- finalistes : incumbent historique, puis score, puis diversite ---
    finalist_bases, min_diff = select_diverse_finalists(
        rough, CONNECTED_ORTOOLS_FINALISTS, protected_keys=protected_keys)
    finalist_keys = {b["partition_key"] for b in finalist_bases}
    diag["connected_candidates_selected_diverse"] = len(finalist_bases)
    diag["connected_candidate_min_difference"] = min_diff
    diag["connected_legacy_in_finalists"] = bool(
        protected_keys and protected_keys[0] in finalist_keys)
    diag["connected_legacy_finalists"] = sum(
        1 for k in protected_keys if k in finalist_keys)
    print(f"  Finalistes: {len(finalist_bases)} sur {len(rough)} candidates, "
          f"ecart d'appartenance minimal={min_diff} points", flush=True)

    # Journal compact des candidates HISTORIQUES : c'est la trace qui manquait
    # pour savoir ce qu'etait devenu sweep_2 d'un run a l'autre.
    for c in sorted(legacy_rough, key=lambda x: x["seed"]):
        print("    [historique] %-18s cle=%s rang=%-3d proxy=%.1fs/%.0fm "
              "finaliste=%s"
              % (c["seed"], _short_key(c["partition_key"]),
                 proxy_rank[c["partition_key"]], c["duration_s"],
                 c["distance_m"],
                 "oui" if c["partition_key"] in finalist_keys else "NON"),
              flush=True)

    # Affinage Or-opt + 2-opt reserve aux finalistes : le faire sur toutes les
    # candidates coute cher et ne change pas la preselection.
    scored = []
    for base in finalist_bases:
        ra = _estimate_group_cost(dur_matrix, base["group_a"], start_idx, end_idx, True)[1]
        rb = _estimate_group_cost(dur_matrix, base["group_b"], start_idx, end_idx, True)[1]
        sol = _make_connected_solution(base, ra, rb, dur_matrix, dist_matrix,
                                       "heuristic")
        scored.append(sol)
        solutions.append(sol)
    scored.sort(key=_selection_key)

    # --- niveau 2 : OR-Tools sur les meilleures, sequencement seul ---
    ortools_sols = []
    if ORTOOLS_AVAILABLE:
        for cand in scored[:CONNECTED_TOP_ORTOOLS]:
            ra = _tsp_order_ortools(dur_matrix, cand["group_a"], start_idx, end_idx)
            rb = _tsp_order_ortools(dur_matrix, cand["group_b"], start_idx, end_idx)
            if ra is None or rb is None:
                continue
            diag["connected_candidates_ortools"] += 1
            sol = _make_connected_solution(cand, ra, rb, dur_matrix, dist_matrix,
                                           "ortools")
            solutions.append(sol)
            ortools_sols.append(sol)
            if cand.get("legacy"):
                print("    [historique] %-18s cle=%s OR-Tools %.1fs / %.0fm"
                      % (cand["seed"], _short_key(cand["partition_key"]),
                         sol["duration_s"], sol["distance_m"]), flush=True)
        # La MEME fonction de selection sert partout : ces deux colonnes
        # decrivent OR-Tools seul, avec la regle appliquee au reste.
        ortools_best = select_best_solution(ortools_sols)
        if ortools_best is not None:
            # Le MEILLEUR OR-Tools, pas l'incumbent courant : ces deux colonnes
            # de Benchmark doivent decrire OR-Tools seul.
            diag["ortools_total_duration_s"] = round(ortools_best["duration_s"], 1)
            diag["ortools_total_distance_m"] = round(ortools_best["distance_m"], 1)

    # --- niveau 3 : Vroom sur les 3 meilleures, 2 appels chacune ---
    # Les reponses sont MEMORISEES par partition : apres la selection, aucune
    # candidate deja evaluee n'est rappelee. C'est ce rappel final qui, avant
    # correction, remplacait silencieusement l'ordre OR-Tools par celui de
    # Vroom tout en laissant selected_sequencer annoncer "ortools".
    vroom_cache = {}
    vroom_sols = []
    finalists = scored[:CONNECTED_TOP_VROOM]
    for cand in finalists:
        pkey = cand["partition_key"]
        if pkey in vroom_cache:
            diag["connected_vroom_cache_hits"] += 1
            continue
        ra, _da, _ = _resequence_single(points, cand["group_a"], start_idx, end_idx, headers)
        rb, _db, _ = _resequence_single(points, cand["group_b"], start_idx, end_idx, headers)
        diag["connected_vroom_calls"] += 2
        if ra is None or rb is None:
            # Rate limit, erreur reseau ou reponse invalide : la meilleure
            # solution OR-Tools reste en lice, le run ne tombe pas.
            vroom_cache[pkey] = None
            diag["connected_fallback_used"] = True
            diag["connected_vroom_error"] = (diag["connected_vroom_error"]
                                             or "vroom unavailable")
            diag["connected_error"] = diag["connected_error"] or "vroom unavailable, kept OR-Tools order"
            break
        diag["connected_candidates_vroom"] += 1
        sol = _make_connected_solution(cand, ra, rb, dur_matrix, dist_matrix,
                                       "vroom")
        vroom_cache[pkey] = sol
        solutions.append(sol)
        vroom_sols.append(sol)
    vroom_best = select_best_solution(vroom_sols)
    if vroom_best is not None:
        diag["vroom_total_duration_s"] = round(vroom_best["duration_s"], 1)
        diag["vroom_total_distance_m"] = round(vroom_best["distance_m"], 1)

    # --- selection finale, sur l'ensemble complet des solutions ---
    incumbent = select_best_solution(solutions)
    if incumbent is None:
        diag["connected_error"] = "no scorable connected solution"
        return None, diag["connected_error"], meta
    diag["final_selection_reason"] = incumbent["selection_reason"]
    diag["connected_solutions_considered"] = len(solutions)
    diag["connected_legacy_is_winner"] = bool(incumbent.get("legacy"))

    # Meilleure solution issue de la partition historique, tous sequenceurs
    # confondus. La comparer au gagnant repond a la seule question qui compte :
    # la diversification a-t-elle ajoute ou remplace ?
    legacy_best = select_best_solution([s for s in solutions if s.get("legacy")])
    if legacy_best is not None:
        diag["connected_legacy_duration_s"] = round(legacy_best["duration_s"], 1)
        diag["connected_legacy_distance_m"] = round(legacy_best["distance_m"], 1)
        if not diag["connected_legacy_is_winner"]:
            print("  Historique battu: %.1fs / %.0fm (%s) contre %.1fs / %.0fm "
                  "retenu (%s)"
                  % (legacy_best["duration_s"], legacy_best["distance_m"],
                     legacy_best["seed"], incumbent["duration_s"],
                     incumbent["distance_m"], incumbent["seed"]), flush=True)

    diag["connected_score_ms"] = int((time.time() - t_score) * 1000)

    # --- verification finale, independante de la construction ---
    ga, gb = incumbent["group_a"], incumbent["group_b"]
    ia = is_connected_partition(ga, adjacency)
    ib = is_connected_partition(gb, adjacency)
    cardinality_ok = (len(ga) == target_a and len(gb) == n - target_a
                      and set(ga) | set(gb) == allset and not (set(ga) & set(gb)))
    if not (ia["connected"] and ib["connected"] and cardinality_ok):
        diag["connected_error"] = (f"final check failed: connected="
                                   f"{ia['connected']}/{ib['connected']}, "
                                   f"cardinality_ok={cardinality_ok}")
        return None, diag["connected_error"], meta

    b = boundary_metrics(ga, gb, adjacency, points)
    diag.update({
        "connected_partition": True,
        "connected_membership_locked": True,
        "connected_components_t1": ia["component_count"],
        "connected_components_t2": ib["component_count"],
        "connected_component_sizes_t1": ia["component_sizes"],
        "connected_component_sizes_t2": ib["component_sizes"],
        "connected_cut_edges": b["cut_edges"],
        "connected_cut_length_m": b["cut_length_m"],
        "connected_cross_neighbors": b["cross_neighbors"],
        "connected_enclave_points": b["enclave_points"],
        "connected_selected_seed": incumbent["seed"],
        "selected_sequencer": incumbent["sequencer"],
        "connected_selected_duration_s": round(incumbent["duration_s"], 1),
        "connected_selected_distance_m": round(incumbent["distance_m"], 1),
    })

    # Les ORDRES gagnants remontent avec l'appartenance. Sans eux, l'appelant
    # devait reconstruire une sequence -- deux appels Vroom de plus -- et
    # l'ordre retourne n'etait alors plus celui annonce par selected_sequencer.
    # Les routes memorisees sont reutilisees telles quelles : aucun rappel.
    meta["connected_routes"] = [list(incumbent["route_a"]),
                                list(incumbent["route_b"])]
    meta["connected_vroom_ok"] = not diag["connected_fallback_used"]
    meta["connected_vroom_error"] = diag["connected_vroom_error"] or None

    print(f"  Retenue: {[len(ga), len(gb)]} pts, seed={incumbent['seed']}, "
          f"sequenceur={incumbent['sequencer']}, composantes 1/1, "
          f"{b['cut_edges']} aretes coupees, {b['enclave_points']} enclaves, "
          f"duree {incumbent['duration_s'] / 60:.1f}min, "
          f"{incumbent['distance_m'] / 1000:.2f}km, "
          f"{len(solutions)} solutions comparees, "
          f"{diag['connected_vroom_calls']} appels Vroom "
          f"({int((time.time() - t0) * 1000)}ms)", flush=True)

    return [ga, gb], None, meta


def _tsp_order_ortools(matrix, group, start_idx, end_idx):
    """Ordonne UN groupe avec OR-Tools. Ne touche jamais a l'appartenance :
    un seul vehicule, donc aucun point ne peut changer de tournee.
    Retourne une route en index globaux, ou None."""
    if not ORTOOLS_AVAILABLE or not group:
        return [start_idx, end_idx] if not group else None
    nodes = [start_idx] + [g for g in group if g != start_idx and g != end_idx]
    if end_idx != start_idx:
        nodes.append(end_idx)
    local = {k: nodes[k] for k in range(len(nodes))}
    try:
        if start_idx == end_idx:
            mgr = pywrapcp.RoutingIndexManager(len(nodes), 1, 0)
        else:
            mgr = pywrapcp.RoutingIndexManager(len(nodes), 1, [0], [len(nodes) - 1])
        routing = pywrapcp.RoutingModel(mgr)

        def cb(i, j):
            return int(matrix[local[mgr.IndexToNode(i)]][local[mgr.IndexToNode(j)]])

        t = routing.RegisterTransitCallback(cb)
        routing.SetArcCostEvaluatorOfAllVehicles(t)
        prm = pywrapcp.DefaultRoutingSearchParameters()
        prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        prm.solution_limit = 30          # limite COURTE : 12 candidates a sequencer
        prm.time_limit.FromSeconds(3)
        prm.log_search = False
        sol = routing.SolveWithParameters(prm)
        if sol is None:
            return None
        out, idx = [], routing.Start(0)
        while not routing.IsEnd(idx):
            out.append(local[mgr.IndexToNode(idx)])
            idx = sol.Value(routing.NextVar(idx))
        out.append(end_idx)
        return out
    except Exception:
        return None


# =========================
# 5. NEAREST-NEIGHBOR FALLBACK
# =========================
def _nearest_neighbor_route(points, vehicle_points, start_idx, end_idx):
    """Fallback TSP : nearest-neighbor quand Vroom echoue."""
    if not vehicle_points:
        return [start_idx, end_idx]

    remaining = set(vehicle_points)
    route = [start_idx]
    current = start_idx

    while remaining:
        nearest = min(remaining, key=lambda j: haversine(
            (points[current]["lat"], points[current]["lon"]),
            (points[j]["lat"], points[j]["lon"])
        ))
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest

    route.append(end_idx)
    return route


def _compute_route_distance(points, route):
    """Calcule la distance totale d'une route en km."""
    total = 0.0
    for i in range(len(route) - 1):
        total += haversine(
            (points[route[i]]["lat"], points[route[i]]["lon"]),
            (points[route[i+1]]["lat"], points[route[i+1]]["lon"])
        )
    return round(total, 2)


# =========================
# 6. POST-PROCESSING : SWAP FRONTIERE
# =========================
def _find_border_points(points, routes_idx, start_idx, end_idx):
    """Identifie les points frontiere : proches d'un point de l'autre tournee."""
    depot = {start_idx, end_idx}
    border = []

    for v, route in enumerate(routes_idx):
        other_v = 1 - v
        other_pts = [p for p in routes_idx[other_v] if p not in depot]

        for pt in route:
            if pt in depot:
                continue
            min_dist = min(
                haversine((points[pt]["lat"], points[pt]["lon"]),
                          (points[op]["lat"], points[op]["lon"]))
                for op in other_pts
            ) if other_pts else float("inf")

            if min_dist < 0.5:  # 500m : zone frontiere
                border.append((v, pt, min_dist))

    border.sort(key=lambda x: x[2])
    return border


def _resequence_single(points, vehicle_pts, start_idx, end_idx, headers):
    """Re-sequence un vehicule avec Vroom. Retourne (route, duration, distance) ou (None, None, None)."""
    if not vehicle_pts:
        return [start_idx, end_idx], 0, 0

    start_coord = [points[start_idx]["lon"], points[start_idx]["lat"]]
    end_coord = [points[end_idx]["lon"], points[end_idx]["lat"]]

    vehicle = {
        "id": 0,
        "profile": "driving-car",
        "start": start_coord,
        "end": end_coord
    }
    jobs = [{"id": idx, "location": [points[idx]["lon"], points[idx]["lat"]]}
            for idx in vehicle_pts]

    try:
        response = _post_vroom(
            {"jobs": jobs, "vehicles": [vehicle]},
            headers,
            timeout=20
        )
        data = response.json()
        if "routes" not in data:
            return None, None, None

        ordered = [start_idx]
        for step in data["routes"][0]["steps"]:
            if step["type"] == "job":
                ordered.append(step["id"])
        ordered.append(end_idx)
        dur = data["routes"][0].get("duration", 0)
        # NB: Vroom ne renvoie pas 'distance' dans les routes -> km calcules via matrice ORS (voir post_process_swaps)
        dist = data["routes"][0].get("distance") or 0
        return ordered, dur, dist

    except Exception:
        return None, None, None


def post_process_swaps(points, routes_idx, start_idx, end_idx, max_per_vehicle,
                       entry_metrics=None, max_candidates=None,
                       max_consecutive_fails=None):
    """Post-processing iteratif : echanges de points frontiere jusqu'a convergence.
    Deux modes :
    - Deplacement : point A (T1) -> T2, si T2 n'est pas plein
    - Echange    : point A (T1) <-> point B (T2), maintient l'equilibre (fonctionne meme 30/30)
    Relance la detection de frontiere apres chaque amelioration (max 5 iterations, 50 appels Vroom).

    D-2 : trois etats strictement distincts, jamais confondus sous un meme nom.

      A. ENTREE PROTEGEE  entry_routes / entry_metrics / entry_duration_s
         L'ordre issu d'Or-opt et ses metriques ORS Matrix exactes.
         Jamais modifie, restaure en cas de doute.

      B. NOTATION         current_pts / current_vroom_durs / current_vroom_dists
                          + current_vroom_total
         Echelle Vroom homogene servant uniquement a departager les candidats
         entre eux. Les deux appels Vroom initiaux n'existent que pour fournir
         cette base : leurs ROUTES sont volontairement ignorees.

      C. SORTIE           accepted_routes / accepted_swaps
         Ecrit uniquement lorsqu'un echange est reellement accepte.

    Retourne (routes, metriques, swap_stats).
    """
    max_candidates = SWAP_MAX_CANDIDATES if max_candidates is None else max_candidates
    max_consecutive_fails = (SWAP_MAX_CONSECUTIVE_FAILS
                             if max_consecutive_fails is None else max_consecutive_fails)

    # Trace de configuration : c'est elle qui prouve ce que le backend a
    # reellement recu. Aucune donnee sensible, uniquement deux entiers.
    print(f"Post-processing config: max_candidates={max_candidates}, "
          f"max_consecutive_fails={max_consecutive_fails}", flush=True)

    total_tested = 0
    accepted_swaps = 0
    swap_stats = {
        "max_swap_candidates": max_candidates,
        "swap_max_consecutive_fails": max_consecutive_fails,
        "swap_candidates_tested": 0,
        "swaps_accepted": 0,
        "swap_resequence_cache_hits": 0,
        "swap_resequence_cache_misses": 0,
        "swap_vroom_calls_saved": 0,
        "swap_stop_reason": "completed",
    }

    def _finish(reason):
        swap_stats["swap_candidates_tested"] = total_tested
        swap_stats["swaps_accepted"] = accepted_swaps
        swap_stats["swap_stop_reason"] = reason
        # Un hit est exactement un appel reseau evite.
        swap_stats["swap_vroom_calls_saved"] = swap_stats["swap_resequence_cache_hits"]
        return swap_stats

    # Moins de 2 tournees : les swaps n'ont pas de sens, aucun appel n'est emis.
    if len(routes_idx) != 2:
        return routes_idx, entry_metrics, _finish("disabled")

    headers = {
        "Authorization": ORS_KEY,
        "Content-Type": "application/json"
    }
    depot = {start_idx, end_idx}

    # ---------- A. ETAT D'ENTREE PROTEGE ----------
    entry_routes = [list(routes_idx[0]), list(routes_idx[1])]

    # ---------- SWAPS DESACTIVES ----------
    # Court-circuit AVANT les deux appels Vroom de notation : desactiver les
    # swaps ne doit rien couter. Aucune detection de frontiere, aucun candidat,
    # aucun appel Matrix final. D-2 garantit le retour exact de l'entree.
    if max_candidates <= 0:
        print("Post-processing: swaps desactives : routes initiales conservees", flush=True)
        return entry_routes, entry_metrics, _finish("disabled")

    # ---------- MEMOISATION EXACTE, PORTEE = CET APPEL ----------
    # Cle = tuple(pts), ordre COMPRIS : _resequence_single construit ses jobs
    # dans l'ordre de la liste et Vroom y est sensible (optimize_with_vroom
    # exploite d'ailleurs ce fait avec ses permutations). Ni set, ni frozenset,
    # ni sorted : la cle doit etre exacte, sinon le cache changerait le resultat.
    #
    # Les candidats dupliques par symetrie (a,b) / (b,a) produisent des listes
    # identiques element par element, ordre compris : le cache est donc exact.
    _reseq_cache = {}

    def reseq(pts):
        key = tuple(pts)
        cached = _reseq_cache.get(key)
        if cached is not None:
            swap_stats["swap_resequence_cache_hits"] += 1
            c_route, c_dur, c_dist = cached
            return list(c_route), c_dur, c_dist          # copie a la restitution
        swap_stats["swap_resequence_cache_misses"] += 1
        route, dur, dist = _resequence_single(points, pts, start_idx, end_idx, headers)
        # Seuls les succes sont memorises. Un echec transitoire -- timeout, 429,
        # 5xx, coupure reseau, reponse Vroom invalide -- doit pouvoir etre
        # retente plus tard dans la meme requete.
        if route is not None and dur is not None:
            _reseq_cache[key] = (list(route), dur, dist)  # copie au stockage
        return route, dur, dist

    entry_duration_s = None
    if entry_metrics and len(entry_metrics) == 2:
        e0 = entry_metrics[0].get("duration_s")
        e1 = entry_metrics[1].get("duration_s")
        if e0 is not None and e1 is not None:
            entry_duration_s = e0 + e1

    pts0 = [p for p in entry_routes[0] if p not in depot]
    pts1 = [p for p in entry_routes[1] if p not in depot]

    # ---------- B. ETAT DE NOTATION ----------
    # Ces deux appels Vroom donnent une base de comparaison homogene avec les
    # candidats, tous mesures par Vroom. Leurs routes sont jetees : les adopter
    # ecraserait l'ordre optimise par Or-opt, c'est la cause exacte du defaut D-2.
    _vroom_route0, dur0, dist0 = reseq(pts0)
    _vroom_route1, dur1, dist1 = reseq(pts1)

    if dur0 is None or dur1 is None:
        print("Post-processing: impossible de calculer durees initiales", flush=True)
        return entry_routes, entry_metrics, _finish("vroom_error")

    current_pts = [list(pts0), list(pts1)]
    current_vroom_durs = [dur0, dur1]
    current_vroom_dists = [dist0, dist1]
    current_vroom_total = dur0 + dur1

    # ---------- C. ETAT DE SORTIE ----------
    accepted_routes = [list(entry_routes[0]), list(entry_routes[1])]

    consecutive_fails = 0
    stop_reason = "completed"
    MAX_ITER = 5

    print(f"Post-processing: base Vroom = {dur0}s + {dur1}s = {current_vroom_total}s "
          f"| plafond candidats={max_candidates}, arret anticipe="
          f"{max_consecutive_fails or 'desactive'}", flush=True)

    for iteration in range(MAX_ITER):
        if total_tested >= max_candidates:
            stop_reason = "candidate_limit"
            break

        border = _find_border_points(points, accepted_routes, start_idx, end_idx)
        print(f"  Iteration {iteration+1}: {len(border)} points frontiere (seuil 500m)", flush=True)

        if not border:
            stop_reason = "no_border_points"
            break

        improved = False

        for v_from, pt_a, dist_a in border[:15]:
            if total_tested >= max_candidates:
                stop_reason = "candidate_limit"
                break
            if max_consecutive_fails and consecutive_fails >= max_consecutive_fails:
                stop_reason = "consecutive_failures"
                break

            v_to = 1 - v_from

            # --- MODE 1 : deplacement si l'autre route a de la place ---
            if len(current_pts[v_to]) < max_per_vehicle:
                new_pts_from = [p for p in current_pts[v_from] if p != pt_a]
                new_pts_to = current_pts[v_to] + [pt_a]

                r_from, d_from, dist_from = reseq(new_pts_from)
                r_to, d_to, dist_to = reseq(new_pts_to)
                total_tested += 1
                consecutive_fails += 1   # remis a zero si le candidat est accepte

                if d_from is not None and d_to is not None:
                    gain = current_vroom_total - (d_from + d_to)
                    # Candidat refuse (gain <= 0) : aucun etat n'est ecrit,
                    # new_pts_* et r_* sont des listes neuves devenues inatteignables.
                    if gain > 0:
                        print(f"    Deplacement pt {pt_a} T{v_from+1}->T{v_to+1}: +{gain}s", flush=True)
                        current_vroom_total = d_from + d_to
                        # D-1 : sans ces 4 lignes, les durees/distances restaient sur
                        # les valeurs d'avant le deplacement et les minutes renvoyees
                        # etaient perimees (symetrie avec le MODE 2 ci-dessous).
                        current_vroom_durs[v_from]  = d_from
                        current_vroom_durs[v_to]    = d_to
                        current_vroom_dists[v_from] = dist_from
                        current_vroom_dists[v_to]   = dist_to
                        accepted_routes[v_from] = r_from
                        accepted_routes[v_to] = r_to
                        current_pts[v_from] = new_pts_from
                        current_pts[v_to] = new_pts_to
                        accepted_swaps += 1
                        consecutive_fails = 0
                        improved = True
                        break  # relancer la detection

            # --- MODE 2 : echange pt_a (v_from) <-> pt_b (v_to) ---
            candidates_b = sorted(
                current_pts[v_to],
                key=lambda p: haversine(
                    (points[pt_a]["lat"], points[pt_a]["lon"]),
                    (points[p]["lat"], points[p]["lon"])
                )
            )[:5]

            for pt_b in candidates_b:
                if total_tested >= max_candidates:
                    stop_reason = "candidate_limit"
                    break
                if max_consecutive_fails and consecutive_fails >= max_consecutive_fails:
                    stop_reason = "consecutive_failures"
                    break

                new_pts_from = [pt_b if p == pt_a else p for p in current_pts[v_from]]
                new_pts_to = [pt_a if p == pt_b else p for p in current_pts[v_to]]

                r_from, d_from, dist_from = reseq(new_pts_from)
                r_to, d_to, dist_to = reseq(new_pts_to)
                total_tested += 1
                consecutive_fails += 1   # remis a zero si le candidat est accepte

                if d_from is None or d_to is None:
                    continue

                gain = current_vroom_total - (d_from + d_to)
                # Candidat refuse (gain <= 0) : aucun etat n'est ecrit.
                if gain > 0:
                    print(f"    Echange pt {pt_a}(T{v_from+1}) <-> pt {pt_b}(T{v_to+1}): +{gain}s", flush=True)
                    current_vroom_total = d_from + d_to
                    current_vroom_durs[v_from] = d_from
                    current_vroom_durs[v_to] = d_to
                    current_vroom_dists[v_from] = dist_from
                    current_vroom_dists[v_to] = dist_to
                    accepted_routes[v_from] = r_from
                    accepted_routes[v_to] = r_to
                    current_pts[v_from] = new_pts_from
                    current_pts[v_to] = new_pts_to
                    accepted_swaps += 1
                    consecutive_fails = 0
                    improved = True
                    break

            if improved:
                break  # relancer la detection de frontiere
            if stop_reason in ("candidate_limit", "consecutive_failures"):
                break

        # Arret dur decide dans la boucle des candidats : ne pas le requalifier.
        if stop_reason in ("candidate_limit", "consecutive_failures"):
            if stop_reason == "consecutive_failures":
                print(f"  Arret anticipe: {consecutive_fails} candidats consecutifs "
                      f"non ameliorants (plafond {max_consecutive_fails})", flush=True)
            else:
                print(f"  Plafond de {max_candidates} candidats atteint", flush=True)
            break

        if not improved:
            print(f"  Convergence atteinte a l'iteration {iteration+1}", flush=True)
            stop_reason = "convergence"
            break

    # ---------- CAS 1 : aucun swap accepte ----------
    # On rend l'etat d'entree tel quel. Les deux appels Matrix finaux ne sont
    # pas emis : ils ne mesureraient que des routes qu'on ne renvoie pas.
    print(f"Post-processing: cache reseq {swap_stats['swap_resequence_cache_hits']} hits / "
          f"{swap_stats['swap_resequence_cache_misses']} appels reels, "
          f"{swap_stats['swap_resequence_cache_hits']} appels Vroom evites "
          f"| arret: {stop_reason}", flush=True)

    if accepted_swaps == 0:
        print(f"Post-processing: aucun echange ameliorant ({total_tested} testes)", flush=True)
        print("Post-processing: aucun swap accepte : routes initiales conservees", flush=True)
        return entry_routes, entry_metrics, _finish(stop_reason)

    # ---------- CAS 2 : au moins un swap accepte ----------
    print(f"Post-processing: {accepted_swaps} echange(s), {total_tested} appels, "
          f"duree Vroom = {current_vroom_total}s", flush=True)

    # Distance ET duree routieres des routes acceptees : Vroom ne renvoie pas
    # 'distance', on les calcule via la matrice ORS (2 appels). La matrice de
    # durees est deja dans la meme reponse : la lire ne coute aucun appel.
    final_dists = [current_vroom_dists[0], current_vroom_dists[1]]
    final_durs_s = [None, None]
    for v in range(2):
        dist_matrix, dur_matrix = _fetch_ors_matrix(points, accepted_routes[v], headers)
        local = list(range(len(accepted_routes[v])))
        if dist_matrix:
            final_dists[v] = _matrix_route_cost(dist_matrix, local)
        if dur_matrix:
            # MEME estimateur que duration_s cote Or-opt : matrice de DUREES ORS.
            final_durs_s[v] = _matrix_route_cost(dur_matrix, local)
        print(f"  T{v+1} finale: {final_dists[v]/1000:.2f}km, ~{current_vroom_durs[v]/60:.1f}min", flush=True)

    final_metrics = [
        {"km": round(final_dists[0] / 1000, 2), "min": round(current_vroom_durs[0] / 60, 1),
         "duration_s": final_durs_s[0]},
        {"km": round(final_dists[1] / 1000, 2), "min": round(current_vroom_durs[1] / 60, 1),
         "duration_s": final_durs_s[1]},
    ]

    final_duration_s = None
    if final_durs_s[0] is not None and final_durs_s[1] is not None:
        final_duration_s = final_durs_s[0] + final_durs_s[1]

    # ---------- GARDE-FOU D-2 ----------
    # Les deux durees comparees viennent du MEME estimateur (matrice de durees
    # ORS) et sont en secondes non arrondies. Aucun melange Vroom / Matrix,
    # aucune minute arrondie.
    if entry_duration_s is None or final_duration_s is None:
        # Pas d'element de comparaison commun. On ne compare pas des unites
        # differentes : on restaure l'entree. C'est le comportement le plus sur,
        # d'autant que sans matrice ORS les distances finales retomberaient sur
        # current_vroom_dists, que Vroom laisse a 0 : les km seraient faux.
        print("Post-processing: durees exactes ORS indisponibles, "
              "garde-fou D-2 sans element de comparaison : "
              "restauration des routes initiales", flush=True)
        return entry_routes, entry_metrics, _finish(stop_reason)

    if final_duration_s >= entry_duration_s:
        # Egalite incluse : on conserve l'entree. Meme convention que le critere
        # d'acceptation existant, qui exige une amelioration STRICTE (gain > 0).
        print(f"Post-processing: resultat swaps non meilleur : restauration des "
              f"routes initiales ({final_duration_s:.0f}s >= {entry_duration_s:.0f}s)",
              flush=True)
        return entry_routes, entry_metrics, _finish(stop_reason)

    print(f"Post-processing: {accepted_swaps} swap(s) accepte(s), duree totale ORS "
          f"{entry_duration_s:.0f}s -> {final_duration_s:.0f}s "
          f"(-{entry_duration_s - final_duration_s:.0f}s)", flush=True)
    return accepted_routes, final_metrics, _finish(stop_reason)


# =========================
# 6b. 2-OPT POST-PROCESSING
# =========================
def _two_opt(points, route):
    """2-opt local search sur une route (distances a vol d'oiseau).
    Teste tous les echanges de 2 aretes et garde les ameliorations.
    Retourne la route optimisee (2-optimale)."""
    best = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                d_current = (
                    haversine((points[best[i-1]]["lat"], points[best[i-1]]["lon"]),
                              (points[best[i]]["lat"],   points[best[i]]["lon"])) +
                    haversine((points[best[j]]["lat"],   points[best[j]]["lon"]),
                              (points[best[j+1]]["lat"], points[best[j+1]]["lon"]))
                )
                d_new = (
                    haversine((points[best[i-1]]["lat"], points[best[i-1]]["lon"]),
                              (points[best[j]]["lat"],   points[best[j]]["lon"])) +
                    haversine((points[best[i]]["lat"],   points[best[i]]["lon"]),
                              (points[best[j+1]]["lat"], points[best[j+1]]["lon"]))
                )
                if d_new < d_current - 1e-6:
                    best[i:j+1] = best[i:j+1][::-1]
                    improved = True
    return best


def apply_two_opt(points, routes_idx):
    """Applique 2-opt sur chaque tournee independamment."""
    improved_routes = []
    for v, route in enumerate(routes_idx):
        before = _compute_route_distance(points, route)
        optimized = _two_opt(points, route)
        after = _compute_route_distance(points, optimized)
        gain = round(before - after, 2)
        if gain > 0:
            print(f"  2-opt T{v+1}: {before}km -> {after}km (-{gain}km)", flush=True)
        improved_routes.append(optimized)
    return improved_routes


# =========================
# 6c. OR-OPT + 2-OPT ROUTIER
# =========================
def _or_opt(points, route, seg_sizes=[1, 2, 3]):
    """Or-opt : deplace des segments de 1-3 points vers la meilleure position.
    Complementaire au 2-opt : trouve des ameliorations que 2-opt ne voit pas."""
    best = list(route)
    improved = True
    while improved:
        improved = False
        for seg_size in seg_sizes:
            for i in range(1, len(best) - seg_size - 1):
                segment = best[i:i + seg_size]
                remaining = best[:i] + best[i + seg_size:]
                d_removed = (
                    haversine((points[best[i-1]]["lat"], points[best[i-1]]["lon"]),
                              (points[best[i]]["lat"],   points[best[i]]["lon"])) +
                    haversine((points[best[i+seg_size-1]]["lat"], points[best[i+seg_size-1]]["lon"]),
                              (points[best[i+seg_size]]["lat"],   points[best[i+seg_size]]["lon"])) -
                    haversine((points[best[i-1]]["lat"], points[best[i-1]]["lon"]),
                              (points[best[i+seg_size]]["lat"],   points[best[i+seg_size]]["lon"]))
                )
                for j in range(1, len(remaining) - 1):
                    for seg in [segment, list(reversed(segment))]:
                        d_inserted = (
                            haversine((points[remaining[j-1]]["lat"], points[remaining[j-1]]["lon"]),
                                      (points[seg[0]]["lat"],         points[seg[0]]["lon"])) +
                            haversine((points[seg[-1]]["lat"],         points[seg[-1]]["lon"]),
                                      (points[remaining[j]]["lat"],   points[remaining[j]]["lon"])) -
                            haversine((points[remaining[j-1]]["lat"], points[remaining[j-1]]["lon"]),
                                      (points[remaining[j]]["lat"],   points[remaining[j]]["lon"]))
                        )
                        if d_removed - d_inserted > 1e-6:
                            best = remaining[:j] + seg + remaining[j:]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best


def _fetch_ors_matrix(points, route_indices, headers):
    """Recupere les matrices de distances (m) et durees (s) ORS pour une route.
    Retourne (dist_matrix, dur_matrix) ou (None, None) en cas d'erreur."""
    locations = [[points[i]["lon"], points[i]["lat"]] for i in route_indices]
    try:
        response = _post_matrix(
            {"locations": locations, "metrics": ["distance", "duration"]},
            headers,
            timeout=20
        )
        data = response.json()
        return data.get("distances", None), data.get("durations", None)
    except Exception:
        return None, None


def _or_opt_matrix(matrix, route_local, seg_sizes=[1, 2, 3]):
    """Or-opt utilisant une matrice de distances routieres reelles."""
    best = list(route_local)
    improved = True
    while improved:
        improved = False
        for seg_size in seg_sizes:
            for i in range(1, len(best) - seg_size - 1):
                d_removed = (
                    matrix[best[i-1]][best[i]] +
                    matrix[best[i+seg_size-1]][best[i+seg_size]] -
                    matrix[best[i-1]][best[i+seg_size]]
                )
                remaining = best[:i] + best[i+seg_size:]
                segment   = best[i:i+seg_size]
                for j in range(1, len(remaining) - 1):
                    d_inserted = (
                        matrix[remaining[j-1]][segment[0]] +
                        matrix[segment[-1]][remaining[j]] -
                        matrix[remaining[j-1]][remaining[j]]
                    )
                    if d_removed - d_inserted > 1e-6:
                        best = remaining[:j] + segment + remaining[j:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best


def _two_opt_matrix(matrix, route_local):
    """2-opt utilisant une matrice de distances routieres reelles.
    Comparaison du cout total (et non du delta partiel) pour matrice asymetrique."""
    best = list(route_local)
    best_cost = _matrix_route_cost(matrix, best)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_cost = _matrix_route_cost(matrix, candidate)
                if new_cost < best_cost - 1e-6:
                    best = candidate
                    best_cost = new_cost
                    improved = True
    return best


def _matrix_route_cost(matrix, route_local):
    """Calcule le cout total d'une route a partir d'une matrice."""
    return sum(matrix[route_local[i]][route_local[i+1]] for i in range(len(route_local)-1))


def apply_or_opt_and_routing_2opt(points, routes_idx):
    """Pour chaque tournee : Or-opt + 2-opt sur matrice ORS duree reelle.
    Optimise pour la duree (coherent avec Vroom). Retourne (routes, road_metrics)."""
    headers = {
        "Authorization": ORS_KEY,
        "Content-Type": "application/json"
    }
    improved_routes = []
    road_metrics = []

    for v, route in enumerate(routes_idx):
        dist_matrix, dur_matrix = _fetch_ors_matrix(points, route, headers)

        if dist_matrix:
            cost_matrix = dur_matrix if dur_matrix else dist_matrix
            n = len(route)
            before_s = _matrix_route_cost(cost_matrix, list(range(n)))

            # Multi-start Or-opt + 2-opt : 3 points de depart differents, on garde le meilleur
            depot_local = 0  # index local du depot (toujours 0)
            interior = list(range(1, n - 1))  # points interieurs (hors depot depart/arrivee)
            starts = [
                list(range(n)),                                       # ordre Vroom
                [0] + list(reversed(interior)) + [n - 1],            # interieur inverse
                [0] + sorted(interior, key=lambda i: cost_matrix[0][i]) + [n - 1],  # par duree au depot
            ]

            best_local = None
            best_s     = float("inf")
            for s_idx, start in enumerate(starts):
                candidate = _or_opt_matrix(cost_matrix, start)
                candidate = _two_opt_matrix(cost_matrix, candidate)
                cost_s = _matrix_route_cost(cost_matrix, candidate)
                print(f"  Or-opt start#{s_idx+1} T{v+1}: {cost_s/60:.1f}min", flush=True)
                if cost_s < best_s:
                    best_s     = cost_s
                    best_local = candidate

            print(f"  T{v+1}: {before_s/60:.1f}min -> {best_s/60:.1f}min (-{(before_s-best_s)/60:.1f}min)", flush=True)
            route = [route[i] for i in best_local]
            road_km  = round(_matrix_route_cost(dist_matrix, best_local) / 1000, 2)
            road_min = round(best_s / 60, 1) if dur_matrix else None
            print(f"  T{v+1}: {road_km}km routiers, ~{road_min}min", flush=True)
            # duration_s : secondes exactes NON arrondies, issues de la matrice de
            # DUREES ORS. None si seule la matrice de distances etait disponible :
            # best_s serait alors un metrage, jamais a placer dans une duree.
            # Champ interne, transporte jusqu'a post_process_swaps, jamais expose
            # dans la reponse JSON publique.
            road_metrics.append({"km": road_km, "min": road_min,
                                 "duration_s": best_s if dur_matrix else None})
        else:
            print(f"  Matrice ORS T{v+1} indisponible, fallback haversine", flush=True)
            route = _or_opt(points, route)
            road_metrics.append({"km": _compute_route_distance(points, route),
                                 "min": None, "duration_s": None})

        improved_routes.append(route)

    return improved_routes, road_metrics


# =========================
# 7. API
# =========================
@app.route("/optimize", methods=["POST"])
def optimize():

    t_start = time.time()
    _reset_api_stats()

    data = request.json
    points = data.get("points", [])
    num_vehicles = data.get("num_vehicles", 2)
    max_per_vehicle = data.get("max_per_vehicle", 35)
    start_id = data.get("start_id", "")
    end_id = data.get("end_id", "")

    if not points:
        return jsonify({"error": "no points"}), 400

    # Capture du payload pour figer une fixture de benchmark (DUMP_PAYLOAD=1)
    if os.environ.get("DUMP_PAYLOAD", "") == "1":
        print("PAYLOAD_DUMP " + json.dumps(data, ensure_ascii=False), flush=True)

    # --- Selecteur de strategie ---
    # Lot 1 : seul 'kmeans' est implemente. Une strategie non disponible renvoie 501,
    # JAMAIS un repli silencieux sur kmeans : sinon la feuille Benchmark accumulerait
    # des lignes etiquetees ortools_* qui contiennent en realite du K-Means.
    strategy = str(data.get("strategy") or "kmeans").strip().lower()
    if strategy not in VALID_STRATEGIES:
        return jsonify({"error": f"unknown strategy '{strategy}'",
                        "valid": list(VALID_STRATEGIES)}), 400
    if strategy not in IMPLEMENTED_STRATEGIES:
        detail = f"strategy '{strategy}' not implemented yet"
        if strategy.startswith("ortools_") and not ORTOOLS_AVAILABLE:
            detail = (f"strategy '{strategy}' unavailable: "
                      f"ortools is not installed on the server")
        return jsonify({"error": detail,
                        "implemented": sorted(IMPLEMENTED_STRATEGIES)}), 501
    strategy_used = strategy  # divergerait en cas de repli (jamais silencieux)

    # LOT 4.1-C : surcharge experimentale de solution_limit, par requete.
    # Absent -> ORTOOLS_SOLUTION_LIMIT (250), comportement actuel inchange.
    # N'agit que sur ortools_ors_matrix ; kmeans et ortools_haversine
    # ne lisent pas ce parametre.
    ortools_solution_limit = data.get("ortools_solution_limit")
    if ortools_solution_limit is not None:
        try:
            ortools_solution_limit = int(ortools_solution_limit)
        except (TypeError, ValueError):
            return jsonify({"error": "ortools_solution_limit must be an integer"}), 400
        if not (1 <= ortools_solution_limit <= 10000):
            return jsonify({"error": "ortools_solution_limit out of range (1..10000)"}), 400

    # Le reporting doit annoncer la limite REELLEMENT utilisee par le solveur,
    # pas celle demandee : une ligne de Benchmark batie sur ce champ serait
    # sinon fausse pour ortools_haversine, qui ignore la surcharge.
    #   kmeans             -> None, aucun solveur OR-Tools n'a tourne
    #   ortools_haversine  -> constante globale, la surcharge ne l'atteint pas
    #   ortools_ors_matrix -> surcharge si fournie, sinon constante globale
    # Lot D-3 : plafond de candidats et arret anticipe, surchargeables par
    # requete. Validation STRICTE avant tout appel reseau : une valeur refusee
    # ne consomme aucun quota.
    max_swap_candidates, err = _strict_int_param(data, "max_swap_candidates", 0, 200,
                                                 SWAP_MAX_CANDIDATES)
    if err:
        return jsonify({"error": err}), 400
    swap_max_consecutive_fails, err = _strict_int_param(data, "swap_max_consecutive_fails",
                                                        0, 200, SWAP_MAX_CONSECUTIVE_FAILS)
    if err:
        return jsonify({"error": err}), 400

    # Depuis le lot territorial, la partition de ortools_ors_matrix ne passe
    # PLUS par _solve_cvrp_ortools : aucune limite de solutions OR-Tools ne s'y
    # applique, et la surcharge par requete n'atteint donc plus aucun solveur.
    # Seule ortools_haversine utilise encore le solveur CVRP.
    #   kmeans             -> None, aucun solveur OR-Tools
    #   ortools_haversine  -> constante globale, la surcharge ne l'atteint pas
    #   ortools_ors_matrix -> None, balayage geometrique
    if strategy == "ortools_haversine":
        partition_solver = "ortools_cvrp"
        ortools_limit_effective = ORTOOLS_SOLUTION_LIMIT
    elif strategy == "ortools_ors_matrix":
        partition_solver = "territorial_projection"
        ortools_limit_effective = None
    elif strategy == "ortools_ors_matrix_connected":
        # OR-Tools n'y sert qu'au SEQUENCEMENT de chaque groupe, avec une
        # limite courte propre a ce niveau. La limite globale ne s'y applique
        # pas : l'annoncer serait faux. La valeur reelle est renseignee plus
        # bas, uniquement si le solveur a effectivement tourne.
        partition_solver = "connected_graph_partition"
        ortools_limit_effective = None
    else:
        # kmeans : le moteur exact, vroom_multi ou kmeans_fallback, est deja
        # rapporte par partition_engine.
        partition_solver = None
        ortools_limit_effective = None

    # Plus aucun solveur ne lit la surcharge : l'annoncer appliquee serait faux.
    # La valeur demandee reste visible dans ortools_solution_limit_requested.
    ortools_limit_override_applied = False

    # Resoudre les index depart / arrivee
    start_idx = 0
    end_idx = 0

    if start_id:
        for i, p in enumerate(points):
            if str(p["id"]) == str(start_id):
                start_idx = i
                break

    if end_id:
        for i, p in enumerate(points):
            if str(p["id"]) == str(end_id):
                end_idx = i
                break
    else:
        end_idx = start_idx

    print(f"Optimisation: {len(points)} points, {num_vehicles} vehicules, max={max_per_vehicle}", flush=True)
    print(f"Strategie: {strategy} | signature jeu: {_points_signature(points)}", flush=True)

    # --- AIGUILLAGE PAR STRATEGIE ---
    # Seules les etapes 1 et 2 (l'affectation) different. Les etapes 3, 4 et 5
    # sont communes a toutes les strategies : c'est ce qui rend la comparaison
    # valide, seule la partition change.
    routes_idx, vroom_ok, vroom_error = None, False, None

    # Routes deja sequencees par la strategie de partition (connexe). Non nul
    # signifie : l'ordre est CHOISI, personne ne doit le refaire.
    presequenced_routes = None

    # optimization_path decrit UNIQUEMENT la strategie de partition et n'est
    # jamais suffixe. Les etapes communes reellement executees sont listees
    # a part dans post_processing.
    optimization_path = strategy
    post_processing = []

    # Moteur d'affectation reellement utilise. Sous strategy=kmeans, la
    # partition vient de Vroom multi (jeu <= 59 points) ou de kmeans_partition :
    # deux methodes differentes sous la meme etiquette de strategie.
    partition_engine = None

    # Diagnostics de la matrice ORS (strategie ortools_ors_matrix uniquement) :
    # appels reellement consommes, cache touche ou non, cellules nulles.
    matrix_meta = None

    # Appartenance verrouillee : positionne par la partition territoriale.
    # Les swaps inter-tournees ne sont alors PAS lances, y compris leurs deux
    # appels Vroom de notation : on evite le cout au lieu de rejeter a la fin.
    membership_locked = False
    swap_lock_reason = "territorial_partition_locked"

    # Statistiques des swaps. Valeurs neutres si post_process_swaps ne tourne
    # pas (routes absentes ou Vroom en echec) : les champs restent presents.
    swap_stats = {
        "max_swap_candidates": max_swap_candidates,
        "swap_max_consecutive_fails": swap_max_consecutive_fails,
        "swap_candidates_tested": 0,
        "swaps_accepted": 0,
        "swap_resequence_cache_hits": 0,
        "swap_resequence_cache_misses": 0,
        "swap_vroom_calls_saved": 0,
        "swap_stop_reason": None,
    }

    if strategy == "kmeans":
        # 1. VROOM MULTI-VEHICULES (affectation + sequencement sur reseau routier reel)
        routes_idx, vroom_ok, vroom_error = optimize_with_vroom(
            points, num_vehicles, max_per_vehicle, start_idx, end_idx
        )
        if routes_idx is not None:
            partition_engine = "vroom_multi"

        # 2. FALLBACK: K-Means + Vroom par vehicule
        if routes_idx is None:
            print("Fallback K-Means + Vroom...", flush=True)
            routes_idx, vroom_ok, vroom_error = kmeans_partition(
                points, num_vehicles, max_per_vehicle, start_idx, end_idx
            )
            if routes_idx is not None:
                partition_engine = "kmeans_fallback"

    else:
        # 1bis. VROOM multi est explicitement saute : sans cela, un jeu <= 59
        #       points verrait sa strategie ignoree en silence.
        # 2bis. Partition locale, puis le MEME _sequence_groups() que kmeans.
        headers = {
            "Authorization": ORS_KEY,
            "Content-Type": "application/json"
        }

        if strategy == "ortools_haversine":
            groups, part_err = ortools_partition_haversine(
                points, num_vehicles, max_per_vehicle, start_idx, end_idx
            )
        elif strategy == "ortools_ors_matrix":
            groups, part_err, matrix_meta = ortools_partition_ors_matrix(
                points, num_vehicles, max_per_vehicle, start_idx, end_idx, headers,
                solution_limit=ortools_solution_limit
            )
            # La partition territoriale est un CONTRAT : Vroom et Or-opt peuvent
            # reordonner chaque tournee, aucun point ne change de vehicule.
            terr_diag = (matrix_meta or {}).get("territorial") or {}
            if terr_diag.get("territorial_membership_locked"):
                membership_locked = True
        elif strategy == "ortools_ors_matrix_connected":
            groups, part_err, matrix_meta = ortools_partition_ors_matrix_connected(
                points, num_vehicles, max_per_vehicle, start_idx, end_idx, headers,
                solution_limit=ortools_solution_limit
            )
            conn_diag = (matrix_meta or {}).get("connected") or {}
            if conn_diag.get("connected_membership_locked"):
                membership_locked = True
                swap_lock_reason = "connected_partition_locked"
            # La strategie connexe a DEJA choisi les ordres : elle a compare
            # OR-Tools et Vroom sur la meme matrice ORS et retourne les routes
            # du gagnant. Les reseqencer ici couterait deux appels Vroom de
            # plus ET remplacerait l'ordre retenu par celui de Vroom, quel que
            # soit selected_sequencer.
            presequenced_routes = (matrix_meta or {}).get("connected_routes")
        else:
            groups, part_err = None, f"no partition function for '{strategy}'"

        # Echec de partition = erreur explicite. Retomber sur K-Means ici
        # produirait une ligne de Benchmark etiquetee ortools_* contenant du
        # K-Means, ce qui fausserait la comparaison sans laisser de trace.
        if groups is None:
            print(f"Partition '{strategy}' echouee: {part_err}", flush=True)
            return jsonify({
                "error": f"partition failed for strategy '{strategy}': {part_err}",
                "strategy_requested": strategy,
                "elapsed_ms": int((time.time() - t_start) * 1000),
                "api_calls": {
                    "vroom": _API_STATS["vroom"],
                    "matrix": _API_STATS["matrix"],
                    "total": _api_calls_total(),
                },
                "ors_matrix": matrix_meta,
            }), 500

        partition_engine = strategy

        if presequenced_routes is not None:
            routes_idx = [list(r) for r in presequenced_routes]
            vroom_ok = bool((matrix_meta or {}).get("connected_vroom_ok"))
            vroom_error = (matrix_meta or {}).get("connected_vroom_error")
            print(f"Sequencement deja fait par {strategy} "
                  f"(sequenceur retenu: "
                  f"{(matrix_meta or {}).get('connected', {}).get('selected_sequencer')}), "
                  f"aucun appel Vroom supplementaire", flush=True)
        else:
            print(f"Sequencement Vroom des groupes {strategy}...", flush=True)
            routes_idx, _seq_dur, vroom_ok, vroom_error = _sequence_groups(
                points, groups, start_idx, end_idx, headers
            )

    # 3. 2-OPT haversine : seulement si Vroom a echoue (Vroom deja optimal pour la duree ORS)
    # Un ordre deja sequence sur la matrice ORS ne passe pas par ce repli :
    # le 2-opt haversine le degraderait au lieu de le sauver.
    if routes_idx and not vroom_ok and presequenced_routes is None:
        print("2-opt par tournee (fallback haversine)...", flush=True)
        routes_idx = apply_two_opt(points, routes_idx)
        post_processing.append("haversine_2opt")

    # 4. Or-opt + 2-opt routier
    road_metrics = []
    d2_probe = None                 # D-2 : mesure seule, aucun changement de comportement
    routes_after_or2opt = None
    if routes_idx:
        print("Or-opt + 2-opt routier...", flush=True)
        _d2_t0 = time.time()
        _d2_calls0 = _api_calls_total()
        routes_before_or2opt = [list(r) for r in routes_idx]
        try:
            routes_idx, road_metrics = apply_or_opt_and_routing_2opt(points, routes_idx)
            post_processing.append("or2opt")
        except Exception as e:
            print(f"Or-opt + 2-opt routier: erreur ignoree ({e}), on continue", flush=True)
        routes_after_or2opt = [list(r) for r in routes_idx]
        d2_probe = {
            "elapsed_ms": int((time.time() - _d2_t0) * 1000),
            "api_calls": _api_calls_total() - _d2_calls0,
            # l'etape 4 a-t-elle reellement modifie l'ordre des routes ?
            "reordered_by_or2opt": [a != b for a, b in zip(routes_before_or2opt, routes_after_or2opt)],
            # renseignes apres l'etape 5
            "swaps_ran": False,
            "order_survived_swaps": None,
            "pointsets_changed_by_swaps": None,
        }

    # 5. POST-PROCESSING : swap des points frontiere
    # Une partition verrouillee interdit les swaps meme si Vroom a echoue :
    # sans cette precision, un echec Vroom rendait swap_stop_reason
    # "vroom_error" alors que la vraie raison reste le verrou territorial.
    if routes_idx and (vroom_ok or presequenced_routes is not None) and membership_locked:
        # Aucun appel : les swaps deplacent des points entre tournees, ce que la
        # partition territoriale interdit par construction.
        swap_stats["swap_stop_reason"] = swap_lock_reason
        print(f"Post-processing: swaps non lances, appartenance verrouillee "
              f"({swap_lock_reason})", flush=True)

    elif routes_idx and vroom_ok:
        routes_idx, swap_metrics, swap_stats = post_process_swaps(
            points, routes_idx, start_idx, end_idx, max_per_vehicle,
            entry_metrics=road_metrics,
            max_candidates=max_swap_candidates,
            max_consecutive_fails=swap_max_consecutive_fails
        )
        if swap_metrics is not None:
            road_metrics = swap_metrics
        post_processing.append("swaps")

        if d2_probe is not None and routes_after_or2opt is not None:
            d2_probe["swaps_ran"] = True
            d2_probe["order_survived_swaps"] = [
                a == b for a, b in zip(routes_after_or2opt, routes_idx)
            ]
            # si l'ensemble des points n'a pas bouge mais que l'ordre a change,
            # c'est que l'etape 5 a rejete puis reconstruit l'ordre de l'etape 4.
            d2_probe["pointsets_changed_by_swaps"] = [
                set(a) != set(b) for a, b in zip(routes_after_or2opt, routes_idx)
            ]

    else:
        # Swaps non executes : Vroom indisponible ou aucune route. Sans cette
        # ligne, swap_stop_reason restait vide et se confondait avec un run
        # complet sans echange accepte -- c'est exactement ce qui a masque
        # l'indisponibilite Vroom pendant la campagne D-3.
        swap_stats["swap_stop_reason"] = "vroom_error"
        print(f"Post-processing: NON EXECUTE (routes={bool(routes_idx)}, "
              f"vroom_ok={vroom_ok}, erreur={vroom_error})", flush=True)

    print(f"Partition: {optimization_path} (moteur: {partition_engine}) | "
          f"post-traitement: {post_processing} | "
          f"vroom_ok={vroom_ok}, erreur={vroom_error}", flush=True)
    print(f"Metriques routes finales: {road_metrics}", flush=True)
    print(f"Appels API: vroom={_API_STATS['vroom']} matrix={_API_STATS['matrix']} "
          f"total={_api_calls_total()} | duree calcul={int((time.time()-t_start)*1000)}ms", flush=True)
    print(f"Sonde D-2: {d2_probe}", flush=True)

    # Metriques des routes REELLEMENT retournees, apres post-optimisation.
    # Elles decrivent routes_idx tel qu'il part dans la reponse : c'est le
    # seul couple duree/distance qu'un lecteur de Benchmark peut confronter
    # aux tournees affichees. None des qu'une tournee n'a pas de mesure ORS.
    final_post_optimizer = "+".join(post_processing) if post_processing else "none"
    if road_metrics and all(m.get("duration_s") is not None for m in road_metrics):
        final_total_duration_s = round(sum(m["duration_s"] for m in road_metrics), 1)
    else:
        final_total_duration_s = None
    if road_metrics and all(m.get("km") is not None for m in road_metrics):
        final_total_distance_m = round(sum(m["km"] for m in road_metrics) * 1000.0, 1)
    else:
        final_total_distance_m = None

    # 6. FORMAT RESPONSE (compatible code.js)
    response = {
        "num_clusters_dbscan": num_vehicles,
        "vroom_used": vroom_ok,
        "vroom_error": vroom_error,
        "optimization_path": optimization_path,

        # --- champs additifs (aucune cle existante supprimee) ---
        "strategy_requested": strategy,
        "strategy_used": strategy_used,
        "partition_engine": partition_engine,
        "post_processing": post_processing,
        "ors_matrix": matrix_meta,
        "ortools_solution_limit": ortools_limit_effective,
        "ortools_solution_limit_requested": ortools_solution_limit,
        "ortools_solution_limit_override_applied": ortools_limit_override_applied,
        "partition_solver": partition_solver,

        # --- lot D-3 : cout et pilotage des swaps ---
        "max_swap_candidates": swap_stats["max_swap_candidates"],
        "swap_max_consecutive_fails": swap_stats["swap_max_consecutive_fails"],
        "swap_candidates_tested": swap_stats["swap_candidates_tested"],
        "swaps_accepted": swap_stats["swaps_accepted"],
        "swap_resequence_cache_hits": swap_stats["swap_resequence_cache_hits"],
        "swap_resequence_cache_misses": swap_stats["swap_resequence_cache_misses"],
        "swap_vroom_calls_saved": swap_stats["swap_vroom_calls_saved"],
        "swap_stop_reason": swap_stats["swap_stop_reason"],

        # --- certificat territorial (ortools_ors_matrix) ---
        "territorial_partition": _terr_get(matrix_meta, "territorial_partition", False),
        "territorial_method": _terr_get(matrix_meta, "territorial_method", None),
        "territorial_membership_locked": membership_locked,
        "territorial_candidates_generated": _terr_get(matrix_meta, "territorial_candidates_generated", 0),
        "territorial_candidates_unique": _terr_get(matrix_meta, "territorial_candidates_unique", 0),
        "territorial_candidates_scored": _terr_get(matrix_meta, "territorial_candidates_scored", 0),
        "territorial_side_violations": _terr_get(matrix_meta, "territorial_side_violations", None),
        "territorial_separator_angle_deg": _terr_get(matrix_meta, "territorial_separator_angle_deg", None),
        "territorial_separator_margin_m": _terr_get(matrix_meta, "territorial_separator_margin_m", None),
        "territorial_overlap_status": _terr_get(matrix_meta, "territorial_overlap_status", None),
        "territorial_fallback_used": _terr_get(matrix_meta, "territorial_fallback_used", False),
        "territorial_error": _terr_get(matrix_meta, "territorial_error", ""),
        "territorial_enum_ms": _terr_get(matrix_meta, "territorial_enum_ms", None),
        "territorial_score_ms": _terr_get(matrix_meta, "territorial_score_ms", None),

        # --- certificat de connexite (ortools_ors_matrix_connected) ---
        "connected_partition": _conn_get(matrix_meta, "connected_partition", False),
        "connected_method": _conn_get(matrix_meta, "connected_method", None),
        "connected_membership_locked": _conn_get(matrix_meta, "connected_membership_locked", False),
        "connected_target_sizes": _conn_get(matrix_meta, "connected_target_sizes", None),
        "connected_components_t1": _conn_get(matrix_meta, "connected_components_t1", None),
        "connected_components_t2": _conn_get(matrix_meta, "connected_components_t2", None),
        "connected_component_sizes_t1": _conn_get(matrix_meta, "connected_component_sizes_t1", None),
        "connected_component_sizes_t2": _conn_get(matrix_meta, "connected_component_sizes_t2", None),
        "connected_candidates_generated": _conn_get(matrix_meta, "connected_candidates_generated", 0),
        "connected_candidates_valid": _conn_get(matrix_meta, "connected_candidates_valid", 0),
        "connected_candidates_scored": _conn_get(matrix_meta, "connected_candidates_scored", 0),
        "connected_candidates_ortools": _conn_get(matrix_meta, "connected_candidates_ortools", 0),
        "connected_candidates_vroom": _conn_get(matrix_meta, "connected_candidates_vroom", 0),
        "connected_cut_edges": _conn_get(matrix_meta, "connected_cut_edges", None),
        "connected_cut_length_m": _conn_get(matrix_meta, "connected_cut_length_m", None),
        "connected_cross_neighbors": _conn_get(matrix_meta, "connected_cross_neighbors", None),
        "connected_enclave_points": _conn_get(matrix_meta, "connected_enclave_points", None),
        "connected_selected_seed": _conn_get(matrix_meta, "connected_selected_seed", None),
        "connected_fallback_used": _conn_get(matrix_meta, "connected_fallback_used", False),
        "connected_error": _conn_get(matrix_meta, "connected_error", ""),
        "connected_graph_k": _conn_get(matrix_meta, "connected_graph_k", None),
        "connected_vroom_calls": _conn_get(matrix_meta, "connected_vroom_calls", 0),
        "selected_sequencer": _conn_get(matrix_meta, "selected_sequencer", None),
        "final_selection_reason": _conn_get(matrix_meta, "final_selection_reason", ""),
        "ortools_total_duration_s": _conn_get(matrix_meta, "ortools_total_duration_s", None),
        "ortools_total_distance_m": _conn_get(matrix_meta, "ortools_total_distance_m", None),
        "vroom_total_duration_s": _conn_get(matrix_meta, "vroom_total_duration_s", None),
        "vroom_total_distance_m": _conn_get(matrix_meta, "vroom_total_distance_m", None),
        "connected_solutions_considered": _conn_get(matrix_meta, "connected_solutions_considered", 0),
        "connected_selection_window_s": _conn_get(matrix_meta, "connected_selection_window_s", None),
        "connected_selected_duration_s": _conn_get(matrix_meta, "connected_selected_duration_s", None),
        "connected_selected_distance_m": _conn_get(matrix_meta, "connected_selected_distance_m", None),
        "connected_vroom_cache_hits": _conn_get(matrix_meta, "connected_vroom_cache_hits", 0),
        "connected_vroom_error": _conn_get(matrix_meta, "connected_vroom_error", ""),

        # --- diversification des partitions connexes ---
        # Ajoutees A LA FIN : aucune colonne existante de Benchmark ne bouge.
        "connected_candidates_raw": _conn_get(matrix_meta, "connected_candidates_raw", 0),
        "connected_candidates_unique": _conn_get(matrix_meta, "connected_candidates_unique", 0),
        "connected_candidates_duplicates": _conn_get(matrix_meta, "connected_candidates_duplicates", 0),
        "connected_candidates_invalid_size": _conn_get(matrix_meta, "connected_candidates_invalid_size", 0),
        "connected_candidates_disconnected": _conn_get(matrix_meta, "connected_candidates_disconnected", 0),
        "connected_candidates_repair_failed": _conn_get(matrix_meta, "connected_candidates_repair_failed", 0),
        "connected_candidates_by_source": _conn_get(matrix_meta, "connected_candidates_by_source", {}),
        # Meme information, aplatie : une feuille de calcul ne sait pas lire un
        # objet, mais elle sait lire "mst=6;sweep=21".
        "connected_candidates_by_source_text": _flatten_counts(
            _conn_get(matrix_meta, "connected_candidates_by_source", {})),
        "connected_candidates_sweep": _conn_get(matrix_meta, "connected_candidates_sweep", 0),
        "connected_candidates_mst": _conn_get(matrix_meta, "connected_candidates_mst", 0),
        "connected_candidates_region_growing": _conn_get(matrix_meta, "connected_candidates_region_growing", 0),
        "connected_candidates_two_means": _conn_get(matrix_meta, "connected_candidates_two_means", 0),
        "connected_candidates_kmedoids": _conn_get(matrix_meta, "connected_candidates_kmedoids", 0),
        "connected_candidates_ors_repair": _conn_get(matrix_meta, "connected_candidates_ors_repair", 0),
        "connected_candidates_perturbation": _conn_get(matrix_meta, "connected_candidates_perturbation", 0),
        "connected_candidate_min_difference": _conn_get(matrix_meta, "connected_candidate_min_difference", 0),
        "connected_candidates_selected_diverse": _conn_get(matrix_meta, "connected_candidates_selected_diverse", 0),
        "connected_graph_method": _conn_get(matrix_meta, "connected_graph_method", None),
        "connected_ors_neighbor_k": _conn_get(matrix_meta, "connected_ors_neighbor_k", None),
        "connected_diversity_error": _conn_get(matrix_meta, "connected_diversity_error", ""),

        # --- protection de l'incumbent historique ---
        "connected_candidates_legacy": _conn_get(matrix_meta, "connected_candidates_legacy", 0),
        "connected_candidates_local_search": _conn_get(matrix_meta, "connected_candidates_local_search", 0),
        "connected_legacy_protected": _conn_get(matrix_meta, "connected_legacy_protected", False),
        "connected_legacy_seed": _conn_get(matrix_meta, "connected_legacy_seed", None),
        "connected_legacy_proxy_rank": _conn_get(matrix_meta, "connected_legacy_proxy_rank", None),
        "connected_legacy_finalist_slots": _conn_get(matrix_meta, "connected_legacy_finalist_slots", 0),
        "connected_legacy_finalists": _conn_get(matrix_meta, "connected_legacy_finalists", 0),
        "connected_legacy_in_finalists": _conn_get(matrix_meta, "connected_legacy_in_finalists", False),
        "connected_legacy_is_winner": _conn_get(matrix_meta, "connected_legacy_is_winner", False),
        "connected_legacy_duration_s": _conn_get(matrix_meta, "connected_legacy_duration_s", None),
        "connected_legacy_distance_m": _conn_get(matrix_meta, "connected_legacy_distance_m", None),
        "connected_per_source": _conn_get(matrix_meta, "connected_per_source", {}),
        "connected_per_source_text": _flatten_per_source(
            _conn_get(matrix_meta, "connected_per_source", {})),
        "connected_generation_expired_after": _conn_get(matrix_meta, "connected_generation_expired_after", None),
        # Empreinte du CONTENU de la matrice ORS : deux runs sur la meme
        # signature de points peuvent avoir recu des durees routieres
        # differentes. Sans ce champ, aucune comparaison de runs n'est fondee.
        "connected_matrix_hash": _conn_get(matrix_meta, "connected_matrix_hash", None),

        # --- post-optimisation intra-tournee, communes a toutes les strategies ---
        "final_post_optimizer": final_post_optimizer,
        "final_total_duration_s": final_total_duration_s,
        "final_total_distance_m": final_total_distance_m,
        "elapsed_ms": int((time.time() - t_start) * 1000),
        "api_calls": {
            "vroom": _API_STATS["vroom"],
            "matrix": _API_STATS["matrix"],
            "total": _api_calls_total(),
        },
        "partition_sizes": [max(0, len(r) - 2) for r in routes_idx] if routes_idx else [],
        "points_signature": _points_signature(points),
        "d2_probe": d2_probe,
    }

    for v in range(num_vehicles):
        key = "tournee_" + str(v + 1)
        if v < len(routes_idx):
            response[key] = [points[i]["id"] for i in routes_idx[v]]
            if v < len(road_metrics) and road_metrics[v]["km"] is not None:
                response[key + "_km"]  = road_metrics[v]["km"]
                response[key + "_min"] = road_metrics[v]["min"]
            else:
                response[key + "_km"]  = _compute_route_distance(points, routes_idx[v])
                response[key + "_min"] = None
        else:
            response[key] = []
            response[key + "_km"]  = 0
            response[key + "_min"] = None

    return jsonify(response)


# =========================
# 8. TEST
# =========================
@app.route("/")
def home():
    return "API OK - Vroom VRP ready"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
