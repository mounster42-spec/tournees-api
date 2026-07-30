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

    _matrix_cache_put(key, (dur_matrix, dist_matrix, nulls))
    print(f"  Matrice ORS {n}x{n} assemblee en {meta['calls']} appel(s), "
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

CONNECTED_KNN_START = 4          # k initial du graphe de voisinage
CONNECTED_KNN_MAX = 12           # au-dela, l'arbre couvrant prend le relais
CONNECTED_TOP_ORTOOLS = 12       # candidates envoyees au solveur (niveau 2)
CONNECTED_TOP_VROOM = 3          # candidates envoyees a Vroom (niveau 3)
CONNECTED_LOCAL_ITERS = 60       # bornes de la recherche locale
CONNECTED_REPAIR_ITERS = 200
CONNECTED_TIE_SECONDS = 30.0     # ecart de duree sous lequel on departage au km


def build_geo_graph(points, indices, k=CONNECTED_KNN_START):
    """Graphe de voisinage non oriente sur les points de collecte.

    k plus proches voisins au sens haversine, k augmente tant que le graphe
    global n'est pas connexe, puis UNION avec un arbre couvrant minimal. Cette
    union garantit la connexite globale meme pour un point tres isole, ce que
    les kNN seuls ne donnent jamais.
    Retourne (adjacency, meta). Aucune dependance, aucun appel reseau.
    """
    n = len(indices)
    adjacency = {i: set() for i in indices}
    if n <= 1:
        return adjacency, {"k": 0, "mst_edges": 0, "connected": True}

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

    # Arbre couvrant minimal (Prim), en union : filet de securite.
    mst_added = 0
    in_tree = {indices[0]}
    rest = set(indices[1:])
    while rest:
        best = min(((u, v) for u in in_tree for v in rest),
                   key=lambda e: (dist[e], e[0], e[1]))
        u, v = best
        key = (min(u, v), max(u, v))
        if key not in edges:
            edges.add(key)
            mst_added += 1
        in_tree.add(v)
        rest.discard(v)

    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    return adjacency, {"k": used_k, "mst_edges": mst_added,
                       "connected": _graph_connected(indices, adjacency),
                       "edges": len(edges)}


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


def repair_to_connected(ga, gb, adjacency, points, dur_matrix,
                        start_idx, end_idx, target_a):
    """Rend deux groupes connexes en conservant la cardinalite exacte.

    Deux temps : absorber les composantes secondaires -- iles et enclaves --
    dans le groupe voisin, puis retablir la cardinalite en deplacant des points
    de FRONTIERE choisis pour degrader le moins possible la duree ORS, et
    seulement s'ils laissent les deux groupes connexes.
    Retourne (ga, gb, ok, moves).
    """
    ga, gb = list(ga), list(gb)
    moves = 0

    for _ in range(CONNECTED_REPAIR_ITERS):
        ia = is_connected_partition(ga, adjacency)
        ib = is_connected_partition(gb, adjacency)
        if ia["connected"] and ib["connected"]:
            break
        # On deplace la plus petite composante secondaire, cote le plus morcele.
        if ia["component_count"] > 1:
            src, dst, comps = ga, gb, ia["components"]
        else:
            src, dst, comps = gb, ga, ib["components"]
        smallest = min(comps[1:], key=lambda c: (len(c), c[0]))
        for node in smallest:
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
        for node in sorted(src):
            # candidat de frontiere uniquement : il doit toucher l'autre groupe
            if not any(nb in set(dst) for nb in adjacency.get(node, ())):
                continue
            new_src = [x for x in src if x != node]
            new_dst = list(dst) + [node]
            if not is_connected_partition(new_src, adjacency)["connected"]:
                continue
            if not is_connected_partition(new_dst, adjacency)["connected"]:
                continue
            delta = _move_cost_delta(dur_matrix, src, dst, node, start_idx, end_idx)
            if best is None or (delta, node) < (best[0], best[1]):
                best = (delta, node, new_src, new_dst)
        if best is None:
            return ga, gb, False, moves
        _, _, new_src, new_dst = best
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


def _two_means_partition(points, indices, target_a):
    """2-moyennes local et deterministe : germes = les deux points les plus
    eloignes. Evite d'importer sklearn ici et reste testable hors production."""
    if len(indices) < 2:
        return list(indices), []
    xy = _local_xy(points, indices)
    seed_a, seed_b, best = indices[0], indices[1], -1.0
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            ia, ib = indices[a], indices[b]
            d = math.hypot(xy[ia][0] - xy[ib][0], xy[ia][1] - xy[ib][1])
            if d > best:
                best, seed_a, seed_b = d, ia, ib
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


def generate_connected_candidates(points, indices, target_a, adjacency,
                                  dur_matrix, hav_matrix, start_idx, end_idx):
    """Partitions initiales diversifiees, puis reparees jusqu'a la connexite.

    Sources : ORS non contrainte (l'incumbent de performance), balayage
    lineaire, Haversine, 2-moyennes local, et plusieurs coupes du balayage.
    Chacune est normalisee en cardinalite PUIS reparee : la solution ORS n'est
    jamais acceptee telle quelle si elle reste morcelee.
    Retourne (candidates, stats).
    """
    stats = {"generated": 0, "valid": 0, "sources": {}}
    xy = _local_xy(points, indices)
    seen = {}
    raw = []

    def add(seed, ga, gb):
        raw.append((seed, list(ga), list(gb)))

    # 1. ORS non contrainte : la performance a reparer.
    if ORTOOLS_AVAILABLE:
        g, _err = _solve_cvrp_ortools(dur_matrix, 2, len(indices), start_idx, end_idx)
        if g and len(g) == 2:
            add("ors_unconstrained", g[0], g[1])
        # 3. Haversine, meme solveur sur la matrice a vol d'oiseau.
        gh, _err = _solve_cvrp_ortools(hav_matrix, 2, len(indices), start_idx, end_idx)
        if gh and len(gh) == 2:
            add("haversine", gh[0], gh[1])

    # 2 et 5. Balayage lineaire : la meilleure et quelques coupes diversifiees.
    sweep, _sstats = enumerate_territorial_partitions(points, indices, target_a)
    for rank, c in enumerate(sweep[:8]):
        add("sweep_%d" % rank, c["group_a"], c["group_b"])

    # 4. 2-moyennes local.
    ka, kb = _two_means_partition(points, indices, target_a)
    add("two_means", ka, kb)

    for seed, ga, gb in raw:
        stats["generated"] += 1
        ga, gb = _normalize_sizes(ga, gb, indices, target_a, xy)
        ga, gb, ok, moves = repair_to_connected(
            ga, gb, adjacency, points, dur_matrix, start_idx, end_idx, target_a)
        if not ok:
            stats["sources"][seed] = "repair_failed"
            continue
        ga, gb, est, swaps = connected_local_search(
            ga, gb, adjacency, dur_matrix, start_idx, end_idx)
        key = _partition_key(ga, gb)
        if key in seen:
            stats["sources"][seed] = "duplicate"
            continue
        stats["sources"][seed] = "ok(repair=%d,swaps=%d)" % (moves, swaps)
        stats["valid"] += 1
        seen[key] = {"group_a": ga, "group_b": gb, "seed": seed,
                     "est_duration_s": est, "repair_moves": moves,
                     "local_swaps": swaps}

    candidates = sorted(seen.values(), key=lambda c: (c["est_duration_s"],
                                                      _partition_key(c["group_a"], c["group_b"])))
    return candidates, stats


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
    """Ordre lexicographique impose. L'equilibre entre tournees n'y figure pas.
    La duree est arrondie au palier de tolerance pour que le kilometrage puisse
    departager deux solutions pratiquement equivalentes en temps."""
    return (0 if cand["connected"] else 1,
            0 if cand["cardinality_ok"] else 1,
            cand["components_total"],
            round(cand["duration_s"] / CONNECTED_TIE_SECONDS),
            cand["distance_m"],
            cand["boundary"]["cut_edges"] + cand["boundary"]["enclave_points"],
            _partition_key(cand["group_a"], cand["group_b"]))


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

    adjacency, gmeta = build_geo_graph(points, indices)
    diag["connected_graph_k"] = gmeta["k"]
    print(f"  Graphe: k={gmeta['k']}, {gmeta['edges']} aretes, "
          f"{gmeta['mst_edges']} ajoutees par l'arbre couvrant, "
          f"connexe={gmeta['connected']}", flush=True)

    hav_matrix = _build_haversine_matrix(points)

    t_gen = time.time()
    cands, cstats = generate_connected_candidates(
        points, indices, target_a, adjacency, dur_matrix, hav_matrix,
        start_idx, end_idx)
    diag["connected_enum_ms"] = int((time.time() - t_gen) * 1000)
    diag["connected_candidates_generated"] = cstats["generated"]
    diag["connected_candidates_valid"] = cstats["valid"]
    print(f"  Candidates: {cstats['generated']} generees, {cstats['valid']} valides "
          f"en {diag['connected_enum_ms']}ms | {cstats['sources']}", flush=True)

    if not cands:
        diag["connected_error"] = "no connected partition could be built"
        return None, diag["connected_error"], meta

    # --- niveau 1 : heuristique locale sur la matrice ORS ---
    t_score = time.time()
    scored = []
    allset = set(indices)
    for c in cands:
        ga, gb = c["group_a"], c["group_b"]
        ia = is_connected_partition(ga, adjacency)
        ib = is_connected_partition(gb, adjacency)
        ra = _estimate_group_cost(dur_matrix, ga, start_idx, end_idx, True)[1]
        rb = _estimate_group_cost(dur_matrix, gb, start_idx, end_idx, True)[1]
        dur, dist = _rescore(dur_matrix, dist_matrix, ra, rb)
        scored.append({
            "group_a": ga, "group_b": gb, "seed": c["seed"],
            "route_a": ra, "route_b": rb, "duration_s": dur, "distance_m": dist,
            "connected": ia["connected"] and ib["connected"],
            "cardinality_ok": (len(ga) == target_a and len(gb) == n - target_a
                               and set(ga) | set(gb) == allset
                               and not (set(ga) & set(gb))),
            "components_total": ia["component_count"] + ib["component_count"] - 2,
            "comp_a": ia, "comp_b": ib,
            "boundary": boundary_metrics(ga, gb, adjacency, points),
            "sequencer": "heuristic",
        })
    diag["connected_candidates_scored"] = len(scored)
    scored.sort(key=_selection_key)

    # Incumbent : jamais remplace par une solution moins bonne.
    incumbent = scored[0]
    diag["final_selection_reason"] = "level1_heuristic"

    # --- niveau 2 : OR-Tools sur les meilleures, sequencement seul ---
    if ORTOOLS_AVAILABLE:
        for cand in scored[:CONNECTED_TOP_ORTOOLS]:
            ra = _tsp_order_ortools(dur_matrix, cand["group_a"], start_idx, end_idx)
            rb = _tsp_order_ortools(dur_matrix, cand["group_b"], start_idx, end_idx)
            if ra is None or rb is None:
                continue
            diag["connected_candidates_ortools"] += 1
            dur, dist = _rescore(dur_matrix, dist_matrix, ra, rb)
            variant = dict(cand)
            variant.update({"route_a": ra, "route_b": rb, "duration_s": dur,
                            "distance_m": dist, "sequencer": "ortools"})
            if _selection_key(variant) < _selection_key(incumbent):
                incumbent = variant
                diag["final_selection_reason"] = "level2_ortools"
        diag["ortools_total_duration_s"] = round(incumbent["duration_s"], 1)
        diag["ortools_total_distance_m"] = round(incumbent["distance_m"], 1)

    # --- niveau 3 : Vroom sur les 3 meilleures, 2 appels chacune ---
    finalists = sorted(scored, key=_selection_key)[:CONNECTED_TOP_VROOM]
    for cand in finalists:
        ra, da, _ = _resequence_single(points, cand["group_a"], start_idx, end_idx, headers)
        rb, db, _ = _resequence_single(points, cand["group_b"], start_idx, end_idx, headers)
        diag["connected_vroom_calls"] += 2
        if ra is None or rb is None:
            # Rate limit ou erreur : on garde la solution OR-Tools, sans echouer.
            diag["connected_fallback_used"] = True
            diag["connected_error"] = diag["connected_error"] or "vroom unavailable, kept OR-Tools order"
            break
        diag["connected_candidates_vroom"] += 1
        dur, dist = _rescore(dur_matrix, dist_matrix, ra, rb)
        variant = dict(cand)
        variant.update({"route_a": ra, "route_b": rb, "duration_s": dur,
                        "distance_m": dist, "sequencer": "vroom"})
        if diag["vroom_total_duration_s"] is None or dur < diag["vroom_total_duration_s"]:
            diag["vroom_total_duration_s"] = round(dur, 1)
            diag["vroom_total_distance_m"] = round(dist, 1)
        if _selection_key(variant) < _selection_key(incumbent):
            incumbent = variant
            diag["final_selection_reason"] = "level3_vroom"

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
    })

    print(f"  Retenue: {[len(ga), len(gb)]} pts, seed={incumbent['seed']}, "
          f"sequenceur={incumbent['sequencer']}, composantes 1/1, "
          f"{b['cut_edges']} aretes coupees, {b['enclave_points']} enclaves, "
          f"duree {incumbent['duration_s'] / 60:.1f}min "
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

        print(f"Sequencement Vroom des groupes {strategy}...", flush=True)
        routes_idx, _seq_dur, vroom_ok, vroom_error = _sequence_groups(
            points, groups, start_idx, end_idx, headers
        )

    # 3. 2-OPT haversine : seulement si Vroom a echoue (Vroom deja optimal pour la duree ORS)
    if routes_idx and not vroom_ok:
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
    if routes_idx and vroom_ok and membership_locked:
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
