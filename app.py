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
VALID_STRATEGIES = ("kmeans", "ortools_haversine", "ortools_ors_matrix")
IMPLEMENTED_STRATEGIES = {"kmeans"}
if ORTOOLS_AVAILABLE:
    IMPLEMENTED_STRATEGIES.add("ortools_haversine")
    IMPLEMENTED_STRATEGIES.add("ortools_ors_matrix")

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
ORTOOLS_SOLUTION_LIMIT = 50
ORTOOLS_TIME_LIMIT_S = 25

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


def ortools_partition_ors_matrix(points, num_vehicles, max_per_vehicle,
                                 start_idx, end_idx, headers,
                                 solution_limit=None):
    """Affectation OR-Tools sur les DUREES routieres reelles ORS.
    Le cout du solveur est la duree, pas la distance : c'est l'objectif de Vroom
    et celui des metriques finales. Retourne (groups, err, meta)."""
    meta = {}
    depot_indices = {start_idx, end_idx}
    delivery_indices = [i for i in range(len(points)) if i not in depot_indices]

    if not delivery_indices:
        return [[] for _ in range(num_vehicles)], None, meta

    if num_vehicles * max_per_vehicle < len(delivery_indices):
        return None, (f"capacity too small: {num_vehicles} x {max_per_vehicle} "
                      f"< {len(delivery_indices)} points"), meta

    print(f"OR-Tools matrice ORS: {len(delivery_indices)} points, {num_vehicles} vehicules, "
          f"capacite {max_per_vehicle}", flush=True)

    dur_matrix, dist_matrix, meta, err = _build_full_matrix_chunked(points, headers)
    if dur_matrix is None:
        return None, f"ORS matrix failed: {err}", meta

    groups, err = _solve_cvrp_ortools(
        dur_matrix, num_vehicles, max_per_vehicle, start_idx, end_idx,
        solution_limit=solution_limit
    )
    if groups is None:
        return None, err, meta

    print(f"  Partition: {[len(g) for g in groups]} pts", flush=True)
    return groups, None, meta


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


def post_process_swaps(points, routes_idx, start_idx, end_idx, max_per_vehicle):
    """Post-processing iteratif : echanges de points frontiere jusqu'a convergence.
    Deux modes :
    - Deplacement : point A (T1) -> T2, si T2 n'est pas plein
    - Echange    : point A (T1) <-> point B (T2), maintient l'equilibre (fonctionne meme 30/30)
    Relance la detection de frontiere apres chaque amelioration (max 5 iterations, 50 appels Vroom).
    """
    if len(routes_idx) != 2:
        return routes_idx, None

    headers = {
        "Authorization": ORS_KEY,
        "Content-Type": "application/json"
    }
    depot = {start_idx, end_idx}

    pts0 = [p for p in routes_idx[0] if p not in depot]
    pts1 = [p for p in routes_idx[1] if p not in depot]

    route0, dur0, dist0 = _resequence_single(points, pts0, start_idx, end_idx, headers)
    route1, dur1, dist1 = _resequence_single(points, pts1, start_idx, end_idx, headers)

    if dur0 is None or dur1 is None:
        print("Post-processing: impossible de calculer durees initiales", flush=True)
        return routes_idx, None

    best_total = dur0 + dur1
    best_routes = [route0, route1]
    best_pts = [list(pts0), list(pts1)]
    best_durs = [dur0, dur1]
    best_dists = [dist0, dist1]
    total_swaps = 0
    total_tested = 0
    MAX_ITER = 5
    MAX_CALLS = 50

    print(f"Post-processing: duree initiale = {dur0}s + {dur1}s = {best_total}s", flush=True)

    for iteration in range(MAX_ITER):
        if total_tested >= MAX_CALLS:
            break

        border = _find_border_points(points, best_routes, start_idx, end_idx)
        print(f"  Iteration {iteration+1}: {len(border)} points frontiere (seuil 500m)", flush=True)

        if not border:
            break

        improved = False

        for v_from, pt_a, dist_a in border[:15]:
            if total_tested >= MAX_CALLS:
                break

            v_to = 1 - v_from

            # --- MODE 1 : deplacement si l'autre route a de la place ---
            if len(best_pts[v_to]) < max_per_vehicle:
                new_pts_from = [p for p in best_pts[v_from] if p != pt_a]
                new_pts_to = best_pts[v_to] + [pt_a]

                r_from, d_from, dist_from = _resequence_single(points, new_pts_from, start_idx, end_idx, headers)
                r_to, d_to, dist_to = _resequence_single(points, new_pts_to, start_idx, end_idx, headers)
                total_tested += 1

                if d_from is not None and d_to is not None:
                    gain = best_total - (d_from + d_to)
                    if gain > 0:
                        print(f"    Deplacement pt {pt_a} T{v_from+1}->T{v_to+1}: +{gain}s", flush=True)
                        best_total = d_from + d_to
                        # D-1 : sans ces 4 lignes, best_durs/best_dists restaient sur les
                        # valeurs d'avant le deplacement et les minutes renvoyees etaient
                        # perimees (symetrie avec le MODE 2 ci-dessous).
                        best_durs[v_from]  = d_from
                        best_durs[v_to]    = d_to
                        best_dists[v_from] = dist_from
                        best_dists[v_to]   = dist_to
                        best_routes[v_from] = r_from
                        best_routes[v_to] = r_to
                        best_pts[v_from] = new_pts_from
                        best_pts[v_to] = new_pts_to
                        total_swaps += 1
                        improved = True
                        break  # relancer la detection

            # --- MODE 2 : echange pt_a (v_from) <-> pt_b (v_to) ---
            candidates_b = sorted(
                best_pts[v_to],
                key=lambda p: haversine(
                    (points[pt_a]["lat"], points[pt_a]["lon"]),
                    (points[p]["lat"], points[p]["lon"])
                )
            )[:5]

            for pt_b in candidates_b:
                if total_tested >= MAX_CALLS:
                    break

                new_pts_from = [pt_b if p == pt_a else p for p in best_pts[v_from]]
                new_pts_to = [pt_a if p == pt_b else p for p in best_pts[v_to]]

                r_from, d_from, dist_from = _resequence_single(points, new_pts_from, start_idx, end_idx, headers)
                r_to, d_to, dist_to = _resequence_single(points, new_pts_to, start_idx, end_idx, headers)
                total_tested += 1

                if d_from is None or d_to is None:
                    continue

                gain = best_total - (d_from + d_to)
                if gain > 0:
                    print(f"    Echange pt {pt_a}(T{v_from+1}) <-> pt {pt_b}(T{v_to+1}): +{gain}s", flush=True)
                    best_total = d_from + d_to
                    best_durs[v_from] = d_from
                    best_durs[v_to] = d_to
                    best_dists[v_from] = dist_from
                    best_dists[v_to] = dist_to
                    best_routes[v_from] = r_from
                    best_routes[v_to] = r_to
                    best_pts[v_from] = new_pts_from
                    best_pts[v_to] = new_pts_to
                    total_swaps += 1
                    improved = True
                    break

            if improved:
                break  # relancer la detection de frontiere

        if not improved:
            print(f"  Convergence atteinte a l'iteration {iteration+1}", flush=True)
            break

    if total_swaps > 0:
        print(f"Post-processing: {total_swaps} echange(s), {total_tested} appels, duree finale = {best_total}s", flush=True)
    else:
        print(f"Post-processing: aucun echange ameliorant ({total_tested} testes)", flush=True)

    # Distance routiere des routes finales : Vroom ne renvoie pas 'distance',
    # on la calcule via la matrice ORS (2 appels).
    final_dists = [best_dists[0], best_dists[1]]
    for v in range(2):
        dist_matrix, _ = _fetch_ors_matrix(points, best_routes[v], headers)
        if dist_matrix:
            final_dists[v] = _matrix_route_cost(dist_matrix, list(range(len(best_routes[v]))))
        print(f"  T{v+1} finale: {final_dists[v]/1000:.2f}km, ~{best_durs[v]/60:.1f}min", flush=True)

    metrics = [
        {"km": round(final_dists[0] / 1000, 2), "min": round(best_durs[0] / 60, 1)},
        {"km": round(final_dists[1] / 1000, 2), "min": round(best_durs[1] / 60, 1)},
    ]
    return best_routes, metrics


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
            road_metrics.append({"km": road_km, "min": road_min})
        else:
            print(f"  Matrice ORS T{v+1} indisponible, fallback haversine", flush=True)
            route = _or_opt(points, route)
            road_metrics.append({"km": _compute_route_distance(points, route), "min": None})

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
    if routes_idx and vroom_ok:
        routes_idx, swap_metrics = post_process_swaps(
            points, routes_idx, start_idx, end_idx, max_per_vehicle
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
        "ortools_solution_limit": ortools_solution_limit or ORTOOLS_SOLUTION_LIMIT,
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
