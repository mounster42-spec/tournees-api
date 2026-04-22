from flask import Flask, request, jsonify
import math
import os
import time
import numpy as np
import requests
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
ORS_VROOM_URL = "https://api.openrouteservice.org/optimization"


# =========================
# 3. VROOM MULTI-VEHICULES (affectation + sequencement simultanes)
# =========================
def _call_vroom_multi(jobs, vehicles, headers):
    """Un appel Vroom multi-vehicules. Retourne (routes_by_vehicle, total_duration) ou (None, err)."""
    try:
        response = requests.post(
            ORS_VROOM_URL,
            json={"jobs": jobs, "vehicles": vehicles},
            headers=headers,
            timeout=30
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


def _auto_num_runs(n_jobs):
    """Ajuste automatiquement le nombre de runs Vroom selon la taille du probleme.
    Moins de jobs = plus de runs (recherche plus exhaustive, calcul plus rapide).
    """
    if n_jobs <= 15:
        return 5
    elif n_jobs <= 25:
        return 4
    elif n_jobs <= 40:
        return 3
    else:
        return 2


def optimize_with_vroom(points, num_vehicles, max_per_vehicle, start_idx, end_idx):
    """Appelle Vroom avec tous les vehicules pour affectation + sequencement simultanes.
       Nombre de runs auto-ajuste selon la taille : plus le probleme est petit, plus on teste d'ordres."""

    # ORS free tier : max 3500 routes = ~59 locations
    # locations = nb_jobs + nb_depots_uniques
    num_locations = len(points) + 1  # jobs + 1 depot (start=end)
    if num_locations > 59:
        print(f"Vroom multi-vehicules: {num_locations} locations > 59 (limite ORS 3500), skip", flush=True)
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
    n_runs = _auto_num_runs(len(delivery_indices))
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

    for name, jobs in orderings_to_run:
        print(f"  Run '{name}'...", flush=True)
        result, err = _call_vroom_multi(jobs, vehicles, headers)
        if result:
            routes, dur = result
            print(f"    -> {dur}s", flush=True)
            if dur < best_duration:
                best_duration = dur
                best = routes
                best_name = name
        else:
            last_err = err
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
            response = requests.post(
                ORS_VROOM_URL,
                json={"jobs": jobs, "vehicles": [vehicle]},
                headers=headers,
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
    """Re-sequence un vehicule avec Vroom. Retourne (route, duration) ou (None, None)."""
    if not vehicle_pts:
        return [start_idx, end_idx], 0

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
        response = requests.post(
            ORS_VROOM_URL,
            json={"jobs": jobs, "vehicles": [vehicle]},
            headers=headers,
            timeout=20
        )
        data = response.json()
        if "routes" not in data:
            return None, None

        ordered = [start_idx]
        for step in data["routes"][0]["steps"]:
            if step["type"] == "job":
                ordered.append(step["id"])
        ordered.append(end_idx)
        dur = data["routes"][0].get("duration", 0)
        return ordered, dur

    except Exception:
        return None, None


def post_process_swaps(points, routes_idx, start_idx, end_idx, max_per_vehicle):
    """Post-processing iteratif : echanges de points frontiere jusqu'a convergence.
    Deux modes :
    - Deplacement : point A (T1) -> T2, si T2 n'est pas plein
    - Echange    : point A (T1) <-> point B (T2), maintient l'equilibre (fonctionne meme 30/30)
    Relance la detection de frontiere apres chaque amelioration (max 5 iterations, 50 appels Vroom).
    """
    if len(routes_idx) != 2:
        return routes_idx

    headers = {
        "Authorization": ORS_KEY,
        "Content-Type": "application/json"
    }
    depot = {start_idx, end_idx}

    pts0 = [p for p in routes_idx[0] if p not in depot]
    pts1 = [p for p in routes_idx[1] if p not in depot]

    route0, dur0 = _resequence_single(points, pts0, start_idx, end_idx, headers)
    route1, dur1 = _resequence_single(points, pts1, start_idx, end_idx, headers)

    if dur0 is None or dur1 is None:
        print("Post-processing: impossible de calculer durees initiales", flush=True)
        return routes_idx

    best_total = dur0 + dur1
    best_routes = [route0, route1]
    best_pts = [list(pts0), list(pts1)]
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

                r_from, d_from = _resequence_single(points, new_pts_from, start_idx, end_idx, headers)
                r_to, d_to = _resequence_single(points, new_pts_to, start_idx, end_idx, headers)
                total_tested += 1

                if d_from is not None and d_to is not None:
                    gain = best_total - (d_from + d_to)
                    if gain > 0:
                        print(f"    Deplacement pt {pt_a} T{v_from+1}->T{v_to+1}: +{gain}s", flush=True)
                        best_total = d_from + d_to
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

                r_from, d_from = _resequence_single(points, new_pts_from, start_idx, end_idx, headers)
                r_to, d_to = _resequence_single(points, new_pts_to, start_idx, end_idx, headers)
                total_tested += 1

                if d_from is None or d_to is None:
                    continue

                gain = best_total - (d_from + d_to)
                if gain > 0:
                    print(f"    Echange pt {pt_a}(T{v_from+1}) <-> pt {pt_b}(T{v_to+1}): +{gain}s", flush=True)
                    best_total = d_from + d_to
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

    return best_routes


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
        response = requests.post(
            "https://api.openrouteservice.org/v2/matrix/driving-car",
            json={"locations": locations, "metrics": ["distance", "duration"]},
            headers=headers,
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
    """2-opt utilisant une matrice de distances routieres reelles."""
    best = list(route_local)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                d_current = matrix[best[i-1]][best[i]] + matrix[best[j]][best[j+1]]
                d_new     = matrix[best[i-1]][best[j]] + matrix[best[i]][best[j+1]]
                if d_new < d_current - 1e-6:
                    best[i:j+1] = best[i:j+1][::-1]
                    improved = True
    return best


def _matrix_route_cost(matrix, route_local):
    """Calcule le cout total d'une route a partir d'une matrice."""
    return sum(matrix[route_local[i]][route_local[i+1]] for i in range(len(route_local)-1))


def apply_or_opt_and_routing_2opt(points, routes_idx):
    """Pour chaque tournee : Or-opt + 2-opt sur matrice ORS distance reelle.
    Retourne (routes_ameliorees, road_metrics) ou road_metrics = [{km, min}, ...]."""
    headers = {
        "Authorization": ORS_KEY,
        "Content-Type": "application/json"
    }
    improved_routes = []
    road_metrics = []

    for v, route in enumerate(routes_idx):
        dist_matrix, dur_matrix = _fetch_ors_matrix(points, route, headers)

        if dist_matrix:
            n = len(route)
            local = list(range(n))

            # Or-opt sur distances routieres
            before_m = _matrix_route_cost(dist_matrix, local)
            local = _or_opt_matrix(dist_matrix, local)
            after_or = _matrix_route_cost(dist_matrix, local)
            if after_or < before_m:
                print(f"  Or-opt T{v+1}: {before_m/1000:.2f}km -> {after_or/1000:.2f}km (-{(before_m-after_or)/1000:.2f}km)", flush=True)

            # 2-opt sur distances routieres
            local = _two_opt_matrix(dist_matrix, local)
            after_2opt = _matrix_route_cost(dist_matrix, local)
            if after_2opt < after_or:
                print(f"  2-opt routier T{v+1}: {after_or/1000:.2f}km -> {after_2opt/1000:.2f}km (-{(after_or-after_2opt)/1000:.2f}km)", flush=True)

            route = [route[i] for i in local]
            road_km  = round(_matrix_route_cost(dist_matrix, local) / 1000, 2)
            road_min = round(_matrix_route_cost(dur_matrix,  local) / 60,   1) if dur_matrix else None
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

    data = request.json
    points = data.get("points", [])
    num_vehicles = data.get("num_vehicles", 2)
    max_per_vehicle = data.get("max_per_vehicle", 35)
    start_id = data.get("start_id", "")
    end_id = data.get("end_id", "")

    if not points:
        return jsonify({"error": "no points"}), 400

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

    # 1. VROOM MULTI-VEHICULES (affectation + sequencement sur reseau routier reel)
    routes_idx, vroom_ok, vroom_error = optimize_with_vroom(
        points, num_vehicles, max_per_vehicle, start_idx, end_idx
    )

    # 2. FALLBACK: K-Means + Vroom par vehicule
    if routes_idx is None:
        print("Fallback K-Means + Vroom...", flush=True)
        routes_idx, vroom_ok, vroom_error = kmeans_partition(
            points, num_vehicles, max_per_vehicle, start_idx, end_idx
        )

    # 3. 2-OPT haversine : amelioration locale de chaque tournee
    if routes_idx:
        print("2-opt par tournee...", flush=True)
        routes_idx = apply_two_opt(points, routes_idx)

    # 4. Or-opt + 2-opt routier
    road_metrics = []
    if routes_idx:
        print("Or-opt + 2-opt routier...", flush=True)
        try:
            routes_idx, road_metrics = apply_or_opt_and_routing_2opt(points, routes_idx)
        except Exception as e:
            print(f"Or-opt + 2-opt routier: erreur ignoree ({e}), on continue", flush=True)

    # 5. POST-PROCESSING : swap des points frontiere
    if routes_idx and vroom_ok:
        routes_idx = post_process_swaps(
            points, routes_idx, start_idx, end_idx, max_per_vehicle
        )

    # 6. FORMAT RESPONSE (compatible code.js)
    response = {
        "num_clusters_dbscan": num_vehicles,
        "vroom_used": vroom_ok,
        "vroom_error": vroom_error
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