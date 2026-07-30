"""Tests de la strategie experimentale hybrid_local_vroom_territorial.

Aucun reseau, aucun subprocess : la matrice ORS est fabriquee sur place et le
solveur VROOM est remplace par une fonction deterministe. Ce qui est teste ici
est l'ORCHESTRATION -- budgets, compteurs, juge commun, contraintes dures,
selection -- pas la qualite d'optimisation de VROOM, qui est prouvee ailleurs.

Le solveur factice est appele exactement comme le vrai, donc le compteur de
resolutions, les gardes de temps et la validation stricte sont bien exerces.
"""

import math
import os
import sys
import time
import types
import unittest

# Les modules lourds sont remplaces avant l'import d'app, comme dans les
# autres fichiers de tests du depot : la suite doit tourner sur la seule
# bibliotheque standard.
if "flask" not in sys.modules:
    def _stub(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class _FakeFlask:
        def __init__(self, *a, **k):
            pass

        def route(self, *a, **k):
            return lambda fn: fn

    _stub("flask", Flask=_FakeFlask, request=None, jsonify=lambda *a, **k: None)
    _stub("requests", post=None, get=None)
    _stub("numpy")
    sys.modules["sklearn"] = types.ModuleType("sklearn")
    _stub("sklearn.cluster", KMeans=object)
    _stub("sklearn.metrics", silhouette_score=None)

import app                                                        # noqa: E402
import local_vroom                                                # noqa: E402


# =========================================================================
# INSTANCE SYNTHETIQUE
# =========================================================================
# Deux blocs adjacents, en grille reguliere, separes par un intervalle valant
# deux pas de grille. Chaque bloc est donc connexe dans G5, la partition
# naturelle est evidente, MAIS il existe de vrais points de frontiere : sans
# eux, les variantes a noyaux n'auraient rien a redessiner et le test ne
# prouverait rien.

DEPOT = (48.8566, 2.3522)
STEP = 0.004
COLUMNS = 6


def make_points(n_tasks=60):
    points = [{"id": "DEPOT", "lat": DEPOT[0], "lon": DEPOT[1]}]
    half = n_tasks - n_tasks // 2
    for i in range(n_tasks):
        cluster = 0 if i < half else 1
        rank = i if cluster == 0 else i - half
        lat = DEPOT[0] + 0.02 + (rank // COLUMNS) * STEP
        lon = DEPOT[1] + (rank % COLUMNS) * STEP
        if cluster == 1:
            lon += COLUMNS * STEP + STEP      # un pas d'ecart entre les blocs
        points.append({"id": "P%02d" % (i + 1), "lat": lat, "lon": lon})
    return points


def haversine_m(a, b):
    radius = 6371000.0
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    x = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0]))
         * math.sin(dlon / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(x))


def make_matrices(points, speed_kmh=28.0):
    speed = speed_kmh * 1000.0 / 3600.0
    size = len(points)
    dur = [[0] * size for _ in range(size)]
    dist = [[0] * size for _ in range(size)]
    for i in range(size):
        pi = (points[i]["lat"], points[i]["lon"])
        for j in range(i + 1, size):
            pj = (points[j]["lat"], points[j]["lon"])
            metres = int(round(haversine_m(pi, pj)))
            seconds = int(round(metres / speed))
            dur[i][j] = dur[j][i] = seconds
            dist[i][j] = dist[j][i] = metres
    return dur, dist


class MatrixStub:
    """Remplace _build_full_matrix_chunked et COMPTE les appels.

    Le comptage est le point du test : la strategie doit tenir dans deux
    appels Matrix au maximum, quel que soit le nombre de solutions evaluees.
    """

    def __init__(self, points, calls=2):
        self.dur, self.dist = make_matrices(points)
        self.calls = 0
        self.declared_calls = calls

    def __call__(self, points, headers):
        self.calls += 1
        meta = {"n": len(points), "calls": self.declared_calls, "blocks": 2,
                "cached": False, "nulls": 0}
        return self.dur, self.dist, meta, None


class SolverStub:
    """Solveur VROOM factice, deterministe et respectueux du contrat.

    Il lit le payload comme le vrai binaire : capacite, skills, indices de
    localisation. Il produit une affectation par plus proche voisin sur la
    matrice fournie, ce qui suffit a exercer la validation stricte et le juge
    commun sans lancer de processus.
    """

    def __init__(self, mode="ok"):
        self.mode = mode
        self.calls = []

    def __call__(self, payload, timeout_s=None, ledger=None,
                 cancellation_deadline=None, config=None):
        config = config or local_vroom.get_config()

        # Le solveur factice reproduit les MEMES gardes que le vrai wrapper,
        # sinon un test pourrait croire qu'un budget est respecte alors que
        # seul le faux solveur l'ignore.
        if not config.enabled:
            raise local_vroom.LocalVroomError(local_vroom.ERR_DISABLED, "off")
        if ledger is not None and ledger.budget_left() <= 0:
            ledger.stop(local_vroom.ERR_BUDGET_EXHAUSTED)
            raise local_vroom.LocalVroomError(local_vroom.ERR_BUDGET_EXHAUSTED,
                                              "plafond atteint")
        remaining = None
        if cancellation_deadline is not None:
            remaining = cancellation_deadline - time.monotonic()
        if remaining is not None and remaining < config.min_remaining_to_start_s:
            if ledger is not None:
                ledger.record_skip_for_time()
            raise local_vroom.LocalVroomError(local_vroom.ERR_GLOBAL_TIME_LIMIT,
                                              "plus assez de temps")
        if ledger is not None:
            ledger.record_attempt()

        self.calls.append(payload)

        if self.mode == "fail":
            if ledger is not None:
                ledger.record_failure(local_vroom.ERR_TIMEOUT, 10)
            raise local_vroom.LocalVroomError(local_vroom.ERR_TIMEOUT, "timeout")

        solution = self._solve(payload)
        if ledger is not None:
            ledger.record_success(10)
        return solution

    def _solve(self, payload):
        durations = payload["matrices"]["car"]["durations"]
        vehicles = payload["vehicles"]
        capacity = vehicles[0]["capacity"][0]
        jobs = {j["id"]: j for j in payload["jobs"]}

        # Skills : un job qui en porte une n'est servable que par le vehicule
        # qui la porte aussi. Le faux solveur les respecte, sinon les tests de
        # noyaux ne prouveraient rien.
        forced = {}
        for vehicle in vehicles:
            for skill in vehicle.get("skills") or []:
                forced[skill] = vehicle["id"]

        buckets = {v["id"]: [] for v in vehicles}
        free = []
        for jid, job in sorted(jobs.items()):
            skills = job.get("skills") or []
            owner = next((forced[s] for s in skills if s in forced), None)
            if owner is not None:
                buckets[owner].append(jid)
            else:
                free.append(jid)

        # Les points libres vont au vehicule dont le barycentre courant est le
        # plus proche au sens de la matrice, sans jamais depasser la capacite.
        for jid in free:
            loc = jobs[jid]["location_index"]
            best_vid, best_cost = None, None
            for vehicle in vehicles:
                vid = vehicle["id"]
                if len(buckets[vid]) >= capacity:
                    continue
                if buckets[vid]:
                    cost = min(durations[jobs[o]["location_index"]][loc]
                               for o in buckets[vid])
                else:
                    cost = durations[vehicle["start_index"]][loc]
                if best_cost is None or cost < best_cost:
                    best_vid, best_cost = vid, cost
            if best_vid is None:
                best_vid = min(buckets, key=lambda v: len(buckets[v]))
            buckets[best_vid].append(jid)

        routes = []
        for vehicle in vehicles:
            vid = vehicle["id"]
            order = self._nearest_neighbour(durations, vehicle["start_index"],
                                            buckets[vid], jobs)
            steps = [{"type": "start", "location_index": vehicle["start_index"]}]
            steps += [{"type": "job", "id": jid} for jid in order]
            steps.append({"type": "end", "location_index": vehicle["end_index"]})
            duration = 0
            cursor = vehicle["start_index"]
            for jid in order:
                duration += durations[cursor][jobs[jid]["location_index"]]
                cursor = jobs[jid]["location_index"]
            duration += durations[cursor][vehicle["end_index"]]
            routes.append({"vehicle": vid, "steps": steps, "duration": duration})

        return {"code": 0, "unassigned": [], "routes": routes,
                "summary": {"duration": sum(r["duration"] for r in routes)}}

    @staticmethod
    def _nearest_neighbour(durations, start, job_ids, jobs):
        remaining = list(job_ids)
        order = []
        cursor = start
        while remaining:
            nxt = min(remaining,
                      key=lambda j: (durations[cursor][jobs[j]["location_index"]], j))
            order.append(nxt)
            cursor = jobs[nxt]["location_index"]
            remaining.remove(nxt)
        return order


class HybridTestCase(unittest.TestCase):
    """Socle commun : environnement active, solveur et matrice injectes."""

    n_tasks = 60
    # Les budgets de recherche sont raccourcis pour la suite : leurs valeurs
    # par defaut sont verifiees dans tests_local_vroom, et laisser tourner
    # six secondes d'ALNS a chaque test rendrait la suite inutilisable. Les
    # tests qui mesurent la recherche elle-meme les relevent explicitement.
    alns_budget_s = "0.25"
    route_first_budget_s = "0.5"

    def setUp(self):
        self.env_backup = dict(os.environ)
        os.environ["LOCAL_VROOM_EXPERIMENT_ENABLED"] = "true"
        os.environ["LOCAL_VROOM_ALNS_BUDGET_S"] = self.alns_budget_s
        os.environ["LOCAL_VROOM_ROUTE_FIRST_BUDGET_S"] = self.route_first_budget_s
        local_vroom.get_config(refresh=True)

        self.points = make_points(self.n_tasks)
        self.matrix = MatrixStub(self.points)
        self.solver = SolverStub()

        self._real_matrix = app._build_full_matrix_chunked
        self._real_solver = local_vroom.solve_vroom_local
        app._build_full_matrix_chunked = self.matrix
        local_vroom.solve_vroom_local = self.solver

        self.addCleanup(self._restore)

    def _restore(self):
        app._build_full_matrix_chunked = self._real_matrix
        local_vroom.solve_vroom_local = self._real_solver
        os.environ.clear()
        os.environ.update(self.env_backup)
        local_vroom.get_config(refresh=True)

    def run_strategy(self, max_per_vehicle=None):
        capacity = max_per_vehicle or app._exact_capacity(self.n_tasks, 2)
        return app.hybrid_local_vroom_territorial(
            self.points, 2, capacity, 0, 0, {"Authorization": "dummy"})


# =========================================================================
# JUGE COMMUN
# =========================================================================

class TestCommonJudge(unittest.TestCase):

    def test_score_is_the_sum_of_arcs(self):
        dur = [[0, 10, 20], [10, 0, 30], [20, 30, 0]]
        score = app.score_routes_on_matrix(dur, None, [[0, 1, 2, 0]])
        self.assertEqual(score["duration_s"], 10 + 30 + 20)
        self.assertEqual(score["stops"], 2)

    def test_service_time_is_counted_once_per_stop(self):
        dur = [[0, 10, 20], [10, 0, 30], [20, 30, 0]]
        plain = app.score_routes_on_matrix(dur, None, [[0, 1, 2, 0]], service_s=0)
        served = app.score_routes_on_matrix(dur, None, [[0, 1, 2, 0]], service_s=60)
        self.assertEqual(served["duration_s"] - plain["duration_s"], 120)

    def test_two_routes_are_scored_together(self):
        dur = [[0, 10, 20], [10, 0, 30], [20, 30, 0]]
        one = app.score_routes_on_matrix(dur, None, [[0, 1, 0], [0, 2, 0]])
        self.assertEqual(one["duration_s"], 20 + 40)
        self.assertEqual(one["stops"], 2)

    def test_distance_uses_the_distance_matrix(self):
        dur = [[0, 10], [10, 0]]
        dist = [[0, 900], [900, 0]]
        score = app.score_routes_on_matrix(dur, dist, [[0, 1, 0]])
        self.assertEqual(score["duration_s"], 20)
        self.assertEqual(score["distance_m"], 1800)

    def test_the_same_routes_always_score_the_same(self):
        points = make_points(20)
        dur, dist = make_matrices(points)
        routes = [[0] + list(range(1, 11)) + [0], [0] + list(range(11, 21)) + [0]]
        first = app.score_routes_on_matrix(dur, dist, routes)
        second = app.score_routes_on_matrix(dur, dist, routes)
        self.assertEqual(first, second)


# =========================================================================
# GRAPHE G5
# =========================================================================

class TestGraphG5(unittest.TestCase):

    def test_k_is_fixed_at_five_and_never_grows(self):
        points = make_points(60)
        indices = list(range(1, 61))
        adjacency, meta = app.build_knn_graph_g5(points, indices, k=5)
        self.assertEqual(meta["k"], 5)
        self.assertEqual(meta["method"], "knn_haversine_strict")
        # Chaque sommet a au moins ses cinq plus proches voisins ; l'union
        # symetrique peut lui en donner davantage, jamais moins.
        for node in indices:
            self.assertGreaterEqual(len(adjacency[node]), 5)

    def test_graph_is_symmetric(self):
        points = make_points(30)
        indices = list(range(1, 31))
        adjacency, _ = app.build_knn_graph_g5(points, indices, k=5)
        for node, neighbours in adjacency.items():
            for other in neighbours:
                self.assertIn(node, adjacency[other])

    def test_graph_is_deterministic(self):
        points = make_points(40)
        indices = list(range(1, 41))
        first, _ = app.build_knn_graph_g5(points, indices, k=5)
        second, _ = app.build_knn_graph_g5(points, indices, k=5)
        self.assertEqual({k: sorted(v) for k, v in first.items()},
                         {k: sorted(v) for k, v in second.items()})

    def test_graph_uses_no_matrix_and_no_network(self):
        # La signature n'accepte aucune matrice : la geographie du graphe ne
        # peut donc pas etre contaminee par les durees routieres.
        import inspect
        params = list(inspect.signature(app.build_knn_graph_g5).parameters)
        self.assertEqual(params, ["points", "indices", "k"])


# =========================================================================
# BLOC A
# =========================================================================

class TestBlocADirect(HybridTestCase):

    def test_sixty_tasks_give_thirty_thirty(self):
        groups, err, meta = self.run_strategy()
        self.assertIsNone(err)
        self.assertEqual(sorted(len(g) for g in groups), [30, 30])
        diag = meta["hybrid"]
        self.assertTrue(diag["joint_direct_valid"])
        self.assertEqual(sorted(diag["joint_direct_sizes"]), [30, 30])

    def test_no_task_is_lost_or_duplicated(self):
        groups, err, meta = self.run_strategy()
        self.assertIsNone(err)
        merged = sorted(groups[0] + groups[1])
        self.assertEqual(merged, list(range(1, self.n_tasks + 1)))

    def test_one_request_with_two_vehicles_counts_as_one_solve(self):
        _, _, meta = self.run_strategy()
        payload = self.solver.calls[0]
        # Une requete porte les DEUX vehicules et les 60 taches, et pese une
        # seule resolution : le compteur suit les requetes, pas les vehicules.
        self.assertEqual(len(payload["vehicles"]), 2)
        self.assertEqual(len(payload["jobs"]), 60)
        self.assertEqual(meta["hybrid"]["local_vroom_attempted"],
                         len(self.solver.calls))

    def test_both_vehicles_share_start_and_end(self):
        self.run_strategy()
        payload = self.solver.calls[0]
        self.assertEqual({v["start_index"] for v in payload["vehicles"]}, {0})
        self.assertEqual({v["end_index"] for v in payload["vehicles"]}, {0})

    def test_no_geometry_is_requested(self):
        self.run_strategy()
        payload = self.solver.calls[0]
        self.assertNotIn("options", payload)

    def test_at_most_two_matrix_calls(self):
        _, _, meta = self.run_strategy()
        self.assertEqual(self.matrix.calls, 1)
        self.assertLessEqual(meta["hybrid"]["hybrid_matrix_calls"], 2)

    def test_declared_duration_is_a_diagnostic_not_the_score(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertIsNotNone(diag["joint_direct_declared_duration_s"])
        # La duree retenue est celle du juge commun : elle inclut le retour au
        # depot des deux tournees et ne coincide pas avec la valeur declaree.
        self.assertIsNotNone(diag["joint_direct_duration_s"])
        self.assertIsNotNone(diag["common_rescore_duration_s"])
        self.assertEqual(diag["common_rescore_duration_s"],
                         diag["joint_selected_duration_s"])

    def test_routes_are_returned_presequenced(self):
        _, _, meta = self.run_strategy()
        routes = meta["hybrid_routes"]
        self.assertEqual(len(routes), 2)
        for route in routes:
            self.assertEqual(route[0], 0)
            self.assertEqual(route[-1], 0)
        visited = sorted(p for r in routes for p in r[1:-1])
        self.assertEqual(visited, list(range(1, self.n_tasks + 1)))

    def test_membership_is_locked(self):
        _, _, meta = self.run_strategy()
        self.assertTrue(meta["hybrid_membership_locked"])

    def test_matrix_hash_is_reported(self):
        _, _, meta = self.run_strategy()
        self.assertTrue(meta["hybrid"]["common_rescore_matrix_hash"])

    def test_fifty_eight_tasks_give_twenty_nine_each(self):
        self.n_tasks = 58
        self.points = make_points(58)
        self.matrix = MatrixStub(self.points)
        app._build_full_matrix_chunked = self.matrix
        groups, err, meta = self.run_strategy()
        self.assertIsNone(err)
        self.assertEqual(sorted(len(g) for g in groups), [29, 29])
        payload = self.solver.calls[0]
        self.assertEqual([v["capacity"] for v in payload["vehicles"]],
                         [[29], [29]])


class TestBlocAResilience(HybridTestCase):

    def test_a_failed_solve_leaves_a_valid_incumbent(self):
        """VROOM peut echouer sans emporter la strategie.

        Route-first ne depend d'aucun solveur externe : il produit encore une
        solution valide, et c'est elle qui est retournee. L'echec VROOM est
        consigne tel quel, jamais maquille en succes."""
        self.solver.mode = "fail"
        groups, err, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertIsNone(err)
        self.assertEqual(sorted(len(g) for g in groups), [30, 30])
        self.assertEqual(diag["joint_direct_error"], local_vroom.ERR_TIMEOUT)
        self.assertFalse(diag["joint_direct_valid"])
        self.assertGreaterEqual(diag["local_vroom_timed_out"], 1)
        self.assertEqual(diag["local_vroom_succeeded"], 0)
        self.assertEqual(diag["joint_selected_source"], "route_first")

    def test_no_solution_at_all_is_an_explicit_failure(self):
        """Quand plus rien n'est valide, on echoue franchement.

        Une partition fabriquee au hasard serait pire qu'une erreur : elle
        serait servie comme une tournee reelle."""
        self.solver.mode = "fail"
        original = app.hybrid_route_first
        app.hybrid_route_first = lambda *a, **k: ([], {"cycles": 0, "cuts": 0,
                                                      "connected": 0, "unique": 0})
        self.addCleanup(setattr, app, "hybrid_route_first", original)
        groups, err, meta = self.run_strategy()
        self.assertIsNone(groups)
        self.assertTrue(err)

    def test_disabled_experiment_is_refused_before_any_work(self):
        os.environ["LOCAL_VROOM_EXPERIMENT_ENABLED"] = "false"
        local_vroom.get_config(refresh=True)
        groups, err, meta = self.run_strategy()
        self.assertIsNone(groups)
        self.assertEqual(err, local_vroom.ERR_DISABLED)
        self.assertEqual(self.matrix.calls, 0)
        self.assertEqual(len(self.solver.calls), 0)

    def test_a_second_optimisation_is_refused_while_one_runs(self):
        lock = local_vroom.LocalVroomRunLock()
        self.assertTrue(lock.acquire())
        self.addCleanup(lock.release)
        groups, err, meta = self.run_strategy()
        self.assertIsNone(groups)
        self.assertEqual(err, local_vroom.ERR_BUSY)
        self.assertEqual(self.matrix.calls, 0)

    def test_three_vehicles_are_refused(self):
        groups, err, meta = app.hybrid_local_vroom_territorial(
            self.points, 3, 20, 0, 0, {})
        self.assertIsNone(groups)
        self.assertIn("2 vehicles", err)

    def test_capacity_too_small_is_refused(self):
        groups, err, meta = self.run_strategy(max_per_vehicle=10)
        self.assertIsNone(groups)
        self.assertIn("cannot split", err)

    def test_ledger_diagnostics_are_complete(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        for key in ("local_vroom_enabled", "local_vroom_max_solves",
                    "local_vroom_attempted", "local_vroom_succeeded",
                    "local_vroom_failed", "local_vroom_timed_out",
                    "local_vroom_reused", "local_vroom_skipped_for_time",
                    "local_vroom_elapsed_ms", "local_vroom_stop_reason",
                    "local_vroom_last_error"):
            self.assertIn(key, diag)
        self.assertEqual(diag["local_vroom_max_solves"], 4)

    def test_stage_discipline_is_recorded(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        stages = {s["stage"] for s in diag["hybrid_stages"]}
        self.assertIn("matrix", stages)
        self.assertIn("joint_direct", stages)
        for stage in diag["hybrid_stages"]:
            for key in ("budget_s", "started_at_s", "elapsed_ms",
                        "stop_reason", "remaining_after_s"):
                self.assertIn(key, stage)
        self.assertIsInstance(diag["total_elapsed_ms"], int)
        self.assertFalse(diag["soft_limit_reached"])


# =========================================================================
# BLOC B
# =========================================================================

class TestBlocBNucleus(HybridTestCase):

    def test_cores_are_pinned_by_skills_and_border_stays_free(self):
        self.run_strategy()
        self.assertGreaterEqual(len(self.solver.calls), 2)
        payload = self.solver.calls[1]
        by_id = {j["id"]: j for j in payload["jobs"]}
        skilled = {jid: job["skills"] for jid, job in by_id.items()
                   if job.get("skills")}
        free = [jid for jid, job in by_id.items() if not job.get("skills")]
        self.assertTrue(skilled, "aucun point de noyau n'a ete fixe")
        self.assertTrue(free, "aucun point de frontiere n'est reste libre")
        self.assertEqual({tuple(v) for v in skilled.values()}, {(1,), (2,)})
        vehicles = {v["id"]: v.get("skills") for v in payload["vehicles"]}
        self.assertEqual(vehicles, {1: [1], 2: [2]})

    def test_a_single_nucleus_variant_by_default(self):
        _, _, meta = self.run_strategy()
        self.assertEqual(meta["hybrid"]["joint_nucleus_attempted"], 1)

    def test_border_and_cores_partition_the_points(self):
        points = make_points(60)
        indices = list(range(1, 61))
        adjacency, _ = app.build_knn_graph_g5(points, indices, k=5)
        group_a, group_b = indices[:30], indices[30:]
        free, core_a, core_b = app._hybrid_border_and_cores(
            group_a, group_b, adjacency, depth=1)
        self.assertEqual(sorted(free + core_a + core_b), indices)
        self.assertFalse(set(core_a) & set(core_b))
        self.assertTrue(set(core_a).issubset(group_a))
        self.assertTrue(set(core_b).issubset(group_b))

    def test_deeper_nucleus_frees_more_points(self):
        points = make_points(60)
        indices = list(range(1, 61))
        adjacency, _ = app.build_knn_graph_g5(points, indices, k=5)
        shallow, _, _ = app._hybrid_border_and_cores(
            indices[:30], indices[30:], adjacency, depth=1)
        deep, _, _ = app._hybrid_border_and_cores(
            indices[:30], indices[30:], adjacency, depth=2)
        self.assertTrue(set(shallow).issubset(set(deep)))
        self.assertGreater(len(deep), len(shallow))

    def test_a_nucleus_without_border_is_skipped_without_spending_a_solve(self):
        """Deux territoires sans aucun contact ne meritent pas de resolution.

        La variante ne pourrait que reproduire sa graine : la lancer serait un
        quart du budget depense pour un resultat deja connu."""
        points = make_points(60)
        indices = list(range(1, 61))
        adjacency = {i: set() for i in indices}      # aucun voisin, donc aucune frontiere
        seed = {"group_a": indices[:30], "group_b": indices[30:]}
        with self.assertRaises(local_vroom.LocalVroomError) as ctx:
            app.hybrid_nucleus_solve(
                points, indices, self.matrix.dur, self.matrix.dist, adjacency,
                0, 0, 30, seed, 1, local_vroom.LocalVroomLedger(),
                app._HybridClock(58.0), 0)
        self.assertEqual(ctx.exception.code, local_vroom.ERR_INVALID_SOLUTION)
        self.assertEqual(len(self.solver.calls), 0)


class TestBlocBRouteFirst(HybridTestCase):

    def _route_first(self, budget_s=3.0):
        indices = list(range(1, self.n_tasks + 1))
        adjacency, _ = app.build_knn_graph_g5(self.points, indices, k=5)
        clock = app._HybridClock(58.0)
        clock.begin("route_first", budget_s)
        solutions, stats = app.hybrid_route_first(
            self.points, indices, self.matrix.dur, self.matrix.dist,
            adjacency, 0, 0, self.n_tasks // 2, clock, 0)
        clock.end("done")
        return solutions, stats, adjacency, indices

    def test_route_first_makes_no_network_call(self):
        before = self.matrix.calls
        solutions, _, _, _ = self._route_first()
        self.assertEqual(self.matrix.calls, before)
        self.assertEqual(len(self.solver.calls), 0)
        self.assertTrue(solutions)

    def test_several_cycles_and_orientations_are_explored(self):
        _, stats, _, _ = self._route_first()
        self.assertGreaterEqual(stats["cycles"], 4)
        # Rotations x deux orientations : bien plus de decoupes que de cycles.
        self.assertGreater(stats["cuts"], stats["cycles"] * 2)
        self.assertGreater(stats["unique"], 1)

    def test_every_candidate_is_a_valid_contiguous_split(self):
        solutions, _, adjacency, indices = self._route_first()
        for solution in solutions:
            self.assertEqual(sorted(solution["group_a"] + solution["group_b"]),
                             indices)
            self.assertEqual(sorted(solution["sizes"]), [30, 30])
            self.assertTrue(solution["connected"])
            self.assertTrue(solution["cardinality_ok"])
            self.assertEqual(solution["components"], [1, 1])

    def test_candidates_are_scored_by_the_common_judge(self):
        solutions, _, _, _ = self._route_first()
        for solution in solutions:
            expected = app.score_routes_on_matrix(
                self.matrix.dur, self.matrix.dist,
                [solution["route_a"], solution["route_b"]], 0)
            self.assertAlmostEqual(solution["duration_s"], expected["duration_s"])
            self.assertAlmostEqual(solution["distance_m"], expected["distance_m"])

    def test_route_first_respects_its_hard_budget(self):
        started = time.monotonic()
        self._route_first(budget_s=0.4)
        # Le budget est un couperet : un depassement de plus d'une seconde
        # mangerait le temps des blocs suivants.
        self.assertLess(time.monotonic() - started, 1.6)

    def test_route_first_is_deterministic(self):
        first, _, _, _ = self._route_first()
        second, _, _, _ = self._route_first()
        self.assertEqual([s["partition_key"] for s in first],
                         [s["partition_key"] for s in second])
        self.assertEqual([s["duration_s"] for s in first],
                         [s["duration_s"] for s in second])

    def test_partitions_are_unique(self):
        solutions, _, _, _ = self._route_first()
        keys = [s["partition_key"] for s in solutions]
        self.assertEqual(len(keys), len(set(keys)))

    def test_refinement_never_loses_a_point(self):
        solutions, _, _, indices = self._route_first()
        for solution in solutions:
            visited = sorted(solution["route_a"][1:-1] + solution["route_b"][1:-1])
            self.assertEqual(visited, indices)


class TestBlocBIntegration(HybridTestCase):

    def test_best_of_all_blocks_wins(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertGreater(diag["joint_solutions_considered"], 2)
        self.assertIn(diag["joint_selected_source"],
                      ("joint_direct", "joint_nucleus_d1", "route_first"))
        # Le gagnant ne peut pas etre moins bon que la solution directe.
        self.assertLessEqual(diag["joint_selected_duration_s"],
                             diag["joint_direct_duration_s"])

    def test_route_first_diagnostics_are_filled(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertGreaterEqual(diag["route_first_cycles"], 4)
        self.assertGreaterEqual(diag["route_first_unique"], 1)
        self.assertIsNotNone(diag["route_first_best_duration_s"])

    def test_every_block_has_its_own_stage_record(self):
        _, _, meta = self.run_strategy()
        stages = {s["stage"] for s in meta["hybrid"]["hybrid_stages"]}
        # Les etapes des blocs A et B sont presentes ; la liste exhaustive de
        # toutes les etapes est verifiee dans TestTimeDiscipline.
        self.assertTrue({"matrix", "joint_direct", "joint_nucleus",
                         "route_first"}.issubset(stages))

    def test_solve_budget_is_never_exceeded(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertLessEqual(diag["local_vroom_attempted"],
                             diag["local_vroom_max_solves"])


# =========================================================================
# BLOC C
# =========================================================================

class TestBlocCAlns(HybridTestCase):

    def _alns(self, budget_s=0.4, seeds=None):
        indices = list(range(1, self.n_tasks + 1))
        adjacency, _ = app.build_knn_graph_g5(self.points, indices, k=5)
        if seeds is None:
            half = self.n_tasks // 2
            seeds = [{"group_a": indices[:half], "group_b": indices[half:]}]
        clock = app._HybridClock(58.0)
        clock.begin("joint_alns", budget_s)
        finalists, stats = app.hybrid_territorial_alns(
            self.points, indices, self.matrix.dur, self.matrix.dist,
            adjacency, 0, 0, seeds, "abc123", clock, 0)
        clock.end("done")
        return finalists, stats, adjacency, indices

    def test_alns_calls_neither_matrix_nor_vroom(self):
        before = self.matrix.calls
        finalists, stats, _, _ = self._alns()
        self.assertEqual(self.matrix.calls, before)
        self.assertEqual(len(self.solver.calls), 0)
        self.assertGreater(stats["iterations"], 0)

    def test_seed_is_derived_from_the_matrix_hash(self):
        _, stats, _, _ = self._alns()
        self.assertEqual(stats["seed"], app._hybrid_matrix_seed("abc123"))
        other = app._hybrid_matrix_seed("def456")
        self.assertNotEqual(stats["seed"], other)

    def test_same_matrix_gives_the_same_finalists(self):
        first, _, _, _ = self._alns(budget_s=0.4)
        second, _, _, _ = self._alns(budget_s=0.4)
        # Le budget etant temporel, le NOMBRE d'iterations varie ; le meilleur
        # finaliste, lui, doit etre le meme puisque la suite de mouvements
        # est identique.
        self.assertEqual(first[0]["partition_key"], second[0]["partition_key"])
        self.assertAlmostEqual(first[0]["proxy_cost"], second[0]["proxy_cost"])

    def test_every_finalist_satisfies_the_hard_constraints(self):
        finalists, _, adjacency, indices = self._alns()
        half = self.n_tasks // 2
        for finalist in finalists:
            merged = sorted(finalist["group_a"] + finalist["group_b"])
            self.assertEqual(merged, indices, "union incomplete ou doublon")
            self.assertEqual(sorted([len(finalist["group_a"]),
                                     len(finalist["group_b"])]),
                             [half, self.n_tasks - half])
            self.assertTrue(app.is_connected_partition(
                finalist["group_a"], adjacency)["connected"])
            self.assertTrue(app.is_connected_partition(
                finalist["group_b"], adjacency)["connected"])

    def test_finalists_are_unique_and_capped_at_twelve(self):
        finalists, _, _, _ = self._alns()
        keys = [f["partition_key"] for f in finalists]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertLessEqual(len(finalists), 12)

    def test_finalists_are_ranked_by_ors_cost(self):
        finalists, _, _, _ = self._alns()
        costs = [f["proxy_cost"] for f in finalists]
        self.assertEqual(costs, sorted(costs))

    def test_operators_cover_the_declared_families(self):
        _, stats, _, _ = self._alns(budget_s=0.6)
        used = set(stats["operators"])
        for operator in ("swap_1_1", "swap_2_2", "chain", "destroy_repair"):
            self.assertIn(operator, used)

    def test_a_move_that_breaks_cardinality_is_refused(self):
        indices = list(range(1, 11))
        adjacency = {i: {j for j in indices if j != i} for i in indices}
        self.assertFalse(app._hybrid_partition_valid(
            indices[:6], indices[6:], indices, (5, 5), adjacency))
        self.assertTrue(app._hybrid_partition_valid(
            indices[:5], indices[5:], indices, (5, 5), adjacency))

    def test_a_move_that_duplicates_a_point_is_refused(self):
        indices = list(range(1, 11))
        adjacency = {i: {j for j in indices if j != i} for i in indices}
        self.assertFalse(app._hybrid_partition_valid(
            [1, 2, 3, 4, 5], [5, 6, 7, 8, 9], indices, (5, 5), adjacency))

    def test_a_disconnected_move_is_refused(self):
        indices = [1, 2, 3, 4]
        adjacency = {1: {2}, 2: {1}, 3: {4}, 4: {3}}
        self.assertTrue(app._hybrid_partition_valid(
            [1, 2], [3, 4], indices, (2, 2), adjacency))
        self.assertFalse(app._hybrid_partition_valid(
            [1, 3], [2, 4], indices, (2, 2), adjacency))

    def test_haversine_only_locates_the_border(self):
        """Le graphe G5 dit OU regarder, la matrice ORS dit CE QUE ca coute."""
        indices = list(range(1, 7))
        adjacency = {1: {2}, 2: {1, 3}, 3: {2, 4}, 4: {3, 5}, 5: {4, 6}, 6: {5}}
        border_a, border_b = app._hybrid_border([1, 2, 3], [4, 5, 6], adjacency)
        self.assertEqual(border_a, [3])
        self.assertEqual(border_b, [4])


class TestBlocCFinalists(HybridTestCase):

    def test_at_most_four_solves_with_default_settings(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertLessEqual(diag["local_vroom_attempted"], 4)
        self.assertEqual(diag["local_vroom_max_solves"], 4)
        self.assertLessEqual(len(self.solver.calls), 4)

    def test_a_fifth_solve_is_impossible(self):
        self.run_strategy()
        # Le budget par defaut est 1 directe + 1 noyau + 2 finalistes.
        self.assertLessEqual(len(self.solver.calls), 4)
        calls_before = len(self.solver.calls)
        # Une resolution supplementaire sur un ledger epuise n'atteint jamais
        # le solveur.
        ledger = local_vroom.LocalVroomLedger(max_solves=4)
        for _ in range(4):
            ledger.record_attempt()
        with self.assertRaises(local_vroom.LocalVroomError) as ctx:
            self.solver({"vehicles": [], "jobs": []}, ledger=ledger,
                        config=local_vroom.get_config())
        self.assertEqual(ctx.exception.code, local_vroom.ERR_BUDGET_EXHAUSTED)
        self.assertEqual(len(self.solver.calls), calls_before)

    def test_budget_can_be_raised_to_eight_by_environment(self):
        os.environ["LOCAL_VROOM_MAX_SOLVES"] = "8"
        os.environ["LOCAL_VROOM_NUCLEUS_SOLVES"] = "2"
        os.environ["LOCAL_VROOM_FINALIST_SOLVES"] = "5"
        local_vroom.get_config(refresh=True)
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertEqual(diag["local_vroom_max_solves"], 8)
        self.assertGreater(diag["local_vroom_attempted"], 4)
        self.assertLessEqual(diag["local_vroom_attempted"], 8)

    def test_at_most_two_finalists_are_solved_by_default(self):
        _, _, meta = self.run_strategy()
        self.assertLessEqual(
            meta["hybrid"]["joint_finalists_local_vroom_solved"], 2)

    def test_an_already_solved_partition_is_never_solved_twice(self):
        _, _, meta = self.run_strategy()
        keys = []
        for payload in self.solver.calls:
            skilled_a = sorted(j["id"] for j in payload["jobs"]
                               if j.get("skills") == [1])
            skilled_b = sorted(j["id"] for j in payload["jobs"]
                               if j.get("skills") == [2])
            if skilled_a and skilled_b and not [
                    j for j in payload["jobs"] if not j.get("skills")]:
                keys.append(app.canonical_partition_key(skilled_a, skilled_b))
        # Les requetes a partition entierement figee portent chacune une
        # partition distincte : aucune n'est resolue deux fois.
        self.assertEqual(len(keys), len(set(keys)))

    def test_a_fixed_partition_goes_in_a_single_request(self):
        _, _, meta = self.run_strategy()
        fixed = [p for p in self.solver.calls
                 if p["jobs"] and all(j.get("skills") for j in p["jobs"])]
        for payload in fixed:
            self.assertEqual(len(payload["vehicles"]), 2)
            self.assertEqual(len(payload["jobs"]), self.n_tasks)

    def test_a_persistent_error_stops_the_finalist_loop(self):
        """Une cause qui se reproduira n'est pas reessayee douze fois.

        Binaire absent, budget epuise, temps ecoule : le finaliste suivant
        echouerait a l'identique. Seule une solution invalide, propre a une
        partition, laisse sa chance a la suivante."""
        self.solver.mode = "fail"
        _, _, meta = self.run_strategy()
        # Direct, noyaux, puis un seul finaliste avant l'arret.
        self.assertLessEqual(len(self.solver.calls), 3)
        self.assertEqual(
            meta["hybrid"]["joint_finalists_local_vroom_solved"], 0)

    def test_nucleus_uses_a_route_first_seed_when_the_direct_solve_fails(self):
        """Route-first passe avant les noyaux justement pour cela.

        Sans graine, la variante a noyaux serait sautee des que la resolution
        directe echoue, et le budget resterait inutilise."""
        self.solver.mode = "fail"
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertGreaterEqual(diag["route_first_unique"], 1)
        self.assertEqual(diag["joint_nucleus_attempted"], 1)
        self.assertNotEqual(diag["joint_nucleus_error"], "no seed solution")

    def test_finalist_partitions_are_respected_by_the_solver(self):
        groups, err, meta = self.run_strategy()
        self.assertIsNone(err)
        self.assertEqual(sorted(groups[0] + groups[1]),
                         list(range(1, self.n_tasks + 1)))


class TestSelection(HybridTestCase):

    def test_minimum_duration_wins_outside_the_window(self):
        solutions = [
            {"connected": True, "cardinality_ok": True, "duration_s": 1000.0,
             "distance_m": 10.0, "boundary": {}, "partition_key": (),
             "sequencer": "a", "route_a": [], "route_b": []},
            {"connected": True, "cardinality_ok": True, "duration_s": 900.0,
             "distance_m": 99999.0, "boundary": {}, "partition_key": (),
             "sequencer": "b", "route_a": [], "route_b": []},
        ]
        best = app.select_best_solution(solutions, tie_seconds=30.0)
        self.assertEqual(best["duration_s"], 900.0)

    def test_inside_the_thirty_second_window_distance_decides(self):
        solutions = [
            {"connected": True, "cardinality_ok": True, "duration_s": 900.0,
             "distance_m": 50000.0, "boundary": {}, "partition_key": (),
             "sequencer": "a", "route_a": [], "route_b": []},
            {"connected": True, "cardinality_ok": True, "duration_s": 925.0,
             "distance_m": 40000.0, "boundary": {}, "partition_key": (),
             "sequencer": "b", "route_a": [], "route_b": []},
        ]
        best = app.select_best_solution(solutions, tie_seconds=30.0)
        self.assertEqual(best["distance_m"], 40000.0)

    def test_the_window_is_exactly_thirty_seconds(self):
        solutions = [
            {"connected": True, "cardinality_ok": True, "duration_s": 900.0,
             "distance_m": 50000.0, "boundary": {}, "partition_key": (),
             "sequencer": "a", "route_a": [], "route_b": []},
            {"connected": True, "cardinality_ok": True, "duration_s": 930.0,
             "distance_m": 40000.0, "boundary": {}, "partition_key": (),
             "sequencer": "b", "route_a": [], "route_b": []},
            {"connected": True, "cardinality_ok": True, "duration_s": 930.1,
             "distance_m": 10.0, "boundary": {}, "partition_key": (),
             "sequencer": "c", "route_a": [], "route_b": []},
        ]
        best = app.select_best_solution(solutions, tie_seconds=30.0)
        # 930,0 est DANS la fenetre, 930,1 est dehors malgre sa distance.
        self.assertEqual(best["distance_m"], 40000.0)

    def test_no_balancing_objective_between_the_two_routes(self):
        """Un ecart de duree entre T1 et T2 n'est jamais penalise."""
        balanced = [
            {"connected": True, "cardinality_ok": True, "duration_s": 1000.0,
             "distance_m": 100.0, "boundary": {}, "partition_key": (),
             "sequencer": "equilibree", "route_a": [], "route_b": []},
            {"connected": True, "cardinality_ok": True, "duration_s": 990.0,
             "distance_m": 100.0, "boundary": {}, "partition_key": (),
             "sequencer": "desequilibree", "route_a": [], "route_b": []},
        ]
        best = app.select_best_solution(balanced, tie_seconds=0.0)
        self.assertEqual(best["sequencer"], "desequilibree")

    def test_selection_is_reported(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertIsNotNone(diag["joint_selected_source"])
        self.assertIsNotNone(diag["joint_selected_duration_s"])
        self.assertIsNotNone(diag["joint_selected_sizes"])
        self.assertEqual(diag["joint_selection_window_s"], 30.0)


class TestTimeDiscipline(HybridTestCase):

    def test_a_short_global_limit_still_returns_a_valid_result(self):
        """Quand la limite souple est courte, la strategie ne rend pas rien.

        Elle arrete la generation, ne lance plus de resolution, choisit parmi
        ce qu'elle a et retourne proprement."""
        os.environ["LOCAL_VROOM_TOTAL_SOFT_LIMIT_S"] = "10"
        local_vroom.get_config(refresh=True)
        groups, err, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertIsNone(err)
        self.assertEqual(sorted(len(g) for g in groups), [30, 30])
        self.assertLessEqual(diag["total_elapsed_ms"], 14000)

    def test_no_solve_starts_below_the_minimum_remaining(self):
        """Sous le seuil, plus aucune resolution ne part.

        La limite souple est fixee si bas qu'aucune resolution ne peut plus
        etre lancee : le compteur doit rester a zero, pas simplement echouer
        apres coup."""
        os.environ["LOCAL_VROOM_TOTAL_SOFT_LIMIT_S"] = "2"
        local_vroom.get_config(refresh=True)
        groups, err, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertEqual(diag["local_vroom_attempted"], 0)
        self.assertEqual(len(self.solver.calls), 0)
        self.assertGreaterEqual(diag["local_vroom_skipped_for_time"], 1)
        # Route-first ne depend d'aucune resolution : un resultat valide sort
        # quand meme.
        self.assertIsNone(err)
        self.assertEqual(sorted(len(g) for g in groups), [30, 30])

    def test_every_stage_records_budget_and_stop_reason(self):
        _, _, meta = self.run_strategy()
        stages = meta["hybrid"]["hybrid_stages"]
        expected = {"matrix", "joint_direct", "joint_nucleus", "route_first",
                    "joint_alns", "alns_refine", "joint_finalists"}
        self.assertEqual({s["stage"] for s in stages}, expected)
        for stage in stages:
            self.assertIsNotNone(stage["stop_reason"])
            self.assertGreaterEqual(stage["elapsed_ms"], 0)
            self.assertGreaterEqual(stage["remaining_after_s"], 0.0)

    def test_stages_never_overrun_the_global_limit(self):
        _, _, meta = self.run_strategy()
        diag = meta["hybrid"]
        self.assertLess(diag["total_elapsed_ms"], 58000)

    def test_matrix_is_called_once_whatever_the_number_of_solutions(self):
        _, _, meta = self.run_strategy()
        self.assertEqual(self.matrix.calls, 1)
        self.assertLessEqual(meta["hybrid"]["hybrid_matrix_calls"], 2)
        self.assertGreater(meta["hybrid"]["joint_solutions_considered"], 5)


if __name__ == "__main__":
    unittest.main()
