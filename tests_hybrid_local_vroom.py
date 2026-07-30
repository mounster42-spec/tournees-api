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
# Deux grappes nettement separees : la partition geographiquement correcte
# est evidente, donc un test qui echoue signale un defaut d'orchestration et
# non une instance ambigue.

DEPOT = (48.8566, 2.3522)


def make_points(n_tasks=60):
    points = [{"id": "DEPOT", "lat": DEPOT[0], "lon": DEPOT[1]}]
    half = n_tasks // 2
    for i in range(n_tasks):
        cluster = 0 if i < half else 1
        rank = i if cluster == 0 else i - half
        lat = DEPOT[0] + (0.05 if cluster == 0 else -0.05) + rank * 0.0015
        lon = DEPOT[1] + (0.07 if cluster == 0 else -0.07) + (rank % 5) * 0.0012
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
            import time as _time
            remaining = cancellation_deadline - _time.monotonic()
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

    def setUp(self):
        self.env_backup = dict(os.environ)
        os.environ["LOCAL_VROOM_EXPERIMENT_ENABLED"] = "true"
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
        self.run_strategy()
        self.assertEqual(len(self.solver.calls), 1)
        payload = self.solver.calls[0]
        self.assertEqual(len(payload["vehicles"]), 2)
        self.assertEqual(len(payload["jobs"]), 60)

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

    def test_a_failed_solve_does_not_crash_the_strategy(self):
        self.solver.mode = "fail"
        groups, err, meta = self.run_strategy()
        # Sans aucune solution valide, l'echec est explicite : on ne fabrique
        # pas de solution de remplacement.
        self.assertIsNone(groups)
        self.assertTrue(err)
        self.assertEqual(meta["hybrid"]["joint_direct_error"],
                         local_vroom.ERR_TIMEOUT)
        self.assertEqual(meta["hybrid"]["local_vroom_timed_out"], 1)

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


if __name__ == "__main__":
    unittest.main()
