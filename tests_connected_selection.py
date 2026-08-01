"""
Tests COMPORTEMENTAUX de la selection OR-Tools / Vroom du mode connexe.

Lancement :
    python -m unittest tests_connected_selection -v

Ces tests ne cherchent pas des chaines dans le source : ils font tourner la
partition connexe -- et, pour les plus complets, l'endpoint /optimize entier --
avec des sequenceurs bouchonnes dont on connait exactement les ordres, puis
verifient que la solution RETOURNEE est bien celle annoncee.

Le bug corrige ici : la partition ne remontait que l'appartenance, jamais les
ordres. L'appelant resequencait donc les deux groupes avec Vroom -- deux appels
de plus -- et l'ordre final etait TOUJOURS celui de Vroom, meme quand le
reporting annoncait selected_sequencer = "ortools".

Aucun acces reseau. Matrice ORS, OR-Tools et Vroom sont tous bouchonnes.
"""

import math
import statistics
import sys
import types
import unittest


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
        def deco(fn):
            return fn
        return deco

    def run(self, *a, **k):
        pass


_stub("flask", Flask=_FakeFlask, request=None, jsonify=lambda *a, **k: None)
_stub("requests", post=lambda *a, **k: None)
_np = _stub("numpy")
_np.mean = lambda seq: statistics.fmean(list(seq))
_np.array = lambda x: x
sys.modules["sklearn"] = types.ModuleType("sklearn")
_stub("sklearn.cluster", KMeans=object)
_stub("sklearn.metrics", silhouette_score=lambda *a, **k: 0.0)

import app  # noqa: E402


# --------------------------------------------------------------- fixtures

def P(pid, lat, lon):
    return {"id": str(pid), "lat": lat, "lon": lon, "address": "pt " + str(pid)}


def two_lines(per_side=4):
    """Deux alignements nettement separes, depot en index 0.

    Sur chaque alignement l'ordre optimal est l'ordre des index : un ordre
    melange coute strictement plus cher, ce qui permet de fabriquer un
    sequenceur volontairement mauvais et un sequenceur optimal.
    """
    pts = [P("DEP", 45.500, 4.400)]
    for k in range(per_side):                      # ouest, alignes en latitude
        pts.append(P("W%d" % k, 45.440 + 0.004 * k, 4.380))
    for k in range(per_side):                      # est, alignes en latitude
        pts.append(P("E%d" % k, 45.440 + 0.004 * k, 4.430))
    return pts


def euclid_matrix(points, scale=1.0):
    """Matrice symetrique deterministe, tenant lieu de matrice ORS."""
    xy = app._local_xy(points, list(range(len(points))))
    n = len(points)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = math.hypot(xy[i][0] - xy[j][0],
                                     xy[i][1] - xy[j][1]) * scale
    return m


def sol(duration_s, distance_m, sequencer="ortools", partition_key=(1,),
        route_a=(0, 1, 0), route_b=(0, 2, 0), cut=0):
    """Solution minimale, telle que select_best_solution la recoit."""
    return {
        "duration_s": float(duration_s),
        "distance_m": float(distance_m),
        "sequencer": sequencer,
        "selection_reason": app._SELECTION_REASONS.get(sequencer, sequencer),
        "partition_key": partition_key,
        "route_a": list(route_a),
        "route_b": list(route_b),
        "connected": True,
        "cardinality_ok": True,
        "boundary": {"cut_edges": cut, "enclave_points": 0},
    }


class _Harness:
    """Bouchonne matrice ORS, OR-Tools et Vroom autour d'app.py.

    Les ordres rendus par chaque sequenceur sont pilotes par des fonctions
    fournies au test : c'est ce qui permet de decider a l'avance QUI doit
    gagner, puis de verifier que c'est bien sa route qui ressort.
    """

    def __init__(self, points, ortools_order, vroom_order, dist_scale=1000.0):
        self.points = points
        self.dur = euclid_matrix(points)
        self.dist = euclid_matrix(points, scale=dist_scale)
        self.ortools_order = ortools_order
        self.vroom_order = vroom_order
        self.vroom_calls = []
        self.matrix_calls = 0
        self._saved = {}

    # --- bouchons ---
    def _matrix(self, points, headers):
        self.matrix_calls += 1
        return self.dur, self.dist, {"ors_matrix": {"stub": True}}, None

    def _tsp(self, matrix, group, start_idx, end_idx):
        if self.ortools_order is None:
            return None
        return self.ortools_order(sorted(group), start_idx, end_idx)

    def _vroom(self, points, group, start_idx, end_idx, headers):
        self.vroom_calls.append(tuple(sorted(group)))
        app._API_STATS["vroom"] += 1
        if self.vroom_order is None:
            return None, None, None
        return self.vroom_order(sorted(group), start_idx, end_idx), 0, 0

    def _fetch_sub_matrix(self, points, route_indices, headers):
        """Sous-matrice de la MEME matrice ORS : la post-optimisation
        intra-tournee travaille donc sur les memes durees, sans reseau."""
        self.matrix_calls += 1
        app._API_STATS["matrix"] += 1
        d = [[self.dist[i][j] for j in route_indices] for i in route_indices]
        t = [[self.dur[i][j] for j in route_indices] for i in route_indices]
        return d, t

    def __enter__(self):
        for name in ("_build_full_matrix_chunked", "_tsp_order_ortools",
                     "_resequence_single", "_fetch_ors_matrix",
                     "ORTOOLS_AVAILABLE", "_solve_cvrp_ortools"):
            self._saved[name] = getattr(app, name)
        app._build_full_matrix_chunked = self._matrix
        app._tsp_order_ortools = self._tsp
        app._resequence_single = self._vroom
        app._fetch_ors_matrix = self._fetch_sub_matrix
        app.ORTOOLS_AVAILABLE = self.ortools_order is not None
        # Le CVRP non contraint n'est pas le sujet ici : les autres sources de
        # candidates (balayage, 2-moyennes) suffisent a peupler la generation.
        app._solve_cvrp_ortools = lambda *a, **k: (None, "stub")
        app._reset_api_stats()
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(app, name, value)
        return False


def ordered(group, start_idx, end_idx):
    """Ordre optimal sur les fixtures : les index croissants."""
    return [start_idx] + sorted(group) + [end_idx]


def shuffled(group, start_idx, end_idx):
    """Ordre volontairement mauvais : le premier point part a la fin."""
    g = sorted(group)
    if len(g) >= 3:
        g = g[1:2] + g[2:] + g[0:1]
    return [start_idx] + g + [end_idx]


def run_partition(points, ortools_order, vroom_order, **kw):
    with _Harness(points, ortools_order, vroom_order, **kw) as h:
        groups, err, meta = app.ortools_partition_ors_matrix_connected(
            points, 2, len(points), 0, 0, {})
    return groups, err, meta, h


# ------------------------------------------------- Test A / B : le gagnant

class TestWinningSequencerIsReturned(unittest.TestCase):
    """Tests A et B : selected_sequencer doit decrire les routes retournees."""

    def _assert_routes_match(self, meta, order_fn, groups):
        diag = meta["connected"]
        routes = meta["connected_routes"]
        expected = [order_fn(sorted(groups[0]), 0, 0),
                    order_fn(sorted(groups[1]), 0, 0)]
        self.assertEqual(routes, expected,
                         "les routes remontees ne sont pas celles du gagnant")
        # duree et distance recalculees DEPUIS les routes retournees
        dur = euclid_matrix(two_lines())
        dist = euclid_matrix(two_lines(), scale=1000.0)
        exp_d, exp_k = app._rescore(dur, dist, routes[0], routes[1])
        self.assertAlmostEqual(diag["connected_selected_duration_s"],
                               round(exp_d, 1), places=1)
        self.assertAlmostEqual(diag["connected_selected_distance_m"],
                               round(exp_k, 1), places=1)

    def test_A_ortools_wins_and_its_routes_are_returned(self):
        pts = two_lines()
        groups, err, meta, h = run_partition(pts, ordered, shuffled)
        self.assertIsNone(err)
        diag = meta["connected"]
        self.assertEqual(diag["selected_sequencer"], "ortools")
        self.assertEqual(diag["final_selection_reason"], "level2_ortools")
        self._assert_routes_match(meta, ordered, groups)
        # aucune route Vroom ne doit s'etre substituee
        for route, group in zip(meta["connected_routes"], groups):
            self.assertNotEqual(route, shuffled(sorted(group), 0, 0))
        self.assertLess(diag["ortools_total_duration_s"],
                        diag["vroom_total_duration_s"])

    def test_B_vroom_wins_and_its_routes_are_returned(self):
        pts = two_lines()
        groups, err, meta, h = run_partition(pts, shuffled, ordered)
        self.assertIsNone(err)
        diag = meta["connected"]
        self.assertEqual(diag["selected_sequencer"], "vroom")
        self.assertEqual(diag["final_selection_reason"], "level3_vroom")
        self._assert_routes_match(meta, ordered, groups)
        for route, group in zip(meta["connected_routes"], groups):
            self.assertNotEqual(route, shuffled(sorted(group), 0, 0))
        self.assertLess(diag["vroom_total_duration_s"],
                        diag["ortools_total_duration_s"])

    def test_membership_is_identical_whoever_wins(self):
        """Le sequenceur ne change JAMAIS l'appartenance : seul le NOM du
        gagnant change entre les deux runs, jamais la composition des groupes.
        Les deux runs retournent ici le meme ordre optimal, produit par
        OR-Tools dans le premier et par Vroom dans le second."""
        pts = two_lines()
        g1, _, m1, _ = run_partition(pts, ordered, shuffled)
        g2, _, m2, _ = run_partition(pts, shuffled, ordered)
        self.assertEqual([sorted(g) for g in g1], [sorted(g) for g in g2])
        self.assertNotEqual(m1["connected"]["selected_sequencer"],
                            m2["connected"]["selected_sequencer"])
        self.assertEqual(m1["connected_routes"], m2["connected_routes"])

    def test_bad_sequencer_order_is_never_returned(self):
        """Le perdant ne doit laisser AUCUNE trace dans les routes finales."""
        for otools, vroom, winner in ((ordered, shuffled, "ortools"),
                                      (shuffled, ordered, "vroom")):
            groups, _, meta, _ = run_partition(two_lines(), otools, vroom)
            self.assertEqual(meta["connected"]["selected_sequencer"], winner)
            for route, group in zip(meta["connected_routes"], groups):
                self.assertEqual(route, ordered(sorted(group), 0, 0))


# --------------------------------------- Test C : la fenetre de 30 secondes

class TestSelectionWindow(unittest.TestCase):
    """Test C : la fenetre porte sur les SECONDES exactes, bornes comprises."""

    def test_window_uses_exact_seconds_not_rounded_minutes(self):
        # 6641 s et 6663 s : 110.7 min et 111.1 min. Arrondies a la minute
        # elles seraient departagees a tort par la duree.
        a = sol(6641, 34732, "ortools", route_a=(0, 1, 0))
        b = sol(6663, 34953, "vroom", route_a=(0, 2, 0))
        best = app.select_best_solution([a, b])
        self.assertEqual(best["sequencer"], "ortools",
                         "dans la fenetre, la distance minimale gagne")
        self.assertEqual(best["distance_m"], 34732.0)

    def test_29_seconds_is_inside_the_window(self):
        a = sol(1000, 50000, "ortools")
        b = sol(1029, 40000, "vroom")
        self.assertEqual(app.select_best_solution([a, b])["sequencer"], "vroom")

    def test_exactly_30_seconds_is_inside_the_window(self):
        a = sol(1000, 50000, "ortools")
        b = sol(1030, 40000, "vroom")
        self.assertEqual(app.select_best_solution([a, b])["sequencer"], "vroom",
                         "la borne de 30 s est INCLUSE")

    def test_31_seconds_is_outside_the_window(self):
        a = sol(1000, 50000, "ortools")
        b = sol(1031, 40000, "vroom")
        self.assertEqual(app.select_best_solution([a, b])["sequencer"], "ortools",
                         "au-dela de 30 s, la duree tranche seule")

    def test_slower_solution_can_win_on_distance(self):
        """Cas explicite du cahier des charges : Vroom est plus rapide de 20 s,
        OR-Tools gagne quand meme par la distance."""
        o = sol(6700, 35000, "ortools")
        v = sol(6680, 35300, "vroom")
        best = app.select_best_solution([o, v])
        self.assertEqual(best["sequencer"], "ortools")

    def test_tie_is_broken_deterministically(self):
        a = sol(1000, 40000, "vroom", partition_key=(2, 3))
        b = sol(1000, 40000, "vroom", partition_key=(1, 2))
        first = app.select_best_solution([a, b])
        second = app.select_best_solution([b, a])
        self.assertEqual(first["partition_key"], (1, 2))
        self.assertEqual(first, second,
                         "l'ordre d'entree ne doit pas changer le gagnant")

    def test_geographic_quality_breaks_equal_duration_and_distance(self):
        a = sol(1000, 40000, "vroom", cut=9)
        b = sol(1000, 40000, "vroom", cut=2)
        self.assertEqual(app.select_best_solution([a, b])["boundary"]["cut_edges"], 2)

    def test_empty_pool_returns_none(self):
        self.assertIsNone(app.select_best_solution([]))

    def test_invalid_solutions_are_discarded_when_a_valid_one_exists(self):
        bad = sol(10, 10, "ortools")
        bad["connected"] = False
        good = sol(9000, 90000, "vroom")
        self.assertEqual(app.select_best_solution([bad, good])["sequencer"], "vroom")


# ------------------------------------------ Test D : selection d'ensemble

class TestGlobalSelection(unittest.TestCase):
    """Test D : la fenetre est calculee sur le MINIMUM GLOBAL, une seule fois.

    Une tolerance appliquee de proche en proche n'est pas transitive : elle
    ferait dependre le gagnant de l'ordre d'examen.
    """

    def test_three_solutions_chained_by_pairs_would_pick_the_wrong_one(self):
        # A ~ B (25 s) et B ~ C (25 s), mais A et C sont distants de 50 s.
        # Une chaine de comparaisons pair a pair laisserait C battre A par la
        # distance ; la fenetre globale l'exclut.
        a = sol(1000, 50000, "ortools", partition_key=(1,))
        b = sol(1025, 45000, "vroom", partition_key=(2,))
        c = sol(1050, 10000, "vroom", partition_key=(3,))
        best = app.select_best_solution([a, b, c])
        self.assertEqual(best["partition_key"], (2,),
                         "C est hors fenetre : 1050 > 1000 + 30")

    def test_result_is_independent_of_input_order(self):
        a = sol(1000, 50000, "ortools", partition_key=(1,))
        b = sol(1025, 45000, "vroom", partition_key=(2,))
        c = sol(1050, 10000, "vroom", partition_key=(3,))
        import itertools
        winners = {tuple(sorted(app.select_best_solution(list(p))["partition_key"]))
                   for p in itertools.permutations([a, b, c])}
        self.assertEqual(len(winners), 1, "gagnant dependant de l'ordre d'entree")

    def test_all_sequencers_compete_in_the_same_pool(self):
        pts = two_lines()
        _, _, meta, _ = run_partition(pts, ordered, shuffled)
        diag = meta["connected"]
        # heuristiques des finalistes + OR-Tools + Vroom : un seul ensemble
        self.assertEqual(diag["connected_solutions_considered"],
                         diag["connected_candidates_selected_diverse"]
                         + diag["connected_candidates_ortools"]
                         + diag["connected_candidates_vroom"])
        self.assertGreater(diag["connected_candidates_ortools"], 0)
        self.assertGreater(diag["connected_candidates_vroom"], 0)


# ------------------------------------------------ Test E : echec de Vroom

class TestVroomFailure(unittest.TestCase):
    """Test E : rate limit, panne reseau ou reponse invalide.
    L'incumbent OR-Tools doit survivre et le run ne doit pas echouer."""

    def _run_with_failing_vroom(self, failure):
        pts = two_lines()
        with _Harness(pts, ordered, None) as h:
            app._resequence_single = failure
            groups, err, meta = app.ortools_partition_ors_matrix_connected(
                pts, 2, len(pts), 0, 0, {})
        return groups, err, meta

    def test_rate_limit_keeps_ortools(self):
        def rate_limited(points, group, s, e, headers):
            return None, None, None            # Vroom renvoie 429
        groups, err, meta = self._run_with_failing_vroom(rate_limited)
        diag = meta["connected"]
        self.assertIsNone(err, "un echec Vroom ne doit pas faire tomber le run")
        self.assertTrue(diag["connected_fallback_used"])
        self.assertIn("vroom", diag["connected_error"].lower())
        self.assertEqual(diag["selected_sequencer"], "ortools")
        self.assertEqual(meta["connected_routes"],
                         [ordered(sorted(groups[0]), 0, 0),
                          ordered(sorted(groups[1]), 0, 0)])
        self.assertFalse(meta["connected_vroom_ok"])

    def test_network_error_keeps_ortools(self):
        def boom(points, group, s, e, headers):
            raise_it = ConnectionError("network down")
            try:
                raise raise_it
            except ConnectionError:
                return None, None, None        # _resequence_single avale deja
        groups, err, meta = self._run_with_failing_vroom(boom)
        self.assertIsNone(err)
        self.assertEqual(meta["connected"]["selected_sequencer"], "ortools")

    def test_invalid_response_keeps_ortools(self):
        def garbage(points, group, s, e, headers):
            return None, None, None            # pas de cle "routes"
        groups, err, meta = self._run_with_failing_vroom(garbage)
        self.assertIsNone(err)
        self.assertEqual(meta["connected"]["selected_sequencer"], "ortools")
        self.assertEqual(meta["connected"]["connected_candidates_vroom"], 0)

    def test_a_valid_incumbent_is_never_lost(self):
        def rate_limited(points, group, s, e, headers):
            return None, None, None
        groups, err, meta = self._run_with_failing_vroom(rate_limited)
        self.assertIsNotNone(groups)
        # aucune perte, aucun doublon : l'incumbent reste une partition valide
        self.assertEqual(set(groups[0]) | set(groups[1]),
                         set(range(1, len(two_lines()))))
        self.assertEqual(set(groups[0]) & set(groups[1]), set())
        self.assertEqual(len(groups[0]), len(groups[1]))
        self.assertTrue(meta["connected"]["connected_partition"])


# ------------------------------------------------- Test F : cache Vroom

class TestVroomCallBudget(unittest.TestCase):
    """Test F : 3 finalistes -> 6 appels Vroom au maximum, et AUCUN apres la
    selection. C'est le rappel final qui ecrasait l'ordre gagnant."""

    def test_at_most_six_vroom_calls_for_three_finalists(self):
        pts = two_lines()
        _, _, meta, h = run_partition(pts, ordered, shuffled)
        diag = meta["connected"]
        self.assertLessEqual(len(h.vroom_calls), 6)
        self.assertEqual(diag["connected_vroom_calls"], len(h.vroom_calls))
        self.assertLessEqual(diag["connected_vroom_calls"],
                             2 * app.CONNECTED_TOP_VROOM)
        # exactement deux appels par finaliste reellement traite
        self.assertEqual(diag["connected_vroom_calls"],
                         2 * min(app.CONNECTED_TOP_VROOM,
                                 diag["connected_candidates_valid"]))

    def test_no_vroom_call_after_selection(self):
        """La partition remonte ses routes : plus personne n'a besoin de
        resequencer, donc le compteur ne bouge plus apres le retour."""
        pts = two_lines()
        _, _, meta, h = run_partition(pts, ordered, shuffled)
        calls_at_return = len(h.vroom_calls)
        self.assertIsNotNone(meta.get("connected_routes"))
        self.assertEqual(len(h.vroom_calls), calls_at_return)

    def test_memorised_routes_are_reused_not_recomputed(self):
        """Quand Vroom gagne, la route retournee est EXACTEMENT celle deja
        obtenue pendant le niveau 3 : aucun second sequencement."""
        pts = two_lines()
        groups, _, meta, h = run_partition(pts, shuffled, ordered)
        self.assertEqual(meta["connected"]["selected_sequencer"], "vroom")
        self.assertEqual(meta["connected_routes"],
                         [ordered(sorted(groups[0]), 0, 0),
                          ordered(sorted(groups[1]), 0, 0)])
        # deux appels par finaliste, jamais un de plus pour le gagnant
        self.assertEqual(len(h.vroom_calls), len(set(h.vroom_calls)),
                         "un meme groupe a ete envoye deux fois a Vroom")

    def test_no_matrix_call_beyond_the_single_load(self):
        pts = two_lines()
        _, _, meta, h = run_partition(pts, ordered, shuffled)
        self.assertEqual(h.matrix_calls, 1,
                         "la matrice ORS ne doit etre chargee qu'une fois")


# -------------------------------------------- Test G : coherence de bout en bout

class _FakeRequest:
    def __init__(self, payload):
        self.json = payload


class TestEndToEndCoherence(unittest.TestCase):
    """Test G : /optimize complet. selected_sequencer, routes JSON, duree,
    distance et payload de carte doivent decrire LES MEMES routes."""

    def _optimize(self, points, ortools_order, vroom_order):
        payload = {
            "points": points,
            "num_vehicles": 2,
            "max_per_vehicle": len(points),
            "start_id": points[0]["id"],
            "end_id": points[0]["id"],
            "strategy": "ortools_ors_matrix_connected",
        }
        saved = (app.request, app.jsonify, app.IMPLEMENTED_STRATEGIES)
        with _Harness(points, ortools_order, vroom_order) as h:
            app.request = _FakeRequest(payload)
            app.jsonify = lambda obj=None, **kw: obj if obj is not None else kw
            app.IMPLEMENTED_STRATEGIES = set(app.IMPLEMENTED_STRATEGIES) | {
                "ortools_ors_matrix_connected"}
            try:
                resp = app.optimize()
            finally:
                app.request, app.jsonify, app.IMPLEMENTED_STRATEGIES = saved
        return resp, h

    def test_G_response_describes_the_selected_routes(self):
        pts = two_lines()
        resp, h = self._optimize(pts, ordered, shuffled)
        self.assertNotIn("error", resp, resp.get("error", ""))
        self.assertEqual(resp["selected_sequencer"], "ortools")
        self.assertEqual(resp["final_selection_reason"], "level2_ortools")

        # 1. les tournees JSON portent bien les identifiants des deux groupes
        by_id = {p["id"]: i for i, p in enumerate(pts)}
        t1 = [by_id[i] for i in resp["tournee_1"]]
        t2 = [by_id[i] for i in resp["tournee_2"]]
        self.assertEqual(t1[0], 0)
        self.assertEqual(t1[-1], 0)
        self.assertEqual(sorted(set(t1[1:-1]) | set(t2[1:-1])),
                         list(range(1, len(pts))))
        self.assertEqual(set(t1[1:-1]) & set(t2[1:-1]), set())

        # 2. duree et distance finales recalculees DEPUIS ces routes
        dur, dist = h.dur, h.dist
        exp_d, exp_k = app._rescore(dur, dist, t1, t2)
        self.assertAlmostEqual(resp["final_total_duration_s"], round(exp_d, 1),
                               places=0)
        # les km par tournee sont arrondis a 2 decimales avant sommation :
        # la tolerance couvre exactement cet arrondi, pas un autre trajet.
        self.assertAlmostEqual(resp["final_total_distance_m"], exp_k,
                               delta=10.0 * len(resp["partition_sizes"]))

        # 3. les km par tournee du Benchmark decrivent les memes routes
        self.assertAlmostEqual(resp["tournee_1_km"],
                               round(app._matrix_route_cost(dist, t1) / 1000, 2),
                               places=2)
        self.assertAlmostEqual(resp["tournee_2_km"],
                               round(app._matrix_route_cost(dist, t2) / 1000, 2),
                               places=2)

        # 4. la post-optimisation n'a pas degrade le gagnant
        self.assertLessEqual(resp["final_total_duration_s"],
                             resp["connected_selected_duration_s"] + 1e-6)

    def test_G_total_vroom_calls_stay_at_six(self):
        pts = two_lines()
        resp, h = self._optimize(pts, ordered, shuffled)
        self.assertLessEqual(resp["api_calls"]["vroom"], 6,
                             "les deux appels de resequencement final sont revenus")
        self.assertEqual(resp["api_calls"]["vroom"], resp["connected_vroom_calls"])

    def test_G_swaps_stay_locked(self):
        pts = two_lines()
        resp, _ = self._optimize(pts, ordered, shuffled)
        self.assertTrue(resp["connected_membership_locked"])
        self.assertEqual(resp["swap_stop_reason"], "connected_partition_locked")
        self.assertEqual(resp["swap_candidates_tested"], 0)
        self.assertEqual(resp["swaps_accepted"], 0)
        self.assertEqual(resp["connected_components_t1"], 1)
        self.assertEqual(resp["connected_components_t2"], 1)
        self.assertTrue(resp["connected_partition"])
        self.assertEqual(resp["partition_solver"], "connected_graph_partition")

    def test_G_vroom_winner_is_not_overwritten_by_ortools(self):
        pts = two_lines()
        resp, h = self._optimize(pts, shuffled, ordered)
        self.assertEqual(resp["selected_sequencer"], "vroom")
        by_id = {p["id"]: i for i, p in enumerate(pts)}
        t1 = [by_id[i] for i in resp["tournee_1"]]
        t2 = [by_id[i] for i in resp["tournee_2"]]
        exp_d, _ = app._rescore(h.dur, h.dist, t1, t2)
        self.assertAlmostEqual(resp["final_total_duration_s"], round(exp_d, 1),
                               places=0)
        self.assertLessEqual(resp["final_total_duration_s"],
                             resp["connected_selected_duration_s"] + 1e-6)

    def test_G_swaps_locked_even_when_vroom_failed(self):
        pts = two_lines()
        resp, _ = self._optimize(pts, ordered, None)
        self.assertEqual(resp["selected_sequencer"], "ortools")
        self.assertEqual(resp["swap_stop_reason"], "connected_partition_locked")
        self.assertEqual(resp["swaps_accepted"], 0)

    def test_G_is_deterministic(self):
        pts = two_lines()
        r1, _ = self._optimize(pts, ordered, shuffled)
        r2, _ = self._optimize(pts, ordered, shuffled)
        for key in ("tournee_1", "tournee_2", "selected_sequencer",
                    "final_selection_reason", "connected_selected_duration_s",
                    "final_total_duration_s"):
            self.assertEqual(r1[key], r2[key], "champ non deterministe : " + key)


# --------------------------------------- non-regression des autres strategies

class TestOtherStrategiesUntouched(unittest.TestCase):

    def setUp(self):
        with open("app.py", encoding="utf-8") as fh:
            self.src = fh.read()

    def test_presequenced_routes_only_for_the_connected_strategy(self):
        """Le court-circuit de _sequence_groups ne doit exister que sous
        ortools_ors_matrix_connected : les autres strategies continuent de
        passer par Vroom exactement comme avant."""
        body = self.src[self.src.index('elif strategy == "ortools_ors_matrix_connected":'):
                        self.src.index("# 3. 2-OPT haversine")]
        self.assertIn('presequenced_routes = (matrix_meta or {}).get("connected_routes")',
                      body)
        terr = self.src[self.src.index('elif strategy == "ortools_ors_matrix":'):
                        self.src.index('elif strategy == "ortools_ors_matrix_connected":')]
        self.assertNotIn("presequenced_routes", terr)

    def test_sequence_groups_still_used_by_other_strategies(self):
        self.assertIn("routes_idx, _seq_dur, vroom_ok, vroom_error = _sequence_groups(",
                      self.src)

    def test_selection_key_has_no_balancing_term(self):
        body = self.src[self.src.index("def _selection_key"):
                        self.src.index("def _solution_tiebreak")]
        self.assertNotIn("abs(", body)

    def test_map_payload_escaping_targets_line_separators(self):
        """L'echappement du payload de carte doit viser U+2028 et U+2029, sous
        forme ECHAPPEE. Ecrits en clair, ces caracteres sont des terminateurs
        de ligne : le litteral de regex se coupe et code.js ne se parse plus.
        Et jamais l'espace ordinaire, qui n'a rien a neutraliser."""
        with open("code.js", encoding="utf-8") as fh:
            js = fh.read()
        block = js[js.index("const safe = payloadJson"):]
        block = block[:block.index(";")]
        self.assertIn("\\u2028/g", block)
        self.assertIn("\\u2029/g", block)
        self.assertNotIn("\u2028", block, "U+2028 ecrit en clair dans la regex")
        self.assertNotIn("\u2029", block, "U+2029 ecrit en clair dans la regex")
        self.assertNotIn(".replace(/ /g", block, "l'espace ordinaire est vise")

    def test_selection_never_rounds_to_minutes(self):
        body = self.src[self.src.index("def select_best_solution"):
                        self.src.index("_SELECTION_REASONS = {")]
        self.assertNotIn("/ 60", body)
        self.assertNotIn("round(", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
