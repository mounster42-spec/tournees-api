"""
Tests du RECENTRAGE ORS-FIRST : reference a cardinalite exacte, budgets par
etape, ordonnancement de la generation, arbitrage de la post-optimisation.

Lancement :
    python -m unittest tests_connected_orsfirst -v

Ce que ces tests verrouillent, et pourquoi :

1. La graine ORS est desormais produite a CARDINALITE EXACTE par le solveur.
   Avant, la capacite valait len(indices) : une repartition 42/18 etait
   admissible, et _normalize_sizes la ramenait ensuite a 30/30 en deplacant les
   points les plus proches du centroide oppose -- un critere HAVERSINE, aveugle
   a la duree ORS. Douze points pouvaient changer de tournee sur un critere
   sans rapport avec l'objectif avant meme le debut de la reparation.

2. Les etapes GARANTIES -- generateur historique, puis reparations ORS -- sont
   examinees avant les sources optionnelles. Sur un run reel, les six sources
   etaient calculees puis jetees sans qu'une seule n'atteigne le banc d'essai :
   le budget expirait pendant la construction de la liste.

3. La post-optimisation Or-opt/2-opt passe sous la MEME regle de selection que
   le reste. Elle n'optimise que la duree ; elle intervenait apres la selection
   et remplacait ses routes sans condition, ce qui pouvait defaire le
   departage a la distance effectue dans la fenetre de trente secondes.

OR-Tools n'est pas requis : le solveur est bouchonne par un faux qui respecte
la capacite qu'on lui passe, ce qui permet de verifier le CONTRAT sans dependre
de l'installation.
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


def scattered_points(n, seed=7):
    """n collectes plus un depot en tete. Generateur congruentiel : la fixture
    est identique d'une machine a l'autre."""
    pts = [P("DEP", 45.4666, 4.3903)]
    x = seed
    for k in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        a = x / 2147483648.0
        x = (1103515245 * x + 12345) % 2147483648
        b = x / 2147483648.0
        pts.append(P(k, 45.400 + 0.09 * a, 4.330 + 0.11 * b))
    return pts


def euclid_matrix(points, scale=1.0):
    xy = app._local_xy(points, list(range(len(points))))
    n = len(points)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = math.hypot(xy[i][0] - xy[j][0],
                                     xy[i][1] - xy[j][1]) * scale
    return m


class _FakeSolver:
    """Faux _solve_cvrp_ortools qui RESPECTE la capacite demandee.

    Il enregistre chaque appel : c'est ainsi qu'on verifie qu'une seule
    resolution part dans le pipeline connexe, et avec quelle capacite.
    `honour_capacity=False` simule un solveur qui rend une repartition
    inadmissible, pour eprouver le repli.
    """

    def __init__(self, honour_capacity=True, fail=False):
        self.calls = []
        self.honour_capacity = honour_capacity
        self.fail = fail

    def __call__(self, cost_matrix, num_vehicles, capacity, start_idx, end_idx,
                 solution_limit=None, time_limit_s=None, detail=None):
        self.calls.append({"capacity": capacity, "num_vehicles": num_vehicles,
                           "time_limit_s": time_limit_s,
                           "matrix_id": id(cost_matrix)})
        if self.fail:
            if detail is not None:
                detail.update({"solver_status": "ROUTING_FAIL", "routes": None,
                               "sizes": None, "solutions": 0,
                               "time_limit_hit": False})
            return None, "no solution"
        nodes = [i for i in range(len(cost_matrix))
                 if i not in (start_idx, end_idx)]
        cut = capacity if self.honour_capacity else max(1, int(len(nodes) * 0.7))
        cut = min(cut, len(nodes))
        groups = [nodes[:cut], nodes[cut:]]
        routes = [[start_idx] + list(g) + [end_idx] for g in groups]
        if detail is not None:
            detail.update({"routes": routes, "sizes": [len(g) for g in groups],
                           "solver_status": "ROUTING_SUCCESS", "solutions": 4,
                           "elapsed_s": 0.01, "time_limit_hit": False,
                           "objective": 1234})
        return groups, None


class _Harness:
    """Matrice ORS, OR-Tools et Vroom bouchonnes autour de la strategie."""

    def __init__(self, points, solver=None, vroom_gain=None):
        self.points = points
        self.dur = euclid_matrix(points)
        self.dist = euclid_matrix(points, scale=1000.0)
        self.vroom_calls = []
        self.matrix_calls = 0
        self.vroom_gain = vroom_gain
        self.solver = solver if solver is not None else _FakeSolver()
        self._saved = {}

    def _matrix(self, points, headers):
        self.matrix_calls += 1
        return self.dur, self.dist, {"ors_matrix": {"stub": True},
                                     "content_hash": "deadbeefcafe"}, None

    def _tsp(self, matrix, group, s, e):
        return app._estimate_group_cost(matrix, group, s, e, True)[1]

    def _vroom(self, points, group, s, e, headers):
        self.vroom_calls.append(tuple(sorted(group)))
        if self.vroom_gain is not None:
            return self.vroom_gain(sorted(group), s, e), 0, 0
        return app._nn_route_matrix(self.dur, sorted(group), s, e), 0, 0

    def __enter__(self):
        for name in ("_build_full_matrix_chunked", "_tsp_order_ortools",
                     "_resequence_single", "ORTOOLS_AVAILABLE",
                     "_solve_cvrp_ortools"):
            self._saved[name] = getattr(app, name)
        app._build_full_matrix_chunked = self._matrix
        app._tsp_order_ortools = self._tsp
        app._resequence_single = self._vroom
        app.ORTOOLS_AVAILABLE = True
        app._solve_cvrp_ortools = self.solver
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(app, name, value)
        return False

    def run(self, num_vehicles=2, cap=None):
        n = len(self.points) - 1
        cap = cap if cap is not None else n
        return app.ortools_partition_ors_matrix_connected(
            self.points, num_vehicles, cap, 0, 0, {})


def _reference(points, solver=None):
    """Construit la reference ORS sur une fixture, hors pipeline complet."""
    idx = [i for i in range(len(points)) if i != 0]
    target = len(idx) // 2
    dur = euclid_matrix(points)
    dist = euclid_matrix(points, scale=1000.0)
    adj, _g = app.build_geo_graph(points, idx, dur_matrix=dur)
    saved = (app.ORTOOLS_AVAILABLE, app._solve_cvrp_ortools)
    app.ORTOOLS_AVAILABLE = True
    app._solve_cvrp_ortools = solver if solver is not None else _FakeSolver()
    try:
        ref = app.build_ors_reference(points, idx, target, dur, dist, adj, 0, 0)
    finally:
        app.ORTOOLS_AVAILABLE, app._solve_cvrp_ortools = saved
    return ref, idx, target, dur, dist, adj


# ------------------------------------- 1 a 3 : cardinalite exacte imposee

class TestExactCardinality(unittest.TestCase):

    def test_01_sixty_collections_split_30_30_without_normalisation(self):
        """60 collectes, deux vehicules : la capacite passee au solveur vaut
        30, la somme des capacites vaut 60, et 30/30 est donc la SEULE
        repartition admissible. Un 42/18 devient infaisable sans qu'aucun terme
        d'equilibrage n'entre dans l'objectif."""
        pts = scattered_points(60)
        solver = _FakeSolver()
        calls = []
        saved = app._normalize_sizes
        app._normalize_sizes = lambda *a, **k: calls.append(1) or saved(*a, **k)
        try:
            ref, idx, target, _d, _k, _a = _reference(pts, solver)
        finally:
            app._normalize_sizes = saved

        self.assertEqual(len(idx), 60)
        self.assertEqual(target, 30)
        self.assertEqual(solver.calls[0]["capacity"], 30,
                         "la capacite ne force pas la cardinalite")
        self.assertEqual(sorted(ref["sizes"]), [30, 30])
        self.assertTrue(ref["cardinality_exact"])
        self.assertFalse(ref["fallback_used"])
        self.assertEqual(ref["selection_reason"], "ors_reference_exact_cardinality")
        self.assertEqual(calls, [],
                         "la voie normale passe encore par _normalize_sizes")

    def test_02_fifty_eight_collections_split_29_29(self):
        pts = scattered_points(58)
        solver = _FakeSolver()
        ref, idx, target, _d, _k, _a = _reference(pts, solver)
        self.assertEqual(len(idx), 58)
        self.assertEqual(target, 29)
        self.assertEqual(solver.calls[0]["capacity"], 29)
        self.assertEqual(sorted(ref["sizes"]), [29, 29])
        self.assertTrue(ref["cardinality_exact"])

    def test_02b_odd_count_deviates_by_one_at_most(self):
        """Nombre impair : l'ecart de 1 est le seul autorise, et il vient de la
        capacite, pas d'un arbitrage."""
        for n in (57, 59, 61):
            pts = scattered_points(n)
            ref, idx, target, _d, _k, _a = _reference(pts)
            self.assertEqual(sum(ref["sizes"]), n, n)
            self.assertLessEqual(max(ref["sizes"]) - min(ref["sizes"]), 1, n)
            self.assertEqual(sorted(ref["sizes"]),
                             sorted((target, n - target)), n)

    def test_02c_exact_capacity_formula(self):
        self.assertEqual(app._exact_capacity(60, 2), 30)
        self.assertEqual(app._exact_capacity(58, 2), 29)
        self.assertEqual(app._exact_capacity(59, 2), 30)
        # somme des capacites : n pour n pair, n+1 pour n impair. C'est ce qui
        # rend toute autre repartition infaisable.
        for n in (56, 57, 58, 59, 60, 61):
            self.assertIn(2 * app._exact_capacity(n, 2), (n, n + 1), n)

    def test_03_a_single_cvrp_runs_in_the_connected_pipeline(self):
        """Une seule resolution initiale. La resolution Haversine a ete
        retiree : elle dupliquait la strategie autonome ortools_haversine au
        prix de plusieurs secondes prises sur les etapes garanties."""
        pts = scattered_points(40)
        solver = _FakeSolver()
        with _Harness(pts, solver=solver) as h:
            groups, err, meta = h.run()
        self.assertIsNone(err)
        self.assertEqual(len(solver.calls), 1,
                         "resolutions CVRP initiales : %d" % len(solver.calls))
        self.assertEqual(solver.calls[0]["capacity"], 20)
        self.assertEqual(solver.calls[0]["time_limit_s"],
                         app.CONNECTED_ORS_CVRP_TIME_LIMIT_S)
        self.assertEqual(sorted(len(g) for g in groups), [20, 20])

    def test_03b_the_removed_haversine_seed_is_gone(self):
        """Le generateur de partitions non contraintes -- deux resolutions,
        dont une Haversine -- n'existe plus."""
        self.assertFalse(hasattr(app, "_unconstrained_partitions"))
        src = open("app.py", encoding="utf-8").read()
        block = src[src.index("def build_ors_reference("):
                    src.index("def _reference_seeds(")]
        self.assertEqual(block.count("_solve_cvrp_ortools("), 2,
                         "voie normale + repli : deux appels au plus")
        self.assertIn("_exact_capacity(", block)

    def test_04_standalone_haversine_strategy_is_untouched(self):
        src = open("app.py", encoding="utf-8").read()
        body = src[src.index("def ortools_partition_haversine"):
                   src.index("# 4c. MATRICE ORS COMPLETE")]
        self.assertIn("_solve_cvrp_ortools(", body)
        self.assertNotIn("connected", body)
        self.assertNotIn("time_limit_s", body,
                         "la strategie autonome ne doit pas changer de budget")
        self.assertNotIn("_exact_capacity", body)

    def test_04b_kmeans_and_linear_territorial_are_untouched(self):
        src = open("app.py", encoding="utf-8").read()
        kmeans = src[src.index("def kmeans_partition("):
                     src.index("def _build_haversine_matrix(")]
        self.assertNotIn("connected", kmeans)
        self.assertNotIn("build_ors_reference", kmeans)
        terr = src[src.index("def ortools_partition_ors_matrix("):
                   src.index("# 4e. PARTITION CONNEXE")]
        self.assertNotIn("build_ors_reference", terr)
        self.assertNotIn("_StageClock", terr)
        self.assertIn('"territorial_method": "sweep_line_projection"', src)


# ------------------------------- 5 a 8 : ordres conserves et reference

class TestReferenceIsKept(unittest.TestCase):

    def test_05_cvrp_orders_are_preserved(self):
        """Plusieurs secondes de recherche guidee ne doivent plus etre payees
        pour n'en garder que l'affectation."""
        pts = scattered_points(40)
        ref, _i, _t, _d, _k, _a = _reference(pts)
        self.assertTrue(ref["available"])
        self.assertEqual(len(ref["routes"]), 2)
        for route, group in ((ref["route_a"], ref["group_a"]),
                             (ref["route_b"], ref["group_b"])):
            self.assertEqual(route[0], 0, "la route ne part pas du depot")
            self.assertEqual(route[-1], 0, "la route ne finit pas au depot")
            self.assertEqual(sorted(i for i in route if i != 0), sorted(group),
                             "la route ne decrit pas son groupe")

    def test_05b_solver_diagnostics_are_reported(self):
        pts = scattered_points(40)
        ref, _i, _t, _d, _k, _a = _reference(pts)
        self.assertEqual(ref["solver_status"], "ROUTING_SUCCESS")
        self.assertEqual(ref["solutions"], 4)
        self.assertFalse(ref["time_limit_hit"])
        self.assertIsInstance(ref["solve_ms"], int)

    def test_06_reference_metrics_are_rescored_from_the_kept_routes(self):
        """La duree et la distance de reference sont recalculees sur la MEME
        matrice ORS que tout le reste, depuis les ordres conserves. Jamais un
        cout interne du solveur."""
        pts = scattered_points(40)
        ref, _i, _t, dur, dist, _a = _reference(pts)
        expected = app._rescore(dur, dist, ref["route_a"], ref["route_b"])
        self.assertAlmostEqual(ref["duration_s"], expected[0], places=6)
        self.assertAlmostEqual(ref["distance_m"], expected[1], places=6)
        # l'affinage matriciel ne peut pas degrader la duree brute
        self.assertLessEqual(ref["duration_s"], ref["duration_raw_s"] + 1e-6)

    def test_07_connectivity_penalty_is_the_gap_to_the_reference(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
        self.assertIsNone(err)
        d = meta["connected"]
        self.assertTrue(d["connected_ors_reference_available"])
        self.assertIsNotNone(d["connectivity_penalty_duration_s"])
        # tolerance de 0,11 s : les trois champs sont arrondis au dixieme
        # chacun de leur cote, l'ecart de double arrondi est borne.
        self.assertAlmostEqual(
            d["connectivity_penalty_duration_s"],
            d["connected_selected_duration_s"]
            - d["connected_ors_reference_duration_s"], delta=0.11)
        self.assertAlmostEqual(
            d["connectivity_penalty_distance_m"],
            d["connected_selected_distance_m"]
            - d["connected_ors_reference_distance_m"], delta=0.11)

    def test_07b_penalty_reliability_is_stated_not_assumed(self):
        """La penalite n'est un plancher que si la reference n'a ete ni
        tronquee par sa limite de temps, ni fabriquee par le repli. Le dire
        vaut mieux que laisser croire a une garantie qui n'existe pas."""
        pts = scattered_points(40)
        with _Harness(pts, solver=_FakeSolver(honour_capacity=False)) as h:
            _g, err, meta = h.run()
        self.assertIsNone(err)
        d = meta["connected"]
        self.assertTrue(d["connected_ors_reference_fallback_used"])
        self.assertFalse(d["connectivity_penalty_reliable"])
        self.assertIn("fallback_used", d["connectivity_penalty_note"])

    def test_08_the_exact_cvrp_fallback_is_explicit_and_diagnosed(self):
        """Repli seulement si la voie normale echoue REELLEMENT, et il est
        clairement distingue : sans cela, une reference batie sur une
        normalisation geographique se ferait passer pour une reference a
        cardinalite exacte."""
        pts = scattered_points(40)
        solver = _FakeSolver(honour_capacity=False)
        ref, _i, target, _d, _k, _a = _reference(pts, solver)
        self.assertTrue(ref["available"])
        self.assertTrue(ref["fallback_used"])
        self.assertFalse(ref["cardinality_exact"])
        self.assertEqual(ref["selection_reason"],
                         "ors_reference_normalized_fallback")
        self.assertTrue(ref["error"], "un repli non diagnostique")
        # meme par le repli, la cardinalite finale reste exacte
        self.assertEqual(sorted(ref["sizes"]), sorted((target, 40 - target)))
        self.assertEqual(len(solver.calls), 2, "la voie normale a ete sautee")

    def test_08b_a_failing_solver_leaves_the_reference_unavailable(self):
        pts = scattered_points(40)
        ref, _i, _t, _d, _k, _a = _reference(pts, _FakeSolver(fail=True))
        self.assertFalse(ref["available"])
        self.assertTrue(ref["error"])
        self.assertIsNone(ref["duration_s"])

    def test_08c_a_failing_reference_does_not_break_the_run(self):
        pts = scattered_points(40)
        with _Harness(pts, solver=_FakeSolver(fail=True)) as h:
            groups, err, meta = h.run()
        self.assertIsNone(err, "la strategie tombe quand la reference manque")
        self.assertEqual(sorted(len(g) for g in groups), [20, 20])
        d = meta["connected"]
        self.assertFalse(d["connected_ors_reference_available"])
        self.assertIsNone(d["connectivity_penalty_duration_s"])
        self.assertIn("no ORS reference", d["connectivity_penalty_note"])


# ------------------------- 9 a 11 : ordonnancement et budgets par etape

class TestStageScheduling(unittest.TestCase):

    def test_09_legacy_candidates_are_always_examined(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
        self.assertIsNone(err)
        d = meta["connected"]
        self.assertGreater(d["connected_candidates_legacy"], 0,
                           "aucune candidate historique n'a atteint consider()")
        self.assertGreaterEqual(d["connected_legacy_finalists"], 1)

    def test_09b_legacy_runs_before_any_optional_source(self):
        """L'ordre est un CONTRAT : une source optionnelle placee avant les
        etapes garanties peut consommer leur budget."""
        src = open("app.py", encoding="utf-8").read()
        body = src[src.index("def generate_connected_candidates("):
                   src.index("def _fallback_connected_candidates(")]
        i_legacy = body.index('clock.begin("legacy"')
        i_repair = body.index('clock.begin("ors_repair"')
        i_resid = body.index('clock.begin("residual"')
        self.assertLess(i_legacy, i_repair)
        self.assertLess(i_repair, i_resid)

    def test_10_ors_repairs_have_a_reserved_budget_and_reach_consider(self):
        """Le defaut principal : les douze reparations etaient calculees puis
        jetees sans qu'une seule n'atteigne le banc d'essai."""
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
        self.assertIsNone(err)
        d = meta["connected"]
        self.assertGreater(d["connected_ors_repair_candidates_raw"], 0,
                           "la source ORS-first n'a rien produit")
        self.assertGreaterEqual(d["connected_ors_repair_candidates_unique"], 1)
        self.assertIn("ors_repair", d["connected_stage_timings_ms"])

    def test_10b_ors_repair_is_a_generator_examined_on_the_fly(self):
        idx = list(range(1, 21))
        pts = scattered_points(20)
        dur = euclid_matrix(pts)
        hav = app._build_haversine_matrix(pts)
        adj, _g = app.build_geo_graph(pts, idx, dur_matrix=dur)
        gen = app._ors_repair_candidates(pts, idx, 10, adj, dur, hav, 0, 0, 99,
                                         seeds=[("ors_reference", idx[:14],
                                                 idx[14:])])
        self.assertTrue(hasattr(gen, "__next__"),
                        "la source construit encore une liste complete")

    def test_10c_ors_repair_slots_are_reserved_among_finalists(self):
        self.assertGreaterEqual(app.CONNECTED_ORS_REPAIR_FINALIST_SLOTS, 1)
        self.assertLessEqual(app.CONNECTED_ORS_REPAIR_FINALIST_SLOTS
                             + app.CONNECTED_LEGACY_FINALIST_SLOTS,
                             app.CONNECTED_ORTOOLS_FINALISTS)

    def test_11_an_expired_residual_source_never_loses_the_incumbent(self):
        """Budget residuel ramene a l'instant present : les sources
        optionnelles ne tournent pas, les etapes garanties restent intactes."""
        pts = scattered_points(40)
        idx = [i for i in range(len(pts)) if i != 0]
        dur = euclid_matrix(pts)
        hav = app._build_haversine_matrix(pts)
        adj, gmeta = app.build_geo_graph(pts, idx, dur_matrix=dur)
        saved = app.CONNECTED_RESIDUAL_SOURCES_BUDGET_S
        app.CONNECTED_RESIDUAL_SOURCES_BUDGET_S = 0.0
        try:
            cands, stats = app.generate_connected_candidates(
                pts, idx, len(idx) // 2, adj, dur, hav, 0, 0,
                tree_edges=gmeta["tree_edges"])
        finally:
            app.CONNECTED_RESIDUAL_SOURCES_BUDGET_S = saved
        self.assertGreater(len(cands), 0, "plus aucune candidate")
        self.assertTrue(any(c["legacy"] for c in cands),
                        "l'incumbent historique a ete perdu")
        self.assertEqual(stats["by_source"].get("sweep", 0), 0,
                         "une source residuelle a tourne malgre le budget nul")

    def test_11b_stage_timings_and_exhaustion_are_reported(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
        self.assertIsNone(err)
        d = meta["connected"]
        timings = d["connected_stage_timings_ms"]
        for stage in ("ors_cvrp", "legacy", "ors_repair", "residual",
                      "prescore", "ortools_finalists", "vroom_finalists"):
            self.assertIn(stage, timings, "etape non chronometree : " + stage)
            self.assertIsInstance(timings[stage], int)
        self.assertIsInstance(d["connected_stage_budget_exhausted"], list)
        text = app._stage_timings_text(timings)
        self.assertIn("legacy=", text)
        self.assertEqual(text, ";".join(sorted(text.split(";"))))

    def test_11c_stage_budgets_are_env_configurable(self):
        src = open("app.py", encoding="utf-8").read()
        for name in ("CONNECTED_ORS_CVRP_TIME_LIMIT_S",
                     "CONNECTED_LEGACY_BUDGET_S",
                     "CONNECTED_ORS_REPAIR_BUDGET_S",
                     "CONNECTED_RESIDUAL_SOURCES_BUDGET_S",
                     "CONNECTED_PRESCORE_BUDGET_S",
                     "CONNECTED_RESIDUAL_PER_SOURCE",
                     "CONNECTED_PRESCORE_REFINE_MAX",
                     "CONNECTED_ORS_REPAIR_FINALIST_SLOTS"):
            self.assertIn('_env_int("%s"' % name, src)
            self.assertIsInstance(getattr(app, name), int)
        self.assertEqual(app.CONNECTED_ORS_CVRP_TIME_LIMIT_S, 8)
        self.assertEqual(app.CONNECTED_LEGACY_BUDGET_S, 5)
        self.assertEqual(app.CONNECTED_ORS_REPAIR_BUDGET_S, 10)
        self.assertEqual(app.CONNECTED_RESIDUAL_SOURCES_BUDGET_S, 5)
        self.assertEqual(app.CONNECTED_PRESCORE_BUDGET_S, 3)

    def test_11d_the_initial_cvrp_no_longer_eats_the_whole_budget(self):
        """ORTOOLS_TIME_LIMIT_S valait 25 s, soit exactement le budget de
        generation : une seule resolution pouvait le consommer entierement."""
        self.assertLess(app.CONNECTED_ORS_CVRP_TIME_LIMIT_S,
                        app.CONNECTED_MAX_GENERATION_S)
        self.assertLessEqual(app.CONNECTED_LEGACY_BUDGET_S
                             + app.CONNECTED_ORS_REPAIR_BUDGET_S
                             + app.CONNECTED_RESIDUAL_SOURCES_BUDGET_S,
                             app.CONNECTED_MAX_GENERATION_S)

    def test_11e_stage_clock_keeps_what_is_already_acquired(self):
        clock = app._StageClock(time_module_now() + 100)
        clock.begin("a", 0.0)
        self.assertTrue(clock.expired())
        clock.end()
        self.assertIn("a", clock.timings_ms)
        self.assertIn("a", clock.exhausted)
        clock.begin("b", 100.0)
        self.assertFalse(clock.expired())
        clock.end()
        self.assertNotIn("b", clock.exhausted)


def time_module_now():
    import time
    return time.time()


# ------------------------------- 12 a 14 : selection des 12 finalistes

def _rough(specs):
    """specs : (partition_key, duration, seed, legacy, ors_repair)."""
    out = []
    for pkey, dur, seed, legacy, repair in specs:
        out.append({
            "group_a": list(pkey[0]), "group_b": list(pkey[1]), "seed": seed,
            "partition_key": pkey, "connected": True, "cardinality_ok": True,
            "components_total": 0, "duration_s": float(dur),
            "distance_m": float(dur) * 10.0,
            "boundary": {"cut_edges": 1, "enclave_points": 0,
                         "cross_neighbors": 2, "cut_length_m": 10.0},
            "legacy": legacy, "ors_repair": repair, "refined": False,
        })
    return out


def _field(n_plain=30, n_legacy=8, n_repair=6):
    """Un champ ou les places reservees suffisent a saturer l'ancien quota."""
    specs = []
    for k in range(n_plain):
        specs.append((((k,), (100 + k,)), 1000 + k, "sweep", False, False))
    for k in range(n_legacy):
        specs.append((((200 + k,), (300 + k,)), 90000 + k,
                      "legacy:sweep_%d" % k, True, False))
    for k in range(n_repair):
        specs.append((((400 + k,), (500 + k,)), 95000 + k, "ors_repair",
                      False, True))
    return _rough(specs)


class TestFinalistSelection(unittest.TestCase):

    def test_12_the_best_candidate_is_always_a_finalist(self):
        """Defaut corrige : les places reservees etaient servies AVANT le
        quota de score, dont la boucle sortait alors immediatement. Avec six
        places historiques et un quota de six, aucune candidate n'etait plus
        retenue au score -- la meilleure du champ pouvait ne jamais atteindre
        OR-Tools alors que douze places etaient disponibles."""
        scored = _field()
        best = min(scored, key=app._selection_key)
        protected = tuple(c["partition_key"] for c in scored if c["legacy"])[:6]
        preferred = tuple(c["partition_key"] for c in scored
                          if c["ors_repair"])[:3]
        chosen, _ = app.select_diverse_finalists(
            scored, 12, protected_keys=protected, preferred_keys=preferred)
        keys = [c["partition_key"] for c in chosen]
        self.assertIn(best["partition_key"], keys,
                      "la meilleure candidate globale a ete evincee")
        self.assertEqual(len(chosen), 12, "des places sont restees vides")

    def test_12b_score_slots_survive_the_reservations(self):
        """Les places reservees ne consomment PAS le quota de score."""
        scored = _field()
        protected = tuple(c["partition_key"] for c in scored if c["legacy"])[:6]
        chosen, _ = app.select_diverse_finalists(scored, 12,
                                                 protected_keys=protected)
        plain = [c for c in chosen if not c["legacy"] and not c["ors_repair"]]
        self.assertGreaterEqual(len(plain), 2,
                                "aucune candidate retenue au score")

    def test_13_protected_slots_do_not_block_a_better_candidate(self):
        scored = _field()
        ordered = sorted(scored, key=app._selection_key)
        protected = tuple(c["partition_key"] for c in scored if c["legacy"])[:6]
        chosen, _ = app.select_diverse_finalists(scored, 12,
                                                 protected_keys=protected)
        keys = [c["partition_key"] for c in chosen]
        self.assertIn(ordered[0]["partition_key"], keys)
        self.assertTrue(any(k in keys for k in protected),
                        "les places reservees ne servent plus")

    def test_13b_preferred_ors_repairs_reach_the_bench(self):
        scored = _field()
        preferred = tuple(c["partition_key"] for c in scored
                          if c["ors_repair"])[:3]
        chosen, _ = app.select_diverse_finalists(scored, 12,
                                                 preferred_keys=preferred)
        keys = [c["partition_key"] for c in chosen]
        self.assertTrue(any(k in keys for k in preferred),
                        "aucune reparation ORS n'atteint OR-Tools")

    def test_14_finalists_are_deterministic_and_unique(self):
        scored = _field()
        protected = tuple(c["partition_key"] for c in scored if c["legacy"])[:6]
        preferred = tuple(c["partition_key"] for c in scored
                          if c["ors_repair"])[:3]
        a, da = app.select_diverse_finalists(scored, 12,
                                             protected_keys=protected,
                                             preferred_keys=preferred)
        b, db = app.select_diverse_finalists(scored, 12,
                                             protected_keys=protected,
                                             preferred_keys=preferred)
        self.assertEqual([c["partition_key"] for c in a],
                         [c["partition_key"] for c in b])
        self.assertEqual(da, db)
        keys = [c["partition_key"] for c in a]
        self.assertEqual(len(keys), len(set(keys)),
                         "douze sequencements de la meme appartenance")
        self.assertLessEqual(len(a), 12)

    def test_14b_no_balancing_term_entered_the_selection(self):
        src = open("app.py", encoding="utf-8").read()
        body = src[src.index("def select_diverse_finalists("):
                   src.index("def _min_pairwise_difference(")]
        for forbidden in ("abs(", "makespan", "balance", "variance"):
            self.assertNotIn(forbidden, body)


# ------------------------------------------- 15 : proxy affine, hors reseau

class TestRefinedProxy(unittest.TestCase):

    def test_15_the_refined_proxy_makes_no_network_call(self):
        src = open("app.py", encoding="utf-8").read()
        body = src[src.index("    # --- proxy AFFINE"):
                   src.index("    # Rang proxy AFFINE")]
        for forbidden in ("_post_matrix", "_post_vroom", "_resequence_single",
                          "_fetch_ors_matrix", "_build_full_matrix_chunked",
                          "requests."):
            self.assertNotIn(forbidden, body, forbidden)
        self.assertIn("CONNECTED_PRESCORE_BUDGET_S", body)

    def test_15b_the_refined_proxy_runs_and_is_reported(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
            calls_before = h.matrix_calls
        self.assertIsNone(err)
        d = meta["connected"]
        self.assertGreater(d["connected_prescore_refined"], 0)
        self.assertEqual(calls_before, 1, "un appel Matrix de plus")

    def test_15c_three_ranks_are_exposed_for_comparison(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
        self.assertIsNone(err)
        d = meta["connected"]
        for field in ("connected_winner_proxy_rank_rough",
                      "connected_winner_proxy_rank_refined",
                      "connected_winner_ortools_rank"):
            self.assertIn(field, d)
        self.assertIsNotNone(d["connected_winner_proxy_rank_rough"])
        self.assertIsNotNone(d["connected_winner_proxy_rank_refined"])

    def test_15d_refined_routes_are_reused_not_recomputed(self):
        src = open("app.py", encoding="utf-8").read()
        body = src[src.index("    scored = []"):
                   src.index("    # --- niveau 2 : OR-Tools")]
        self.assertIn('base.get("refined")', body)


# ------------------- 16 a 18 : la post-optimisation passe sous contrat

def _metrics(after_s, after_km, before_s, before_km):
    return [{"km": after_km, "min": round(after_s / 60, 1),
             "duration_s": after_s, "before_km": before_km,
             "before_duration_s": before_s}]


class TestPostOptimizationContract(unittest.TestCase):

    def test_16_a_worse_post_optimization_is_rejected(self):
        """Un gain d'une seconde paye par trois cents metres : l'ecart de duree
        tient dans la fenetre de trente secondes, la distance departage, donc
        l'ordre d'avant gagne."""
        before = [[0, 1, 2, 0]]
        after = [[0, 2, 1, 0]]
        metrics = _metrics(after_s=5999.0, after_km=35.3,
                           before_s=6000.0, before_km=35.0)
        routes, kept_metrics, kept, note = app.arbitrate_post_optimization(
            before, after, metrics, "ortools")
        self.assertFalse(kept)
        self.assertEqual(note, "or2opt_rejected")
        self.assertEqual(routes, before)
        self.assertEqual(kept_metrics[0]["duration_s"], 6000.0)
        self.assertEqual(kept_metrics[0]["km"], 35.0,
                         "les metriques ne decrivent pas les routes rendues")

    def test_17_a_better_post_optimization_is_kept(self):
        """Trente et une secondes de gain : hors fenetre, la duree tranche."""
        before = [[0, 1, 2, 0]]
        after = [[0, 2, 1, 0]]
        metrics = _metrics(after_s=5969.0, after_km=35.3,
                           before_s=6000.0, before_km=35.0)
        routes, kept_metrics, kept, note = app.arbitrate_post_optimization(
            before, after, metrics, "ortools")
        self.assertTrue(kept)
        self.assertEqual(note, "or2opt_kept")
        self.assertEqual(routes, after)
        self.assertEqual(kept_metrics[0]["duration_s"], 5969.0)

    def test_17b_equal_duration_and_shorter_distance_is_kept(self):
        before = [[0, 1, 2, 0]]
        after = [[0, 2, 1, 0]]
        metrics = _metrics(after_s=6000.0, after_km=34.0,
                           before_s=6000.0, before_km=35.0)
        _r, _m, kept, _n = app.arbitrate_post_optimization(before, after,
                                                           metrics, "ortools")
        self.assertTrue(kept)

    def test_17c_a_strict_tie_keeps_the_selected_order(self):
        """Egalite parfaite : rien ne justifie de remplacer la gagnante."""
        before = [[0, 1, 2, 0]]
        after = [[0, 2, 1, 0]]
        metrics = _metrics(after_s=6000.0, after_km=35.0,
                           before_s=6000.0, before_km=35.0)
        routes, _m, kept, _n = app.arbitrate_post_optimization(
            before, after, metrics, "ortools")
        self.assertFalse(kept)
        self.assertEqual(routes, before)

    def test_17d_a_membership_change_is_refused_outright(self):
        """Or-opt et 2-opt reordonnent, ils ne deplacent aucun point entre
        tournees. Si l'appartenance bouge, on garde l'ordre d'avant."""
        before = [[0, 1, 2, 0]]
        after = [[0, 1, 3, 0]]
        metrics = _metrics(5000.0, 30.0, 6000.0, 35.0)
        routes, _m, kept, note = app.arbitrate_post_optimization(
            before, after, metrics, "ortools")
        self.assertFalse(kept)
        self.assertEqual(note, "membership_changed")
        self.assertEqual(routes, before)

    def test_17e_incomplete_metrics_keep_the_previous_behaviour(self):
        before = [[0, 1, 2, 0]]
        after = [[0, 2, 1, 0]]
        metrics = [{"km": 35.0, "min": None, "duration_s": None,
                    "before_km": None, "before_duration_s": None}]
        routes, _m, kept, note = app.arbitrate_post_optimization(
            before, after, metrics, "ortools")
        self.assertTrue(kept)
        self.assertEqual(note, "metrics_incomplete")
        self.assertEqual(routes, after)

    def test_17f_the_arbitration_is_wired_for_the_connected_strategy_only(self):
        src = open("app.py", encoding="utf-8").read()
        body = src[src.index("        # 4bis."):
                   src.index("    # 5. POST-PROCESSING")]
        self.assertIn("presequenced_routes is not None", body)
        self.assertIn("arbitrate_post_optimization(", body)

    def test_18_final_metrics_always_describe_the_returned_routes(self):
        """Invariant : quel que soit le verdict, les metriques rendues sont
        celles des routes rendues."""
        before = [[0, 1, 2, 0]]
        after = [[0, 2, 1, 0]]
        for a_s, a_km, b_s, b_km, expect in (
                (5999.0, 35.3, 6000.0, 35.0, "before"),
                (5969.0, 35.3, 6000.0, 35.0, "after"),
                (6000.0, 34.0, 6000.0, 35.0, "after")):
            routes, metrics, _k, _n = app.arbitrate_post_optimization(
                before, after, _metrics(a_s, a_km, b_s, b_km), "ortools")
            want_s, want_km = (b_s, b_km) if expect == "before" else (a_s, a_km)
            self.assertEqual(routes, before if expect == "before" else after)
            self.assertEqual(metrics[0]["duration_s"], want_s)
            self.assertEqual(metrics[0]["km"], want_km)

    def test_18b_selected_metrics_match_the_returned_connected_routes(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
        self.assertIsNone(err)
        d = meta["connected"]
        ra, rb = meta["connected_routes"]
        dur, dist = app._rescore(h.dur, h.dist, ra, rb)
        self.assertAlmostEqual(d["connected_selected_duration_s"],
                               round(dur, 1), places=1)
        self.assertAlmostEqual(d["connected_selected_distance_m"],
                               round(dist, 1), places=1)


# ------------------- 19 a 25 : invariants que le lot ne doit pas casser

class TestPreservedInvariants(unittest.TestCase):

    def test_19_ortools_wins_when_it_is_better(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:           # Vroom rend un simple plus-proche-voisin
            _g, err, meta = h.run()
        self.assertIsNone(err)
        d = meta["connected"]
        self.assertEqual(d["selected_sequencer"], "ortools")
        self.assertLessEqual(d["ortools_total_duration_s"],
                             d["vroom_total_duration_s"])

    def test_20_vroom_wins_when_it_is_better(self):
        pts = scattered_points(40)

        def _better(group, s, e):
            # ordre affine : strictement meilleur que le plus-proche-voisin
            base = app._nn_route_matrix(euclid_matrix(pts), sorted(group), s, e)
            return app._two_opt_matrix(euclid_matrix(pts), base)

        h = _Harness(pts, vroom_gain=_better)
        with h:
            saved = app._tsp_order_ortools
            app._tsp_order_ortools = lambda m, g, s, e: app._nn_route_matrix(
                m, sorted(g), s, e)
            try:
                _g, err, meta = h.run()
            finally:
                app._tsp_order_ortools = saved
        self.assertIsNone(err)
        d = meta["connected"]
        self.assertLessEqual(d["vroom_total_duration_s"],
                             d["ortools_total_duration_s"])
        self.assertEqual(d["selected_sequencer"], "vroom")
        self.assertEqual(d["final_selection_reason"], "level3_vroom")

    def test_21_at_most_six_vroom_calls(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
            calls = len(h.vroom_calls)
        self.assertIsNone(err)
        self.assertLessEqual(calls, 6, "budget Vroom depasse : %d" % calls)
        self.assertLessEqual(meta["connected"]["connected_vroom_calls"], 6)
        self.assertEqual(app.CONNECTED_VROOM_FINALISTS, 3)
        self.assertEqual(app.CONNECTED_ORTOOLS_FINALISTS, 12)

    def test_22_membership_stays_locked_no_inter_route_swap(self):
        pts = scattered_points(40)
        with _Harness(pts) as h:
            groups, err, meta = h.run()
        self.assertIsNone(err)
        self.assertTrue(meta["connected"]["connected_membership_locked"])
        ra, rb = meta["connected_routes"]
        self.assertEqual(sorted(i for i in ra if i != 0), sorted(groups[0]))
        self.assertEqual(sorted(i for i in rb if i != 0), sorted(groups[1]))

    def test_23_cardinality_union_uniqueness_and_connectivity(self):
        pts = scattered_points(40)
        idx = [i for i in range(len(pts)) if i != 0]
        with _Harness(pts) as h:
            groups, err, meta = h.run()
        self.assertIsNone(err)
        ga, gb = groups
        self.assertEqual(len(ga), 20)
        self.assertEqual(len(gb), 20)
        self.assertEqual(sorted(ga + gb), sorted(idx), "point perdu ou double")
        self.assertEqual(set(ga) & set(gb), set())
        d = meta["connected"]
        self.assertEqual(d["connected_components_t1"], 1)
        self.assertEqual(d["connected_components_t2"], 1)
        self.assertTrue(d["connected_partition"])

    def test_24_the_sequencer_fix_and_selection_engine_are_intact(self):
        src = open("app.py", encoding="utf-8").read()
        self.assertIn("def select_best_solution(", src)
        self.assertIn("def _rescore(", src)
        self.assertIn('meta["connected_routes"]', src)
        self.assertIn("presequenced_routes", src)
        body = src[src.index("def select_best_solution("):
                   src.index("_SELECTION_REASONS = {")]
        self.assertIn("best_duration + tie_seconds", body)
        for forbidden in ("abs(", "makespan", "balance", "variance"):
            self.assertNotIn(forbidden, body)

    def test_25_d1_d2_d3_probes_are_intact(self):
        src = open("app.py", encoding="utf-8").read()
        # D-2 : la sonde et ses champs
        for field in ("reordered_by_or2opt", "order_survived_swaps",
                      "pointsets_changed_by_swaps", "swaps_ran"):
            self.assertIn(field, src, "sonde D-2 amputee : " + field)
        self.assertIn("d2_probe", src)
        # D-3 : pilotage et compteurs de swaps
        for field in ("max_swap_candidates", "swap_max_consecutive_fails",
                      "swap_candidates_tested", "swaps_accepted",
                      "swap_resequence_cache_hits", "swap_vroom_calls_saved",
                      "swap_stop_reason"):
            self.assertIn(field, src, "lot D-3 ampute : " + field)
        # D-1 : pilotage de solution_limit
        for field in ("ortools_solution_limit",
                      "ortools_solution_limit_requested",
                      "ortools_solution_limit_override_applied"):
            self.assertIn(field, src, "lot D-1 ampute : " + field)

    def test_25b_new_benchmark_columns_are_appended_at_the_end(self):
        """Aucune colonne existante ne bouge : les nouvelles sont ajoutees
        apres d2_probe, la derniere du bloc precedent."""
        src = open("app.py", encoding="utf-8").read()
        start = src.index('        "d2_probe": d2_probe,')
        block = src[start:src.index('    for v in range(num_vehicles):', start)]
        for field in ("connected_ors_reference_available",
                      "connected_ors_reference_duration_s",
                      "connected_ors_reference_distance_m",
                      "connected_ors_reference_sizes",
                      "connected_ors_reference_components_t1",
                      "connected_ors_reference_components_t2",
                      "connected_ors_reference_time_limit_hit",
                      "connected_ors_reference_fallback_used",
                      "connectivity_penalty_duration_s",
                      "connectivity_penalty_distance_m",
                      "connected_stage_timings_ms",
                      "connected_stage_budget_exhausted",
                      "connected_ors_repair_candidates_raw",
                      "connected_ors_repair_candidates_unique",
                      "connected_ors_repair_reached_ortools"):
            self.assertIn('"%s"' % field, block,
                          "colonne absente ou mal placee : " + field)
        # les colonnes conservees restent presentes
        for field in ("connected_matrix_hash", "connected_per_source_text",
                      "connected_generation_expired_after",
                      "selected_sequencer", "final_selection_reason",
                      "final_total_duration_s", "final_total_distance_m"):
            self.assertIn('"%s"' % field, src, "colonne perdue : " + field)

    def test_25c_the_arbitration_costs_no_extra_api_call(self):
        """L'arbitrage compare les deux ordres sur les MEMES matrices
        par-tournee deja recuperees par l'etape 4 : aucune requete de plus, et
        aucune comparaison entre estimateurs differents."""
        src = open("app.py", encoding="utf-8").read()
        body = src[src.index("def arbitrate_post_optimization("):
                   src.index("# =========================\n# 7. API")]
        for forbidden in ("_post_matrix", "_post_vroom", "_fetch_ors_matrix",
                          "_resequence_single", "requests."):
            self.assertNotIn(forbidden, body, forbidden)
        self.assertIn("before_duration_s", body)
        self.assertIn("select_best_solution(", body)
        # la strategie connexe ne doit exporter aucune matrice dans sa meta :
        # elle finirait dans la reponse JSON.
        pts = scattered_points(40)
        with _Harness(pts) as h:
            _g, err, meta = h.run()
        self.assertIsNone(err)
        self.assertEqual([k for k in meta if k.startswith("_")], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
